import torch
import numpy as np
import argparse
import os
import wandb

from src.models.feature_extractor import MnistFeatureExtractor
from src.models.feature_extractor_rgb import CifarFeatureExtractor, CifarWrapper
from src.models.bayesian_layers import BayesianLinearAnalytic, BayesianLinearMFVI
from src.models.ood_detector import MahalanobisScorer, GMMScorer
from src.models.train_utils import train_feature_extractor
from src.utils.data_loader_ood import CombinedDataManager
from src.core.active_loop_ood import RegressionActiveLoopOOD

def main():
    parser = argparse.ArgumentParser(description='OOD-Aware Active Learning')
    parser.add_argument('--method', type=str, default='analytic', choices=['analytic', 'mfvi'])
    parser.add_argument('--acquisition', type=str, default='robust_pv', 
                        choices=['random', 'predictive_variance', 'robust_pv', 'robust_pv_gmm', 'robust_pv_gmm_filtered', 'sin_only'])
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default='mnist-active-learning-ood')
    parser.add_argument('--id_dataset', type=str, default='mnist', help='In-Distribution Dataset')
    parser.add_argument('--ood_dataset', type=str, default='fashion_mnist', help='Out-of-Distribution Dataset')
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize WANDB
    run_name = f"{args.id_dataset}_{args.ood_dataset}_{args.method}_{args.acquisition}"
    wandb.init(project=args.wandb_project, name=run_name, config=args)
    
    # --- 1. Data Setup ---
    print(f"Initializing OOD Data Manager ({args.id_dataset} vs {args.ood_dataset})...")
    data_manager = CombinedDataManager(
        id_name=args.id_dataset, 
        ood_name=args.ood_dataset,
        pool_size=10000, 
        ood_ratio=0.5
    )
    
    # --- 2. Feature Extractor Training ---
    print("Training Feature Extractor (on InD)...")
    if data_manager.mode == 'grayscale':
        feature_extractor = MnistFeatureExtractor().to(device)
    else:
        # RGB Models (CIFAR/SVHN)
        base_fe = CifarFeatureExtractor(output_dim=128)
        feature_extractor = CifarWrapper(base_fe, num_classes=10).to(device)
        
    fe_loader = data_manager.get_pretrain_loader()
    feature_extractor = train_feature_extractor(feature_extractor, fe_loader, device, epochs=5)
    
    # Freeze Parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    
    # --- 3. Bayesian Model ---
    if args.method == 'analytic':
        bayes_layer = BayesianLinearAnalytic(in_features=128, out_features=10).to(device)
    else:
        bayes_layer = BayesianLinearMFVI(in_features=128, out_features=10).to(device)
        
    # --- 4. OOD Scorer Initialization ---
    print("Initializing OOD Scorer...")
    
    # Extract features from ID training set for fitting
    all_feats = []
    max_samples = 10000
    samples_count = 0
    with torch.no_grad():
        for x_batch, _ in fe_loader:
            x_batch = x_batch.to(device)
            feats = feature_extractor.extract_features(x_batch)
            all_feats.append(feats)
            samples_count += x_batch.size(0)
            if samples_count >= max_samples: break
    phi_train = torch.cat(all_feats, dim=0)

    if 'gmm' in args.acquisition:
        print("Fitting GMM Scorer...")
        scorer = GMMScorer(n_components=10, covariance_type='full')
        scorer.fit(phi_train)
    else:
        print("Fitting Mahalanobis Scorer...")
        scorer = MahalanobisScorer(temperature=1.0) 
        scorer.fit(phi_train)
        # Auto-calibration
        diff = phi_train - scorer.mu
        term1 = diff @ scorer.precision
        d_squared = torch.sum(term1 * diff, dim=1)
        mean_d2 = d_squared.mean().item()
        scorer.temperature = mean_d2
        print(f"Auto-calibrated Temperature: {mean_d2:.4f}")
    
    # --- 5. Active Learning Loop ---
    al_loop = RegressionActiveLoopOOD(feature_extractor, bayes_layer, scorer, device)
    
    rmse_history = []
    ood_rate_history = []
    
    print("Starting Active Learning...")
    for r in range(args.rounds):
        rmse, query_indices, ood_rate, avg_ood_var = al_loop.run_round(
            data_manager, args.query_size, method=args.method, acquisition=args.acquisition
        )
        
        rmse_history.append(rmse)
        ood_rate_history.append(ood_rate)
        data_manager.update_sets(query_indices)
        
        wandb.log({
            "round": r+1,
            "val_rmse": rmse,
            "ood_query_rate": ood_rate,
            "avg_ood_variance": avg_ood_var,
            "labeled_pool_size": len(data_manager.X_train)
        })
        print(f"Round {r+1}: RMSE={rmse:.4f}, OOD Rate={ood_rate:.2f}")
        
    # --- 6. Save Results ---
    if not os.path.exists('results_ood'):
        os.makedirs('results_ood')
    
    filename_suffix = f"{args.id_dataset}_{args.ood_dataset}_{args.method}_{args.acquisition}"
    np.save(f'results_ood/rmse_{filename_suffix}.npy', np.array(rmse_history))
    np.save(f'results_ood/ood_rate_{filename_suffix}.npy', np.array(ood_rate_history))
    
    wandb.finish()

if __name__ == "__main__":
    main()
