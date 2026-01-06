import torch
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset

from src.models.feature_extractor import MnistFeatureExtractor
from src.models.bayesian_layers import BayesianLinearAnalytic, BayesianLinearMFVI
from src.models.train_utils import train_feature_extractor, train_mfvi
from src.utils.data_loader import MnistDataManager
from src.core.active_loop import ActiveLearningLoop

def main():
    parser = argparse.ArgumentParser(description='Bayesian Active Regression on MNIST')
    parser.add_argument('--method', type=str, default='analytic', choices=['analytic', 'mfvi'],
                        help='Inference method: analytic or mfvi')
    parser.add_argument('--acquisition', type=str, default='predictive_variance', choices=['predictive_variance', 'random'],
                        help='Acquisition function')
    parser.add_argument('--rounds', type=int, default=10, help='Number of AL rounds')
    parser.add_argument('--query_size', type=int, default=10, help='Number of queries per round')
    parser.add_argument('--epochs_fe', type=int, default=5, help='Epochs for Feature Extractor training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='mnist-active-learning', help='WANDB project name')
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize WANDB
    import wandb
    wandb.init(project=args.wandb_project, name=f"reg_{args.method}_{args.acquisition}_seed{args.seed}", config=args)


    # 1. Data Setup
    print("Initializing Data Manager...")
    data_manager = MnistDataManager()
    
    # 2. Feature Extractor Training
    print("Setting up Feature Extractor...")
    feature_extractor = MnistFeatureExtractor().to(device)
    
    
    # Let's use the full train_subset for pre-training to get good features.
    # Updated: Use get_pretrain_loader to use ALL data (Train + Pool) for strong features.
    fe_train_loader = data_manager.get_pretrain_loader(batch_size=64)
    feature_extractor = train_feature_extractor(feature_extractor, fe_train_loader, device, epochs=args.epochs_fe)
    
    # Freeze Feature Extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    
    # 3. Bayesian Layer Setup
    print(f"Setting up Bayesian Layer ({args.method})...")
    if args.method == 'analytic':
        bayes_layer = BayesianLinearAnalytic(in_features=128, out_features=10).to(device)
    else:
        bayes_layer = BayesianLinearMFVI(in_features=128, out_features=10).to(device)
        
    # 4. Active Learning Loop
    al_loop = ActiveLearningLoop(feature_extractor, bayes_layer, device)
    
    rmse_history = []
    
    print("Starting Active Learning Loop...")
    for r in range(args.rounds):
        print(f"\n--- Round {r+1}/{args.rounds} ---")
        
        # For MFVI, we need to train the layer inside the loop
        if args.method == 'mfvi':
            # Get current train data
            data = data_manager.get_regression_data()
            X_train = data['X_train'].to(device)
            y_train = data['y_train'].to(device)
            
            # Extract features
            with torch.no_grad():
                phi_train = feature_extractor.extract_features(X_train)
            
            # Train MFVI
            bayes_layer = train_mfvi(bayes_layer, phi_train, y_train)
            
        # Run AL Round
        rmse, query_indices = al_loop.run_round(data_manager, args.query_size, method=args.method, acquisition=args.acquisition)
        
        print(f"Validation RMSE: {rmse:.4f}")
        rmse_history.append(rmse)
        
        # Update Data
        data_manager.update_sets(query_indices)
        print(f"Queried {len(query_indices)} points. New Train Size: {len(data_manager.X_train)}")
        
        # Log to WANDB
        wandb.log({
            "round": r+1, 
            "val_rmse": rmse, 
            "labeled_pool_size": len(data_manager.X_train)
        })


    # Save Results
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = f'results/rmse_{args.method}_{args.acquisition}.npy'
    np.save(filename, np.array(rmse_history))
    print(f"Results saved to {filename}")
    wandb.finish()


if __name__ == "__main__":
    main()
