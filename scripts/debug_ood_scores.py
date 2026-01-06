import torch
import numpy as np
from src.models.feature_extractor import MnistFeatureExtractor
from src.models.bayesian_layers import BayesianLinearAnalytic
from src.models.ood_detector import MahalanobisScorer
from src.utils.data_loader_ood import CombinedDataManager

def debug_scores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    print("Loading Data...")
    dm = CombinedDataManager(pool_size=1000, ood_ratio=0.5)
    data = dm.get_regression_data()
    X_pool = data['X_pool'].to(device)
    ood_pool = data['ood_pool'].to(device)
    
    # 2. Load Models (Untrained/Randomly initialized for now, or load from saved?)
    # We need a trained FE to see real feature space behavior.
    # Let's quickly train one.
    print("Training FE...")
    from src.models.train_utils import train_feature_extractor
    fe = MnistFeatureExtractor().to(device)
    fe_loader = dm.get_pretrain_loader()
    fe = train_feature_extractor(fe, fe_loader, device, epochs=2)
    fe.eval()
    
    # Extract Features
    with torch.no_grad():
        phi_pool = fe.extract_features(X_pool)
        
    # 3. Fit Scorer (on InD part of pool for demo, or just random InD)
    # Let's use the InD part of the pool to fit, to see ideal separation
    ind_mask = ~ood_pool
    phi_ind = phi_pool[ind_mask]
    
    print("Fitting Scorer on InD data...")
    scorer = MahalanobisScorer()
    scorer.fit(phi_ind)
    
    # 4. Compute Scores & Distances
    # We need to access raw distances. Let's modify scorer or just re-compute here for debug.
    # Or better, let's just inspect the scorer internals if possible, or add a method.
    # Let's just assume we can get them by calling score with T=1 and taking -log.
    
    # But wait, scorer.score returns exp(-d^2/T). 
    # If it returns 0, we can't take log.
    # Let's modify the scorer to print distances inside 'score' method temporarily?
    # Or just subclass/monkeypatch.
    
    # Let's just manually compute distance here to check.
    diff = phi_pool - scorer.mu
    term1 = diff @ scorer.precision
    d_squared = torch.sum(term1 * diff, dim=1)
    
    d2_ind = d_squared[ind_mask].cpu().numpy()
    d2_ood = d_squared[ood_pool].cpu().numpy()
    
    print(f"\n--- In-Distribution (MNIST) d^2 ---")
    print(f"Mean: {d2_ind.mean():.4f}, Std: {d2_ind.std():.4f}")
    
    print(f"\n--- Out-of-Distribution (Fashion) d^2 ---")
    print(f"Mean: {d2_ood.mean():.4f}, Std: {d2_ood.std():.4f}")
    
    # Suggest T
    suggested_T = d2_ind.mean()
    print(f"\nSuggested Temperature T: {suggested_T:.4f}")
    
    # 5. Analyze
    scores_ind = scores[ind_mask].cpu().numpy()
    scores_ood = scores[ood_pool].cpu().numpy()
    
    print(f"\n--- In-Distribution (MNIST) Scores ---")
    print(f"Mean: {scores_ind.mean():.4f}, Std: {scores_ind.std():.4f}")
    print(f"Min: {scores_ind.min():.4f}, Max: {scores_ind.max():.4f}")
    
    print(f"\n--- Out-of-Distribution (Fashion) Scores ---")
    print(f"Mean: {scores_ood.mean():.4f}, Std: {scores_ood.std():.4f}")
    print(f"Min: {scores_ood.min():.4f}, Max: {scores_ood.max():.4f}")
    
    # Check overlap
    threshold = scores_ind.mean() - 2 * scores_ind.std()
    false_neg = (scores_ind < threshold).mean()
    false_pos = (scores_ood > threshold).mean()
    print(f"\nOverlap Check (Threshold {threshold:.4f}):")
    print(f"False Neg (InD rejected): {false_neg*100:.1f}%")
    print(f"False Pos (OOD accepted): {false_pos*100:.1f}%")

if __name__ == "__main__":
    debug_scores()
