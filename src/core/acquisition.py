import torch
import numpy as np

def acquisition_random(pool_size, n_query):
    """Random acquisition."""
    return np.random.choice(range(pool_size), n_query, replace=False)

def acquisition_predictive_variance(mean, var, n_query):
    """
    Select points with highest predictive variance.
    Var: (N, K) -> Sum over K outputs -> (N,)
    """
    # Sum of variances across all outputs (K=10 for MNIST regression)
    total_var = var.sum(dim=1)
    
    # Get top n_query indices
    _, indices = torch.topk(total_var, n_query)
    return indices.cpu().numpy()

def acquisition_bald(predictions, n_query):
    """
    BALD for classification (retained for reference/completeness).
    predictions: (T, N, K) - Monte Carlo samples
    """
    # Expected Entropy
    expected_entropy = -torch.mean(torch.sum(predictions * torch.log(predictions + 1e-10), dim=2), dim=0)
    
    # Entropy of Expected
    expected_p = torch.mean(predictions, dim=0)
    entropy_expected = -torch.sum(expected_p * torch.log(expected_p + 1e-10), dim=1)
    
    bald_score = entropy_expected - expected_entropy
    _, indices = torch.topk(bald_score, n_query)
    return indices.cpu().numpy()

def acquisition_entropy(predictions, n_query):
    """
    Max Entropy acquisition.
    predictions: (T, N, K)
    """
    # Mean prediction across MC samples
    mean_preds = predictions.mean(dim=0) # (N, K)
    
    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(mean_preds * torch.log(mean_preds + 1e-10), dim=1)
    
    _, indices = torch.topk(entropy, n_query)
    return indices.cpu().numpy()

def acquisition_var_ratios(predictions, n_query):
    """
    Variation Ratios acquisition.
    1 - max_y p(y|x)
    """
    # Get hard predictions from each MC sample
    # predictions: (T, N, K)
    # We need to count class votes
    mc_preds = predictions.argmax(dim=2) # (T, N)
    
    # Count mode
    # torch.mode returns (values, indices)
    mode_val, _ = torch.mode(mc_preds, dim=0) # (N,)
    
    # Count how many times mode was predicted
    count = (mc_preds == mode_val).float().sum(dim=0) # (N,)
    
    # Variation Ratio = 1 - (count / T)
    var_ratio = 1.0 - (count / predictions.size(0))
    
    _, indices = torch.topk(var_ratio, n_query)
    return indices.cpu().numpy()

def acquisition_mean_std(predictions, n_query):
    """
    Mean Standard Deviation acquisition.
    Mean of std dev across classes.
    """
    # Std dev across MC samples
    std_dev = predictions.std(dim=0) # (N, K)
    
    # Mean of std devs
    mean_std = std_dev.mean(dim=1) # (N,)
    
    _, indices = torch.topk(mean_std, n_query)
    return indices.cpu().numpy()

