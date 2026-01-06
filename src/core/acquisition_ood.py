import torch
import numpy as np

def acquisition_robust_pv(pred_mean, pred_var, features, scorer, n_query):
    """
    Robust Predictive Variance: a_rob(x) = a_PV(x) * s_in(x)
    Selection based on high uncertainty AND high in-distribution likelihood.
    """
    # 1. PV Score (Sum/Trace if multi-output)
    if pred_var.dim() > 1:
        pv_score = pred_var.sum(dim=1)
    else:
        pv_score = pred_var
        
    # 2. In-Distribution Score
    with torch.no_grad():
        s_in = scorer.score(features)
        
    # 3. Combine and Select
    robust_score = pv_score * s_in
    _, indices = torch.topk(robust_score, n_query)
    
    return indices.cpu().numpy()

def acquisition_sin_only(features, scorer, n_query):
    """Baseline: Select points with highest In-Distribution score."""
    with torch.no_grad():
        s_in = scorer.score(features)
        
    _, indices = torch.topk(s_in, n_query)
    return indices.cpu().numpy()

def acquisition_robust_pv_filtered(pred_mean, pred_var, features, scorer, n_query, quantile=0.2):
    """
    Robust Filtered PV:
    1. Filter out points with s_in(x) < quantile.
    2. Select top PV from remainder.
    """
    if pred_var.dim() > 1:
        pv_score = pred_var.sum(dim=1)
    else:
        pv_score = pred_var
        
    with torch.no_grad():
        s_in = scorer.score(features)
        
    threshold = torch.quantile(s_in, quantile)
    mask = s_in >= threshold
    valid_indices = torch.nonzero(mask).squeeze()
    
    if valid_indices.numel() < n_query:
        # Fallback to standard PV if filter is too aggressive
        _, indices = torch.topk(pv_score, n_query)
        return indices.cpu().numpy()
        
    cand_pv = pv_score[valid_indices]
    _, top_k_local = torch.topk(cand_pv, n_query)
    
    final_indices = valid_indices[top_k_local]
    return final_indices.cpu().numpy()
