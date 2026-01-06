import torch
import numpy as np
from sklearn.mixture import GaussianMixture

class MahalanobisScorer:
    """
    In-Distribution scorer using Mahalanobis distance.
    High score = In-Distribution.
    """
    def __init__(self, temperature=1.0, jitter=1e-6):
        self.temperature = temperature
        self.jitter = jitter
        self.mu = None
        self.precision = None
        
    def fit(self, features):
        self.mu = features.mean(dim=0)
        X_centered = features - self.mu
        
        N = features.size(0)
        if N < 2:
            self.precision = torch.eye(features.size(1), device=features.device)
            return
            
        cov = (X_centered.t() @ X_centered) / (N - 1)
        cov += torch.eye(cov.size(0), device=cov.device) * 1e-3
        
        try:
            self.precision = torch.inverse(cov)
        except RuntimeError:
            self.precision = torch.pinverse(cov)
            
    def score(self, features):
        if self.mu is None:
            raise ValueError("Scorer not fitted.")
            
        diff = features - self.mu
        term1 = diff @ self.precision
        d_squared = torch.sum(term1 * diff, dim=1)
        
        # Score = exp(-Distance / T)
        return torch.exp(-d_squared / self.temperature)


class GMMScorer:
    """
    In-Distribution scorer using a Gaussian Mixture Model.
    """
    def __init__(self, n_components=10, covariance_type='full'):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        self.train_log_probs_mean = None
        self.train_log_probs_std = None

    def fit(self, features):
        X = features.cpu().numpy()
        self.gmm.fit(X)
        
        log_probs = self.gmm.score_samples(X)
        self.train_log_probs_mean = log_probs.mean()
        self.train_log_probs_std = log_probs.std() + 1e-6
        
    def score(self, features):
        if self.train_log_probs_mean is None:
            raise ValueError("Scorer not fitted.")
            
        X = features.cpu().numpy()
        log_probs = self.gmm.score_samples(X)
        
        # Z-score normalization and Sigmoid mapping
        z_scores = (log_probs - self.train_log_probs_mean) / self.train_log_probs_std
        scores = 1 / (1 + np.exp(-z_scores))
        
        return torch.tensor(scores, device=features.device, dtype=torch.float32)
