import torch
import torch.nn as nn
import numpy as np

class BayesianLinearAnalytic(nn.Module):
    """
    Bayesian Linear Regression layer with closed-form posterior updates.
    """
    def __init__(self, in_features, out_features, prior_var=1.0, noise_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        self.noise_var = noise_var
        
        self.register_buffer('prior_precision', torch.eye(in_features) / prior_var)
        self.register_buffer('posterior_precision', self.prior_precision.clone())
        self.register_buffer('posterior_mean', torch.zeros(in_features, out_features))

    def fit(self, Phi, Y):
        """Update posterior statistics given new data."""
        # Precision update
        phi_t_phi = Phi.t() @ Phi
        self.posterior_precision = self.prior_precision + (phi_t_phi / self.noise_var)
        
        # Mean update
        posterior_covariance = torch.inverse(self.posterior_precision)
        phi_t_y = Phi.t() @ Y
        self.posterior_mean = posterior_covariance @ (phi_t_y / self.noise_var)
        self.posterior_covariance = posterior_covariance 

    def predict(self, Phi):
        """Returns predictive mean and total variance (epistemic + aleatoric)."""
        pred_mean = Phi @ self.posterior_mean
        
        # Diagonal of Phi @ Sigma @ Phi.T
        term1 = Phi @ self.posterior_covariance
        epistemic_var = torch.sum(term1 * Phi, dim=1, keepdim=True)
        
        total_var = epistemic_var + self.noise_var
        return pred_mean, total_var.expand_as(pred_mean)


class BayesianLinearMFVI(nn.Module):
    """
    Bayesian Linear Regression layer using Mean-Field Variational Inference.
    """
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.zeros(in_features, out_features))
        self.weight_log_sigma = nn.Parameter(torch.zeros(in_features, out_features) - 5.0)
        
        self.prior_mu = 0.0
        self.prior_std = prior_std
        
    def forward(self, x, sample=True):
        if sample:
            std = torch.exp(self.weight_log_sigma)
            eps = torch.randn_like(std)
            weight = self.weight_mu + eps * std
        else:
            weight = self.weight_mu
        return x @ weight

    def kl_divergence(self):
        """Compute Gaussian KL divergence."""
        std = torch.exp(self.weight_log_sigma)
        var = std ** 2
        prior_var = self.prior_std ** 2
        
        kl = 0.5 * (
            (var + (self.weight_mu - self.prior_mu)**2) / prior_var
            - 1 
            + 2 * np.log(self.prior_std) 
            - 2 * self.weight_log_sigma
        )
        return torch.sum(kl)

    def predict(self, Phi):
        """Approximate predictive mean and variance."""
        pred_mean = Phi @ self.weight_mu
        
        std = torch.exp(self.weight_log_sigma)
        var_weights = std ** 2
        
        # Epistemic Variance approximation
        epistemic_var = (Phi ** 2) @ var_weights
        total_var = epistemic_var + 1.0 # Assuming unit noise variance
        
        return pred_mean, total_var
