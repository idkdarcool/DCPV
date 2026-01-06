import torch
import numpy as np
import wandb
from src.core.acquisition import acquisition_random, acquisition_predictive_variance
from src.core.acquisition_ood import acquisition_robust_pv, acquisition_sin_only, acquisition_robust_pv_filtered

class RegressionActiveLoopOOD:
    def __init__(self, feature_extractor, bayes_layer, scorer, device):
        self.feature_extractor = feature_extractor
        self.bayes_layer = bayes_layer
        self.scorer = scorer
        self.device = device
        
    def run_round(self, data_manager, n_query, method='analytic', acquisition='robust_pv'):
        """Executes a single round of OOD-aware active learning."""
        
        data = data_manager.get_regression_data()
        X_train, y_train = data['X_train'].to(self.device), data['y_train'].to(self.device)
        X_pool = data['X_pool'].to(self.device)
        X_val, y_val = data['X_val'].to(self.device), data['y_val'].to(self.device)
        ood_pool = data['ood_pool'].to(self.device)
        ood_train = data['ood_train'].to(self.device)

        with torch.no_grad():
            self.feature_extractor.eval()
            phi_train = self.feature_extractor.extract_features(X_train)
            phi_pool = self.feature_extractor.extract_features(X_pool)
            phi_val = self.feature_extractor.extract_features(X_val)
            
        if method == 'analytic':
            self.bayes_layer.fit(phi_train, y_train)
        elif method == 'mfvi':
            from src.models.train_utils import train_mfvi
            # Retrain MFVI with new data
            self.bayes_layer = train_mfvi(self.bayes_layer, phi_train, y_train, epochs=50)

        # Validation Performance
        pred_mean_val, _ = self.bayes_layer.predict(phi_val)
        rmse_val = torch.sqrt(torch.mean((pred_mean_val - y_val)**2)).item()
        
        # Pool Predictions
        pred_mean_pool, pred_var_pool = self.bayes_layer.predict(phi_pool)

        ind_mask = ~ood_train
        if ind_mask.sum() > 0:
            self.scorer.fit(phi_train[ind_mask])
        else:
            print("Warning: No InD data available to fit the scorer.")
            
        if acquisition == 'random':
            query_indices = acquisition_random(len(X_pool), n_query)
        elif acquisition == 'predictive_variance':
            query_indices = acquisition_predictive_variance(pred_mean_pool, pred_var_pool, n_query)
        elif acquisition == 'robust_pv':
            query_indices = acquisition_robust_pv(pred_mean_pool, pred_var_pool, phi_pool, self.scorer, n_query)
        elif acquisition == 'robust_pv_gmm':
            query_indices = acquisition_robust_pv(pred_mean_pool, pred_var_pool, phi_pool, self.scorer, n_query)
        elif acquisition == 'robust_pv_gmm_filtered':
            query_indices = acquisition_robust_pv_filtered(pred_mean_pool, pred_var_pool, phi_pool, self.scorer, n_query, quantile=0.2)
        elif acquisition == 'sin_only':
            query_indices = acquisition_sin_only(phi_pool, self.scorer, n_query)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")
            

        n_ood_selected = ood_pool[query_indices].sum().item()
        ood_rate = n_ood_selected / n_query
        
        ood_pool_mask = ood_pool.bool()
        avg_ood_var = 0.0
        if ood_pool_mask.sum() > 0:
            var_ood = pred_var_pool[ood_pool_mask]
            if var_ood.dim() > 1:
                var_ood = var_ood.sum(dim=1)
            avg_ood_var = var_ood.mean().item()

        return rmse_val, query_indices, ood_rate, avg_ood_var
