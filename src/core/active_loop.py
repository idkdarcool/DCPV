import torch
import numpy as np
from src.core.acquisition import acquisition_predictive_variance, acquisition_random

class ActiveLearningLoop:
    def __init__(self, feature_extractor, bayes_layer, device):
        self.feature_extractor = feature_extractor
        self.bayes_layer = bayes_layer
        self.device = device
        
    def run_round(self, data_manager, n_query, method='analytic', acquisition='predictive_variance'):
        """
        Executes one round of active learning.
        1. Fit model on current labeled train set.
        2. Evaluate on Validation set (RMSE).
        3. Select new points from Pool.
        4. Update Train/Pool sets.
        """
        # Get current data
        data = data_manager.get_regression_data()
        X_train = data['X_train'].to(self.device)
        y_train = data['y_train'].to(self.device)
        X_pool = data['X_pool'].to(self.device)
        X_val = data['X_val'].to(self.device)
        y_val = data['y_val'].to(self.device)
        
        # 1. Extract Features (Frozen)
        with torch.no_grad():
            self.feature_extractor.eval()
            phi_train = self.feature_extractor.extract_features(X_train)
            phi_pool = self.feature_extractor.extract_features(X_pool)
            phi_val = self.feature_extractor.extract_features(X_val)
            
        # 2. Fit Bayesian Layer
        if method == 'analytic':
            self.bayes_layer.fit(phi_train, y_train)
            
            # Validation RMSE
            pred_mean_val, _ = self.bayes_layer.predict(phi_val)
            rmse_val = torch.sqrt(torch.mean((pred_mean_val - y_val)**2)).item()
            
            # Pool Predictions for Acquisition
            pred_mean_pool, pred_var_pool = self.bayes_layer.predict(phi_pool)
            
            # Acquisition
            if acquisition == 'random':
                query_indices = acquisition_random(len(X_pool), n_query)
            else:
                query_indices = acquisition_predictive_variance(pred_mean_pool, pred_var_pool, n_query)
            
        elif method == 'mfvi':
            # Validation RMSE
            # Use closed-form predict
            pred_mean_val, _ = self.bayes_layer.predict(phi_val)
            rmse_val = torch.sqrt(torch.mean((pred_mean_val - y_val)**2)).item()
            
            # Pool Predictions for Acquisition
            pred_mean_pool, pred_var_pool = self.bayes_layer.predict(phi_pool)
            
            # Acquisition
            if acquisition == 'random':
                query_indices = acquisition_random(len(X_pool), n_query)
            else:
                query_indices = acquisition_predictive_variance(pred_mean_pool, pred_var_pool, n_query)

        
        return rmse_val, query_indices

class ClassificationActiveLoop:
    def __init__(self, model, device, use_wandb=False):
        self.model = model
        self.device = device
        self.use_wandb = use_wandb
        
    def run_round(self, data_manager, n_query, acquisition_func, dropout_iter=100, deterministic=False):
        """
        Executes one round of active learning for classification.
        """
        import wandb
        from src.models.train_utils import train_classifier
        from src.core.acquisition import (
            acquisition_entropy, acquisition_bald, 
            acquisition_var_ratios, acquisition_mean_std, acquisition_random
        )
        

        train_loader = data_manager.get_train_loader()
        val_loader = data_manager.get_val_loader()
        pool_loader = data_manager.get_pool_loader() 

        self.model.apply(self._reset_weights)
        self.model, val_acc = train_classifier(
            self.model, train_loader, val_loader, self.device, 
            epochs=50, use_wandb=self.use_wandb
        )

        if deterministic:
            self.model.eval() 
            dropout_iter = 1
        else:
            self.model.train() 
        

        X_pool, _ = next(iter(pool_loader))
        X_pool = X_pool.to(self.device)
        

        predictions = []
        with torch.no_grad():
            for _ in range(dropout_iter):
                output = torch.softmax(self.model(X_pool), dim=1)
                predictions.append(output)
        predictions = torch.stack(predictions)

        
        # Select Acquisition Function
        if acquisition_func == 'random':
            query_indices = acquisition_random(len(X_pool), n_query)
        elif acquisition_func == 'entropy':
            query_indices = acquisition_entropy(predictions, n_query)
        elif acquisition_func == 'bald':
            query_indices = acquisition_bald(predictions, n_query)
        elif acquisition_func == 'var_ratios':
            query_indices = acquisition_var_ratios(predictions, n_query)
        elif acquisition_func == 'mean_std':
            query_indices = acquisition_mean_std(predictions, n_query)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_func}")
            
        return val_acc, query_indices

    def _reset_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            m.reset_parameters()


