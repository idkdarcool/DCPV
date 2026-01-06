import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from src.models.feature_extractor import MnistFeatureExtractor
from src.models.bayesian_layers import BayesianLinearAnalytic
from src.models.ood_detector import MahalanobisScorer
from src.utils.data_loader_ood import CombinedDataManager
from src.models.train_utils import train_feature_extractor

def get_premium_layout(title):
    return dict(
        title=dict(text=title, font=dict(family="Arial, sans-serif", size=24), x=0.5, xanchor='center'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=14),
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

def analyze_distributions():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load Data (Pool with 50% OOD)
    print("Loading Data...")
    dm = CombinedDataManager(pool_size=2000, ood_ratio=0.5)
    data = dm.get_regression_data()
    X_pool = data['X_pool'].to(device)
    ood_pool = data['ood_pool'].to(device) # Boolean mask
    
    # 2. Train Feature Extractor (InD only)
    print("Training Feature Extractor...")
    fe = MnistFeatureExtractor().to(device)
    fe_loader = dm.get_pretrain_loader()
    fe = train_feature_extractor(fe, fe_loader, device, epochs=3)
    fe.eval()
    
    # 3. Extract Features
    with torch.no_grad():
        phi_pool = fe.extract_features(X_pool)
        
    # 4. Fit Bayesian Model (Analytic)
    # We need some labels to fit the posterior. Let's use the initial training set.
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    
    with torch.no_grad():
        phi_train = fe.extract_features(X_train)
        
    print("Fitting Bayesian Model...")
    bayes = BayesianLinearAnalytic(in_features=128, out_features=10).to(device)
    bayes.fit(phi_train, y_train)
    
    # 5. Fit OOD Scorer (Mahalanobis)
    # Fit on InD training data
    print("Fitting Mahalanobis Scorer...")
    scorer = MahalanobisScorer(temperature=123000.0) # Recalibrated T
    scorer.fit(phi_train)
    
    # 6. Compute Metrics for Pool
    print("Computing metrics...")
    with torch.no_grad():
        # Predictive Variance
        _, pred_var = bayes.predict(phi_pool)
        # pred_var is (N, K) or (N,). If (N, K) take sum/trace.
        if pred_var.dim() > 1:
            pv_scores = pred_var.sum(dim=1).cpu().numpy()
        else:
            pv_scores = pred_var.cpu().numpy()
            
        # InD Scores
        # Debug: Print raw distances first
        diff = phi_pool - scorer.mu
        term1 = diff @ scorer.precision
        d_squared = torch.sum(term1 * diff, dim=1)
        print(f"D^2 Stats: Min {d_squared.min().item():.4f}, Max {d_squared.max().item():.4f}, Mean {d_squared.mean().item():.4f}")
        
        sin_scores = scorer.score(phi_pool).cpu().numpy()
        
    # Separate InD vs OOD
    ood_mask = ood_pool.cpu().numpy().astype(bool)
    ind_mask = ~ood_mask
    
    pv_ind = pv_scores[ind_mask]
    pv_ood = pv_scores[ood_mask]
    
    sin_ind = sin_scores[ind_mask]
    sin_ood = sin_scores[ood_mask]
    
    print(f"InD samples: {len(pv_ind)}, OOD samples: {len(pv_ood)}")
    print(f"PV InD Mean: {pv_ind.mean():.4f}, OOD Mean: {pv_ood.mean():.4f}")
    print(f"Sin InD Mean: {sin_ind.mean():.4f}, OOD Mean: {sin_ood.mean():.4f}")
    
    # --- PLOTTING ---
    if not os.path.exists('results_analysis'):
        os.makedirs('results_analysis')
        
    # Plot 1: Histogram of Predictive Variance
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Histogram(x=pv_ind, name='In-Distribution (MNIST)', opacity=0.7, marker_color='#2E86C1'))
    fig_pv.add_trace(go.Histogram(x=pv_ood, name='OOD (Fashion)', opacity=0.7, marker_color='#E74C3C'))
    fig_pv.update_layout(get_premium_layout("Distribution of Predictive Variance"))
    fig_pv.update_xaxes(title_text="Predictive Variance (Uncertainty)")
    fig_pv.update_layout(barmode='overlay')
    fig_pv.write_image("results_analysis/hist_pv.png", width=1000, height=600, scale=2)
    
    # Plot 2: Histogram of InD Score
    fig_sin = go.Figure()
    fig_sin.add_trace(go.Histogram(x=sin_ind, name='In-Distribution (MNIST)', opacity=0.7, marker_color='#2E86C1'))
    fig_sin.add_trace(go.Histogram(x=sin_ood, name='OOD (Fashion)', opacity=0.7, marker_color='#E74C3C'))
    fig_sin.update_layout(get_premium_layout("Distribution of In-Distribution Score"))
    fig_sin.update_xaxes(title_text="In-Distribution Score (s_in)")
    fig_sin.update_layout(barmode='overlay')
    fig_sin.write_image("results_analysis/hist_sin.png", width=1000, height=600, scale=2)
    
    # Plot 3: Scatter PV vs Sin
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=pv_ind, y=sin_ind, mode='markers', name='In-Distribution',
        marker=dict(color='#2E86C1', size=6, opacity=0.6)
    ))
    fig_scatter.add_trace(go.Scatter(
        x=pv_ood, y=sin_ood, mode='markers', name='OOD',
        marker=dict(color='#E74C3C', size=6, opacity=0.6)
    ))
    fig_scatter.update_layout(get_premium_layout("Predictive Variance vs. InD Score"))
    fig_scatter.update_xaxes(title_text="Predictive Variance (Uncertainty)")
    fig_scatter.update_yaxes(title_text="In-Distribution Score (Density)")
    fig_scatter.write_image("results_analysis/scatter_pv_sin.png", width=1000, height=600, scale=2)
    
    print("Plots saved to results_analysis/")

if __name__ == "__main__":
    analyze_distributions()
