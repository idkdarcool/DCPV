import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def get_premium_layout(title, x_title, y_title):
    return dict(
        title=dict(text=title, font=dict(family="Arial, sans-serif", size=20), x=0.5, xanchor='center'),
        xaxis=dict(title=dict(text=x_title, font=dict(size=14)), showgrid=True, gridcolor='#E5E5E5'),
        yaxis=dict(title=dict(text=y_title, font=dict(size=14)), showgrid=True, gridcolor='#E5E5E5'),
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

def plot_ood():
    # Define Experiments
    experiments = [
        ('MNIST vs Fashion (Original)', 'MNIST', 'Fashion'),
        ('Fashion vs MNIST (Swap)', 'fashion_mnist', 'mnist'),
        ('MNIST vs KMNIST', 'mnist', 'kmnist'),
        ('CIFAR-10 vs CIFAR-100', 'cifar10', 'cifar100'),
        ('CIFAR-10 vs SVHN', 'cifar10', 'svhn')
    ]
    
    methods = [
        ('Random', 'analytic', 'random', '#E74C3C', 'dash'),
        ('Predictive Variance', 'analytic', 'predictive_variance', '#2E86C1', 'solid'),
        ('Robust PV (Ours)', 'analytic', 'robust_pv', '#28B463', 'solid'),
        ('Robust GMM PV', 'analytic', 'robust_pv_gmm', '#E67E22', 'solid'), # Orange
    ]
    
    # Create Subplots: 3 rows, 2 cols (to fit 5 experiments)
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=[e[0] for e in experiments],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, (exp_title, id_name, ood_name) in enumerate(experiments):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        for name, method, acq, color, dash in methods:
            # Handle legacy filenames
            path = None
            if id_name == 'MNIST' and ood_name == 'Fashion':
                # Try legacy first
                p1 = f'results_ood/ood_rate_{method}_{acq}.npy'
                # Try new format (mnist_fashion_mnist) if legacy not found
                p2 = f'results_ood/ood_rate_mnist_fashion_mnist_{method}_{acq}.npy'
                
                if os.path.exists(p1):
                    path = p1
                elif os.path.exists(p2):
                    path = p2
            else:
                path = f'results_ood/ood_rate_{id_name}_{ood_name}_{method}_{acq}.npy'
                
            if path and os.path.exists(path):
                data = np.load(path)
                # Cumulative OOD count (query_size=10)
                cum_ood = np.cumsum(data * 10)
                x = np.arange(1, len(data)+1)
                
                show_legend = (i == 0) # Only show legend for first plot
                
                fig.add_trace(
                    go.Scatter(x=x, y=cum_ood, name=name, line=dict(color=color, dash=dash), showlegend=show_legend),
                    row=row, col=col
                )
                
    fig.update_layout(height=800, width=1200, title_text="Cumulative OOD Queries across Experiments", title_x=0.5)
    fig.update_xaxes(title_text="Round", showgrid=True, gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="Total OOD Samples", showgrid=True, gridcolor='#E5E5E5')
    fig.update_layout(plot_bgcolor='white')
    
    fig.write_image("results_ood/ood_grid_plot.png", scale=2)
    print("Grid plot saved to results_ood/ood_grid_plot.png")

if __name__ == "__main__":
    plot_ood()
