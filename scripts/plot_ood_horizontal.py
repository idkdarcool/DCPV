import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_ood_horizontal():
    # Define Experiments (Order matches previous)
    experiments = [
        ('MNIST vs Fashion', 'MNIST', 'Fashion'),
        ('Fashion vs MNIST', 'fashion_mnist', 'mnist'),
        ('MNIST vs KMNIST', 'mnist', 'kmnist'),
        ('CIFAR-10 vs C-100', 'cifar10', 'cifar100'), # Shortened titles
        ('CIFAR-10 vs SVHN', 'cifar10', 'svhn')
    ]
    
    methods = [
        ('Random', 'analytic', 'random', '#E74C3C', 'dash'),
        ('PV', 'analytic', 'predictive_variance', '#2E86C1', 'solid'),
        ('Robust (1G)', 'analytic', 'robust_pv', '#28B463', 'solid'),
        ('Robust (GMM)', 'analytic', 'robust_pv_gmm', '#E67E22', 'solid'), # Orange
    ]
    
    # Create Subplots: 1 row, 5 cols
    fig = make_subplots(
        rows=1, cols=5, 
        subplot_titles=[e[0] for e in experiments],
        horizontal_spacing=0.02, # Tight spacing
        shared_yaxes=True # Share Y-Axis to save space
    )
    
    for i, (exp_title, id_name, ood_name) in enumerate(experiments):
        col = i + 1
        
        for name, method, acq, color, dash in methods:
            # Handle legacy filenames
            path = None
            if id_name == 'MNIST' and ood_name == 'Fashion':
                p1 = f'results_ood/ood_rate_{method}_{acq}.npy'
                p2 = f'results_ood/ood_rate_mnist_fashion_mnist_{method}_{acq}.npy'
                if os.path.exists(p1): path = p1
                elif os.path.exists(p2): path = p2
            else:
                path = f'results_ood/ood_rate_{id_name}_{ood_name}_{method}_{acq}.npy'
            
            # GMM might be missing for some
            if path and os.path.exists(path):
                data = np.load(path)
                cum_ood = np.cumsum(data * 10)
                x = np.arange(1, len(data)+1)
                
                # Show legend only on first trace of first plot?
                # Actually, horizontal legend on top is better.
                # We add traces. Legend handling: show for first plot, hide for others.
                # But we iterate methods inside plots.
                # Only show legend for the first plot's methods.
                show_legend = (i == 0)
                
                fig.add_trace(
                    go.Scatter(x=x, y=cum_ood, name=name, line=dict(color=color, dash=dash, width=2), showlegend=show_legend),
                    row=1, col=col
                )

    # Layout
    fig.update_layout(
        height=300, # Short height
        width=1500, # Wide
        # title_text="OOD Selection Trajectories",
        # title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="Round", showgrid=True, gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="Cumul. OOD", showgrid=True, gridcolor='#E5E5E5', row=1, col=1) # Y-label only on first
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5', row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5', row=1, col=3)
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5', row=1, col=4)
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5', row=1, col=5)
    
    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig.write_image("figs/ood_strip_horizontal.png", scale=3)
    print("Saved figs/ood_strip_horizontal.png")

if __name__ == "__main__":
    plot_ood_horizontal()
