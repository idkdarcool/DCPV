import numpy as np
import plotly.graph_objects as go
import os

def plot_detailed_mn_fashion():
    # 1. Define Methods to Plot
    methods = [
        ('Random', 'results_ood/ood_rate_analytic_random.npy', '#E74C3C', 'dash'),
        ('Predictive Variance', 'results_ood/ood_rate_analytic_predictive_variance.npy', '#2E86C1', 'solid'),
        ('Robust PV (Ours)', 'results_ood/ood_rate_analytic_robust_pv.npy', '#28B463', 'solid'),
        ('Robust PV (GMM)', 'results_ood/ood_rate_mnist_fashion_mnist_analytic_robust_pv_gmm.npy', '#E67E22', 'solid'), # Orange/Red
        ('InD Score Only', 'results_ood/ood_rate_analytic_sin_only.npy', '#8E44AD', 'dot'),
        ('MFVI - PV', 'results_ood/ood_rate_mfvi_predictive_variance.npy', '#3498DB', 'dash'),
        ('MFVI - Robust PV', 'results_ood/ood_rate_mfvi_robust_pv.npy', '#2ECC71', 'dash'),
    ]
    
    fig = go.Figure()
    
    for name, path, color, dash in methods:
        # Fallback for legacy naming if needed (though GMM usually uses long name)
        if not os.path.exists(path):
            # Try short name
            short_path = path.replace('mnist_fashion_mnist_', '')
            if os.path.exists(short_path):
                path = short_path
            else:
                # Try long name
                long_path = path.replace('ood_rate_', 'ood_rate_mnist_fashion_mnist_')
                if os.path.exists(long_path):
                    path = long_path
                else:
                    print(f"Warning: Could not find {path}")
                    continue
                    
        data = np.load(path)
        cum_ood = np.cumsum(data * 10)
        x = np.arange(1, len(data)+1)
        
        fig.add_trace(go.Scatter(x=x, y=cum_ood, name=name, line=dict(color=color, dash=dash, width=3)))

    fig.update_layout(
        title={
            'text': "Cumulative OOD Queries (MNIST -> Fashion-MNIST)",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Round",
        yaxis_title="Total OOD Samples",
        font=dict(family="Arial", size=14),
        plot_bgcolor='white',
        # Move legend higher and add margin
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        margin=dict(t=100), # Add top margin for legend
        height=600, width=1000
    )
    fig.update_xaxes(showgrid=True, gridcolor='#E5E5E5')
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5')
    
    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig.write_image("figs/mnist_fashion_detailed_gmm.png", scale=3)
    print("Saved figs/mnist_fashion_detailed_gmm.png")

if __name__ == "__main__":
    plot_detailed_mn_fashion()
