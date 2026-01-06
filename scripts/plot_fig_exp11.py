import numpy as np
import plotly.graph_objects as go
import os

def get_premium_layout(title, x_title, y_title):
    return dict(
        title=dict(text=title, font=dict(family="Arial, sans-serif", size=20), x=0.5, xanchor='center'),
        xaxis=dict(title=dict(text=x_title, font=dict(size=14)), showgrid=True, gridcolor='#E5E5E5'),
        yaxis=dict(title=dict(text=y_title, font=dict(size=14)), showgrid=True, gridcolor='#E5E5E5'),
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

def plot_exp11():
    # Experiment 1.1: MNIST (ID) vs Fashion (OOD)
    # We use the 'analytic' method results we generated earlier.
    
    methods = [
        ('Random', 'analytic', 'random', '#E74C3C', 'dash'),
        ('Predictive Variance', 'analytic', 'predictive_variance', '#2E86C1', 'solid'),
        ('Robust PV (Ours)', 'analytic', 'robust_pv', '#28B463', 'solid'),
        ('InD Score Only', 'analytic', 'sin_only', '#884EA0', 'dot'),
    ]
    
    fig = go.Figure()
    
    for name, method, acq, color, dash in methods:
        # Load from the 'original' results (MNIST/Fashion)
        path = f'results_ood/rmse_{method}_{acq}.npy'
        
        if os.path.exists(path):
            data = np.load(path)
            # x-axis: Number of labeled images. 
            # We start with 20. Add query_size (10) per round.
            # data length = rounds.
            rounds = np.arange(1, len(data)+1)
            labeled_images = 20 + rounds * 10
            
            fig.add_trace(go.Scatter(x=labeled_images, y=data, name=name, line=dict(color=color, dash=dash)))
            
    fig.update_layout(get_premium_layout(
        "MNIST regression: Active Learning performance (ID only)", 
        "Number of Labelled Images", 
        "Test RMSE"
    ))
    
    # Save as requested
    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig.write_image("figs/exp11_acq_funcs.png", width=800, height=500, scale=3)
    print("Saved figs/exp11_acq_funcs.png")

if __name__ == "__main__":
    plot_exp11()
