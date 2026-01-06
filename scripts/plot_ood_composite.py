import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_composite_v3():
    # Define Experiments for first 5 slots
    experiments = [
        ('1. MNIST vs Fashion (Paradox)', 'MNIST', 'Fashion'),
        ('2. Fashion vs MNIST (Swap)', 'fashion_mnist', 'mnist'),
        ('3. MNIST vs KMNIST', 'mnist', 'kmnist'),
        ('4. CIFAR-10 vs C-100 (Density Fail)', 'cifar10', 'cifar100'),
        ('5. CIFAR-10 vs SVHN (Easy)', 'cifar10', 'svhn')
    ]
    
    methods = [
        ('Random', 'analytic', 'random', '#E74C3C', 'dash'),
        ('PV', 'analytic', 'predictive_variance', '#2E86C1', 'solid'),
        ('Robust (1G)', 'analytic', 'robust_pv', '#28B463', 'solid'),
        ('Robust (GMM)', 'analytic', 'robust_pv_gmm', '#E67E22', 'solid'), 
    ]
    
    # Create Subplots: 2 rows, 3 cols
    # Specs: Last one (2,3) might have different axis types? 
    # make_subplots handles isolated axes automatically.
    fig = make_subplots(
        rows=2, cols=3, 
        subplot_titles=[e[0] for e in experiments] + ['6. Robustness-Performance Trade-off'],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    # --- PLOTS 1-5: Trajectories ---
    for i, (exp_title, id_name, ood_name) in enumerate(experiments):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        for name, method, acq, color, dash in methods:
            # Handle filenames
            path = None
            if id_name == 'MNIST' and ood_name == 'Fashion':
                p1 = f'results_ood/ood_rate_{method}_{acq}.npy'
                p2 = f'results_ood/ood_rate_mnist_fashion_mnist_{method}_{acq}.npy'
                if os.path.exists(p1): path = p1
                elif os.path.exists(p2): path = p2
            else:
                path = f'results_ood/ood_rate_{id_name}_{ood_name}_{method}_{acq}.npy'
            
            if path and os.path.exists(path):
                data = np.load(path)
                cum_ood = np.cumsum(data * 10)
                x = np.arange(1, len(data)+1)
                
                # Legend only on first plot
                show_legend = (i == 0)
                
                fig.add_trace(
                    go.Scatter(x=x, y=cum_ood, name=name, line=dict(color=color, dash=dash), showlegend=show_legend, legendgroup=name),
                    row=row, col=col
                )
                
        # Axis Titles
        if col == 1:
            fig.update_yaxes(title_text="Cumul. OOD", row=row, col=col)
        if row == 2:
            fig.update_xaxes(title_text="Round", row=row, col=col)

    # --- PLOT 6: Trade-off Scatter ---
    # Load dynamic data
    from analyze_ood_results import analyze_ood
    df = analyze_ood()
    
    def get_points(method_name):
        xs, ys = [], []
        # Fallback order if strict matching fails
        # 1. M->F
        row = df[(df['Experiment'] == 'MNIST vs Fashion') & (df['Method'] == method_name)]
        if not row.empty: xs.append(row.iloc[0]['OOD %']); ys.append(row.iloc[0]['Final RMSE'])
        # 2. F->M
        row = df[(df['Experiment'] == 'fashion_mnist vs mnist') & (df['Method'] == method_name)]
        if not row.empty: xs.append(row.iloc[0]['OOD %']); ys.append(row.iloc[0]['Final RMSE'])
        # 3. M->K
        row = df[(df['Experiment'] == 'mnist vs kmnist') & (df['Method'] == method_name)]
        if not row.empty: xs.append(row.iloc[0]['OOD %']); ys.append(row.iloc[0]['Final RMSE'])
        # 4. C->C
        row = df[(df['Experiment'] == 'cifar10 vs cifar100') & (df['Method'] == method_name)]
        if not row.empty: xs.append(row.iloc[0]['OOD %']); ys.append(row.iloc[0]['Final RMSE'])
        # 5. C->S
        row = df[(df['Experiment'] == 'cifar10 vs svhn') & (df['Method'] == method_name)]
        if not row.empty: xs.append(row.iloc[0]['OOD %']); ys.append(row.iloc[0]['Final RMSE'])
        return xs, ys

    r_x, r_y = get_points('Random')
    pv_x, pv_y = get_points('PV')
    r1g_x, r1g_y = get_points('Robust PV')
    rgmm_x, rgmm_y = get_points('Robust GMM PV')
    
    # Legend for markers (explicitly named)
    fig.add_trace(go.Scatter(x=r_x, y=r_y, mode='markers', name='Random (Final)', 
                            marker=dict(color='#E74C3C', symbol='circle', size=10, line=dict(width=1, color='black'), opacity=0.7), showlegend=True), 
                  row=2, col=3)
                  
    fig.add_trace(go.Scatter(x=pv_x, y=pv_y, mode='markers', name='PV (Final)', 
                            marker=dict(color='#2E86C1', symbol='square', size=10, line=dict(width=1, color='black'), opacity=0.7), showlegend=True), 
                  row=2, col=3)
                  
    fig.add_trace(go.Scatter(x=r1g_x, y=r1g_y, mode='markers', name='Robust 1G (Final)', 
                            marker=dict(color='#28B463', symbol='triangle-up', size=12, line=dict(width=1, color='black'), opacity=0.7), showlegend=True), 
                  row=2, col=3)
                  
    fig.add_trace(go.Scatter(x=rgmm_x, y=rgmm_y, mode='markers', name='Robust GMM (Final)', 
                            marker=dict(color='#E67E22', symbol='star', size=16, line=dict(width=1, color='black'), opacity=0.8), showlegend=True), 
                  row=2, col=3)

    fig.update_xaxes(title_text="OOD %", row=2, col=3)
    fig.update_yaxes(title_text="ID RMSE", row=2, col=3)

    # Global Layout
    fig.update_layout(
        height=700, width=1200, 
        title_text="Comprehensive OOD Analysis: Trajectories and Trade-offs",
        title_x=0.5,
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Grid lines
    fig.update_xaxes(showgrid=True, gridcolor='#E5E5E5')
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5')

    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig.write_image("figs/ood_composite_2x3.png", scale=2)
    print("Saved figs/ood_composite_2x3.png")

if __name__ == "__main__":
    plot_composite_v3()
