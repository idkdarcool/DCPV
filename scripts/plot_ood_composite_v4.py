import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_composite_v4():
    # 2x4 Layout
    # Row 1: M->F, F->M, M->K, Hist(M->F)
    # Row 2: C10->C100, C10->SVHN, Scatter, Hist(C10->C100)
    
    experiments = [
        # Row 1 (Cols 1-3)
        ('1. MNIST vs Fashion', 'MNIST', 'Fashion'),
        ('2. Fashion vs MNIST', 'fashion_mnist', 'mnist'),
        ('3. MNIST vs KMNIST', 'mnist', 'kmnist'),
        # Row 2 (Cols 1-2)
        ('4. CIFAR-10 vs C-100', 'cifar10', 'cifar100'),
        ('5. CIFAR-10 vs SVHN', 'cifar10', 'svhn')
    ]
    
    methods = [
        ('Random', 'analytic', 'random', '#E74C3C', 'dash'),
        ('PV', 'analytic', 'predictive_variance', '#2E86C1', 'solid'),
        ('Robust (1G)', 'analytic', 'robust_pv', '#28B463', 'solid'),
        ('Robust (GMM)', 'analytic', 'robust_pv_gmm', '#E67E22', 'solid'), 
    ]
    
    fig = make_subplots(
        rows=2, cols=4, 
        subplot_titles=[
            '1. M->F (Paradox)', '2. F->M', '3. M->K', 'Density Paradox (M->F)',
            '4. C10->C100 (Fail)', '5. C10->SVHN', '6. Summary Trade-off', 'Density Paradox (C10->C100)'
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )
    
    # --- Trajectories (Plots 1-5) ---
    positions = [
        (1, 1), (1, 2), (1, 3), # Row 1
        (2, 1), (2, 2)          # Row 2
    ]
    
    for i, (exp_title, id_name, ood_name) in enumerate(experiments):
        row, col = positions[i]
        
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
                # Removed truncation: Plot 50 rounds for all
                
                cum_ood = np.cumsum(data * 10)
                x = np.arange(1, len(data)+1)
                
                show_legend = (i == 0)
                # Legend names for lines
                fig.add_trace(
                    go.Scatter(x=x, y=cum_ood, name=name, line=dict(color=color, dash=dash), showlegend=show_legend, legendgroup=name),
                    row=row, col=col
                )
        
        if col == 1:
            fig.update_yaxes(title_text="Cumul. OOD", row=row, col=col)
        if row == 2:
            fig.update_xaxes(title_text="Round", row=row, col=col)

    # --- Plot 7: Trade-off Scatter (Row 2, Col 3) ---
    # Load dynamic data
    from analyze_ood_results import analyze_ood
    df = analyze_ood()
    
    # Define desired order of experiments for plotting consistency
    # (Matches the scatter labels: M->F, F->M, M->K, C->C, C->S)
    ordered_exps = ["MNIST vs Fashion", "Fashion $\\to$ MNIST", "MNIST $\\to$ KMNIST", "CIFAR-10 $\\to$ C-100", "CIFAR-10 $\\to$ SVHN"]
    # Check if we need to map names from df which uses raw filenames sometimes
    # analyze_ood returns standardized labels now in 'Experiment' col?
    # Let's check analyze_ood implementation.
    # It sets: "MNIST vs Fashion", or "id vs ood" (e.g. "fashion_mnist vs mnist")
    # I should update analyze_ood to normalize names perfectly or handle it here.
    # analyze_ood returns "Experiment" column.
    
    # Helper to get ordered points
    def get_points(method_name):
        xs, ys = [], []
        # Fallback order if strict matching fails
        found_exps = []
        
        # 1. M->F
        row = df[(df['Experiment'] == 'MNIST vs Fashion') & (df['Method'] == method_name)]
        if not row.empty: xs.append(row.iloc[0]['OOD %']); ys.append(row.iloc[0]['Final RMSE'])
        
        # 2. F->M
        # analyze_ood currently outputs "fashion_mnist vs mnist".
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

    # FIX 3: Add Legend for Scatter Markers
    # Improved visibility: Alpha blending, clear borders
    fig.add_trace(go.Scatter(x=r_x, y=r_y, mode='markers', name='Random (Final)', 
                            marker=dict(color='#E74C3C', symbol='circle', size=10, line=dict(width=1, color='black'), opacity=0.7), showlegend=True), row=2, col=3)
    fig.add_trace(go.Scatter(x=pv_x, y=pv_y, mode='markers', name='PV (Final)', 
                            marker=dict(color='#2E86C1', symbol='square', size=10, line=dict(width=1, color='black'), opacity=0.7), showlegend=True), row=2, col=3)
    fig.add_trace(go.Scatter(x=r1g_x, y=r1g_y, mode='markers', name='Robust 1G (Final)', 
                            marker=dict(color='#28B463', symbol='triangle-up', size=12, line=dict(width=1, color='black'), opacity=0.7), showlegend=True), row=2, col=3)
    fig.add_trace(go.Scatter(x=rgmm_x, y=rgmm_y, mode='markers', name='Robust GMM (Final)', 
                            marker=dict(color='#E67E22', symbol='star', size=16, line=dict(width=1, color='black'), opacity=0.8), showlegend=True), row=2, col=3)
    
    fig.update_xaxes(title_text="OOD %", row=2, col=3)
    fig.update_yaxes(title_text="ID RMSE", row=2, col=3)

    # --- Plot 4: Hist M->F (Row 1, Col 4) ---
    # FIX 2: Unify Histogram Legend
    path_mf = 'results_ood/gmm_scores_mnist_fashion.npz'
    if os.path.exists(path_mf):
        d = np.load(path_mf)
        fig.add_trace(go.Histogram(x=d['id'], name='ID Density', marker_color='blue', opacity=0.5, bingroup=1, showlegend=True, legendgroup='id_hist'), row=1, col=4)
        fig.add_trace(go.Histogram(x=d['ood'], name='OOD Density', marker_color='red', opacity=0.5, bingroup=1, showlegend=True, legendgroup='ood_hist'), row=1, col=4)
        fig.update_xaxes(title_text="GMM Score", row=1, col=4)
    
    # --- Plot 8: Hist C10->C100 (Row 2, Col 4) ---
    path_c = 'results_ood/gmm_scores_cifar10_cifar100.npz'
    if os.path.exists(path_c):
        d = np.load(path_c)
        fig.add_trace(go.Histogram(x=d['id'], name='ID Density', marker_color='blue', opacity=0.5, bingroup=2, showlegend=False, legendgroup='id_hist'), row=2, col=4)
        fig.add_trace(go.Histogram(x=d['ood'], name='OOD Density', marker_color='red', opacity=0.5, bingroup=2, showlegend=False, legendgroup='ood_hist'), row=2, col=4)
        fig.update_xaxes(title_text="GMM Score", row=2, col=4)

    fig.update_layout(height=600, width=1600, title_text="Figure 1: Comprehensive Analysis (Method Trajectories, Trade-offs, and Density Diagnostics)", plot_bgcolor='white')
    fig.update_xaxes(showgrid=True, gridcolor='#E5E5E5')
    fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5')
    
    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig.write_image("figs/ood_composite_2x4.png", scale=2)
    print("Saved figs/ood_composite_2x4.png")

if __name__ == "__main__":
    plot_composite_v4()
