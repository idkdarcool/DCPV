import matplotlib.pyplot as plt
import numpy as np

def plot_summary_v2():
    # Data Data format: (OOD%, RMSE, Label)
    
    # Random (Blue Circle)
    random_data = [
        (46.4, 0.172, 'M→F'),
        (50.0, 0.225, 'F→M'),
        (47.0, 0.259, 'M→K'),
        (49.5, 0.277, 'C10→C100'),
        (49.5, 0.284, 'C10→S'),
    ]
    
    # PV (Orange Square)
    pv_data = [
        (2.4, 0.104, 'M→F'),
        (44.5, 0.178, 'F→M'),
        (27.0, 0.190, 'M→K'),
        (25.5, 0.254, 'C10→C100'),
        (2.0, 0.230, 'C10→S'),
    ]
    
    # Robust 1G (Green Triangle)
    rob1g_data = [
        (16.8, 0.267, 'M→F'),
        (11.5, 0.291, 'F→M'),
        (11.5, 0.321, 'M→K'),
        (63.0, 0.421, 'C10→C100'),
        (16.0, 0.366, 'C10→S'),
    ]
    
    # Robust GMM (Red Star) - Only run on 3 pairs
    robgmm_data = [
        (47.5, 0.337, 'M→F'), # Updated result
        (47.5, 0.285, 'C10→C100'),
        (47.5, 0.283, 'C10→S'),
    ]
    
    plt.figure(figsize=(10, 7))
    
    # Plot Methods
    # Random
    x, y, labels = zip(*random_data)
    plt.scatter(x, y, c='cornflowerblue', marker='o', s=100, label='Random', edgecolors='black', alpha=0.8)
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=8, color='blue')
        
    # PV
    x, y, labels = zip(*pv_data)
    plt.scatter(x, y, c='darkorange', marker='s', s=100, label='PV', edgecolors='black', alpha=0.8)
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, (x[i], y[i]), xytext=(5, -15), textcoords='offset points', fontsize=8, color='darkorange')

    # Robust 1G
    x, y, labels = zip(*rob1g_data)
    plt.scatter(x, y, c='forestgreen', marker='^', s=120, label='Robust PV (1G)', edgecolors='black', alpha=0.8)
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, (x[i], y[i]), xytext=(-15, 5), textcoords='offset points', fontsize=8, color='green')

    # Robust GMM
    x, y, labels = zip(*robgmm_data)
    plt.scatter(x, y, c='crimson', marker='*', s=180, label='Robust PV (GMM)', edgecolors='black', alpha=0.8)
    for i, txt in enumerate(labels):
        # Annotate GMM points to highlight them
        plt.annotate(txt, (x[i], y[i]), xytext=(10, 0), textcoords='offset points', fontsize=9, color='crimson', weight='bold')

    # Formatting
    plt.xlabel('OOD Percentage in Query Batch (%)', fontsize=14)
    plt.ylabel('In-Distribution Test RMSE (lower is better)', fontsize=14)
    plt.title('Trade-off: Robustness vs. Model Performance', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    
    # Axes limits
    plt.xlim(-5, 70) 
    plt.ylim(0.08, 0.45)
    
    plt.tight_layout()
    plt.savefig('figs/summary_tradeoff_v2.png', dpi=300)
    print("Saved figs/summary_tradeoff_v2.png")

if __name__ == "__main__":
    plot_summary_v2()
