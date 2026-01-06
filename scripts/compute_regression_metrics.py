import numpy as np
import os
import pandas as pd

def compute_regression_metrics():
    # Define the 4 experiments
    experiments = [
        ('BLR (analytic)', 'Predictive variance', 'results/rmse_analytic_predictive_variance.npy'),
        ('MFVI',           'Predictive variance', 'results/rmse_mfvi_predictive_variance.npy'),
        ('BLR (analytic)', 'Random acquisition',  'results/rmse_analytic_random.npy'),
        ('MFVI',           'Random acquisition',  'results/rmse_mfvi_random.npy')
    ]
    
    table1_data = []
    table2_data = []
    
    for method, acq, filepath in experiments:
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            continue
            
        rmse_history = np.load(filepath)
        final_rmse = rmse_history[-1]
        
        # Table 1: Final RMSE
        table1_data.append({
            "Method": method,
            "Acquisition": acq,
            "Final test RMSE": f"{final_rmse:.4f}"
        })
        
        # Table 2: Thresholds
        # Initial size = 20, Query size = 10
        labels_20 = "> 1000"
        labels_15 = "> 1000"
        
        for r, rmse in enumerate(rmse_history):
            n_labels = 20 + r * 10
            if rmse <= 0.20 and labels_20 == "> 1000":
                labels_20 = n_labels
            if rmse <= 0.15 and labels_15 == "> 1000":
                labels_15 = n_labels
                # Don't break, need to find both
        
        # Only add to Table 2 if it's BLR (as per user request, but let's show all for completeness)
        table2_data.append({
            "Method": f"{method} ({acq})",
            "Labels for RMSE <= 0.20": labels_20,
            "Labels for RMSE <= 0.15": labels_15
        })

    print("\n=== Table 1: Final RMSE (1000 labels) ===")
    print(pd.DataFrame(table1_data))
    
    print("\n=== Table 2: Efficiency (Labels needed) ===")
    print(pd.DataFrame(table2_data))

if __name__ == "__main__":
    compute_regression_metrics()
