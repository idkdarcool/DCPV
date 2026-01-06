import numpy as np
import os
import pandas as pd

def generate_table():
    experiments = [
        ('MNIST', 'Fashion', 'analytic', 'random', 'Random'),
        ('MNIST', 'Fashion', 'analytic', 'predictive_variance', 'PV'),
        ('MNIST', 'Fashion', 'analytic', 'robust_pv', 'Robust PV'),
        ('MNIST', 'Fashion', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
        
        ('fashion_mnist', 'mnist', 'analytic', 'random', 'Random'),
        ('fashion_mnist', 'mnist', 'analytic', 'predictive_variance', 'PV'),
        ('fashion_mnist', 'mnist', 'analytic', 'robust_pv', 'Robust PV'),
        ('fashion_mnist', 'mnist', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
        
        ('mnist', 'kmnist', 'analytic', 'random', 'Random'),
        ('mnist', 'kmnist', 'analytic', 'predictive_variance', 'PV'),
        ('mnist', 'kmnist', 'analytic', 'robust_pv', 'Robust PV'),
        ('mnist', 'kmnist', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
        
        ('cifar10', 'cifar100', 'analytic', 'random', 'Random'),
        ('cifar10', 'cifar100', 'analytic', 'predictive_variance', 'PV'),
        ('cifar10', 'cifar100', 'analytic', 'robust_pv', 'Robust PV'),
        ('cifar10', 'cifar100', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
        
        ('cifar10', 'svhn', 'analytic', 'random', 'Random'),
        ('cifar10', 'svhn', 'analytic', 'predictive_variance', 'PV'),
        ('cifar10', 'svhn', 'analytic', 'robust_pv', 'Robust PV'),
        ('cifar10', 'svhn', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
    ]
    
    rows = []
    
    for id_name, ood_name, method, acq, label in experiments:
        # Resolve filenames
        if id_name == 'MNIST' and ood_name == 'Fashion':
             rmse_path = f'results_ood/rmse_{method}_{acq}.npy'
             rate_path = f'results_ood/ood_rate_{method}_{acq}.npy'
             exp_name = "MNIST $\\to$ Fashion"
        else:
             rmse_path = f'results_ood/rmse_{id_name}_{ood_name}_{method}_{acq}.npy'
             rate_path = f'results_ood/ood_rate_{id_name}_{ood_name}_{method}_{acq}.npy'
             if id_name == 'fashion_mnist': exp_name = "Fashion $\\to$ MNIST"
             elif ood_name == 'kmnist': exp_name = "MNIST $\\to$ KMNIST"
             elif ood_name == 'cifar100': exp_name = "CIFAR-10 $\\to$ C-100"
             elif ood_name == 'svhn': exp_name = "CIFAR-10 $\\to$ SVHN"
             else: exp_name = f"{id_name} -> {ood_name}"

        if os.path.exists(rmse_path) and os.path.exists(rate_path):
            r = np.load(rmse_path)
            rate = np.load(rate_path)
            final_rmse = r[-1]
            total_ood = np.sum(rate * 10) # 10 queries per round
            ood_pct = (total_ood / (len(rate)*10)) * 100
            
            # Highlight Best (Lowest RMSE, Lowest OOD) logic could be added here
            # For now, just raw data
            rows.append(f"{exp_name} & {label} & {ood_pct:.1f}\% & {final_rmse:.3f} \\\\")
            if 'GMM' in label:
                rows.append("\\midrule")
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{llcc}")
    print("\\toprule")
    print("Experiment & Method & OOD Selection (\\%) & Final ID RMSE \\\\")
    print("\\midrule")
    for r in rows:
        print(r)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Summary of OOD Active Learning Performance (Extended to 50 Rounds)}")
    print("\\label{tab:ood_summary}")
    print("\\end{table}")

if __name__ == "__main__":
    generate_table()
