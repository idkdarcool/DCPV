import numpy as np
import pandas as pd
import os

def analyze_ood():
    # Define experiments to analyze
    experiments = [
        # Phase 2 Original (MNIST vs Fashion)
        ('MNIST', 'Fashion', 'analytic', 'random', 'Random'),
        ('MNIST', 'Fashion', 'analytic', 'predictive_variance', 'PV'),
        ('MNIST', 'Fashion', 'analytic', 'robust_pv', 'Robust PV'),
        ('MNIST', 'Fashion', 'analytic', 'sin_only', 'InD Score'),
        ('mnist', 'fashion_mnist', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'), # New GMM run
        ('MNIST', 'Fashion', 'mfvi', 'predictive_variance', 'MFVI-PV'),
        
        # Tier 1: Exp 1 (Fashion vs MNIST)
        ('fashion_mnist', 'mnist', 'analytic', 'random', 'Random'),
        ('fashion_mnist', 'mnist', 'analytic', 'predictive_variance', 'PV'),
        ('fashion_mnist', 'mnist', 'analytic', 'robust_pv', 'Robust PV'),
        ('fashion_mnist', 'mnist', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
        
        # Tier 1: Exp 2 (MNIST vs KMNIST)
        ('mnist', 'kmnist', 'analytic', 'random', 'Random'),
        ('mnist', 'kmnist', 'analytic', 'predictive_variance', 'PV'),
        ('mnist', 'kmnist', 'analytic', 'robust_pv', 'Robust PV'),
        ('mnist', 'kmnist', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
        
        # Tier 2: Exp 3 (CIFAR10 vs CIFAR100)
        ('cifar10', 'cifar100', 'analytic', 'random', 'Random'),
        ('cifar10', 'cifar100', 'analytic', 'predictive_variance', 'PV'),
        ('cifar10', 'cifar100', 'analytic', 'robust_pv', 'Robust PV'),
        ('cifar10', 'cifar100', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
        
        # Tier 2: Exp 4 (CIFAR10 vs SVHN)
        ('cifar10', 'svhn', 'analytic', 'random', 'Random'),
        ('cifar10', 'svhn', 'analytic', 'predictive_variance', 'PV'),
        ('cifar10', 'svhn', 'analytic', 'robust_pv', 'Robust PV'),
        ('cifar10', 'svhn', 'analytic', 'robust_pv_gmm', 'Robust GMM PV'),
    ]
    
    results = []
    
    for id_name, ood_name, method, acq, label in experiments:
        # Handle legacy filenames for original experiment
        if id_name == 'MNIST' and ood_name == 'Fashion':
            rmse_path = f'results_ood/rmse_{method}_{acq}.npy'
            rate_path = f'results_ood/ood_rate_{method}_{acq}.npy'
            exp_label = "MNIST vs Fashion"
        else:
            rmse_path = f'results_ood/rmse_{id_name}_{ood_name}_{method}_{acq}.npy'
            rate_path = f'results_ood/ood_rate_{id_name}_{ood_name}_{method}_{acq}.npy'
            exp_label = f"{id_name} vs {ood_name}"
            
        # Normalize labels for grouping
        if id_name == 'mnist' and ood_name == 'fashion_mnist':
            exp_label = "MNIST vs Fashion"
            
        if os.path.exists(rmse_path) and os.path.exists(rate_path):
            rmse = np.load(rmse_path)
            rate = np.load(rate_path)
            
            final_rmse = rmse[-1]
            # Total OOD queries = sum(rate * query_size). query_size=10.
            total_ood = np.sum(rate * 10)
            ood_percent = (total_ood / (len(rate) * 10)) * 100
            
            results.append({
                'Experiment': exp_label,
                'Method': label,
                'Final RMSE': final_rmse,
                'Total OOD': int(total_ood),
                'OOD %': ood_percent
            })
            
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    df = analyze_ood()
    if not df.empty:
        print(df.to_string())
    else:
        print("No results found.")
