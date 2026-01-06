import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rmse():
    if not os.path.exists('results'):
        print("No results directory found.")
        return

    plt.figure(figsize=(10, 6))
    
    methods = ['analytic', 'mfvi']
    for method in methods:
        path = f'results/rmse_{method}.npy'
        if os.path.exists(path):
            rmse = np.load(path)
            plt.plot(rmse, label=f'BLR-{method.upper()}')
            print(f"Loaded {method}: {rmse}")
    
    plt.xlabel('AL Rounds')
    plt.ylabel('RMSE')
    plt.title('Regression Active Learning on MNIST')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/rmse_plot.png')
    print("Plot saved to results/rmse_plot.png")

if __name__ == "__main__":
    plot_rmse()
