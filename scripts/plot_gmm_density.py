import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.models.feature_extractor import MnistFeatureExtractor
from src.models.ood_detector import GMMScorer
from src.utils.data_loader_ood import CombinedDataManager

def plot_gmm_density():
    # Load Scores (Fast - from NPZ)
    path_mf = 'results_ood/gmm_scores_mnist_fashion.npz'
    path_cc = 'results_ood/gmm_scores_cifar10_cifar100.npz'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MNIST vs Fashion
    if os.path.exists(path_mf):
        d = np.load(path_mf)
        sns.histplot(d['id'], color='blue', label='ID (MNIST)', kde=True, stat="density", binwidth=0.02, ax=axes[0], alpha=0.5)
        sns.histplot(d['ood'], color='red', label='OOD (Fashion)', kde=True, stat="density", binwidth=0.02, ax=axes[0], alpha=0.5)
        axes[0].set_title("Paradox (M->F): OOD has Higher Density!", fontsize=12)
        axes[0].set_xlabel("GMM Score (Normalized)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: C10 vs C100
    if os.path.exists(path_cc):
        d = np.load(path_cc)
        sns.histplot(d['id'], color='blue', label='ID (CIFAR-10)', kde=True, stat="density", binwidth=0.02, ax=axes[1], alpha=0.5)
        sns.histplot(d['ood'], color='red', label='OOD (CIFAR-100)', kde=True, stat="density", binwidth=0.02, ax=axes[1], alpha=0.5)
        axes[1].set_title("Failure (C10->C100): Total Overlap", fontsize=12)
        axes[1].set_xlabel("GMM Score (Normalized)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs/gmm_density_paradox.png', dpi=300)
    print("Saved figs/gmm_density_paradox.png")

if __name__ == "__main__":
    plot_gmm_density()
