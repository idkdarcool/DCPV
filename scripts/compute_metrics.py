import numpy as np
import os
import pandas as pd

def compute_milestones():
    if not os.path.exists('results/classification_results_bayes.npy'):
        print("Results file not found.")
        return

    results = np.load('results/classification_results_bayes.npy', allow_pickle=True).item()
    
    milestones = []
    
    for strategy, accuracies in results.items():
        # Initial set size = 20, Query size = 10
        # Round 0 = 20 labels
        # Round r = 20 + r * 10 labels
        
        labels_90 = "> 1000"
        labels_95 = "> 1000"
        
        for r, acc in enumerate(accuracies):
            n_labels = 20 + r * 10
            if acc >= 0.90 and labels_90 == "> 1000":
                labels_90 = n_labels
            if acc >= 0.95 and labels_95 == "> 1000":
                labels_95 = n_labels
                break # Found both
        
        milestones.append({
            "Strategy": strategy,
            "Labels for 90%": labels_90,
            "Labels for 95%": labels_95,
            "Final Acc": accuracies[-1]
        })
        
    df = pd.DataFrame(milestones)
    print("\n=== Efficiency Table (Bayesian CNN) ===")
    print(df)

if __name__ == "__main__":
    compute_milestones()
