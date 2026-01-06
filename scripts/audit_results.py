import numpy as np
import glob
import os

def audit():
    files = glob.glob('results_ood/ood_rate_*.npy')
    print(f"{'File':<60} | {'Rounds':<6}")
    print("-" * 70)
    for f in sorted(files):
        # Skip legacy if short
        data = np.load(f)
        print(f"{os.path.basename(f):<60} | {len(data):<6}")

if __name__ == "__main__":
    audit()
