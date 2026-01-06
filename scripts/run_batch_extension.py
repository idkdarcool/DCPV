import os
import subprocess
import time

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

experiments = [
    # Fashion -> MNIST
    ('--id_dataset fashion_mnist --ood_dataset mnist', ['random', 'predictive_variance', 'robust_pv', 'robust_pv_gmm']),
    # MNIST -> KMNIST
    ('--id_dataset mnist --ood_dataset kmnist', ['random', 'predictive_variance', 'robust_pv', 'robust_pv_gmm']),
    # CIFAR10 -> C100
    ('--id_dataset cifar10 --ood_dataset cifar100', ['random', 'predictive_variance', 'robust_pv', 'robust_pv_gmm']),
    # CIFAR10 -> SVHN
    ('--id_dataset cifar10 --ood_dataset svhn', ['random', 'predictive_variance', 'robust_pv', 'robust_pv_gmm'])
]

def main():
    for dataset_args, acqs in experiments:
        for acq in acqs:
            # Check if 50 rounds already exist?
            # We can force overwrite or just run. 
            # Ideally we check audit, but let's just run.
            cmd = f"python3 main_ood.py {dataset_args} --acquisition {acq} --rounds 50 --query_size 10"
            try:
                run_cmd(cmd)
            except subprocess.CalledProcessError as e:
                print(f"Error running {cmd}: {e}")

if __name__ == "__main__":
    main()
