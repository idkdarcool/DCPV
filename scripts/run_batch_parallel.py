import os
import subprocess
import threading

def run_experiment_set(dataset_args, acqs):
    for acq in acqs:
        cmd = f"python3 main_ood.py {dataset_args} --acquisition {acq} --rounds 50 --query_size 10"
        print(f"[{dataset_args}] Starting {acq}...")
        try:
            subprocess.check_call(cmd, shell=True)
            print(f"[{dataset_args}] Model {acq} DONE.")
        except subprocess.CalledProcessError as e:
            print(f"[{dataset_args}] Model {acq} FAILED: {e}")

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
    threads = []
    print("Starting 4 parallel experiment threads...")
    for dataset_args, acqs in experiments:
        t = threading.Thread(target=run_experiment_set, args=(dataset_args, acqs))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    print("All parallel experiments completed.")

if __name__ == "__main__":
    main()
