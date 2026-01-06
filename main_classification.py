import torch
import numpy as np
import argparse
import os
import wandb
from src.models.feature_extractor import MnistFeatureExtractor
from src.utils.data_loader import MnistDataManager
from src.core.active_loop import ClassificationActiveLoop
from src.utils.visualization import plot_al_curve

def main():
    parser = argparse.ArgumentParser(description='Bayesian Active Classification on MNIST')
    parser.add_argument('--acq_func', type=str, default='all', 
                        choices=['all', 'random', 'entropy', 'bald', 'var_ratios', 'mean_std'],
                        help='Acquisition function')
    parser.add_argument('--rounds', type=int, default=100, help='Number of AL rounds')
    parser.add_argument('--query_size', type=int, default=10, help='Number of queries per round')
    parser.add_argument('--dropout_iter', type=int, default=100, help='MC Dropout iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='mnist-active-learning', help='WANDB project name')
    parser.add_argument('--deterministic', action='store_true', help='Run in deterministic mode (no MC dropout)')
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define methods to run
    if args.acq_func == 'all':
        methods = ['random', 'entropy', 'bald', 'var_ratios', 'mean_std']
    else:
        methods = [args.acq_func]

    all_results = {}

    for method in methods:
        print(f"\n=== Running Acquisition: {method} ===")
        
        # Initialize WANDB run
        run_name = f"{method}_seed{args.seed}"
        if args.deterministic:
            run_name += "_det"
            
        wandb.init(project=args.wandb_project, name=run_name, config=args)

        
        # Reset Data Manager for each method (fresh start)
        data_manager = MnistDataManager()
        
        # Model
        model = MnistFeatureExtractor().to(device)
        
        # Active Loop
        loop = ClassificationActiveLoop(model, device, use_wandb=True)
        
        accuracies = []
        
        # Initial Training (Round 0)
        print("Initial Training...")

        from src.models.train_utils import train_classifier
        train_loader = data_manager.get_train_loader()
        val_loader = data_manager.get_val_loader()
        model, val_acc = train_classifier(model, train_loader, val_loader, device, epochs=50, use_wandb=True)
        accuracies.append(val_acc)
        print(f"Round 0 Acc: {val_acc:.4f}")
        
        # AL Loop
        for r in range(args.rounds):
            print(f"--- Round {r+1}/{args.rounds} ---")
            
            # Run Round
            val_acc, query_indices = loop.run_round(
                data_manager, args.query_size, method, 
                dropout_iter=args.dropout_iter, deterministic=args.deterministic
            )

            
            accuracies.append(val_acc)
            print(f"Round {r+1} Acc: {val_acc:.4f}")
            
            # Update Data
            data_manager.update_sets(query_indices)
            
            # Log to WANDB
            wandb.log({"round": r+1, "test_acc": val_acc, "labeled_pool_size": len(data_manager.X_train)})
            
        all_results[method] = accuracies
        wandb.finish()

    # Save and Plot Final Results
    if not os.path.exists('results'):
        os.makedirs('results')
        
    suffix = "_det" if args.deterministic else "_bayes"
    filename = f'results/classification_results{suffix}.npy'
    np.save(filename, all_results)
    print(f"Results saved to {filename}")
    
    plot_al_curve(all_results, title=f"MNIST Active Classification ({'Deterministic' if args.deterministic else 'Bayesian'})")


if __name__ == "__main__":
    main()
