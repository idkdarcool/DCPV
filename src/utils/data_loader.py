import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class MnistDataManager:
    """
    Manages data loading and splitting for MNIST Active Learning.
    Supports both classification and regression targets.
    """
    def __init__(self, val_size=100, train_size=10000, pool_size=49900, root_dir="./data"):
        self.val_size = val_size
        self.train_size = train_size
        self.pool_size = pool_size
        self.root_dir = root_dir
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self._prepare_datasets()
        self._split_data()
        self._init_training_set()

    def _prepare_datasets(self):
        """Download and load MNIST datasets."""
        # We assume data might already be there or we download it
        # Using download=True usually handles existence checks
        self.train_dataset = datasets.MNIST(
            self.root_dir, train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            self.root_dir, train=False, download=True, transform=self.transform
        )

    def _split_data(self):
        """Split training data into Train, Validation, and Pool."""
        # Ensure sizes match
        total_train = len(self.train_dataset)
        if self.train_size + self.val_size + self.pool_size > total_train:
             # Adjust pool size if needed to fit 60000
             self.pool_size = total_train - self.train_size - self.val_size
        
        self.train_subset, self.val_subset, self.pool_subset = random_split(
            self.train_dataset, 
            [self.train_size, self.val_size, self.pool_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create loaders to extract data easily
        # Note: In AL we often work with indices or full arrays, 
        # but here we'll extract them to numpy/tensors for the AL loop
        self.X_pool, self.y_pool = self._extract_data(self.pool_subset)
        self.X_val, self.y_val = self._extract_data(self.val_subset)
        self.X_test, self.y_test = self._extract_data(self.test_dataset)
        
        # Initial training pool (subset of train_subset)
        # We don't use the whole train_subset, we pick from it
        self.X_train_candidates, self.y_train_candidates = self._extract_data(self.train_subset)

    def _extract_data(self, dataset):
        """Helper to extract all data from a dataset/subset."""
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        return next(iter(loader))

    def _init_training_set(self, n_per_class=2):
        """Initialize a balanced small training set."""
        initial_indices = []
        classes = np.unique(self.y_train_candidates.numpy())
        
        for c in classes:
            # Find indices of this class
            idx = np.where(self.y_train_candidates.numpy() == c)[0]
            # Select n_per_class random indices
            selected = np.random.choice(idx, size=n_per_class, replace=False)
            initial_indices.extend(selected)
            
        self.X_train = self.X_train_candidates[initial_indices]
        self.y_train = self.y_train_candidates[initial_indices]
        
        # Remove these from candidates if we were treating candidates as a pool, 
        # but here 'pool' is separate. 
        # The 'train_subset' is just a source for the initial set in this logic?
        # Actually, usually 'pool' is where we query from. 
        # The original code had a specific logic. We'll stick to:
        # X_train: Initial labeled set
        # X_pool: Unlabeled set (we query from here)
        
        print(f"Initialized training set size: {len(self.X_train)}")

    def get_regression_data(self):
        """
        Returns data formatted for regression (One-Hot Encoded Targets).
        """
        y_train_oh = self._to_one_hot(self.y_train)
        y_val_oh = self._to_one_hot(self.y_val)
        y_pool_oh = self._to_one_hot(self.y_pool)
        y_test_oh = self._to_one_hot(self.y_test)
        
        return {
            "X_train": self.X_train, "y_train": y_train_oh,
            "X_val": self.X_val, "y_val": y_val_oh,
            "X_pool": self.X_pool, "y_pool": y_pool_oh,
            "X_test": self.X_test, "y_test": y_test_oh
        }

    def update_sets(self, query_indices):
        """Move queried indices from pool to train."""
        # Get data to move
        new_X = self.X_pool[query_indices]
        new_y = self.y_pool[query_indices]
        
        # Append to train
        self.X_train = torch.cat([self.X_train, new_X])
        self.y_train = torch.cat([self.y_train, new_y])
        
        # Remove from pool
        # Create mask
        mask = torch.ones(len(self.X_pool), dtype=torch.bool)
        mask[query_indices] = False
        self.X_pool = self.X_pool[mask]
        self.y_pool = self.y_pool[mask]
        
    def _to_one_hot(self, labels, num_classes=10):
        """Convert class labels to one-hot vectors."""
        return torch.eye(num_classes)[labels]

    def get_train_loader(self, batch_size=64):
        # Create a dataset from current X_train, y_train
        # Note: y_train is stored as tensor, for classification we need LongTensor labels
        # But our X_train/y_train are updated in update_sets.
        # Check if y_train is one-hot or labels?
        # In __init__, we extracted them. 
        # In get_regression_data, we convert to one-hot.
        # So self.y_train stores raw labels (from _extract_data).
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size=1000):
        dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_pool_loader(self, batch_size=None):
        if batch_size is None:
            batch_size = len(self.X_pool)
        dataset = torch.utils.data.TensorDataset(self.X_pool, self.y_pool)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_pretrain_loader(self, batch_size=64):
        """Returns loader for pre-training FE on Train + Pool (all available training data)."""
        # Combine train and pool subsets
        combined_dataset = torch.utils.data.ConcatDataset([self.train_subset, self.pool_subset])
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)


