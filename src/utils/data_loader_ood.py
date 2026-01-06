import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class CombinedDataManager:
    """
    Manages ID and OOD data loading for Active Learning.
    Constructs a mixed pool of In-Distribution and Out-of-Distribution samples.
    """
    def __init__(self, id_name='mnist', ood_name='fashion_mnist', 
                 val_size=100, train_size=10000, pool_size=40000, ood_ratio=0.5, root_dir="./data"):
        self.id_name = id_name
        self.ood_name = ood_name
        self.val_size = val_size
        self.train_size = train_size
        self.pool_size = pool_size
        self.ood_ratio = ood_ratio
        self.root_dir = root_dir
        
        if self.id_name in ['mnist', 'fashion_mnist', 'kmnist']:
            self.mode = 'grayscale'
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif self.id_name in ['cifar10', 'cifar100', 'svhn']:
            self.mode = 'rgb'
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            raise ValueError(f"Unknown ID dataset: {self.id_name}")
            
        self._prepare_datasets()
        self._create_contaminated_pool()
        self._init_training_set()

    def _get_dataset_class(self, name):
        name = name.lower()
        if name == 'mnist': return datasets.MNIST
        elif name == 'fashion_mnist': return datasets.FashionMNIST
        elif name == 'kmnist': return datasets.KMNIST
        elif name == 'cifar10': return datasets.CIFAR10
        elif name == 'cifar100': return datasets.CIFAR100
        elif name == 'svhn': return datasets.SVHN
        else: raise ValueError(f"Unknown dataset: {name}")

    def _load_dataset(self, name, train):
        cls = self._get_dataset_class(name)
        split_arg = {'split': 'train' if train else 'test'} if name == 'svhn' else {'train': train}
        return cls(self.root_dir, download=True, transform=self.transform, **split_arg)

    def _prepare_datasets(self):
        print(f"Loading ID: {self.id_name}...")
        self.id_train_set = self._load_dataset(self.id_name, train=True)
        self.id_test_set = self._load_dataset(self.id_name, train=False)
        
        print(f"Loading OOD: {self.ood_name}...")
        self.ood_train_set = self._load_dataset(self.ood_name, train=True)

    def _extract_data(self, dataset, is_ood=False):
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        X, y = next(iter(loader))
        y = y.long()
        ood_flag = torch.ones(len(y), dtype=torch.bool) if is_ood else torch.zeros(len(y), dtype=torch.bool)
        return X, y, ood_flag

    def _create_contaminated_pool(self):
        X_id, y_id, ood_id = self._extract_data(self.id_train_set, is_ood=False)
        X_ood, y_ood, ood_ood = self._extract_data(self.ood_train_set, is_ood=True)
        
        n_id = len(X_id)
        indices = np.random.permutation(n_id)
        
        # 1. Validation Set
        val_idx = indices[:self.val_size]
        self.X_val, self.y_val, self.ood_val = X_id[val_idx], y_id[val_idx], ood_id[val_idx]
        
        # 2. Initial Train Candidates
        train_source_idx = indices[self.val_size : self.val_size + self.train_size]
        self.X_train_candidates = X_id[train_source_idx]
        self.y_train_candidates = y_id[train_source_idx]
        
        # 3. Pool Construction
        pool_id_idx = indices[self.val_size + self.train_size :]
        X_pool_id, y_pool_id, ood_pool_id = X_id[pool_id_idx], y_id[pool_id_idx], ood_id[pool_id_idx]
        
        n_ood_target = int(self.pool_size * self.ood_ratio)
        n_ind_target = self.pool_size - n_ood_target
        
        # Concatenate and Shuffle
        self.X_pool = torch.cat([X_pool_id[:n_ind_target], X_ood[:n_ood_target]])
        self.y_pool = torch.cat([y_pool_id[:n_ind_target], y_ood[:n_ood_target]])
        self.ood_pool = torch.cat([ood_pool_id[:n_ind_target], ood_ood[:n_ood_target]])
        
        perm = torch.randperm(len(self.X_pool))
        self.X_pool = self.X_pool[perm]
        self.y_pool = self.y_pool[perm]
        self.ood_pool = self.ood_pool[perm]
        
        print(f"Pool Created: {len(self.X_pool)} samples. OOD Ratio: {self.ood_pool.float().mean():.2f}")
        
        # Test Set
        self.X_test, self.y_test, _ = self._extract_data(self.id_test_set, is_ood=False)

    def _init_training_set(self, n_per_class=2):
        """Initialize balanced training set."""
        initial_indices = []
        classes = np.unique(self.y_train_candidates.numpy())
        
        for c in classes:
            idx = np.where(self.y_train_candidates.numpy() == c)[0]
            if len(idx) < n_per_class:
                selected = idx
            else:
                selected = np.random.choice(idx, size=n_per_class, replace=False)
            initial_indices.extend(selected)
            
        self.X_train = self.X_train_candidates[initial_indices]
        self.y_train = self.y_train_candidates[initial_indices]
        self.ood_train = torch.zeros(len(self.y_train), dtype=torch.bool)
        
        print(f"Initialized Training Set: {len(self.X_train)} (Pure ID)")

    def get_regression_data(self):
        n_classes = 100 if self.id_name == 'cifar100' else 10
            
        def to_oh(y, n_classes=n_classes):
            # Clamp for safety with mismatched OOD labels
            y_clamped = y % n_classes
            return torch.eye(n_classes)[y_clamped]

        return {
            "X_train": self.X_train, "y_train": to_oh(self.y_train), "ood_train": self.ood_train,
            "X_val": self.X_val,     "y_val": to_oh(self.y_val),     "ood_val": self.ood_val,
            "X_pool": self.X_pool,   "y_pool": to_oh(self.y_pool),   "ood_pool": self.ood_pool,
            "X_test": self.X_test,   "y_test": to_oh(self.y_test)
        }

    def update_sets(self, query_indices):
        """Move queried points from Pool to Train."""
        new_X = self.X_pool[query_indices]
        new_y = self.y_pool[query_indices]
        new_ood = self.ood_pool[query_indices]
        
        self.X_train = torch.cat([self.X_train, new_X])
        self.y_train = torch.cat([self.y_train, new_y])
        self.ood_train = torch.cat([self.ood_train, new_ood])
        
        mask = torch.ones(len(self.X_pool), dtype=torch.bool)
        mask[query_indices] = False
        
        self.X_pool = self.X_pool[mask]
        self.y_pool = self.y_pool[mask]
        self.ood_pool = self.ood_pool[mask]
        
    def get_pretrain_loader(self, batch_size=64):
        return DataLoader(self.id_train_set, batch_size=batch_size, shuffle=True)
