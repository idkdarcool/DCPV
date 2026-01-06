import torch
import numpy as np
import os
from src.models.feature_extractor import MnistFeatureExtractor
from src.models.feature_extractor_rgb import CifarFeatureExtractor, CifarWrapper
from src.models.ood_detector import GMMScorer
from src.utils.data_loader_ood import CombinedDataManager
from src.models.train_utils import train_feature_extractor
import torchvision
from torchvision import transforms

def generate_scores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists('results_ood'):
        os.makedirs('results_ood')

    # --- 1. MNIST vs Fashion ---
    print("\n--- Processing MNIST vs Fashion ---")
    data_manager = CombinedDataManager('mnist', 'fashion_mnist', pool_size=1000, ood_ratio=0.5)
    fe = MnistFeatureExtractor().to(device)
    fe_loader = data_manager.get_pretrain_loader()
    fe = train_feature_extractor(fe, fe_loader, device, epochs=3) # 3 epochs enough for viz
    fe.eval()
    
    # Fit GMM
    all_feats = []
    with torch.no_grad():
        for x, _ in fe_loader:
            all_feats.append(fe.extract_features(x.to(device)))
    phi_train = torch.cat(all_feats)
    scorer = GMMScorer(n_components=10)
    scorer.fit(phi_train)
    
    # Score
    # ID Test
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=500, shuffle=False)
    id_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            s = scorer.score(fe.extract_features(x.to(device)))
            id_scores.append(s.cpu().numpy())
    id_scores = np.concatenate(id_scores)
    
    # OOD Test
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    ood_loader = torch.utils.data.DataLoader(fashion_test, batch_size=500, shuffle=False)
    ood_scores = []
    with torch.no_grad():
        for x, _ in ood_loader:
             s = scorer.score(fe.extract_features(x.to(device)))
             ood_scores.append(s.cpu().numpy())
    ood_scores = np.concatenate(ood_scores)
    
    np.savez('results_ood/gmm_scores_mnist_fashion.npz', id=id_scores, ood=ood_scores)
    print("Saved results_ood/gmm_scores_mnist_fashion.npz")

    # --- 2. CIFAR-10 vs CIFAR-100 ---
    print("\n--- Processing CIFAR-10 vs CIFAR-100 ---")
    # Clean up memory
    del data_manager, fe, fe_loader, scorer, phi_train
    torch.cuda.empty_cache()
    
    data_manager = CombinedDataManager('cifar10', 'cifar100', pool_size=1000, ood_ratio=0.5)
    base_fe = CifarFeatureExtractor(output_dim=128)
    fe = CifarWrapper(base_fe, num_classes=10).to(device)
    fe_loader = data_manager.get_pretrain_loader()
    fe = train_feature_extractor(fe, fe_loader, device, epochs=5) # 5 epochs for RGB
    fe.eval()
    
    # Fit GMM
    all_feats = []
    with torch.no_grad():
        for x, _ in fe_loader:
            all_feats.append(fe.extract_features(x.to(device)))
    phi_train = torch.cat(all_feats)
    scorer = GMMScorer(n_components=10)
    scorer.fit(phi_train)
    
    # Score
    # ID Test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=500, shuffle=False)
    id_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            s = scorer.score(fe.extract_features(x.to(device)))
            id_scores.append(s.cpu().numpy())
    id_scores = np.concatenate(id_scores)
    
    # OOD Test
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    ood_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=500, shuffle=False)
    ood_scores = []
    with torch.no_grad():
        for x, _ in ood_loader:
             s = scorer.score(fe.extract_features(x.to(device)))
             ood_scores.append(s.cpu().numpy())
    ood_scores = np.concatenate(ood_scores)
    
    np.savez('results_ood/gmm_scores_cifar10_cifar100.npz', id=id_scores, ood=ood_scores)
    print("Saved results_ood/gmm_scores_cifar10_cifar100.npz")

if __name__ == "__main__":
    generate_scores()
