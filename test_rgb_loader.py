import torch
from src.utils.data_loader_ood import CombinedDataManager
from src.models.feature_extractor_rgb import CifarFeatureExtractor, CifarWrapper

def test_rgb():
    print("Testing RGB Data Loading (CIFAR10 vs SVHN)...")
    # Small pool for testing
    dm = CombinedDataManager(
        id_name='cifar10', 
        ood_name='svhn', 
        pool_size=100, 
        ood_ratio=0.5,
        train_size=100
    )
    
    data = dm.get_regression_data()
    X_train = data['X_train']
    print(f"X_train shape: {X_train.shape}")
    assert X_train.shape[1:] == (3, 32, 32), f"Expected (3, 32, 32), got {X_train.shape[1:]}"
    
    print("Testing CifarFeatureExtractor...")
    base_fe = CifarFeatureExtractor(output_dim=128)
    model = CifarWrapper(base_fe, num_classes=10)
    
    out = model(X_train[:5])
    print(f"Model Output shape: {out.shape}")
    assert out.shape == (5, 10), f"Expected (5, 10), got {out.shape}"
    
    feats = model.extract_features(X_train[:5])
    print(f"Features shape: {feats.shape}")
    assert feats.shape == (5, 128), f"Expected (5, 128), got {feats.shape}"
    
    print("RGB Test Passed!")

if __name__ == "__main__":
    test_rgb()
