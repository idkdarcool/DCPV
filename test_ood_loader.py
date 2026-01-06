from src.utils.data_loader_ood import CombinedDataManager
import torch

def test_loader():
    print("Initializing CombinedDataManager...")
    dm = CombinedDataManager(pool_size=1000, ood_ratio=0.5)
    
    data = dm.get_regression_data()
    
    print("\n--- Verification ---")
    print(f"Train Size: {len(data['X_train'])}")
    print(f"Train OOD Count: {data['ood_train'].sum()} (Should be 0)")
    
    print(f"Pool Size: {len(data['X_pool'])}")
    print(f"Pool OOD Count: {data['ood_pool'].sum()} (Should be ~500)")
    
    print(f"Val Size: {len(data['X_val'])}")
    print(f"Val OOD Count: {data['ood_val'].sum()} (Should be 0)")
    
    # Check shapes
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    
    assert data['ood_train'].sum() == 0, "Train set has OOD data!"
    assert data['ood_pool'].sum() > 0, "Pool has no OOD data!"
    
    print("\nSUCCESS: OOD Loader works as expected.")

if __name__ == "__main__":
    test_loader()
