import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import the standalone balancing module we built earlier
from data_balancer import detect_and_balance

def format_images_for_pytorch(raw_images):
    """ Helper to swap channels and normalize pixels to [-1.0, 1.0] """
    if len(raw_images.shape) == 4:
        formatted_images = np.transpose(raw_images, (0, 3, 1, 2))
    elif len(raw_images.shape) == 3:
        formatted_images = np.expand_dims(raw_images, axis=1)
    
    tensor_x = torch.tensor(formatted_images, dtype=torch.float32) / 255.0
    return (tensor_x - 0.5) / 0.5

# --- 1. The Local (Private) Data Loader ---
def get_local_hospital_loader(hospital_id, num_classes=9, batch_size=16):
    """ Reads isolated Train files. Applies SMOTE ONLY to Train. """
    print(f"\n📂 [Data Loader] Accessing private database for Hospital {hospital_id}...")
    
    folder_path = f"clients/hospital_{hospital_id}_data"
    
    try:
        # Notice we ONLY load the training data now!
        raw_train_imgs = np.load(os.path.join(folder_path, "train_images.npy"))
        raw_train_lbls = np.load(os.path.join(folder_path, "train_labels.npy"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing data for Hospital {hospital_id}. Run setup_local_dbs.py first!")

    # Balance ONLY the Training Data
    print(f"   📥 Balancing {len(raw_train_lbls)} Train records via SMOTE...")
    
    # We apply the PyTorch normalization before sending it to SMOTE
    normalized_numpy_imgs = format_images_for_pytorch(raw_train_imgs).numpy()
    balanced_train_imgs, balanced_train_lbls = detect_and_balance(normalized_numpy_imgs, raw_train_lbls, num_classes)
    
    # Convert to Tensors
    tensor_train_x = torch.tensor(balanced_train_imgs, dtype=torch.float32)
    tensor_train_y = torch.tensor(balanced_train_lbls, dtype=torch.long)
    
    # Wrap and Load
    train_loader = DataLoader(TensorDataset(tensor_train_x, tensor_train_y), batch_size=batch_size, shuffle=True)
    in_channels = tensor_train_x.shape[1]
    
    print(f"   ✅ Local Loader ready! Train: {len(train_loader)} batches")
    
    # Returns only 2 things now!
    return train_loader, in_channels


# --- 2. The Global (Shared) Validation Loader ---
def get_global_val_loader(batch_size=16):
    """ Reads the shared validation dataset for global model evaluation. """
    print(f"\n🌍 [Data Loader] Accessing shared Global Validation database...")
    
    folder_path = "clients/global_test_data"
    
    try:
        raw_val_imgs = np.load(os.path.join(folder_path, "val_images.npy"))
        raw_val_lbls = np.load(os.path.join(folder_path, "val_labels.npy"))
    except FileNotFoundError:
        raise FileNotFoundError("Missing Global Validation data. Run setup_local_dbs.py first!")

    # Format and Convert
    tensor_val_x = format_images_for_pytorch(raw_val_imgs)
    tensor_val_y = torch.tensor(raw_val_lbls, dtype=torch.long)
    
    val_loader = DataLoader(TensorDataset(tensor_val_x, tensor_val_y), batch_size=batch_size, shuffle=False)
    
    print(f"   ✅ Global Val Loader ready! {len(val_loader)} batches")
    return val_loader


# --- 3. The Dummy Loader (For rapid testing) ---
def get_dummy_loaders(hospital_id, num_classes=9, batch_size=5):
    """ Generates fake data for Train and Val to test the pipeline instantly. """
    print(f"\n🧪 [Dummy Loader] Generating fake dataset for Hospital {hospital_id}...")
    
    # Train
    train_dataset = TensorDataset(torch.randn(10, 3, 28, 28), torch.randint(0, num_classes, (10,)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Val
    val_dataset = TensorDataset(torch.randn(5, 3, 28, 28), torch.randint(0, num_classes, (5,)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    in_channels = 3 
    print("   ✅ Dummy Loaders ready!")
    
    # Returns 3 things to match our new logic!
    return train_loader, val_loader, in_channels