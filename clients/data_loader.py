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
def get_local_hospital_loader(hospital_id, num_classes=9, batch_size=8, use_smote=True): 
    print(f"\n📂 [Data Loader] Accessing private database for Hospital {hospital_id}...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, f"hospital_{hospital_id}_data")
    
    # 1. Load raw integers
    raw_train_imgs = np.load(os.path.join(folder_path, "train_images.npy"))
    raw_train_lbls = np.load(os.path.join(folder_path, "train_labels.npy"))

    # 2. Conditional Balancing
    if use_smote:
        print(f"   📥 Balancing {len(raw_train_lbls)} Train records via SMOTE...")
        final_imgs, final_lbls = detect_and_balance(raw_train_imgs, raw_train_lbls, num_classes)
    else:
        print(f"   ⚠️ SMOTE DISABLED: Feeding {len(raw_train_lbls)} raw, skewed records to loader...")
        final_imgs, final_lbls = raw_train_imgs, raw_train_lbls
    
    # 3. Format and normalize
    tensor_train_x = format_images_for_pytorch(final_imgs)
    tensor_train_y = torch.tensor(final_lbls, dtype=torch.long)
    
    # 4. RAM-Safe DataLoader
    train_loader = DataLoader(
        TensorDataset(tensor_train_x, tensor_train_y), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,    
        pin_memory=False  
    )
    
    in_channels = tensor_train_x.shape[1]
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