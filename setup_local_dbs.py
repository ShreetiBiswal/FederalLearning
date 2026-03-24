import os
import numpy as np
import medmnist
from medmnist import INFO

def generate_biased_hospital_databases(num_hospitals=4, alpha=0.05):
    print("🚀 Starting STRICT DIRICHLET Skewed Local Database Generation...")

    # 1. Load MedMNIST (Train AND Val splits)
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    print("📥 Downloading/Loading PathMNIST Train Set...")
    train_dataset = DataClass(split='train', download=True, root='./data') 
    
    print("📥 Downloading/Loading PathMNIST Val (Evaluation) Set...")
    val_dataset = DataClass(split='val', download=True, root='./data') 

    all_train_images = train_dataset.imgs
    all_train_labels = train_dataset.labels.squeeze()   
    num_classes = len(info['label'])
    
    print(f"Total raw training images loaded: {len(all_train_images)}")

    # 2. Group the training data strictly by disease class
    class_indices = {}
    for c in range(num_classes):
        class_indices[c] = np.where(all_train_labels == c)[0].tolist()

    hospital_indices = {i: [] for i in range(num_hospitals)}

    # 3. The Dirichlet Split (True Non-IID)
    print(f"📉 Distributing diseases using Dirichlet Distribution (Alpha = {alpha})...")
    np.random.seed(42) # Seeded for reproducibility
    
    for c in range(num_classes):
        idx_c = class_indices[c]
        np.random.shuffle(idx_c)
        
        # Draw random proportions for this specific disease across the 4 hospitals
        # With alpha=0.05, one hospital usually gets almost all of the images for this class!
        proportions = np.random.dirichlet(np.repeat(alpha, num_hospitals))
        
        # Convert proportions into actual image counts
        counts = np.round(proportions * len(idx_c)).astype(int)
        
        # Fix any minor rounding errors to ensure we don't lose or invent images
        counts[-1] = len(idx_c) - np.sum(counts[:-1])
        
        # Distribute the images for this class to the hospitals
        current_idx = 0
        for h in range(num_hospitals):
            hospital_indices[h].extend(idx_c[current_idx : current_idx + counts[h]])
            current_idx += counts[h]

    # 4. Extract and Save to Physical Folders
    base_dir = "clients"
    os.makedirs(base_dir, exist_ok=True)

    # A. Save the individual Hospital Training Data
    for h in range(num_hospitals):
        hospital_id = h + 1
        folder_path = os.path.join(base_dir, f"hospital_{hospital_id}_data")
        os.makedirs(folder_path, exist_ok=True)
        
        # Shuffle the hospital's final pile so classes aren't clustered together in batches
        np.random.shuffle(hospital_indices[h])
        
        final_h_images = all_train_images[hospital_indices[h]]
        final_h_labels = all_train_labels[hospital_indices[h]]
        
        np.save(os.path.join(folder_path, "train_images.npy"), final_h_images)
        np.save(os.path.join(folder_path, "train_labels.npy"), final_h_labels)
        
        print(f"✅ Hospital {hospital_id} populated: {len(final_h_labels)} training images saved.")

    # B. Save the Common Evaluation Data
    print("\n🌍 Creating the Common Global Evaluation Folder...")
    eval_folder = os.path.join(base_dir, "global_test_data")
    os.makedirs(eval_folder, exist_ok=True)
    
    np.save(os.path.join(eval_folder, "val_images.npy"), val_dataset.imgs)
    np.save(os.path.join(eval_folder, "val_labels.npy"), val_dataset.labels.squeeze())
    
    print(f"✅ Common Evaluation Set saved: {len(val_dataset.labels)} images.")

    print("📥 Downloading/Loading PathMNIST Test Set...")
    test_dataset = DataClass(split='test', download=True, root='./data') 
    
    np.save(os.path.join(eval_folder, "test_images.npy"), test_dataset.imgs)
    np.save(os.path.join(eval_folder, "test_labels.npy"), test_dataset.labels.squeeze())

if __name__ == "__main__":
    generate_biased_hospital_databases()