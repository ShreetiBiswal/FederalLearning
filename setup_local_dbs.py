import os
import numpy as np
import medmnist
from medmnist import INFO

def generate_biased_hospital_databases(num_hospitals=4, min_samples=15):
    print("🚀 Starting Skewed Local Database Generation...")

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

    # 2. Group the training data by disease class
    class_indices = {}
    for c in range(num_classes):
        class_indices[c] = np.where(all_train_labels == c)[0].tolist()

    hospital_indices = {i: [] for i in range(num_hospitals)}

    # 3. The SMOTE Lifeline (Training Data Only)
    print(f"🛡️ Distributing baseline of {min_samples} samples per class to each hospital...")
    for c in range(num_classes):
        for h in range(num_hospitals):
            baseline_chunk = class_indices[c][:min_samples]
            hospital_indices[h].extend(baseline_chunk)
            class_indices[c] = class_indices[c][min_samples:]

    # 4. Gather everything that is left over
    remaining_indices = []
    for c in range(num_classes):
        remaining_indices.extend(class_indices[c])
    np.random.shuffle(remaining_indices)

    # 5. The Massive Skew (60%, 25%, 10%, 5%)
    proportions = [0.60, 0.25, 0.10, 0.05]
    total_remaining = len(remaining_indices)
    
    print(f"📉 Distributing the remaining {total_remaining} images with a massive skew...")
    current_idx = 0
    for h in range(num_hospitals):
        if h == num_hospitals - 1:
            count = total_remaining - current_idx 
        else:
            count = int(total_remaining * proportions[h])
            
        hospital_indices[h].extend(remaining_indices[current_idx : current_idx + count])
        current_idx += count

    # 6. Extract and Save to Physical Folders
    base_dir = "clients"
    os.makedirs(base_dir, exist_ok=True)

    # A. Save the individual Hospital Training Data
    for h in range(num_hospitals):
        hospital_id = h + 1
        folder_path = os.path.join(base_dir, f"hospital_{hospital_id}_data")
        os.makedirs(folder_path, exist_ok=True)
        
        final_h_images = all_train_images[hospital_indices[h]]
        final_h_labels = all_train_labels[hospital_indices[h]]
        
        np.save(os.path.join(folder_path, "train_images.npy"), final_h_images)
        np.save(os.path.join(folder_path, "train_labels.npy"), final_h_labels)
        
        print(f"✅ Hospital {hospital_id} populated: {len(final_h_labels)} training images saved.")

    # B. Save the Common Evaluation Data
    print("\n🌍 Creating the Common Global Evaluation Folder...")
    eval_folder = os.path.join(base_dir, "global_test_data")
    os.makedirs(eval_folder, exist_ok=True)
    
    # Save the entire validation dataset here for all hospitals to share
    np.save(os.path.join(eval_folder, "val_images.npy"), val_dataset.imgs)
    np.save(os.path.join(eval_folder, "val_labels.npy"), val_dataset.labels.squeeze())
    
    print(f"✅ Common Evaluation Set saved: {len(val_dataset.labels)} images.")

    print("📥 Downloading/Loading PathMNIST Test Set...")
    test_dataset = DataClass(split='test', download=True, root='./data') 
    
    # And then save it next to the val_images in the global_test_data folder
    np.save(os.path.join(eval_folder, "test_images.npy"), test_dataset.imgs)
    np.save(os.path.join(eval_folder, "test_labels.npy"), test_dataset.labels.squeeze())

if __name__ == "__main__":
    generate_biased_hospital_databases()