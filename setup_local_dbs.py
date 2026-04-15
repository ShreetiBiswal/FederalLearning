import os
import numpy as np
import medmnist
from medmnist import INFO

def generate_hospital_databases(dataset_name='pathmnist', num_hospitals=4, alpha=0.05, hospital_proportions=None, iid=False):
    distribution_type = "IID (Uniform)" if iid else f"STRICT DIRICHLET Skewed (Alpha={alpha})"
    print(f"🚀 Starting {distribution_type} Local Database Generation for {dataset_name}...")

    # Validate dataset
    if dataset_name not in INFO:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(INFO.keys())}")

    # 1. Load MedMNIST (Train AND Val splits)
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    print(f"📥 Downloading/Loading {info['python_class']} Train Set...")
    train_dataset = DataClass(split='train', download=True, root='./data') 
    
    print(f"📥 Downloading/Loading {info['python_class']} Val (Evaluation) Set...")
    val_dataset = DataClass(split='val', download=True, root='./data') 

    all_train_images = train_dataset.imgs
    all_train_labels = train_dataset.labels.squeeze()   
    num_classes = len(info['label'])
    
    print(f"Total raw training images loaded: {len(all_train_images)}")

    hospital_indices = {i: [] for i in range(num_hospitals)}

    # 2. Handle Proportions and Capacities
    if hospital_proportions is None:
        hospital_proportions = np.ones(num_hospitals) / num_hospitals
    else:
        hospital_proportions = np.array(hospital_proportions, dtype=float)
        if len(hospital_proportions) != num_hospitals:
            raise ValueError("Length of hospital_proportions array must match num_hospitals.")
        hospital_proportions = hospital_proportions / np.sum(hospital_proportions)

    total_images = len(all_train_images)
    hospital_capacities = np.floor(hospital_proportions * total_images).astype(int)
    
    # Distribute any leftover images caused by floor rounding
    remainder = total_images - np.sum(hospital_capacities)
    for i in range(remainder):
        hospital_capacities[i % num_hospitals] += 1

    print(f"📊 Target capacities per hospital: {hospital_capacities}")
    np.random.seed(42) # Seeded for reproducibility

    # ---------------------------------------------------------
    # 3. DISTRIBUTION LOGIC (IID vs Non-IID)
    # ---------------------------------------------------------
    if iid:
        print("⚖️ Distributing data uniformly (IID Random Partitioning)...")
        all_indices = np.arange(total_images)
        np.random.shuffle(all_indices) # Shuffle the entire dataset
        
        start_idx = 0
        for h in range(num_hospitals):
            take = hospital_capacities[h]
            hospital_indices[h] = all_indices[start_idx : start_idx + take].tolist()
            start_idx += take
            
    else:
        print(f"📉 Distributing diseases using Capacity-Constrained Dirichlet (Alpha = {alpha})...")
        class_indices = {}
        for c in range(num_classes):
            class_indices[c] = np.where(all_train_labels == c)[0].tolist()

        for c in range(num_classes):
            idx_c = class_indices[c].copy()
            np.random.shuffle(idx_c)
            
            while len(idx_c) > 0:
                available_hospitals = np.where(hospital_capacities > 0)[0]
                if len(available_hospitals) == 0:
                    break 
                
                dirichlet_props = np.random.dirichlet(np.repeat(alpha, len(available_hospitals)))
                
                desired_allocation = np.round(dirichlet_props * len(idx_c)).astype(int)
                desired_allocation[-1] = len(idx_c) - np.sum(desired_allocation[:-1]) 
                desired_allocation = np.maximum(desired_allocation, 0)
                
                actual_allocation = np.minimum(desired_allocation, hospital_capacities[available_hospitals])
                
                start_idx = 0
                for i, h in enumerate(available_hospitals):
                    take = actual_allocation[i]
                    if take > 0:
                        hospital_indices[h].extend(idx_c[start_idx : start_idx + take])
                        hospital_capacities[h] -= take
                        start_idx += take
                
                idx_c = idx_c[start_idx:]
                
    # ---------------------------------------------------------
    # 4. Extract and Save to Physical Folders 
    # ---------------------------------------------------------
    base_dir = "clients"
    os.makedirs(base_dir, exist_ok=True)

    # A. Save the individual Hospital Training Data
    for h in range(num_hospitals):
        hospital_id = h + 1
        folder_path = os.path.join(base_dir, f"hospital_{hospital_id}_data")
        os.makedirs(folder_path, exist_ok=True)
        
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

    print(f"📥 Downloading/Loading {info['python_class']} Test Set...")
    test_dataset = DataClass(split='test', download=True, root='./data') 
    
    np.save(os.path.join(eval_folder, "test_images.npy"), test_dataset.imgs)
    np.save(os.path.join(eval_folder, "test_labels.npy"), test_dataset.labels.squeeze())

if __name__ == "__main__":
    # To run perfectly IID, simply set iid=True. 
    # (The 'alpha' parameter is ignored when iid=True)
    generate_hospital_databases(
        dataset_name='pneumoniamnist', 
        num_hospitals=4, 
        hospital_proportions=[0.25, 0.25, 0.25, 0.25], 
        iid=False 
    )