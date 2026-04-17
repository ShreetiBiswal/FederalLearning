import numpy as np
import os
from shared.config import FL_CONFIG

def calculate_alpha():
    print("========== 📊 DIRICHLET ALPHA ESTIMATOR ==========")
    
    num_hospitals = 4
    num_classes = FL_CONFIG["NUM_CLASSES"]
    
    # 1. Store the distribution of classes for each hospital (m x n)
    distributions = np.zeros((num_hospitals, num_classes))
    
    # New: Store the total dataset size for each hospital
    client_sizes = np.zeros(num_hospitals, dtype=int)
    
    # 2. Read the labels from each hospital's folder
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'clients'))
    
    for i in range(1, num_hospitals + 1):
        folder_path = os.path.join(base_dir, f"hospital_{i}_data")
        label_file = os.path.join(folder_path, "train_labels.npy")
        
        try:
            labels = np.load(label_file)
            total_samples = len(labels)
            
            # Count how many of each class this hospital has
            counts = np.bincount(labels, minlength=num_classes)
            
            # Convert to proportions (e.g., 0.45 of dataset is Class 0)
            proportions = counts / total_samples
            
            distributions[i-1] = proportions
            client_sizes[i-1] = total_samples
            
        except FileNotFoundError:
            print(f"[🚨] ERROR: Could not find labels for Hospital {i} at {label_file}")
            return None, None, None

    # 3. Calculate the Sample Variance across the hospitals
    variances = np.var(distributions, axis=0, ddof=1)
    mean_variance = np.mean(variances)
    
    # 4. Method of Moments calculation for Alpha
    p = 1.0 / num_classes
    
    if mean_variance == 0:
        print("\n[✅] Variance is 0. Data is perfectly IID. Alpha approaches Infinity.")
        estimated_alpha = "∞"
    else:
        estimated_alpha = (1.0 / num_classes) * (((p * (1 - p)) / mean_variance) - 1)
        # Alpha cannot be mathematically negative; threshold it
        estimated_alpha = max(0.001, estimated_alpha)

        print(f"\n[📈] Mean Class Variance: {mean_variance:.4f}")
        print(f"[🎯] Estimated Effective Alpha (α): {estimated_alpha:.4f}")
        
        if estimated_alpha < 0.1:
            print("   -> ⚠️ EXTREME HETEROGENEITY (Highly Non-IID)")
        elif estimated_alpha < 1.0:
            print("   -> 📊 MODERATE HETEROGENEITY")
        else:
            print("   -> 🟢 LOW HETEROGENEITY (Near IID)")
            
    # Return Alpha, the m x n distribution matrix, and the array of dataset sizes
    return estimated_alpha, distributions, client_sizes

if __name__ == '__main__':
    alpha, dists, sizes = calculate_alpha()
    if alpha is not None:
        print(f"\nSizes per client: {sizes}")
        print(f"\nDis:- {dists}")