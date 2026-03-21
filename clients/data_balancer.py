import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.stats import chisquare

def detect_and_balance(images, labels, num_classes):
    """
    Pure function: Takes raw image and label arrays, detects bias, 
    applies SMOTE if necessary, and returns balanced arrays.
    """
    print("\n[⚙️ Data Balancer] Analyzing local dataset for bias...")
    
    # 1. Measure the Reality (Observed Counts)
    counts = [np.sum(labels == i) for i in range(num_classes)]
    
    # 2. Measure the Ideal (Expected Counts)
    expected_counts = [len(labels) / num_classes] * num_classes
    expected_counts = [max(e, 1e-5) for e in expected_counts] # Prevent division by zero
    
    # 3. Run Chi-Square
    chi2_stat, p_val = chisquare(f_obs=counts, f_exp=expected_counts)
    print(f"   📊 Class distribution: {counts}")
    print(f"   📉 Chi-Square p-value: {p_val:.4f}")
    
    # 4. Apply SMOTE if statistically biased
    if p_val < 0.05:
        print("   🚨 Severe bias detected! Synthesizing balanced data with SMOTE...")
        
        # Extract dynamic shape: (Samples, Channels, Height, Width)
        num_samples, channels, height, width = images.shape
        
        # Flatten for SMOTE
        flattened_images = images.reshape(num_samples, -1)
        
        # Synthesize
        smote = SMOTE(random_state=42)
        balanced_flat, balanced_labels = smote.fit_resample(flattened_images, labels)
        
        # Reconstruct into dynamic 3D image shape
        final_images = balanced_flat.reshape(-1, channels, height, width)
        final_labels = balanced_labels
        
        print(f"   ✅ Data balanced! Synthesized {final_images.shape[0] - num_samples} new minority samples.")
    else:
        print("   ✅ Dataset is naturally balanced. Skipping SMOTE.")
        final_images = images
        final_labels = labels

    return final_images, final_labels