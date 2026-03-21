import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_client_metrics():
    # 1. Find all client CSV files in the current folder (if you are inside clients/)
    csv_files = glob.glob("hospital_*_metrics.csv")
    
    # Check inside the clients/ folder (if your terminal is in the root FRP folder)
    if not csv_files:
        csv_files = glob.glob("../hospital_*_metrics.csv")
        
    if not csv_files:
        print("❌ Could not find any hospital CSV files. Make sure the clients have run first!")
        return

    print(f"📊 Found {len(csv_files)} client metric files. Generating plots...")

    # 2. Setup the Canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Local Hospital Training Performance', fontsize=16, fontweight='bold')

    # 3. Loop through every CSV and plot its line
    for file in csv_files:
        # Extract the hospital ID and algorithm from the filename
        basename = os.path.basename(file)
        parts = basename.split('_')
        
        if len(parts) >= 4:
            h_id = parts[1]
            algo = parts[2].upper()
            label_name = f"Hospital {h_id} ({algo})"
        else:
            label_name = basename

        # Read the data
        df = pd.read_csv(file)
        
        # Plot Accuracy (Multiply by 100 for percentage)
        ax1.plot(df['Round'], df['Accuracy'] * 100, marker='o', linewidth=2, label=label_name)
        
        # Plot Loss
        ax2.plot(df['Round'], df['Loss'], marker='x', linewidth=2, linestyle='--', label=label_name)

    # 4. Format the Accuracy Graph
    ax1.set_title('Local Test Accuracy over Rounds')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim([0, 100]) # Force Y-axis to 0-100%
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 5. Format the Loss Graph
    ax2.set_title('Local Training Loss over Rounds')
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Make it look clean and show the plot
    plt.tight_layout()
    plt.show()

# --- THIS IS THE TRIGGER THAT MAKES IT ACTUALLY RUN ---
if __name__ == '__main__':
    plot_client_metrics()