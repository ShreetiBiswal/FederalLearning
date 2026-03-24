import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse

def plot_client_metrics(algo, is_iid):
    # 1. Determine the folder suffix and title based on the flag
    folder_suffix = "_results_iid" if is_iid else "_results"
    data_type = "IID (Perfectly Balanced)" if is_iid else "Non-IID (Highly Skewed)"

    # 2. Bulletproof Path Resolution
    plotter_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(plotter_dir, '..', 'results'))
    
    # Target the specific folder (e.g., results/fedavg_results_iid/hospital_*_metrics.csv)
    search_pattern = os.path.join(results_dir, f"{algo}{folder_suffix}", "hospital_*_metrics.csv")
    
    csv_files = glob.glob(search_pattern)
        
    if not csv_files:
        print(f"❌ Could not find any CSV files for algorithm '{algo.upper()}' in {algo}{folder_suffix}.")
        print(f"   Make sure you ran the clients and the folder 'results/{algo}{folder_suffix}/' exists.")
        return

    print(f"📊 Found {len(csv_files)} client metric files for {algo.upper()} [{data_type}]. Generating plots...")

    # 3. Setup the Canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Local Hospital Training Performance ({algo.upper()})\n[{data_type} Data]', fontsize=16, fontweight='bold', y=0.96)

    # Sort files so Hospital 1, 2, 3, 4 appear in order in the legend
    csv_files.sort()

    # 4. Loop through the specific algorithm's CSVs and plot
    for file in csv_files:
        # Get the filename (e.g., 'hospital_1_metrics.csv') and extract the ID
        basename = os.path.basename(file)
        try:
            h_id = basename.split('_')[1] 
            label_name = f"Hospital {h_id}"
        except IndexError:
            label_name = basename # Fallback if naming convention breaks

        # Read the data and clean headers
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        
        # Plot Accuracy (Multiply by 100 for percentage)
        ax1.plot(df['Round'], df['Accuracy'] * 100, marker='o', linewidth=2, label=label_name)
        
        # Plot Loss
        ax2.plot(df['Round'], df['Loss'], marker='x', linewidth=2, linestyle='--', label=label_name)

    # 5. Format the Accuracy Graph
    ax1.set_title('Local Test Accuracy over Rounds', fontsize=13)
    ax1.set_xlabel('Communication Round', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_ylim([0, 100]) # Force Y-axis to 0-100%
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 6. Format the Loss Graph
    ax2.set_title('Local Training Loss over Rounds', fontsize=13)
    ax2.set_xlabel('Communication Round', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    
    # Dynamically scale loss Y-axis based on the highest value across all lines
    max_loss = ax2.get_ylim()[1]
    ax2.set_ylim([0, max_loss + 0.2])
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 7. --- THE FIX: Create a single unified legend at the bottom ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=4, fontsize=11, fancybox=True, shadow=True)

    # Make it look clean and show the plot
    # We leave a 12% margin at the bottom (0.12) to make plenty of room for the legend box
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot local hospital metrics for a specific FL algorithm.")
    parser.add_argument('--algo', type=str, default='fedavg', 
                        help="The algorithm to plot (e.g., fedavg, wsm_ce_fedavg_nosmote). Defaults to fedavg.")
    parser.add_argument('--iid', action='store_true', 
                        help="Flag to read local client data from the IID-specific results folder")
    
    args = parser.parse_args()
    
    # Pass the requested algorithm and the flag to the plotting function
    plot_client_metrics(args.algo, args.iid)