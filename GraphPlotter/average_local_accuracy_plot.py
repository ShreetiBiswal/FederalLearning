import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_server_metrics():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Plot Global Server Metrics.")
    parser.add_argument('--iid', action='store_true', help="Flag to plot IID-specific metrics")
    args = parser.parse_args()

    suffix = "_iid" if args.iid else ""
    data_type = "IID (Perfectly Balanced)" if args.iid else "Non-IID (Highly Skewed)"

    # --- 2. Bulletproof Path Resolution ---
    plotter_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.abspath(os.path.join(plotter_dir, '..', 'server'))
    
    # Points to the new modular folder structure!
    csv_file = os.path.join(server_dir, 'serverAvgMetrics', f'server_avg_metrics{suffix}.csv')
    
    # Check if the server has actually generated the file yet
    if not os.path.exists(csv_file):
        print(f"❌ Could not find {csv_file}.")
        print("Make sure you run the training pipeline and evaluator first!")
        return

    # --- 3. Load the data ---
    print(f"📊 Loading {data_type} data from {csv_file}...")
    df = pd.read_csv(csv_file)
    # Load the data
    print(f"📊 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # 👉 ADD THIS LINE:
    print("Columns found in CSV:", df.columns.tolist())

    # We want to plot each algorithm (e.g., 'fedavg', 'ce_fedavg') as a separate line
    algorithms = df['Algorithm'].unique()

    # Create a figure with 2 subplots (1 for Accuracy, 1 for Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Federated Learning Global Model Performance\n[{data_type} Data]', fontsize=16, fontweight='bold')

    # --- Plot 1: Global Accuracy ---
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        # Multiply by 100 to make it a clean percentage
        ax1.plot(algo_data['Round'], algo_data['Global_Accuracy'] * 100, 
                 marker='o', linewidth=2, label=algo.upper())

    ax1.set_title('Global Test Accuracy over Rounds')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim([0, 100]) # Force the Y-axis to be 0-100%
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- Plot 2: Global Loss ---
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        ax2.plot(algo_data['Round'], algo_data['Global_Loss'], 
                 marker='x', linewidth=2, linestyle='--', label=algo.upper())

    ax2.set_title('Global Training Loss over Rounds')
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Make it look clean and show the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_server_metrics()