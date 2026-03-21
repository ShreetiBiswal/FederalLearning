import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_server_metrics():
    csv_file = '../hospital_1_fedavg_metrics.csv'
    
    # Check if the server has actually generated the file yet
    if not os.path.exists(csv_file):
        print(f"❌ Could not find {csv_file}. Make sure you run the training pipeline first!")
        return

    # Load the data
    print(f"📊 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # We want to plot each algorithm (e.g., 'fedavg', 'ce_fedavg') as a separate line
    algorithms = df['Algorithm'].unique()

    # Create a figure with 2 subplots (1 for Accuracy, 1 for Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Federated Learning Global Model Performance', fontsize=16, fontweight='bold')

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