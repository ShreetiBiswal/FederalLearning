import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():
    # 1. Load the unified CSV file
    csv_path = '../true_global_metrics.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{csv_path}'. Make sure you are running this script in the same folder as the CSV.")
        return

    # 2. Clean up column names (removes hidden spaces)
    df.columns = df.columns.str.strip()

    # Ensure we have the right columns
    if not all(col in df.columns for col in ['Algorithm', 'Round', 'Accuracy', 'Loss']):
        print(f"❌ Error: CSV must contain 'Algorithm', 'Round', 'Accuracy', and 'Loss'. Found: {df.columns.tolist()}")
        return

    # 3. Setup the Plot Canvas with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('True Global Evaluation: Baseline vs Custom Algorithms', fontsize=16, fontweight='bold', y=0.98)

    # 4. Set up styling
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e'] # Blue, Red, Green, Purple, Orange
    algorithms = df['Algorithm'].unique()
    
    handles = []
    labels = []

    # 5. Loop through each algorithm and plot
    for i, algo in enumerate(algorithms):
        algo_data = df[df['Algorithm'] == algo]
        color = colors[i % len(colors)]
        
        # Plot Accuracy on Left (ax1)
        line, = ax1.plot(algo_data['Round'], algo_data['Accuracy'], color=color, marker='o', 
                         linestyle='-', linewidth=2, label=f'{algo.upper()}')
        
        # Plot Loss on Right (ax2) - using dashed lines and 'x' markers to differentiate
        ax2.plot(algo_data['Round'], algo_data['Loss'], color=color, marker='x', 
                 linestyle='--', linewidth=2)
        
        # Collect legend handles (we only need one set since colors match across both graphs)
        handles.append(line)
        labels.append(algo.upper())

    # Format Accuracy Plot (Left)
    ax1.set_title('Global Validation Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Communication Round', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim(0, 1.0) 
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Format Loss Plot (Right)
    ax2.set_title('Global Validation Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Communication Round', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    max_loss = df['Loss'].max()
    ax2.set_ylim(0, max_loss + 0.5) 
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 6. Add a Single Unified Legend at the bottom of the whole figure
    fig.legend(handles, labels, loc='lower center', ncol=len(algorithms), 
               bbox_to_anchor=(0.5, 0.01), fancybox=True, shadow=True, fontsize=12)

    # 7. Show the plot interactively
    # We leave 8% margin at the bottom so the legend doesn't overlap the X-axis
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_metrics()