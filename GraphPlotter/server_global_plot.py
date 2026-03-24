import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_metrics(is_iid):
    # 1. Determine the suffix and title based on the flag
    suffix = "_iid" if is_iid else ""
    data_type = "IID (Perfectly Balanced)" if is_iid else "Non-IID (Highly Skewed)"

    # 2. Bulletproof Path Resolution
    plotter_dir = os.path.dirname(os.path.abspath(__file__))
    # Points to the root folder where true_global_metrics.csv lives
    csv_path = os.path.abspath(os.path.join(plotter_dir, '..', f'true_global_metrics{suffix}.csv'))

    try:
        print(f"📊 Loading {data_type} data from {csv_path}...")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{csv_path}'.")
        print("Make sure your evaluator node generated this file in the root directory!")
        return

    # 3. Clean up column names (removes hidden spaces)
    df.columns = df.columns.str.strip()

    # Ensure we have the right columns
    required_cols = ['Algorithm', 'Round', 'Accuracy', 'Loss']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: CSV must contain 'Algorithm', 'Round', 'Accuracy', and 'Loss'. Found: {df.columns.tolist()}")
        return

    # 4. Setup the Plot Canvas with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'True Global Evaluation: Baseline vs Custom Algorithms\n[{data_type} Data]', fontsize=16, fontweight='bold', y=0.98)

    # 5. Set up styling
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e'] # Blue, Red, Green, Purple, Orange
    algorithms = df['Algorithm'].unique()
    
    handles = []
    labels = []

    # 6. Loop through each algorithm and plot
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
    
    # Check if the dataframe is empty before getting the max loss
    if not df.empty and not df['Loss'].isna().all():
        max_loss = df['Loss'].max()
        ax2.set_ylim(0, max_loss + 0.5) 
        
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 7. Add a Single Unified Legend at the bottom of the whole figure
    fig.legend(handles, labels, loc='lower center', ncol=len(algorithms), 
               bbox_to_anchor=(0.5, 0.01), fancybox=True, shadow=True, fontsize=12)

    # 8. Show the plot interactively
    # We leave 8% margin at the bottom so the legend doesn't overlap the X-axis
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot true global evaluation metrics.")
    parser.add_argument('--iid', action='store_true', help="Flag to plot IID-specific results")
    args = parser.parse_args()
    
    plot_metrics(args.iid)