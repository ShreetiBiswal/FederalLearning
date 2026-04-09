import argparse
import matplotlib.pyplot as plt
import csv
import os
import base64
from io import BytesIO

def plot_final_results(iid=False, return_base64=False):
    """
    Plots the final global accuracy and loss bar charts.
    Can return a base64 encoded image string for HTML reports if return_base64=True.
    """
    # --- 1. Determine Suffix & Path ---
    suffix = "_iid" if iid else ""
    data_type = "IID (Perfectly Balanced)" if iid else "Non-IID (Highly Skewed)"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Pointing directly to your modular folder
    csv_filename = os.path.abspath(os.path.join(script_dir, '..', 'server', 'serverFinalAccuracy', f'server_final_accuracy{suffix}.csv'))

    algorithms = []
    accuracies = []
    losses = []

    if not os.path.exists(csv_filename):
        print(f"\n[🚨] ERROR: Could not find CSV file at {csv_filename}")
        print("Run 'python evaluate_final_model.py' first to generate the final accuracy logs.")
        return None

    if not return_base64:
        print(f"[📂] Reading {data_type} data from {csv_filename}...")
        
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algorithms.append(row['Algorithm'])
            accuracies.append(float(row['Accuracy(%)']))
            losses.append(float(row['Loss']))

    if not algorithms:
        print("\n[🚨] ERROR: The CSV file is empty!")
        return None

    # --- 2. Setup the Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Final Global Model Performance\n[{data_type} Medical Data]', fontsize=16, fontweight='bold', y=1.05)

    # Dynamic colors
    color_palette = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#e67e22']
    colors = [color_palette[i % len(color_palette)] for i in range(len(algorithms))]

    # --- 3. Accuracy Plot (Higher is Better) ---
    bars1 = ax1.bar(algorithms, accuracies, color=colors, edgecolor='black')
    ax1.set_title('Final Global Accuracy (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate the X-axis labels
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=25, ha='right', fontsize=11)

    # Add value labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # --- 4. Loss Plot (Lower is Better) ---
    bars2 = ax2.bar(algorithms, losses, color=colors, edgecolor='black')
    ax2.set_title('Final Global Cross-Entropy Loss', fontsize=14)

    # Dynamically set Y limit based on the maximum loss value
    max_loss = max(losses)
    ax2.set_ylim(0, max_loss * 1.25)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate the X-axis labels
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=25, ha='right', fontsize=11)

    # Add value labels on top of bars
    for bar in bars2:
        yval = bar.get_height()
        offset = max_loss * 0.03 
        ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # --- 5. Render, Return, or Show ---
    plt.tight_layout()
    
    if return_base64:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig) # Prevent it from displaying/blocking when generating the report
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    else:
        # Standard display for command line execution
        plt.show()

if __name__ == '__main__':
    # --- 1. Argument Parsing for Terminal Execution ---
    parser = argparse.ArgumentParser(description="Plot final global accuracy and loss bar charts.")
    parser.add_argument('--iid', action='store_true', help="Flag to plot IID-specific results")
    args = parser.parse_args()
    
    # Run the function natively
    plot_final_results(iid=args.iid)