import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Plot final global accuracy and loss bar charts.")
    parser.add_argument('--iid', action='store_true', help="Flag to plot IID-specific results")
    args = parser.parse_args()

    suffix = "_iid" if args.iid else ""
    data_type = "IID (Perfectly Balanced)" if args.iid else "Non-IID (Highly Skewed)"

    # --- 2. Locate and Load Data from CSV ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Pointing directly to your new modular folder!
    csv_filename = os.path.abspath(os.path.join(script_dir, '..', 'server', 'serverFinalAccuracy', f'server_final_accuracy{suffix}.csv'))

    algorithms = []
    accuracies = []
    losses = []

    if not os.path.exists(csv_filename):
        print(f"\n[🚨] ERROR: Could not find CSV file at {csv_filename}")
        print("Run 'python evaluate_final_model.py' first to generate the final accuracy logs.")
        return

    print(f"[📂] Reading {data_type} data from {csv_filename}...")
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Removed the hacky '\n', we will use rotation instead!
            algorithms.append(row['Algorithm'])
            accuracies.append(float(row['Accuracy(%)']))
            losses.append(float(row['Loss']))

    if not algorithms:
        print("\n[🚨] ERROR: The CSV file is empty!")
        return

    # --- 3. Setup the Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Final Global Model Performance\n[{data_type} Medical Data]', fontsize=16, fontweight='bold', y=1.05)

    # Dynamic colors
    color_palette = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#e67e22']
    colors = [color_palette[i % len(color_palette)] for i in range(len(algorithms))]

    # --- 4. Accuracy Plot (Higher is Better) ---
    bars1 = ax1.bar(algorithms, accuracies, color=colors, edgecolor='black')
    ax1.set_title('Final Global Accuracy (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 🔥 THE FIX: Rotate the X-axis labels so they don't overlap
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=25, ha='right', fontsize=11)

    # Add value labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # --- 5. Loss Plot (Lower is Better) ---
    bars2 = ax2.bar(algorithms, losses, color=colors, edgecolor='black')
    ax2.set_title('Final Global Cross-Entropy Loss', fontsize=14)

    # Dynamically set Y limit based on the maximum loss value to prevent text cutoff
    max_loss = max(losses)
    ax2.set_ylim(0, max_loss * 1.25)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 🔥 THE FIX: Rotate the X-axis labels so they don't overlap
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=25, ha='right', fontsize=11)

    # Add value labels on top of bars
    for bar in bars2:
        yval = bar.get_height()
        offset = max_loss * 0.03 
        ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # --- 6. Render and Save ---
    plt.tight_layout()
    save_path = os.path.join(script_dir, f'final_results_comparison{suffix}.png')
    
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"\n[✅] Graph saved as '{save_path}'. Ready for the PPT!")

    plt.show()

if __name__ == '__main__':
    main()