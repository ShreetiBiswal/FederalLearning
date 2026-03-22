import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# --- 1. Locate and Load Data from CSV ---
# Build an absolute path to ../server/server_final_accuracy.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.abspath(os.path.join(script_dir, '..', 'server', 'server_final_accuracy.csv'))

algorithms = []
accuracies = []
losses = []

if not os.path.exists(csv_filename):
    print(f"\n[🚨] ERROR: Could not find CSV file at {csv_filename}")
    exit(1)

print(f"[📂] Reading data from {csv_filename}...")
with open(csv_filename, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Appending a newline to the algorithm name makes the X-axis labels look cleaner
        algorithms.append(row['Algorithm'] + '\n')
        accuracies.append(float(row['Accuracy(%)']))
        losses.append(float(row['Loss']))

if not algorithms:
    print("\n[🚨] ERROR: The CSV file is empty!")
    exit(1)

# --- 2. Setup the Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Final Global Model Performance on Non-IID Medical Data', fontsize=16, fontweight='bold', y=1.05)

# Dynamic colors just in case you test more than 2 algorithms later
color_palette = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#e67e22']
colors = [color_palette[i % len(color_palette)] for i in range(len(algorithms))]

# --- 3. Accuracy Plot (Higher is Better) ---
bars1 = ax1.bar(algorithms, accuracies, color=colors, edgecolor='black')
ax1.set_title('Final Global Accuracy (%)', fontsize=14)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

# --- 4. Loss Plot (Lower is Better) ---
bars2 = ax2.bar(algorithms, losses, color=colors, edgecolor='black')
ax2.set_title('Final Global Cross-Entropy Loss', fontsize=14)

# Dynamically set Y limit based on the maximum loss value to prevent text cutoff
max_loss = max(losses)
ax2.set_ylim(0, max_loss * 1.25)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars2:
    yval = bar.get_height()
    # Dynamic text offset for loss plot based on the highest loss
    offset = max_loss * 0.03 
    ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

# --- 5. Render and Save ---
plt.tight_layout()
#save_path = os.path.join(script_dir, 'final_results_comparison.png')
#plt.savefig(save_path, dpi=300, bbox_inches='tight')
#print(f"\n[✅] Graph saved as '{save_path}'. Ready for the PPT!")

# Optional: Comment out if running on a headless server without a display
plt.show()