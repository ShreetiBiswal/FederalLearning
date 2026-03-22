import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Final Metrics ---
algorithms = ['Standard FedAvg\n', 'CE-FedAvg (WSM)\n']
accuracies = [67.95, 92.80]
losses = [1.1136, 0.2474]

# --- 2. Setup the Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Final Global Model Performance on Non-IID Medical Data', fontsize=16, fontweight='bold', y=1.05)

# --- 3. Accuracy Plot (Higher is Better) ---
bars1 = ax1.bar(algorithms, accuracies, color=['#e74c3c', '#2ecc71'], edgecolor='black')
ax1.set_title('Final Global Accuracy (%)', fontsize=14)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

# --- 4. Loss Plot (Lower is Better) ---
bars2 = ax2.bar(algorithms, losses, color=['#e74c3c', '#2ecc71'], edgecolor='black')
ax2.set_title('Final Global Cross-Entropy Loss', fontsize=14)
ax2.set_ylim(0, max(losses) * 1.2)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

# --- 5. Render and Save ---
plt.tight_layout()
plt.savefig('final_results_comparison.png', dpi=300, bbox_inches='tight')
print("\n[✅] Graph saved as 'final_results_comparison.png'. Ready for the PPT!")
plt.show()