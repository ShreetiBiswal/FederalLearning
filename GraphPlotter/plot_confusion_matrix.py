import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Plot a saved Confusion Matrix CSV with Metrics.")
    parser.add_argument('--algo', type=str, required=True, help="The algorithm name (e.g., wsm_ce_fedavg_nosmote)")
    parser.add_argument('--iid', action='store_true', help="Flag to load the IID-specific confusion matrix")
    args = parser.parse_args()

    # 1. Determine suffix and title
    suffix = "_iid" if args.iid else ""
    display_algo = f"{args.algo}{suffix}"
    data_type = "IID (Perfectly Balanced)" if args.iid else "Non-IID (Highly Skewed)"

    # 2. Bulletproof Path Resolution
    plotter_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.abspath(os.path.join(plotter_dir, '..', 'server'))
    
    # Target the new modular confusionMatrix folder
    cm_filename = os.path.join(server_dir, 'confusionMatrix', f'confusion_matrix_{display_algo}.csv')

    if not os.path.exists(cm_filename):
        print(f"[🚨] ERROR: Could not find {cm_filename}.")
        print("Run 'python evaluate_final_model.py' first to generate it!")
        return

    # 3. Load the matrix data
    print(f"📊 Loading matrix data from {cm_filename}...")
    cm = np.loadtxt(cm_filename, delimiter=",", dtype=int)
    num_classes = cm.shape[0]

    # --- 🧮 4. Calculate Strict Academic Metrics ---
    total = np.sum(cm)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    
    # "Support" is the actual number of true images for that class
    actual_support = TP + FN 

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.true_divide(TP, TP + FP)
        precision[np.isnan(precision)] = 0

        recall = np.true_divide(TP, actual_support)
        recall[np.isnan(recall)] = 0

        f1 = 2 * (precision * recall) / (precision + recall)
        f1[np.isnan(f1)] = 0

    overall_acc = np.sum(TP) / total

    # --- 🎨 5. Set up the Side-by-Side Plot ---
    fig, (ax_cm, ax_table) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={'width_ratios': [2, 1.4]})
    sns.set_theme(style="white")
    
    # --- A. Plot the Heatmap (Left Side) ---
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                linewidths=.5, square=True, cbar_kws={"shrink": .75}, ax=ax_cm)

    ax_cm.set_title(f'Global Confusion Matrix\nAlgorithm: {display_algo.upper()}\n[{data_type}]', fontsize=16, fontweight='bold', pad=20)
    ax_cm.set_xlabel('Predicted Medical Class', fontsize=14, fontweight='bold')
    ax_cm.set_ylabel('True Medical Class', fontsize=14, fontweight='bold')
    
    class_labels = [f"Class {i}" for i in range(num_classes)]
    ax_cm.set_xticklabels(class_labels, rotation=45, ha='right')
    ax_cm.set_yticklabels(class_labels, rotation=0)

    # --- B. Plot the Metrics Table (Right Side) ---
    ax_table.axis('off')
    
    table_data = []
    columns = ['Class', 'Total\nActual', 'TP', 'FP', 'FN', 'True Acc\n(Recall %)', 'Precision', 'F1']
    
    for i in range(num_classes):
        row = [
            f"Class {i}",
            f"{actual_support[i]}", f"{TP[i]}", f"{FP[i]}", f"{FN[i]}",
            f"{recall[i]*100:.1f}%", 
            f"{precision[i]:.3f}", 
            f"{f1[i]:.3f}"
        ]
        table_data.append(row)

    table = ax_table.table(cellText=table_data, 
                           colLabels=columns, 
                           cellLoc='center', 
                           loc='center',
                           bbox=[0.0, 0.1, 1.0, 0.8]) 
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5) 

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4A90E2') 

    ax_table.text(0.5, 0.05, f"Overall Global Accuracy: {overall_acc*100:.2f}%", 
                  ha='center', va='center', fontsize=14, fontweight='bold', transform=ax_table.transAxes)

    # --- 6. Save and Show ---
    # Saves the image right inside your GraphPlotter folder!
    output_img = os.path.join(plotter_dir, f'plot_cm_metrics_{display_algo}.png')
    plt.tight_layout()
    #plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n[✅] Beautiful Matrix + Table saved as {output_img}!")
    plt.show()

if __name__ == '__main__':
    main()