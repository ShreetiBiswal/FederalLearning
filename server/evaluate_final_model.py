import argparse
import json
import torch
import torch.nn as nn
import os
import sys
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

# --- THE BULLETPROOF PATH FIX ---
# 1. Base server directory
server_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define Subdirectories based on your new Project Structure
models_dir = os.path.join(server_dir, 'models')
cm_dir = os.path.join(server_dir, 'confusionMatrix')
acc_dir = os.path.join(server_dir, 'serverFinalAccuracy')

# Ensure output directories exist just in case
os.makedirs(cm_dir, exist_ok=True)
os.makedirs(acc_dir, exist_ok=True)

# 3. Find the main 'FRP' root directory
root_dir = os.path.abspath(os.path.join(server_dir, '..'))
os.chdir(root_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'clients'))
# ---------------------------------

from shared.cnn_model import GenericClientModel
from shared.tensor_utils import json_ready_to_state_dict
from clients.data_loader import get_global_val_loader
from shared.config import FL_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Evaluate the final aggregated JSON model.")
    parser.add_argument('--algo', type=str, required=True, help="The exact algorithm name (e.g., wsm_ce_fedavg_nosmote)")
    parser.add_argument('--iid', action='store_true', help="Flag to load/save from IID-specific files")
    args = parser.parse_args()

    # Determine suffix based on the flag
    suffix = "_iid" if args.iid else ""
    display_algo = f"{args.algo}{suffix}"

    print(f"========== 🏆 FINAL MODEL EVALUATOR [{display_algo.upper()}] ==========")

    # --- 📂 ROUTING: Read from 'models/' folder ---
    json_filename = os.path.join(models_dir, f"final_master_model_{display_algo}.json")
    
    if not os.path.exists(json_filename):
        print(f"\n[🚨] ERROR: Could not find {json_filename}.")
        print("Did the server finish all rounds? Is it in the 'server/models/' folder?")
        sys.exit(1)

    print(f"\n[📂] Loading weights from {json_filename}...")
    with open(json_filename, 'r', encoding='utf-8') as f:
        json_weights = json.load(f)
    
    state_dict = json_ready_to_state_dict(json_weights)

    in_channels = FL_CONFIG["IN_CHANNELS"]
    num_classes = FL_CONFIG["NUM_CLASSES"]
    
    print("   [⚙️] Initializing CNN and injecting master weights...")
    model = GenericClientModel(
        in_channels=in_channels, 
        num_classes=num_classes,
        image_size=FL_CONFIG["IMAGE_SIZE"]
    )
    model.load_state_dict(state_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print("\n[🌍] Loading pristine Global Test dataset...")
    test_loader = get_global_val_loader(batch_size=32)

    print("\n[🔍] Running final inference on the test set...")
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0

    all_true_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    final_accuracy = correct / total if total > 0 else 0
    final_loss = running_loss / len(test_loader) if len(test_loader) > 0 else 0

    acc_str = f"{final_accuracy * 100:.2f}"
    loss_str = f"{final_loss:.4f}"

    print("\n" + "="*50)
    print(" 🌟 FINAL OFFICIAL METRICS 🌟")
    print("="*50)
    print(f"   Algorithm : {display_algo}")
    print(f"   Accuracy  : {acc_str}%")
    print(f"   Loss      : {loss_str}")
    print("="*50 + "\n")

    # --- 💾 ROUTING: Save to 'confusionMatrix/' folder ---
    print("[💾] Generating Confusion Matrix...")
    cm = confusion_matrix(all_true_labels, all_predictions, labels=range(num_classes))
    
    cm_filename = os.path.join(cm_dir, f'confusion_matrix_{display_algo}.csv')
    np.savetxt(cm_filename, cm, delimiter=",", fmt='%d')
    print(f"[✅] Confusion Matrix saved to {cm_filename}")

    # --- 💾 ROUTING: Save to 'serverFinalAccuracy/' folder ---
    csv_filename = os.path.join(acc_dir, f'server_final_accuracy{suffix}.csv')
    records = {}
    fieldnames = ['Algorithm', 'Accuracy(%)', 'Loss']

    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                records[row['Algorithm']] = row

    records[display_algo] = {
        'Algorithm': display_algo,
        'Accuracy(%)': acc_str,
        'Loss': loss_str
    }

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for algo_name in sorted(records.keys()): 
            writer.writerow(records[algo_name])
            
    print(f"[✅] Accuracy CSV successfully updated at {csv_filename}!")

if __name__ == '__main__':
    main()