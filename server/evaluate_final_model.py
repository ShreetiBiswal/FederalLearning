import argparse
import json
import torch
import torch.nn as nn
import os
import sys

# --- THE BULLETPROOF PATH FIX ---
# 1. Remember exactly where the 'server' folder is so we can find the JSON later
server_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Find the main 'FRP' root directory (one folder up from 'server')
root_dir = os.path.abspath(os.path.join(server_dir, '..'))

# 3. Force Python to treat the 'FRP' root as our current working directory
os.chdir(root_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'clients'))
# ---------------------------------

from shared.cnn_model import GenericClientModel
from shared.tensor_utils import json_ready_to_state_dict
from clients.data_loader import get_global_val_loader

def main():
    parser = argparse.ArgumentParser(description="Evaluate the final aggregated JSON model.")
    parser.add_argument('--algo', type=str, required=True, help="The exact algorithm name used to save the file (e.g., wsm_ce_fedavg_nosmote)")
    args = parser.parse_args()

    print(f"========== 🏆 FINAL MODEL EVALUATOR [{args.algo.upper()}] ==========")

    # 1. Locate the JSON file explicitly in the server folder
    json_filename = os.path.join(server_dir, f"final_master_model_{args.algo}.json")
    
    if not os.path.exists(json_filename):
        print(f"\n[🚨] ERROR: Could not find {json_filename}. Did the server finish the 50 rounds?")
        sys.exit(1)

    # 2. Load and Parse the JSON Weights
    print(f"\n[📂] Loading weights from {json_filename}...")
    with open(json_filename, 'r', encoding='utf-8') as f:
        json_weights = json.load(f)
    
    # Convert the JSON arrays back into PyTorch Tensors
    state_dict = json_ready_to_state_dict(json_weights)

    # 3. Initialize the Empty Model & Inject Weights
    in_channels = 3  # Based on your data loader setup
    num_classes = 9
    
    print("   [⚙️] Initializing CNN and injecting master weights...")
    model = GenericClientModel(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(state_dict)
    
    # Move to GPU if available for faster inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 4. Load the Pristine Global Validation/Test Dataset
    print("\n[🌍] Loading pristine Global Test dataset...")
    test_loader = get_global_val_loader(batch_size=32)

    # 5. Run the Final Inference
    print("\n[🔍] Running final inference on the test set...")
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 6. Calculate and Display Final Metrics
    final_accuracy = correct / total if total > 0 else 0
    final_loss = running_loss / len(test_loader) if len(test_loader) > 0 else 0

    print("\n" + "="*50)
    print(" 🌟 FINAL OFFICIAL METRICS 🌟")
    print("="*50)
    print(f"   Algorithm : {args.algo}")
    print(f"   Accuracy  : {final_accuracy * 100:.2f}%")
    print(f"   Loss      : {final_loss:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()