import argparse
import torch
import torch.nn as nn
import socketio
import os

# Import your shared architecture and utilities
from shared.cnn_model import GenericClientModel
from shared.tensor_utils import json_ready_to_state_dict

# Import the global data loader
from clients.data_loader import get_global_val_loader

# --- Global State ---
sio = socketio.Client()
model = None
val_loader = None
csv_filename = ""

# --- The Evaluation Engine ---
def evaluate_master_model(current_round, json_weights):
    print(f"\n[🔍] Evaluating Master Model for Round {current_round}...")
    
    # 1. Convert JSON arrays back to PyTorch Tensors
    state_dict = json_ready_to_state_dict(json_weights)
    
    # 2. Inject weights into the CNN
    model.load_state_dict(state_dict)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    
    # 3. Run Inference on the untouched Global Validation Set
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total if total > 0 else 0
    avg_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    print(f"   📊 True Global Accuracy: {accuracy*100:.2f}% | Loss: {avg_loss:.4f}")
    
    # 4. Log the True Metrics to a CSV
    with open(csv_filename, mode='a', encoding='utf-8') as f:
        f.write(f"{current_round},{accuracy},{avg_loss}\n")

# --- WebSocket Listeners ---
@sio.event
def connect():
    print(f"\n[🔌] Evaluator Node Connected to Aggregator!")

@sio.on('apply_global_update')
def on_apply_update(data):
    # Whenever the server broadcasts new weights, evaluate them!
    evaluate_master_model(data['round'], data['global_weights'])

@sio.on('training_finished')
def on_training_finished():
    print("\n[✅] Server announced training is complete! Evaluator shutting down.")
    sio.disconnect()

# --- Main Execution ---
def main():
    global model, val_loader, csv_filename

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='fedavg', help="Algorithm name for the CSV file naming")
    args = parser.parse_args()

    print("========== 🌍 GLOBAL EVALUATOR NODE STARTING ==========")

    # 1. Load the pristine Global Validation Data
    val_loader = get_global_val_loader(batch_size=32)
    in_channels = 3 
    num_classes = 9

    # 2. Setup the "True" Metrics CSV
    csv_filename = f"true_global_{args.algo}_metrics.csv"
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', encoding='utf-8') as f:
            f.write("Round,Accuracy,Loss\n")

    # 3. Initialize the Empty Model
    model = GenericClientModel(in_channels=in_channels, num_classes=num_classes)

    # 4. Connect to Server and Wait Silently
    try:
        sio.connect('http://localhost:3000')
        print("[🛑] Waiting silently for server to broadcast master weights...")
        sio.wait()
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == '__main__':
    main()