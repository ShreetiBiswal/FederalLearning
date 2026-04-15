import argparse
import threading
import torch
import torch.nn as nn
import socketio
import sys
import os
import gc

# Add the parent directory to the path so we can import from 'shared' and 'trainers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.cnn_model import GenericClientModel
from shared.tensor_utils import state_dict_to_json_ready, json_ready_to_state_dict
from data_loader import get_dummy_loaders, get_local_hospital_loader

# --- Import the Modular Algorithms ---
from trainers.fedavg import run_fedavg
from trainers.wsm_ce_fedavg import run_wsm_ce_fedAvg
from trainers.wsm_class_weighted import run_wsm_class_weighted
from trainers.scaffold import run_scaffold  # 🔥 NEW: Import SCAFFOLD
from shared.config import FL_CONFIG

# --- 1. Global State & Synchronization ---
sio = socketio.Client()
server_response_event = threading.Event()  
training_finished = False

global_model_weights = None
current_round = 1

# 🔥 NEW: SCAFFOLD State Memory
global_c_state = None  
local_c_state = None   

# --- 2. WebSocket Event Listeners (Background Threads) ---
@sio.event
def connect():
    print(f"\n[🔌] Successfully connected to Node.js Aggregator as {sio.get_sid()}")
    sio.emit('join_as_hospital')

@sio.on('start_training_round')
def on_start_training(data):
    global current_round
    current_round = data['round']
    print(f"\n[🚦] Server says: GREEN LIGHT for Round {current_round}")
    server_response_event.set()

@sio.on('apply_global_update')
def on_apply_update(data):
    global global_model_weights, current_round, global_c_state
    print(f"\n[📥] Downloading updated global master weights from Server...")
    global_model_weights = json_ready_to_state_dict(data['global_weights'])
    
    # 🔥 NEW: Safely extract and convert the global control variate for SCAFFOLD
    raw_global_c = data.get('global_c')
    if raw_global_c is not None:
        global_c_state = json_ready_to_state_dict(raw_global_c)
    else:
        global_c_state = None
        
    current_round = data['round']
    server_response_event.set() 

@sio.on('training_finished')
def on_training_finished():
    global training_finished
    print("\n[✅] Server announced training is complete!")
    training_finished = True
    server_response_event.set()

@sio.on('server_error')
def on_server_error(data):
    print(f"\n[🚨] SERVER ERROR: {data['message']}")
    server_response_event.set()

@sio.event
def disconnect():
    print("\n[🔌] Disconnected from Server.")

# --- 3. The Evaluation Engine ---
def evaluate_global_model(model, data_loader, device='cpu'):
    """ Tests the downloaded master model before local training begins to track accuracy """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total if total > 0 else 0
    avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0
    return accuracy, avg_loss

# --- 4. The Main Execution Flow ---
def main():
    global global_model_weights, training_finished, global_c_state, local_c_state

    parser = argparse.ArgumentParser()
    parser.add_argument('--hospital_id', type=int, required=True, help="ID of this hospital (1-4)")
    parser.add_argument('--algo', type=str, default='fedavg', choices=['fedavg', 'wsm_ce_fedavg', 'scaffold', 'wsm_class_weighted', 'wsm_hm_class_weighted'], help="The FL algorithm to use")
    parser.add_argument('--disable_smote', action='store_true', help="Pass this flag to disable SMOTE balancing locally")
    args = parser.parse_args()
    
    print(f"========== 🏥 HOSPITAL {args.hospital_id} [{args.algo.upper()}] NODE STARTING ==========")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[⚙️] Hardware: Using device {device}")

    # To support low power gareeb system
    batch_size = 8

    # A. Load the local database 
    train_loader, in_channels = get_local_hospital_loader(
        hospital_id=args.hospital_id, 
        batch_size=batch_size, 
        use_smote=not args.disable_smote 
    )
    dataset_size = len(train_loader.dataset)
    num_classes = FL_CONFIG["NUM_CLASSES"]

    # --- Setup the Local CSV Logger ---
    folder_suffix = "_nosmote" if args.disable_smote else ""
    results_dir = os.path.join("results", f"{args.algo}{folder_suffix}_results")
    
    os.makedirs(results_dir, exist_ok=True) 

    csv_filename = os.path.join(results_dir, f"hospital_{args.hospital_id}_metrics.csv")
    
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', encoding='utf-8') as f:
            f.write("Round,Accuracy,Loss\n")

    # B. Initialize the Model (Shared Initialization)
    model = GenericClientModel(
        in_channels=in_channels, 
        num_classes=num_classes, 
        image_size=FL_CONFIG["IMAGE_SIZE"]
    )

    # C. Connect to the Central Aggregator
    try:
        sio.connect('http://localhost:3000')
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return

    # D. The Federated Learning Loop
    while not training_finished:
        print("\n[🛑] Waiting for Server instructions (Red Light)...")
        server_response_event.wait()
        server_response_event.clear()
        
        if training_finished:
            break

        print(f"\n--- 🔄 FL ROUND {current_round} ---")

        # 1. Apply Master Weights & Evaluate for the Graph!
        if global_model_weights is not None:
            model.load_state_dict(global_model_weights)
            
            # Memory Management for 8GB RAM
            del global_model_weights 
            global_model_weights = None 
            gc.collect() 
            
            print("   [⚙️] Global model weights injected into local CNN.")
            
            # Evaluate on the LOCAL TRAINING DATA
            print("   [📊] Evaluating Global Model on local training data...")
            global_eval_acc, global_eval_loss = evaluate_global_model(model, train_loader, device=device)
            print(f"   [🎯] Local Accuracy: {global_eval_acc*100:.2f}%")
            
            # Append the metrics to this hospital's personal CSV file
            with open(csv_filename, mode='a', encoding='utf-8') as f:
                f.write(f"{current_round},{global_eval_acc},{global_eval_loss}\n")
        else:
            # First round has no global weights yet
            global_eval_acc, global_eval_loss = 0.0, 0.0

        # 🔥 CRITICAL FIX 3: Dynamic Learning Rate based on Round
        # 🔥 UPDATED for 100 Rounds: Stretch the decay schedule
        dynamic_lr = 0.001
        if current_round > 40:      # Settle phase
            dynamic_lr = 0.0001
        if current_round > 75:      # Final fine-tuning phase
            dynamic_lr = 0.00001

        # 2. Dynamic Algorithm Routing
        # 2. Dynamic Algorithm Routing
        if args.algo == 'fedavg':
            # 🔥 CRITICAL FIX 2: Pass dynamic_lr and device
            training_results = run_fedavg(
                model, 
                train_loader, 
                epochs=2, 
                lr=dynamic_lr, 
                device=device
            )
        elif args.algo == 'wsm_ce_fedavg':
            training_results = run_wsm_ce_fedAvg(model, train_loader, epochs=2, lr=dynamic_lr, device=device)
            
        elif args.algo == 'scaffold':
            print(f"\n[⚙️] Using SCAFFOLD Algorithm...")
            weights, post_train_acc, post_train_loss, new_local_c = run_scaffold(
                model, train_loader, 
                global_c=global_c_state, 
                local_c=local_c_state, 
                epochs=3, lr=dynamic_lr, device=device
            )
            local_c_state = new_local_c # Store locally for the next round!
            
            training_results = {
                "weights": weights,
                "extra_fields": {
                    "local_c": state_dict_to_json_ready(new_local_c) # Serialize control variate for server
                }
            }
            
        elif args.algo in ['wsm_class_weighted','wsm_hm_class_weighted']:
            print(f"\n[⚙️] Using custom WSM Algorithm ({args.algo}) with LR {dynamic_lr}...")
            weights, post_train_acc, post_train_loss, beta_array, total_samples = run_wsm_class_weighted(
                model, train_loader, epochs=3, lr=dynamic_lr, device=device
            )
            training_results = {
                "weights": weights,
                "extra_fields": {
                    "beta": beta_array
                }
            }

        # 3. Serialize and Construct Payload
        print("   [⬆️] Preparing payload...")
        
        payload = {
            "algo": args.algo,
            "dataset_size": dataset_size,
            "metrics": { "accuracy": global_eval_acc, "loss": global_eval_loss },
            "is_compressed": False,
            "smote_disabled": args.disable_smote
        }
        
        # Attach the extra fields dynamically (beta for WSM, local_c for SCAFFOLD)
        if "extra_fields" in training_results:
            payload.update(training_results["extra_fields"])

        if "delta_weights" not in payload:
             payload["delta_weights"] = state_dict_to_json_ready(training_results["weights"])

        sio.emit('upload_weights', payload)

    print("\n🎉 Federated Learning pipeline executed successfully! Shutting down.")
    sio.disconnect()

if __name__ == '__main__':
    main()