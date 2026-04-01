import argparse
import threading
import torch
import torch.nn as nn
import socketio
import sys
import os

# Add the parent directory to the path so we can import from 'shared' and 'trainers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.cnn_model import GenericClientModel
from shared.tensor_utils import state_dict_to_json_ready, json_ready_to_state_dict
from data_loader import get_dummy_loaders, get_local_hospital_loader

# --- Import the Modular Algorithms ---
from trainers.fedavg import run_fedavg
from trainers.wsm_ce_fedavg import run_wsm_ce_fedAvg
from trainers.wsm_class_weighted import run_wsm_class_weighted

# --- 1. Global State & Synchronization ---
sio = socketio.Client()
server_response_event = threading.Event()  
training_finished = False

global_model_weights = None
current_round = 1

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
    global global_model_weights, current_round
    print(f"\n[📥] Downloading updated global master weights from Server...")
    global_model_weights = json_ready_to_state_dict(data['global_weights'])
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
def evaluate_global_model(model, data_loader):
    """ Tests the downloaded master model before local training begins to track accuracy """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
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
    global global_model_weights, training_finished

    parser = argparse.ArgumentParser()
    parser.add_argument('--hospital_id', type=int, required=True, help="ID of this hospital (1-4)")
    parser.add_argument('--algo', type=str, default='fedavg', choices=['fedavg', 'wsm_ce_fedavg', 'scaffold', 'wsm_class_weighted', 'wsm_hm_class_weighted'], help="The FL algorithm to use")
    # --- NEW: Flag to toggle SMOTE ---
    parser.add_argument('--disable_smote', action='store_true', help="Pass this flag to disable SMOTE balancing locally")
    args = parser.parse_args()
    
    print(f"========== 🏥 HOSPITAL {args.hospital_id} [{args.algo.upper()}] NODE STARTING ==========")

    #To support low power gareeb system
    batch_size = 8

    # A. Load the local database (Now passing the SMOTE flag)
    train_loader, in_channels = get_local_hospital_loader(
        hospital_id=args.hospital_id, 
        batch_size=batch_size, 
        use_smote=not args.disable_smote # Inverse logic: if disable_smote is True, use_smote is False
    )
    dataset_size = len(train_loader.dataset)
    num_classes = 9

    # --- Setup the Local CSV Logger ---
    folder_suffix = "_nosmote" if args.disable_smote else ""
    results_dir = os.path.join("results", f"{args.algo}{folder_suffix}_results")
    
    os.makedirs(results_dir, exist_ok=True) 

    csv_filename = os.path.join(results_dir, f"hospital_{args.hospital_id}_metrics.csv")
    
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', encoding='utf-8') as f:
            f.write("Round,Accuracy,Loss\n")

    # B. Initialize the Model (Shared Initialization)
    torch.manual_seed(42)
    model = GenericClientModel(in_channels=in_channels, num_classes=num_classes)

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
            # --- ADD THIS FOR 8GB RAM ---
            import gc
            del global_model_weights # Remove the dictionary to free RAM
            global_model_weights = None 
            gc.collect() 
            # ----------------------------
            print("   [⚙️] Global model weights injected into local CNN.")
            
            # Evaluate on the LOCAL TRAINING DATA as requested
            print("   [📊] Evaluating Global Model on local training data...")
            global_eval_acc, global_eval_loss = evaluate_global_model(model, train_loader)
            print(f"   [🎯] Local Accuracy: {global_eval_acc*100:.2f}%")
            
            # Append the metrics to this hospital's personal CSV file
            with open(csv_filename, mode='a', encoding='utf-8') as f:
                f.write(f"{current_round},{global_eval_acc},{global_eval_loss}\n")
        else:
            # First round has no global weights yet
            global_eval_acc, global_eval_loss = 0.0, 0.0

        # 2. Dynamic Algorithm Routing
        if args.algo == 'fedavg':
            training_results = run_fedavg(model, train_loader, epochs=2)
        elif args.algo == 'wsm_ce_fedavg':
            training_results = run_wsm_ce_fedAvg(model, train_loader, epochs=2)
        if args.algo in ['wsm_class_weighted','wsm_hm_class_weighted']:
            print(f"\n[⚙️] Using custom WSM Class-Weighted Algorithm...")
            # It now returns 5 items!
            weights, post_train_acc, post_train_loss, beta_array, total_samples = run_wsm_class_weighted(
            model, train_loader, epochs=3, lr=0.001
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
        
        if "extra_fields" in training_results:
            payload.update(training_results["extra_fields"])

        if "delta_weights" not in payload:
             payload["delta_weights"] = state_dict_to_json_ready(training_results["weights"])

        sio.emit('upload_weights', payload)

    print("\n🎉 Federated Learning pipeline executed successfully! Shutting down.")
    sio.disconnect()

if __name__ == '__main__':
    main()