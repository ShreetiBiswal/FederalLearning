import subprocess
import time
import sys
import os

# Define the experiments you want to run.
EXPERIMENTS = [
    # ("fedavg", True),   
    # ("wsm_ce_fedavg", True),   
    ("scaffold", True),              
    # ("wsm_hm_class_weighted", True), 
]

NUM_HOSPITALS = 4

def run_experiment(algo, disable_smote):
    processes = []
    file_handles = [] # Keep track of log files to close them later
    
    log_name = f"{algo}_nosmote" if disable_smote else algo
    
    # Create a directory to hold the logs
    log_dir = os.path.join("logs", log_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT: {log_name.upper()}")
    print(f"📂 Logs are being written to: ./{log_dir}/")
    print(f"{'='*60}")

    try:
        # 1. Start the Node.js Server
        print("[1] Starting Node.js Aggregator Server... (Running in background)")
        server_log = open(os.path.join(log_dir, "server.log"), "w")
        file_handles.append(server_log)
        
        # 🔥 PATH FIX: Pointing to "server/server.js"
        server_process = subprocess.Popen(
            ["node", "server/server.js"], 
            stdout=server_log, 
            stderr=subprocess.STDOUT
        )
        processes.append(server_process)
        time.sleep(3) 

        # 2. Start the Global Evaluator
        print(f"[2] Starting Global Evaluator... (Running in background)")
        evaluator_log = open(os.path.join(log_dir, "evaluator.log"), "w")
        file_handles.append(evaluator_log)
        
        evaluator_cmd = [sys.executable, "server_evaluator.py", "--algo", log_name]
        evaluator_process = subprocess.Popen(
            evaluator_cmd, 
            stdout=evaluator_log, 
            stderr=subprocess.STDOUT
        )
        processes.append(evaluator_process)
        time.sleep(2)

        # 3. Start the Hospital Clients
        print(f"[3] Starting {NUM_HOSPITALS} Hospital Clients... (Running in background)")
        for i in range(1, NUM_HOSPITALS + 1):
            client_log = open(os.path.join(log_dir, f"hospital_{i}.log"), "w")
            file_handles.append(client_log)
            
            client_cmd = [
                sys.executable, "clients/client_node.py", 
                "--hospital_id", str(i), 
                "--algo", algo
            ]
            if disable_smote:
                client_cmd.append("--disable_smote")
                
            p = subprocess.Popen(
                client_cmd, 
                stdout=client_log, 
                stderr=subprocess.STDOUT
            )
            processes.append(p)
            time.sleep(1)

        # 4. Wait for the Server to finish
        print(f"\n⏳ All processes running. Waiting for 250 rounds to complete...")
        
        # This completely freezes Python until the Node server stops running.
        # We capture the exit code to see HOW it stopped.
        exit_code = server_process.wait()
        
        if exit_code == 0:
            print(f"\n✅ EXPERIMENT {log_name.upper()} COMPLETED SUCCESSFULLY.")
        else:
            print(f"\n❌ FATAL ERROR: The Node.js server crashed prematurely (Exit Code: {exit_code}).")
            print(f"   Python thought the experiment was done because the server died.")
            print(f"   👉 OPEN THIS FILE NOW: logs/{log_name}/server.log")
            
            # Stop the whole queue so it doesn't chain-fail through the rest of the algorithms
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[🚨] User Aborted! Terminating all processes...")
    finally:
        # 5. Cleanup Phase
        print("[🧹] Sweeping up lingering processes and closing logs...")
        for p in processes:
            if p.poll() is None: 
                p.kill()
                p.wait()
                
        for f in file_handles:
            f.close()
        
        time.sleep(5)

if __name__ == "__main__":
    for algo, disable_smote in EXPERIMENTS:
        run_experiment(algo, disable_smote)
        
    print("\n🎉 ALL CONFIGURED EXPERIMENTS HAVE FINISHED!")