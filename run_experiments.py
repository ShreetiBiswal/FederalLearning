import subprocess
import time
import sys
import os

# Define the experiments you want to run.
EXPERIMENTS = [
    ("fedavg", True),   
    ("wsm_ce_fedavg", True),   
    ("wsm_hm_class_weighted", True), 
    ("scaffold", True),              
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
         # 🔥 FIX: Run the node server from INSIDE the server folder
        server_process = subprocess.Popen(
            ["node", "server.js"], 
            cwd="server",
            stdout=server_log, 
            stderr=subprocess.STDOUT
        )
        processes.append(server_process)
        time.sleep(3) 

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

        print(f"\n⏳ All processes running. Waiting for 250 rounds to complete...")
        
        exit_code = server_process.wait()
        
        if exit_code == 0:
            print(f"\n✅ EXPERIMENT {log_name.upper()} COMPLETED SUCCESSFULLY.")
            
            print(f"\n[📊] Running Final Evaluation for {log_name}...")
            
            eval_cmd = [sys.executable, "server/evaluate_final_model.py", "--algo", log_name]

            eval_process = subprocess.run(eval_cmd)
            
            if eval_process.returncode == 0:
                print(f"[✅] Final evaluation for {log_name} saved successfully.")
            else:
                print(f"[❌] Warning: Final evaluation script returned an error code for {log_name}.")
                
        else:
            print(f"\n❌ FATAL ERROR: The Node.js server crashed prematurely (Exit Code: {exit_code}).")
            print(f"   Python thought the experiment was done because the server died.")
            print(f"   👉 OPEN THIS FILE NOW: logs/{log_name}/server.log")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[🚨] User Aborted! Terminating all processes...")
    finally:
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
    
    print("\n[📈] Generating Master Report...")
    report_cmd = [sys.executable, "master_report_generator.py"]
    
    report_process = subprocess.run(report_cmd)
    
    if report_process.returncode == 0:
        print("\n[🏆] PIPELINE COMPLETE: Master Report generated successfully!")
    else:
        print("\n[❌] PIPELINE COMPLETE: Master Report generation failed. Check the errors above.")