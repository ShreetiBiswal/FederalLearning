# Federated Learning Framework for Skewed Medical Data

> **Overview**: A custom Federated Learning (FL) architecture designed to tackle catastrophic gradient explosion and client drift in highly skewed (**Non-IID**) environments. It utilizes advanced mathematical aggregations like **Harmonic Mean Class-Weighted (WSM-HM)** and **SCAFFOLD** to protect minority hospitals with rare diseases from being overwritten by majority dictator nodes, and compares them against standard **FedAvg** baselines.

This project is specifically engineered for **Full-Participation Cross-Silo FL environments** (e.g., sophisticated hospital networks) running heavily on secure **CPU-bound** computational pipelines.

---

## 🏗️ Architecture

This project strictly utilizes a hybrid, full-participation architecture:
* **Central Aggregator (Server)**: A Node.js `socket.io` server responsible for routing, receiving weights, and calculating advanced global aggregations (`fedavg`, `scaffold`, `wsm_hm_class_weighted`).
* **Hospital Nodes (Clients)**: Python-based PyTorch clients simulating individual hospitals. They train locally on their isolated data partitions using SGD (no Adam to preserve SCAFFOLD identities) and upload delta weights or control variates to the server.
* **Global Evaluator**: A dedicated Python script (`server_evaluator.py`) that passively monitors the true global accuracy of the master model on a balanced test set during training.
* **Automation Orchestrator**: An overarching Python manager (`run_experiments.py`) that automates the deployment of the server, evaluator, and clients simultaneously, managing background processes and logs safely.

---

## ⚙️ Setup & Installation

### 1. Prerequisites
* **Python 3.8+**
* **Node.js (v16+)** & **npm**
* **Target Environment:** macOS/CPU-bound compute environment. 

### 2. Install Dependencies
```bash
# Server Dependencies
cd server
npm install express socket.io

# Client Dependencies
cd ..
pip install -r requirements.txt
```

### 3. Initialize Local Databases
Run the setup script to partition the underlying datasets and dynamically distribute them among the simulated hospitals. This creates the highly skewed, Non-IID distributions required for testing.

```bash
python setup_local_dbs.py
```

---

## 🚀 Execution Guide

### Option A: The Automated Master Pipeline (Recommended)
You do not need to manually spin up terminal windows. A master orchestration script is included to automatically queue, deploy, and evaluate your algorithms over a full 250-round lifecycle.

```bash
python run_experiments.py
```

**What this does automatically:**
1. Spins up the Node.js Server in the background.
2. Deploys the Python validation Evaluator.
3. Deploys 4 individual Hospital clients simultaneously.
4. Funnels all terminal outputs into isolated text files under `./logs/<algo_name>/`.
5. Runs multiple variations back-to-back (e.g., `fedavg`, `scaffold`, `wsm_hm_class_weighted`).
6. Triggers `evaluate_final_model.py` and `master_report_generator.py` upon completion.

*Note: If you wish to enable/disable SMOTE or change algorithms, simply edit the `EXPERIMENTS` array at the top of `run_experiments.py`.*

---

### Option B: Manual Debugging Execution
If you are modifying the code and need to trace errors in real-time, you can run the suite manually by opening **separate terminal windows**.

**Step 1: Start the Central Server**
```bash
cd server
node server.js
```

**Step 2: Start the Global Evaluator**
Needs to know what algorithm you are tracking so it saves the data correctly.
```bash
python server_evaluator.py --algo wsm_hm_class_weighted
```

**Step 3: Start the Hospital Clients**
Run the client node script, specifying the hospital ID and the algorithm:
* `fedavg`
* `scaffold`
* `wsm_class_weighted`
* `wsm_hm_class_weighted`

*Terminal 3 (Hosp 1):*
```bash
python clients/client_node.py --hospital_id 1 --algo wsm_hm_class_weighted --disable_smote
```
*Terminal 4 (Hosp 2):*
```bash
python clients/client_node.py --hospital_id 2 --algo wsm_hm_class_weighted --disable_smote
```
*(Repeat for Hospitals 3 and 4...)*

---

## 📊 Evaluation & Graph Plotting

Upon the completion of the experiments (either manually or via the Orchestrator), you can generate visual analysis reports. 

### 1. The Master Report
If you ran `run_experiments.py`, this master script will automatically digest all logs and metrics and generate a final holistic report summary.
```bash
python master_report_generator.py
```

### 2. Visual Graph Suite
You have a dedicated standalone suite inside `GraphPlotter/` capable of rendering the internal mathematical dynamics. 

**Global Master Model Learning Curve:**
Plots the true baseline global test curve across all rounds.
```bash
python GraphPlotter/server_global_plot.py
```

**Local Hospital Training Performance:**
Generates individual local training loss/acc curves. Crucial for identifying Client Drift and gradient explosions on pure Non-IID subsets.
```bash
python GraphPlotter/client_local_accuracy_plot.py
```

**Global Confusion Matrix:**
Analyzes the final model against a balanced test set to unveil the Minority Overcorrection paradox (distinguishing True rare disease representations against False Positive biases).
```bash
python GraphPlotter/plot_confusion_matrix.py --algo wsm_hm_class_weighted_nosmote
```

**Final Results Comparison:**
A conclusive grouped bar chart directly comparing the final Round 250 accuracy of Standard `FedAvg` against `SCAFFOLD` and the advanced `WSM-HM` custom architecture.
```bash
python GraphPlotter/plot_final_results.py
```