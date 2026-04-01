# Federated Learning Framework for Skewed Medical Data

> **Overview**: A custom Federated Learning (FL) architecture designed to tackle catastrophic gradient explosion and client drift in highly skewed (**Non-IID**) environments. It introduces the **Harmonic Mean Class-Weighted (WSM-HM)** aggregation algorithm to protect minority hospitals with rare diseases from being overwritten by majority dictator nodes, and compares it against standard **IID** baselines.

---

## 🏗️ Architecture

This project uses a hybrid architecture:
* **Central Aggregator (Server)**: A Node.js `socket.io` server responsible for routing, receiving weights, and calculating advanced mathematical aggregations (FedAvg, Class-Weighted, Harmonic Mean).
* **Hospital Nodes (Clients)**: Python-based PyTorch clients simulating individual hospitals. They train locally on their isolated data partitions and upload delta weights to the server.
* **Global Evaluator**: A dedicated Python script that monitors the true global accuracy of the master model on a balanced test set, tracking specific algorithms and parameters.

---

## ⚙️ Setup & Installation

### 1. Prerequisites
* **Python 3.8+**
* **Node.js (v16+)** & **npm**

### 2. Install Dependencies
```bash
# Server Dependencies
cd server
npm install express socket.io

# Client Dependencies
cd ..
pip install -r requirements.txt
```

### 3. Initialize Local Databases (IID & Non-IID)
Run the setup script to partition the datasets and distribute them among the simulated hospitals. 
* **Non-IID**: Creates highly skewed distributions where some hospitals monopolize rare diseases.
* **IID**: Creates a perfectly uniform, shuffled baseline distribution.

```bash
python setup_local_dbs.py
```

---

## 🧪 Experimental Toggles: SMOTE & Data Distribution

Before running the execution steps below, you must decide what type of experiment you are running:

1. **SMOTE vs. No-SMOTE (`--disable_smote`)**
   * **With SMOTE (Default)**: Hospitals use Synthetic Minority Over-sampling to artificially balance their local data before training.
   * **No-SMOTE (`--disable_smote`)**: Forces the algorithm to rely *entirely* on your mathematical WSM architecture to handle the imbalance. (This is where the Harmonic Mean shines).
2. **Non-IID vs. IID (`--iid`)**
   * Run your tests in the skewed environment first, then append the `--iid` flag to your clients to prove the algorithm also performs safely in standard uniform environments.

---

## 🚀 Execution Guide

To run a full Federated Learning training session, open **separate terminal windows** for the Server, Evaluator, and Clients.

### Step 1: Start the Central Server
The Node.js server will wait for the target number of hospitals (4) to connect before starting Round 1.
```bash
cd server
node server.js
```

### Step 2: Start the Global Evaluator
The evaluator MUST be told which algorithm it is tracking so it saves the `final_master_model` and `server_avg_metrics` correctly.
```bash
# Example for Harmonic Mean without SMOTE
python server_evaluator.py --algo wsm_hm_class_weighted_nosmote
```

### Step 3: Start the Hospital Clients
Run the client node script, specifying the hospital ID, the algorithm, and your experimental toggles. 

**Available Algorithms (`--algo`):**
* `fedavg`: Standard Federated Averaging.
* `wsm_class_weighted`: Pure Class-Weighted aggregation (boosts rare diseases).
* `wsm_hm_class_weighted`: Hybrid Harmonic Mean aggregation (stabilizes gradient explosions).

**Start Hospital 1:**
```bash
python clients/client_node.py --hospital_id 1 --algo wsm_hm_class_weighted --disable_smote
```

**Start Hospital 2:**
```bash
python clients/client_node.py --hospital_id 2 --algo wsm_hm_class_weighted --disable_smote
```

**Start Hospital 3:**
```bash
python clients/client_node.py --hospital_id 3 --algo wsm_hm_class_weighted --disable_smote
```

**Start Hospital 4:**
```bash
python clients/client_node.py --hospital_id 4 --algo wsm_hm_class_weighted --disable_smote
```
*(Once Hospital 4 connects, Round 1 will begin automatically).*

---

## 📊 Graph Plotting & Evaluation Suite

Run these scripts from the `GraphPlotter/` directory **after** your 50 rounds finish to analyze the results.

### 1. Global Master Model Learning Curve
**Definition**: Plots the true learning curve of the Global Master Model (Accuracy vs. Loss) across all 50 rounds, comparing IID vs Non-IID or SMOTE vs No-SMOTE.
```bash
python GraphPlotter/server_global_plot.py
```

### 2. Local Hospital Training Performance
**Definition**: Generates individual performance curves for each specific hospital on their own skewed datasets. Crucial for identifying **Client Drift** and local gradient explosions.
```bash
python GraphPlotter/client_local_accuracy_plot.py
```

### 3. Average Local Accuracy
**Definition**: Aggregates and averages the local test accuracies of all hospitals before local training begins, proving whether the newly downloaded global weights benefited the minority hospitals.
```bash
python GraphPlotter/average_local_accuracy_plot.py
```

### 4. Global Confusion Matrix
**Definition**: Generates a heatmap matrix of predictions vs. true labels. It unveils the "Minority Overcorrection" paradox, proving whether the model learned the rare diseases (True Positives) or blindly guessed them (False Positives).
```bash
python GraphPlotter/plot_confusion_matrix.py --algo wsm_hm_class_weighted_nosmote
```

### 5. Final Results Comparison
**Definition**: The ultimate thesis defense chart. Visually compares the final Round 50 accuracy of Standard `FedAvg` against your custom algorithms across both IID and Non-IID datasets.
```bash
python GraphPlotter/plot_final_results.py
```