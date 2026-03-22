import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WSMCrossEntropyLoss(nn.Module):
    """
    Implements the Re-weighted Softmax (WSM) Loss from Legate et al., 2023.
    """
    def __init__(self, beta_proportions, device='cpu'):
        super(WSMCrossEntropyLoss, self).__init__()
        # Add a tiny epsilon (1e-8) to prevent log(0) for missing classes
        self.log_beta = torch.log(beta_proportions + 1e-8).to(device)

    def forward(self, logits, targets):
        # Math trick: beta * exp(logits) == exp(logits + log(beta))
        adjusted_logits = logits + self.log_beta
        return F.cross_entropy(adjusted_logits, targets)

def calculate_local_beta(data_loader, num_classes=9):
    """
    Calculates the proportion (beta) of each class present in the local dataset.
    """
    class_counts = torch.zeros(num_classes)
    total_samples = 0
    
    for _, labels in data_loader:
        class_counts += torch.bincount(labels, minlength=num_classes)
        total_samples += labels.size(0)
        
    beta_proportions = class_counts / total_samples
    print(f"   [📊] Local Class Proportions (Beta): {beta_proportions.tolist()}")
    return beta_proportions

def run_wsm_ce_fedAvg(model, train_loader, epochs=2):
    """
    Executes local training using the WSM loss function and Adam Optimizer.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    print("   [🧮] Calculating WSM Beta proportions...")
    beta_proportions = calculate_local_beta(train_loader)
    
    criterion = WSMCrossEntropyLoss(beta_proportions, device=device)
    
    # --- UPDATED: Now using Adam Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"   [🚀] Starting WSM_CE_FedAvg Local Training (Adam) for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
    model.to('cpu')
    
    return {
        "weights": model.state_dict()
    }