import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from shared.config import FL_CONFIG

class WSMCrossEntropyLoss(nn.Module):
    """
    Implements the Re-weighted Softmax (WSM) Loss.
    Updated to handle missing classes by setting log_beta to -10.0.
    """
    def __init__(self, beta_proportions, device='cpu'):
        super(WSMCrossEntropyLoss, self).__init__()
        
        # 1. Start with a tensor of -10.0 (the floor for missing classes)
        log_beta = torch.full_like(beta_proportions, -10.0).to(device)
        
        # 2. Only calculate the actual log for classes that exist in the local data (beta > 0)
        mask = beta_proportions > 0
        if mask.any():
            log_beta[mask] = torch.log(beta_proportions[mask])
            
        self.log_beta = log_beta

    def forward(self, logits, targets):
        # Math: adjusted_logits = raw_logits + log(beta)
        # For missing classes, this subtracts 10 from the logit, 
        # heavily penalizing them without causing NaNs.
        adjusted_logits = logits + self.log_beta
        return F.cross_entropy(adjusted_logits, targets)

def calculate_local_beta(data_loader, num_classes=None):
    """
    Calculates the proportion (beta) of each class present in the local dataset.
    """
    num_classes = num_classes or FL_CONFIG["NUM_CLASSES"]
    class_counts = torch.zeros(num_classes)
    total_samples = 0
    
    for _, labels in data_loader:
        class_counts += torch.bincount(labels, minlength=num_classes)
        total_samples += labels.size(0)
        
    beta_proportions = class_counts / total_samples
    print(f"   [📊] Local Class Proportions (Beta): {beta_proportions.tolist()}")
    return beta_proportions

def run_wsm_ce_fedAvg(model, train_loader, epochs=2, lr=0.001, device='cpu'):
    """
    Executes local training using the updated WSM loss with a stable -10 penalty.
    """
    model.to(device)
    model.train()
    
    print("   [🧮] Calculating WSM Beta proportions...")
    beta_proportions = calculate_local_beta(train_loader)
    
    # The updated loss class now handles the -10.0 logic internally
    criterion = WSMCrossEntropyLoss(beta_proportions, device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"   [🚀] Starting WSM_CE_FedAvg Local Training (Adam) for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping: Final layer of protection against non-IID spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
    # Move model back to CPU to save GPU RAM
    model.to('cpu')
    
    return {
        "weights": model.state_dict()
    }