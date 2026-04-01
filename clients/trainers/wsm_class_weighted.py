import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def run_wsm_class_weighted(model, train_loader, epochs=2, lr=0.001, device='cpu'):
    print("\n   [🧠] Initializing WSM (Class-Weighted) Local Training...")
    model.to(device)
    model.train()

    # 1. 📊 Calculate Local Class Proportions (Beta) dynamically
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    num_classes = 9
    class_counts = np.bincount(all_labels, minlength=num_classes)
    total_samples = len(all_labels)
    
    # Beta array: proportion of each class
    beta = class_counts / total_samples

    # 2. 🧮 Prepare the WSM Mathematical Log-Adjustments
    # We use the Logit-Adjustment trick for absolute numerical stability.
    log_beta = torch.zeros(num_classes, dtype=torch.float32).to(device)
    for i in range(num_classes):
        if beta[i] > 0:
            log_beta[i] = np.log(beta[i])
        else:
            log_beta[i] = -1e9  # Mathematically drops the class from the softmax denominator
            
    # 3. Standard Optimizer and Criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 4. 🚀 The Local Training Loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (Raw logits: z)
            logits = model(images)
            
            # WSM Magic: Apply the Beta adjustment to the logits BEFORE CrossEntropy
            adjusted_logits = logits + log_beta
            
            # Calculate loss and backward pass
            loss = criterion(adjusted_logits, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(adjusted_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate final local metrics
    local_acc = correct / total if total > 0 else 0
    local_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

    print(f"   [✅] Local WSM Training Complete! Acc: {local_acc*100:.2f}% | Loss: {local_loss:.4f}")

    # Return the state dict, metrics, AND the beta/total_samples for the server payload!
    return model.state_dict(), local_acc, local_loss, beta.tolist(), total_samples