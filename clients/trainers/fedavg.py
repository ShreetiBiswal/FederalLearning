import torch
import torch.nn as nn
import torch.optim as optim

def run_fedavg(model, train_loader, epochs=2, lr=0.001, device='cpu'):
    print(f"\n   [🧠] Starting FedAvg local training for {epochs} epochs at LR={lr}...")
    model.to(device)
    model.train()
    
    # Standard Optimizer (Adam is great here)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # 🔥 CRITICAL FIX 1: Move data to the correct device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            
            # (Optional but recommended) Clip gradients for FedAvg too
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"      Epoch {epoch+1}/{epochs} | Acc: {epoch_acc*100:.2f}% | Loss: {epoch_loss:.4f}")
    
    # Return dictionary for the server payload
    return {
        "weights": model.state_dict()
    }