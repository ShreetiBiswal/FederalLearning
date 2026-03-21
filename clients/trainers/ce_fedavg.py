import torch.nn as nn
import torch.optim as optim

def run_ce_fedavg(model, train_loader, epochs=2):
    print(f"   [⚡] Starting CE-FedAvg local training (Adam Optimizer)...")
    model.train()
    
    # CE-FedAvg Optimization: Adam instead of SGD
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"      Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")
    
    return {
        "weights": model.state_dict()
    }