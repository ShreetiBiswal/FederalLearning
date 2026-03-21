import torch.nn as nn
import torch.optim as optim

def run_fedavg(model, train_loader, epochs=2):
    print(f"   [🧠] Starting FedAvg local training for {epochs} epochs...")
    model.train()
    
    # Standard SGD
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
    
    # Return a dictionary so we can easily add extra fields for other algorithms later
    return {
        "weights": model.state_dict()
    }