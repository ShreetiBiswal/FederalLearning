import torch
import torch.nn as nn
import torch.optim as optim
import copy

def run_scaffold(model, train_loader, global_c, local_c, epochs=3, lr=0.001, device='cpu'):
    """
    SCAFFOLD Local Training (Option II from the paper)
    """
    model.train()
    model.to(device)
    
    # Keep a copy of the original global model weights (x) for the Option II update later
    global_model_weights = copy.deepcopy(model.state_dict())
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Keep your safety measures!
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    total_loss = 0.0
    total_samples = 0
    correct = 0
    num_steps = 0 # Track total gradient steps (K) for the math

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 🔥 THE SCAFFOLD MAGIC: Correct the client drift before stepping!
            # Math: gradient = gradient - local_c + global_c
            if global_c is not None and local_c is not None:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Move control variates to the correct device
                        c_global_tensor = global_c[name].to(device)
                        c_local_tensor = local_c[name].to(device)
                        
                        # Apply the correction to the gradient physically
                        param.grad.data += (c_global_tensor - c_local_tensor)
            
            # 🔥 CRITICAL FIX: Removed gradient clipping.
            # Gradient clipping dynamically scales the learning step, breaking 
            # the fundamental Option II math identity: weight_diff == -lr * sum(gradients).
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            num_steps += 1
            
            # Metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 🔥 OPTION II: Update the Local Control Variate (c_i) without extra data passes
    # Math: c_i_new = c_i_old - c_global + (1 / (K * lr)) * (x_global - y_local)
    
    new_local_c = {}
    new_weights = model.state_dict()
    
    # Calculate effective learning rate (K * eta_l)
    # Note: Adam's effective step is complex, but using raw lr * steps is the standard approximation
    effective_lr = num_steps * lr 
    
    # 🔥 CRITICAL FIX: Only track control variates for actual trainable parameters!
    # Buffers (like BatchNorm running_mean) update via EMA, not gradients, making Option II invalid for them.
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    
    for name in trainable_names:
        x_global = global_model_weights[name].to(device)
        y_local = new_weights[name].to(device)
        
        # Calculate the weight difference
        weight_diff = x_global - y_local
        
        if local_c is None or global_c is None:
            # First round initialization: c_i = (1 / effective_lr) * weight_diff
            new_local_c[name] = (1.0 / effective_lr) * weight_diff
        else:
            # Standard SCAFFOLD update
            c_old = local_c[name].to(device)
            c_glob = global_c[name].to(device)
            new_local_c[name] = c_old - c_glob + (1.0 / effective_lr) * weight_diff
            
        # Move back to CPU to save RAM before sending to server
        new_local_c[name] = new_local_c[name].cpu()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return new_weights, accuracy, avg_loss, new_local_c