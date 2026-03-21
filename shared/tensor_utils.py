import torch
import numpy as np

def state_dict_to_json_ready(state_dict):
    """
    Converts a PyTorch model's state_dict (which is full of Tensors) 
    into a standard Python dictionary of nested lists. 
    This format can be safely JSON-serialized and sent over WebSockets to Node.js.
    """
    json_ready_dict = {}
    
    for layer_name, tensor in state_dict.items():
        # 1. .cpu() pulls the tensor off the GPU (crucial if training on Nvidia hardware)
        # 2. .numpy() converts the PyTorch math object to a standard NumPy array
        # 3. .tolist() flattens the NumPy array into pure Python lists
        json_ready_dict[layer_name] = tensor.cpu().numpy().tolist()
        
    return json_ready_dict

def json_ready_to_state_dict(json_dict):
    """
    Takes a JSON payload received from the Node.js server (standard arrays)
    and converts them back into PyTorch Tensors for the local CNN.
    """
    state_dict = {}
    
    for layer_name, list_data in json_dict.items():
        # --- THE FIX: Protect integer tracking variables ---
        if 'num_batches_tracked' in layer_name:
            state_dict[layer_name] = torch.tensor(list_data, dtype=torch.int64)
        else:
            # Safely cast all standard weights/biases to float32
            state_dict[layer_name] = torch.tensor(list_data, dtype=torch.float32)
        
    return state_dict

# --- Quick Test ---
if __name__ == "__main__":
    print("Testing Tensor Serialization Engine...\n")
    
    # Create a fake PyTorch weight (e.g., a tiny 2x2 convolutional layer)
    fake_tensor = torch.tensor([[0.123, 0.456], [0.789, 0.101]], dtype=torch.float32)
    fake_state_dict = {"conv1.weight": fake_tensor}
    
    print(f"1. Original PyTorch Type: {type(fake_state_dict['conv1.weight'])}")
    
    # Serialize it (Simulating the upload to Node.js)
    json_ready = state_dict_to_json_ready(fake_state_dict)
    print(f"2. Serialized Python Type: {type(json_ready['conv1.weight'])}")
    print(f"   Payload looks like: {json_ready}")
    
    # Reconstruct it (Simulating the download from Node.js)
    reconstructed = json_ready_to_state_dict(json_ready)
    print(f"3. Reconstructed Type: {type(reconstructed['conv1.weight'])}")