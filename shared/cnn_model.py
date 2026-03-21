import torch
import torch.nn as nn
import torch.nn.functional as F

class GenericClientModel(nn.Module):
    """
    A dynamic Convolutional Neural Network architecture for Federated Learning.
    By default, it assumes MedMNIST image dimensions (28x28 pixels).
    """
    def __init__(self, in_channels, num_classes):
        super(GenericClientModel, self).__init__()
        
        # --- 1. Feature Extraction ---
        # in_channels is now dynamic (e.g., 1 for Grayscale, 3 for RGB)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # --- 2. Classification ---
        # Note: Assuming 28x28 input images (standard for MedMNIST), 
        # the spatial size shrinks to 7x7 after two MaxPool layers.
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        
        # num_classes is now dynamic (e.g., 2 for Pneumonia, 9 for PathMNIST)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return x

# --- Quick Test to prove it is generic ---
if __name__ == "__main__":
    print("Testing Generic Architecture...\n")

    # 1. Test with PathMNIST settings
    path_model = GenericClientModel(in_channels=3, num_classes=9)
    path_data = torch.randn(4, 3, 28, 28) 
    print(f"PathMNIST Output: {path_model(path_data).shape} -> Expect [4, 9]")

    # 2. Test with PneumoniaMNIST settings
    pneumonia_model = GenericClientModel(in_channels=1, num_classes=2)
    pneumonia_data = torch.randn(4, 1, 28, 28)
    print(f"Pneumonia Output: {pneumonia_model(pneumonia_data).shape} -> Expect [4, 2]")