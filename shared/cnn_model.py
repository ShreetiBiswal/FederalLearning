import torch
import torch.nn as nn
import torch.nn.functional as F

class GenericClientModel(nn.Module):
    """
    A dynamic Convolutional Neural Network architecture for Federated Learning.
    Fully parameterized for different image sizes, channels, and output classes.
    """
    def __init__(self, in_channels, num_classes, image_size=28):
        super(GenericClientModel, self).__init__()
        
        # --- 1. Feature Extraction ---
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # --- 2. Classification ---
        # Calculate the size after two 2x2 max pooling operations
        final_spatial_size = image_size // 4 
        flattened_size = 32 * final_spatial_size * final_spatial_size
        
        # 🔥 NEW: Dropout layer to prevent violent overfitting on Non-IID data
        self.dropout = nn.Dropout(p=0.5) 
        
        self.fc1 = nn.Linear(flattened_size, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        
        # 🔥 NEW: Apply dropout before the first dense layer
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return x

# --- Quick Test ---
if __name__ == "__main__":
    print("Testing Fully Generic Architecture...\n")

    # 1. Test with PathMNIST settings (28x28)
    path_model = GenericClientModel(in_channels=3, num_classes=9, image_size=28)
    path_data = torch.randn(4, 3, 28, 28) 
    print(f"PathMNIST Output: {path_model(path_data).shape} -> Expect [4, 9]")

    # 2. Test with a massive hypothetical 256x256 medical scan
    massive_model = GenericClientModel(in_channels=1, num_classes=5, image_size=256)
    massive_data = torch.randn(4, 1, 256, 256)
    print(f"Massive Scan Output: {massive_model(massive_data).shape} -> Expect [4, 5]")