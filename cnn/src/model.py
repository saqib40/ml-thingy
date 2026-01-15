import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # LAYER 1
        # Input: 3 channels (RGB), 32x32 image
        # Output: 32 feature maps, 16x16 image (after pool)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # LAYER 2
        # Input: 32 feature maps (from prev layer)
        # Output: 64 feature maps, 8x8 image (after pool)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # CLASSIFIER aka dense layers
        # The Math: 64 channels * 8 height * 8 width = 4096
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10) # 10 Output classes

    def forward(self, x):
        # Block 1 flow
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2 flow
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flattening flow
        # x.size(0) is the batch size. -1 tells PyTorch to calculate the remaining size (4096)
        x = x.view(x.size(0), -1) 
        
        # Classification flow
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x