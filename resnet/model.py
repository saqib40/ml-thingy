%%writefile model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A ResNet block that performs F(x) + x.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # --- F(x) Main Path ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # --- Shortcut Path (x) ---
        self.shortcut = nn.Sequential()
        
        # If dimensions change (stride=2 or channels increase), 
        # we need to adapt 'x' to match 'F(x)' so we can add them.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # The Magic: F(x) + x
        out += self.shortcut(x)
        
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        # --- Initial Conv Layer ---
        # Note: We use 3x3 kernel for CIFAR (standard ResNet uses 7x7 for larger images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # --- The 4 ResNet Layers ---
        # Each layer contains 2 Residual Blocks
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)  # Image size halves (32->16)
        self.layer3 = self._make_layer(128, 256, stride=2) # Image size halves (16->8)
        self.layer4 = self._make_layer(256, 512, stride=2) # Image size halves (8->4)
        
        # --- Classifier ---
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Forces output to 1x1
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        # First block handles the stride/channel change
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Second block is always stride 1
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out