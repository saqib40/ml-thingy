%%writefile model.py
import torch
import torch.nn as nn

class VGG_CIFAR(nn.Module):
    def __init__(self):
        super(VGG_CIFAR, self).__init__()
        
        self.block1 = nn.Sequential(
            # Conv 1: 3 -> 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # Batch Norm speeds up convergence
            nn.ReLU(),
            # MaxPool: 32x32 -> 16x16
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Conv 3b: 256 -> 256 (VGG Style: stacking convs)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        # --- CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Math: Final Depth (512) * Final Height (2) * Final Width (2) = 2048
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevents overfitting
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 10) # 10 Output Classes
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x