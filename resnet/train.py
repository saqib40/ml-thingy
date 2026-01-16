import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from model import ResNet18 

def train():
    # --- CONFIG ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    BATCH_SIZE = 128
    LEARNING_RATE = 0.01 # ResNet with SGD works well with 0.01 or 0.1
    EPOCHS = 30          # Increased to 30 to allow time for learning augmented features

    # --- DATA PREPARATION (WITH AUGMENTATION) ---
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # 1. Define the Augmentation Pipeline
    train_transform = transforms.Compose([
        # Randomly shift the image by up to 4 pixels (padding fills the gap)
        # Forces model to recognize objects that aren't perfectly centered
        transforms.RandomCrop(32, padding=4),
        
        # Randomly flip the image left-to-right (50% probability)
        # Forces model to learn symmetry (a car facing left is still a car)
        transforms.RandomHorizontalFlip(),
        
        # Convert to Tensor and Normalize (Standard steps)
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    print("Downloading Data...")
    # Apply train_transform here
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                              download=True, transform=train_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=2)

    # --- MODEL ---
    model = ResNet18().to(device)

    # CrossEntropyLoss + SGD with Momentum is the "Gold Standard" for ResNet
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    # --- TRAINING LOOP ---
    print(f"🚀 Starting Training for {EPOCHS} epochs with Data Augmentation...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    print("✨ Training Finished.")

    # --- SAVE ---
    os.makedirs("output", exist_ok=True)
    save_path = "output/resnet18_aug.pth" # Renamed file to indicate augmentation
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")

if __name__ == '__main__':
    train()