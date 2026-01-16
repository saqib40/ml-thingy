import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from model import VGG_CIFAR

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001 # If loss explodes (NaN), try reducing to 0.0001
    EPOCHS = 15           # VGG needs more epochs than SimpleCNN

    # --- DATA PREPARATION ---
    # Standard CIFAR-10 stats
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    print("Downloading Data...")
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Num_workers=2 speeds up data loading (use 0 if on Windows and it crashes)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # --- MODEL SETUP ---
    model = VGG_CIFAR().to(device) # Move model to GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train() # Set to training mode (enables Dropout)
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Print average loss per epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    print("✨ Training Finished.")

    # --- SAVE ---
    os.makedirs("output", exist_ok=True)
    save_path = "output/cifar_vgg.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")

if __name__ == '__main__':
    train()