import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30

stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])
print("Downloading Data...")
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting Training...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss / len(train_loader):.4f}")

print("Training Finished.")

torch.save(model.state_dict(), "../output/model_cnn.pth")
print("Model saved to ../output/model_cnn.pth")