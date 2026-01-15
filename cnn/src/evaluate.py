import torch
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN 

# --- CONFIG ---
BATCH_SIZE = 64

# --- DATA ---
stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- LOAD MODEL ---
# No .to(device) needed here
model = SimpleCNN() 

# map_location='cpu' ensures that even if you downloaded a model trained on a GPU,
# it will force it to load onto your CPU.
model.load_state_dict(torch.load("../output/model_cnn.pth", map_location='cpu'))
model.eval()

# --- EVALUATE ---
correct = 0
total = 0

print("Starting Evaluation...")
with torch.no_grad():
    for inputs, labels in test_loader:
        # No .to(device) needed here. PyTorch tensors are on CPU by default.
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")