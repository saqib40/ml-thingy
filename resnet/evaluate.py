import torch
import torchvision
import torchvision.transforms as transforms
from model import ResNet18

def evaluate():
    # --- CONFIG ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128

    # --- DATA ---
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- LOAD MODEL ---
    model = ResNet18().to(device)
    
    try:
        model.load_state_dict(torch.load("output/resnet18_aug.pth", map_location=device))
        print("✅ Weights loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: 'output/resnet18.pth' not found. Run train.py first!")
        return

    model.eval()

    # --- EVALUATE ---
    correct = 0
    total = 0

    print("Starting Evaluation...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"🏆 Accuracy on 10,000 test images: {accuracy:.2f}%")

if __name__ == '__main__':
    evaluate()