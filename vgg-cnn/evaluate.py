import torch
import torchvision
import torchvision.transforms as transforms
from model import VGG_CIFAR # <--- Updated Import

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = VGG_CIFAR().to(device)
    
    # Load weights (handle CPU/GPU mismatch automatically)
    try:
        model.load_state_dict(torch.load("output/cifar_vgg.pth", map_location=device))
        print("✅ Weights loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: 'output/cifar_vgg.pth' not found. Run train.py first!")
        return

    # Switch to Evaluation Mode (DISABLES Dropout)
    model.eval()

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