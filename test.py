import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.cnn import CNN

transform = transforms.Compose([transforms.ToTensor()])

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = CNN()
model.load_state_dict(torch.load('./weights/mnist_cnn.pth'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

print('Accuracy:', correct / total)
