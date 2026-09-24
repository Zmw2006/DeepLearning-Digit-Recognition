import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from model.cnn import CNN

transform = transforms.Compose([transforms.ToTensor()])

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = CNN()
model.load_state_dict(torch.load('./weights/mnist_cnn_best.pth', map_location='cpu'))
model.eval()

true = []
pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        result = outputs.argmax(1)
        true.extend(labels.numpy())
        pred.extend(result.numpy())

acc = np.mean(np.array(true) == np.array(pred))

print('Accuracy:', acc)
print(classification_report(true, pred))
print('Confusion Matrix:')
print(confusion_matrix(true, pred))
