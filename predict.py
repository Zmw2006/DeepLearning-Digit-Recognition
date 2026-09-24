import torch
from PIL import Image
from torchvision import transforms
from model.cnn import CNN

model = CNN()
model.load_state_dict(torch.load('./weights/mnist_cnn.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

image = Image.open('test.png')
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    result = torch.argmax(output, dim=1)

print('Prediction:', result.item())
