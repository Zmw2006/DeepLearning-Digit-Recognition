import argparse
import torch
from PIL import Image
from torchvision import transforms
from model.cnn import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='test.png')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)
model.load_state_dict(torch.load('./weights/mnist_cnn_best.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image = Image.open(args.image)
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    prob = torch.softmax(output, dim=1)
    confidence, result = torch.max(prob, dim=1)

print('Prediction:', result.item())
print('Confidence:', f'{confidence.item()*100:.2f}%')
