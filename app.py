import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model.cnn import CNN

st.title('Handwritten Digit Recognition')

model = CNN()
model.load_state_dict(torch.load('./weights/mnist_cnn_best.pth', map_location='cpu'))
model.eval()

file = st.file_uploader('Upload digit image', type=['png','jpg','jpeg'])

if file:
    image = Image.open(file)
    st.image(image, width=200)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, 1)
        confidence, pred = torch.max(prob, 1)

    st.success(f'Prediction: {pred.item()}')
    st.info(f'Confidence: {confidence.item()*100:.2f}%')
