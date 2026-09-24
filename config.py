import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = './weights/mnist_cnn.pth'
