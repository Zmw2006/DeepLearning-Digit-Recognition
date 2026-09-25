import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import DATA_DIR

NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))
TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])

def get_data(batch=64, seed=42):
    training = datasets.MNIST(DATA_DIR, train=True, download=True, transform=TRANSFORM)
    testing = datasets.MNIST(DATA_DIR, train=False, download=True, transform=TRANSFORM)
    train, validation = random_split(
        training, [len(training) - 5000, 5000],
        generator=torch.Generator().manual_seed(seed)
    )
    return (
        DataLoader(train, batch_size=batch, shuffle=True),
        DataLoader(validation, batch_size=batch),
        DataLoader(testing, batch_size=batch),
    )
