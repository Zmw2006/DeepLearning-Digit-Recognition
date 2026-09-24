from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(batch=64):
    tf=transforms.Compose([transforms.ToTensor()])
    train=datasets.MNIST('./data',train=True,download=True,transform=tf)
    test=datasets.MNIST('./data',train=False,download=True,transform=tf)
    return DataLoader(train,batch_size=batch,shuffle=True),DataLoader(test,batch_size=batch)
