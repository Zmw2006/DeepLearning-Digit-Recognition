import torch
from model.cnn import CNN
from utils.dataset import get_data

train_loader,_=get_data()
model=CNN()
opt=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for x,y in train_loader:
        out=model(x)
        loss=loss_fn(out,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('epoch',epoch+1,'loss',float(loss))

torch.save(model.state_dict(),'mnist_cnn.pth')
