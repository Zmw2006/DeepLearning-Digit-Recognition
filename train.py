import argparse
import random
import numpy as np
import torch
from config import BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, MODEL_PATH
from model.cnn import CNN
from utils.dataset import get_data

def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            outputs = model(x.to(DEVICE))
            correct += (outputs.argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / total

def main():
    parser = argparse.ArgumentParser(description="训练 MNIST 手写数字识别模型")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs 和 batch-size 必须大于 0")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    train_loader, validation_loader, _ = get_data(args.batch_size)
    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    best = -1.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        score = accuracy(model, validation_loader)
        print(f"Epoch {epoch}/{args.epochs}: loss={loss_sum / len(train_loader.dataset):.4f}, validation_accuracy={score:.4%}")
        if score > best:
            best = score
            torch.save(model.cpu().state_dict(), MODEL_PATH)
            model.to(DEVICE)
    print(f"最佳验证准确率：{best:.4%}；权重：{MODEL_PATH}")

if __name__ == "__main__":
    main()
