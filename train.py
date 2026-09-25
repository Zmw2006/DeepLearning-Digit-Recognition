import argparse
import csv
import random
import numpy as np
import torch
from config import BATCH_SIZE, DEVICE, EPOCHS, LEARNING_RATE, MODEL_PATH, ROOT
from model.cnn import CNN
from utils.dataset import get_data

def accuracy(model, loader, device=DEVICE):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            outputs = model(x.to(device))
            correct += (outputs.argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / total

def main():
    parser = argparse.ArgumentParser(description="训练 MNIST 手写数字识别模型")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs、batch-size 和 learning-rate 必须大于 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_loader, validation_loader, _ = get_data(args.batch_size, args.seed)
    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    best = -1.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    history_path = ROOT / "results" / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "validation_accuracy"])
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
            train_loss = loss_sum / len(train_loader.dataset)
            score = accuracy(model, validation_loader)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{score:.6f}"])
            file.flush()
            print(f"Epoch {epoch}/{args.epochs}: loss={train_loss:.4f}, validation_accuracy={score:.4%}", flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), MODEL_PATH)
    print(f"最佳验证准确率：{best:.4%}；权重：{MODEL_PATH}；训练记录：{history_path}")

if __name__ == "__main__":
    main()
