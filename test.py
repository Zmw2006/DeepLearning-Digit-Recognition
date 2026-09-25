import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix
from config import BATCH_SIZE
from utils.dataset import get_data
from utils.inference import load_model

def main():
    parser = argparse.ArgumentParser(description="评估独立 MNIST 测试集")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须大于 0")
    _, _, loader = get_data(args.batch_size)
    model = load_model()
    truth, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            truth.extend(labels.tolist())
            predicted.extend(model(images).argmax(1).tolist())
    print(classification_report(truth, predicted, digits=4, zero_division=0))
    print("混淆矩阵（行是真实标签，列是预测标签）：")
    print(confusion_matrix(truth, predicted))

if __name__ == "__main__":
    main()
