import argparse
import json
import torch
from sklearn.metrics import classification_report, confusion_matrix
from config import BATCH_SIZE, ROOT
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
    report = classification_report(truth, predicted, labels=list(range(10)), digits=4, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=list(range(10)))
    result = {
        "dataset": "MNIST test",
        "samples": len(truth),
        "accuracy": sum(a == b for a, b in zip(truth, predicted)) / len(truth),
        "classification_report": classification_report(
            truth, predicted, labels=list(range(10)), output_dict=True, zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
    }
    path = ROOT / "results" / "test_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report)
    print("混淆矩阵（行是真实标签，列是预测标签）：")
    print(matrix)
    print(f"评估报告：{path}")

if __name__ == "__main__":
    main()
