import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def draw_curve(values, title, ylabel, path):
    plt.figure(figsize=(8, 5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def draw_confusion(labels, preds, path):
    matrix = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix)
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.colorbar()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
