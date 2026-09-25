from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "weights" / "mnist_cnn_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
