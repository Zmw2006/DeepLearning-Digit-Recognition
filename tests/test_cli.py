import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset
import train
import test as evaluate

class CommandTests(unittest.TestCase):
    def test_train_and_test_generate_reports(self):
        torch.manual_seed(42)
        data = TensorDataset(torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 0, 1]))
        loader = DataLoader(data, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "weights" / "mnist_cnn_best.pth"
            with patch.object(train, "get_data", return_value=(loader, loader, loader)), \
                 patch.object(train, "MODEL_PATH", model_path), \
                 patch.object(train, "ROOT", root), \
                 patch("sys.argv", ["train.py", "--epochs", "1"]):
                train.main()
            self.assertTrue(model_path.is_file())
            history = (root / "results" / "history.csv").read_text(encoding="utf-8")
            self.assertIn("validation_accuracy", history)
            with patch.object(evaluate, "get_data", return_value=(loader, loader, loader)), \
                 patch.object(evaluate, "load_model", side_effect=lambda: __import__(
                     "utils.inference", fromlist=["load_model"]
                 ).load_model(model_path)), \
                 patch.object(evaluate, "ROOT", root), \
                 patch("sys.argv", ["test.py"]):
                evaluate.main()
            result = json.loads((root / "results" / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["samples"], 4)
            self.assertEqual(len(result["confusion_matrix"]), 10)
            self.assertTrue(0 <= result["accuracy"] <= 1)

if __name__ == "__main__":
    unittest.main()
