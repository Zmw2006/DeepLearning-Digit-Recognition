import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from model.cnn import CNN
from utils.inference import load_model, prepare_image, predict
from utils.dataset import get_data

class PipelineTests(unittest.TestCase):
    def test_forward_and_training_step(self):
        model = CNN()
        inputs = torch.rand(2, 1, 28, 28)
        labels = torch.tensor([0, 1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = torch.nn.CrossEntropyLoss()(model(inputs), labels)
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        optimizer.step()
        self.assertEqual(tuple(model(inputs).shape), (2, 10))

    def test_prepare_both_backgrounds(self):
        transformed = []
        for background, foreground in ((255, 0), (0, 255)):
            image = Image.new("L", (80, 80), background)
            for x in range(35, 45):
                for y in range(15, 65):
                    image.putpixel((x, y), foreground)
            result = prepare_image(image)
            self.assertEqual(tuple(result.shape), (1, 1, 28, 28))
            self.assertTrue(torch.isfinite(result).all())
            transformed.append(result)
        self.assertTrue(torch.allclose(*transformed))

    def test_save_load_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save(CNN().state_dict(), path)
            model = load_model(path)
            self.assertFalse(model.training)
            label, confidence = predict(Image.new("L", (28, 28)), model)
            self.assertIn(label, range(10))
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)

    def test_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "先运行 python train.py"):
                load_model(Path(directory) / "missing.pth")

    def test_train_validation_test_sizes(self):
        with patch("utils.dataset.datasets.MNIST") as mnist:
            mnist.side_effect = [list(range(60000)), list(range(10000))]
            train, validation, test = get_data()
        self.assertEqual(len(train.dataset), 55000)
        self.assertEqual(len(validation.dataset), 5000)
        self.assertEqual(len(test.dataset), 10000)

if __name__ == "__main__":
    unittest.main()
