import unittest
from unittest.mock import patch
import torch
from PIL import Image
from model.cnn import CNN
from utils.inference import prepare_image, predict
from utils.dataset import get_data

class PipelineTests(unittest.TestCase):
    def test_forward(self):
        self.assertEqual(tuple(CNN()(torch.zeros(2, 1, 28, 28)).shape), (2, 10))

    def test_prepare_both_backgrounds(self):
        for background, foreground in ((255, 0), (0, 255)):
            image = Image.new("L", (80, 80), background)
            for x in range(35, 45):
                for y in range(15, 65):
                    image.putpixel((x, y), foreground)
            result = prepare_image(image)
            self.assertEqual(tuple(result.shape), (1, 1, 28, 28))
            self.assertTrue(torch.isfinite(result).all())

    def test_inference(self):
        label, confidence = predict(Image.new("L", (28, 28)), CNN().eval())
        self.assertIn(label, range(10))
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_train_validation_test_sizes(self):
        with patch("utils.dataset.datasets.MNIST") as mnist:
            mnist.side_effect = [list(range(60000)), list(range(10000))]
            train, validation, test = get_data()
        self.assertEqual(len(train.dataset), 55000)
        self.assertEqual(len(validation.dataset), 5000)
        self.assertEqual(len(test.dataset), 10000)

if __name__ == "__main__":
    unittest.main()
