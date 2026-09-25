from PIL import Image, ImageOps
import torch
from torchvision import transforms
from config import MODEL_PATH
from model.cnn import CNN
from utils.dataset import NORMALIZE

def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("L")
    # MNIST 是黑底白字；常见白底图片先反色。
    if sum(image.getdata()) / (image.width * image.height) > 127:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    # 裁去留白，缩放到 20 像素方框并放在 28 像素画布中央。
    mask = image.point(lambda value: 255 if value > 25 else 0)
    bounds = mask.getbbox()
    if bounds:
        image = image.crop(bounds)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    return NORMALIZE(transforms.ToTensor()(canvas)).unsqueeze(0)

def load_model(path=MODEL_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型权重：{path}。请先运行 python train.py")
    model = CNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def predict(image, model):
    with torch.inference_mode():
        probabilities = torch.softmax(model(prepare_image(image)), dim=1)[0]
    return int(probabilities.argmax()), float(probabilities.max())
