import argparse
from PIL import Image
from utils.inference import load_model, predict

def main():
    parser = argparse.ArgumentParser(description="识别一张手写数字图片")
    parser.add_argument("--image", required=True, help="图片路径")
    args = parser.parse_args()
    with Image.open(args.image) as image:
        digit, confidence = predict(image, load_model())
    print(f"预测数字：{digit}，置信度：{confidence:.2%}")

if __name__ == "__main__":
    main()
