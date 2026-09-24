# DeepLearning-Digit-Recognition 🚀

基于 PyTorch CNN 的手写数字识别系统。

## Features

- CNN image classification
- MNIST handwritten digit recognition
- GPU accelerated training
- Model evaluation
- Single image prediction
- Web Demo

## Structure

```text
model/       CNN网络
utils/       数据与评估工具
weights/     模型权重
results/     实验结果
train.py     训练
 test.py     测试
predict.py   图片预测
app.py       Web服务
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Predict

```bash
python predict.py --image demo/test.png
```

## Model

Input: 28x28 grayscale image

CNN -> BatchNorm -> ReLU -> Pooling -> Dropout -> Fully Connected -> 10 classes

## Results

Expected accuracy: about 98% on MNIST test set.

## License

MIT
