# DeepLearning-Digit-Recognition 🚀

基于 PyTorch CNN 的手写数字识别项目。

## 项目简介

本项目使用 MNIST 数据集训练卷积神经网络，实现 0-9 手写数字自动分类。

## 技术栈

- Python
- PyTorch
- CNN 卷积神经网络
- MNIST 数据集
- Matplotlib 可视化

## 项目功能

- MNIST 数据加载
- CNN 模型训练
- 测试集准确率评估
- 模型保存与加载
- 单张图片数字预测
- 训练过程可视化

## 项目结构

```text
DeepLearning-Digit-Recognition
│
├── README.md
├── requirements.txt
├── train.py
├── test.py
├── predict.py
│
├── model
│   └── cnn.py
│
└── utils
    └── dataset.py
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 运行

训练模型：

```bash
python train.py
```

测试模型：

```bash
python test.py
```

预测数字：

```bash
python predict.py
```

## CNN结构

输入 28×28 灰度图片，通过卷积层提取图像特征，再经过全连接层输出10分类结果。

## 目标效果

模型训练后测试准确率预计达到 98% 左右。

## License

MIT License
