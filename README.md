# DeepLearning-Digit-Recognition 🚀

基于 PyTorch CNN 的手写数字识别系统。

## 项目简介

本项目使用经典 MNIST 数据集，通过卷积神经网络（CNN）完成 0-9 手写数字分类任务。

从数据加载、模型训练、性能评估到单张图片预测，完整实现深度学习图像分类流程。

## 技术栈

- Python
- PyTorch
- CNN 卷积神经网络
- MNIST 数据集
- Matplotlib

## 项目结构

```text
DeepLearning-Digit-Recognition
│
├── README.md
├── requirements.txt
├── train.py              # 模型训练
├── test.py               # 模型测试
├── predict.py            # 图片预测
│
├── model
│   └── cnn.py             # CNN网络结构
│
├── utils
│   ├── dataset.py         # 数据加载
│   └── visualization.py  # 结果可视化
│
├── weights
│   └── README.md          # 模型文件说明
│
└── results
    └── README.md          # 实验结果说明
```

## 模型结构

输入：

```
28 × 28 灰度图片
```

网络流程：

```
Input
 ↓
Convolution
 ↓
ReLU
 ↓
Pooling
 ↓
Fully Connected
 ↓
10分类输出
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 使用方法

训练：

```bash
python train.py
```

测试：

```bash
python test.py
```

预测：

```bash
python predict.py
```

## 实验结果

训练完成后将在测试集上达到约 98% 的分类准确率。

结果文件包括：

- loss变化曲线
- accuracy变化曲线
- 数字预测结果

## 后续计划

- [ ] 增加 Web 在线数字识别界面
- [ ] 添加模型部署接口
- [ ] 支持用户手写图片上传
- [ ] 使用更先进网络结构优化准确率

## License

MIT License
