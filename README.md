# 手写数字识别（PyTorch + MNIST）

使用卷积神经网络识别 0–9 的手写数字。提供训练、独立测试集评估、单张图片预测和 Streamlit 网页。仓库默认不包含训练权重，首次使用需要训练。

## 快速开始

需要 Python 3.10+。在仓库根目录运行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 10
python test.py
streamlit run app.py
```

训练时自动下载 MNIST 到 `data/`，在 60000 张训练图片中固定划出 5000 张验证图片，依验证准确率保存 `weights/mnist_cnn_best.pth`。测试集 10000 张仅用于运行 `python test.py` 时评估。首次运行需要下载数据，离线环境请提前准备数据。也可用 `python predict.py --image path/to/digit.png` 预测单张图片。

图片推理会对常见白底黑字反色，裁剪留白、缩放并居中到 28×28；较复杂背景、多个数字、裁剪不全的图片可能识别错误。所示置信度是 softmax 输出，不代表经校准的概率。训练和测试均使用 MNIST 均值、标准差归一化。CNN 由两层卷积与池化、批归一化、Dropout 和全连接层构成。

## Docker 部署

先在宿主机生成 `weights/mnist_cnn_best.pth`，然后运行 `docker compose up --build` 并访问 http://localhost:8501。容器挂载宿主机权重目录；详见 [部署说明](docs/README.md)。

## 验证

```bash
python -m unittest discover -s tests -v
```

CI 运行离线单元测试，包括网络前向和反向传播、权重读写、图片预处理及数据划分。训练准确率随随机性、设备与轮次而变；仓库没有预先测得的准确率或可直接使用的模型文件。模型文件、MNIST 数据及生成结果由 .gitignore 排除。

## 目录

- `model/cnn.py`：网络结构
- `utils/dataset.py`：数据集与划分
- `utils/inference.py`：图片预处理与推理
- `train.py`、`test.py`：训练与评估
- `predict.py`、`app.py`：命令行与网页预测

MIT License。
