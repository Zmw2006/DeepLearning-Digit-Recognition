# 手写数字识别：从 MNIST 训练到网页预测

[![Python Test](https://github.com/Zmw2006/DeepLearning-Digit-Recognition/actions/workflows/python-test.yml/badge.svg)](https://github.com/Zmw2006/DeepLearning-Digit-Recognition/actions/workflows/python-test.yml)

本项目用 PyTorch 构建卷积神经网络（CNN），识别 0–9 的单个手写数字。可以从零训练、在独立测试集上评估、识别本地图片，或打开 Streamlit 网页上传图片。适合学习完整的图像分类流程。界面和操作说明使用中文。

> **先说明实际状态：**仓库不附带训练好的权重，也没有已测得的准确率。训练会生成 `weights/mnist_cnn_best.pth`，之后才能使用测试和预测功能。CI 通过只说明代码的离线测试通过，不代表模型已经在 MNIST 上完成全量训练。

## 你能做什么

| 功能 | 命令 | 产物 |
| --- | --- | --- |
| 训练模型 | `python train.py` | `weights/mnist_cnn_best.pth`、`results/history.csv` |
| 测试模型 | `python test.py` | 终端报告、`results/test_metrics.json` |
| 识别图片 | `python predict.py --image digit.png` | 终端预测数字和置信度 |
| 打开网页 | `streamlit run app.py` | 可上传图片的本地网页 |
| 自动检查 | `python -m unittest discover -s tests -v` | 离线单元测试结果 |

## 1. 安装环境

建议使用 Python 3.11；项目的 GitHub Actions 在 Python 3.11 上运行。需要能访问网络以安装依赖和首次下载 MNIST，训练权重约由本机训练产生。

**Windows PowerShell：**

```powershell
git clone https://github.com/Zmw2006/DeepLearning-Digit-Recognition.git
cd DeepLearning-Digit-Recognition
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 禁止激活脚本，可直接运行 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`，后续命令中的 `python` 换成该路径。

**macOS/Linux：**

```bash
git clone https://github.com/Zmw2006/DeepLearning-Digit-Recognition.git
cd DeepLearning-Digit-Recognition
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

全部命令都在仓库根目录运行。首次安装 PyTorch 需要下载较大的包；如果使用特定 GPU，PyTorch 的安装方法可能与默认 `pip install` 不同。没有 GPU 也能用 CPU 训练，但耗时会更长。

## 2. 训练与保存模型

```bash
python train.py --epochs 10 --batch-size 64 --seed 42 --learning-rate 0.001
```

参数都有默认值，可以直接使用 `python train.py`。运行时会自动下载 MNIST 到 `data/`：官方训练部分共有 60000 张，代码用固定随机种子划出 55000 张训练图片和 5000 张验证图片；另外 10000 张测试图片不会用于更新参数或挑选最佳权重。网络输入是归一化后的单通道 28×28 图像，输出 10 个类别分数。

每轮打印训练损失和验证准确率；当验证准确率创下新高时保存权重到 `weights/mnist_cnn_best.pth`。`results/history.csv` 会记录 `epoch`、`train_loss` 和 `validation_accuracy` 三列，准确率在 CSV 中是 0–1 的小数。例如 `0.98` 表示 98%。再次训练会覆盖之前的权重和训练记录，如需保留请自行复制这些文件。

固定种子有助于复现数据划分和随机初始化，但不同硬件、PyTorch 版本及并行计算环境仍可能产生小幅差异。训练时若只有 CPU，建议先用 `--epochs 1` 确认流程，再用正式轮数重新训练。

## 3. 在独立测试集上评估

```bash
python test.py
```

脚本加载已保存的最佳验证权重，输出准确率相关的逐类报告和 10×10 混淆矩阵，同时写入 `results/test_metrics.json`。报告包含样本总数、整体准确率、逐类 precision、recall、F1 和混淆矩阵。混淆矩阵**行是真实数字，列是预测数字**；对角线越大，说明正确分类的样本越多。测试集用于最终评估，不应用它反复调整训练参数后仍宣称是一次独立评估。

## 4. 识别自己的图片

```bash
python predict.py --image path/to/digit.png
streamlit run app.py
```

打开 Streamlit 在终端显示的网址，上传 PNG/JPG 图片。最好只包含**一个清晰、完整的数字**。预处理会转灰度，识别常见白底黑字并反色，裁去留白，把数字按比例缩放、居中到 28×28，再使用与 MNIST 相同的均值和标准差归一化。长宽比会保持，但复杂背景、彩色纹理、多个数字、裁剪不全或与 MNIST 风格差异较大的字迹仍可能预测错误。

网页所示「置信度」来自 softmax 分数；它没有经过概率校准，不能解释为正确概率。模型仅预测单个数字，不提供文字识别、身份证号识别或多数字分割功能。

## 在 GitHub 上运行完整实验（可选）

如果本机不方便训练，可以打开仓库的 **Actions → Train and evaluate MNIST → Run workflow**，选择训练轮数并启动。该工作流会下载 MNIST，训练模型，在独立测试集上评估，并把权重、训练 CSV 和测试 JSON 打包为 `mnist-model-and-reports` artifact，供你在该次 Actions 页面下载。运行耗时与 GitHub Actions 可用计算资源有关；默认不会在每次提交时自动全量训练。下载后将权重文件放在仓库的 `weights/` 目录即可运行本地预测。只有该工作流真正成功运行后，下载包里的数值才是该次训练的实测结果。

## 5. Docker 网页部署

先在宿主机按第 1–2 步生成 `weights/mnist_cnn_best.pth`，再运行：

```bash
docker compose up --build
```

浏览器访问 [http://localhost:8501](http://localhost:8501)。Compose 会把宿主机 `weights/` 以只读方式挂载到容器；镜像本身不包含数据或训练权重，也不会自动训练。若只想在本机体验网页，直接运行 `streamlit run app.py` 更简单。更多信息见 [部署说明](docs/README.md)。

## 6. 验证与排错

```bash
python -m unittest discover -s tests -v
```

CI 在每次提交时运行语法检查和离线单元测试：网络前向与反向传播、保存和读取权重、两种背景图片的预处理、缺失权重提示以及训练/验证/测试数据规模。测试无需下载 MNIST，但实际训练和评估需要。

| 现象 | 检查方法 |
| --- | --- |
| 找不到 `mnist_cnn_best.pth` | 先在仓库根目录运行 `python train.py`；确认训练正常结束且权重存在。 |
| 下载 MNIST 失败 | 检查网络；首次下载成功后数据缓存在 `data/`。 |
| `ModuleNotFoundError` | 确认已激活虚拟环境，在仓库根目录执行 `python -m pip install -r requirements.txt`。 |
| 图片识别错误 | 使用单个、清晰、居中的数字，减小背景杂色；查看图像是否被截断。 |
| Docker 页面打不开 | 检查 `docker compose up --build` 是否在运行，以及本机 8501 端口是否被占用。 |

## 项目结构

```text
├── app.py                 Streamlit 网页
├── train.py               训练并选取最佳验证权重
├── test.py                独立测试集评估
├── predict.py             本地单张图片预测
├── config.py              默认参数与路径
├── model/cnn.py           CNN 模型
├── utils/dataset.py       MNIST 数据与固定划分
├── utils/inference.py     图片预处理、权重加载和预测
├── tests/test_pipeline.py 离线单元测试
├── results/               运行后保存实验指标
├── weights/               运行后保存模型权重
├── docs/                  架构与部署说明
└── .github/workflows/     GitHub Actions 自动检查
```

早期实验留下的 `web/` 静态页面和空 `notebooks/` 不参与目前的 Streamlit 应用。完整的数据流参见 [架构说明](docs/architecture.md)。

## 许可

项目代码采用 [MIT License](LICENSE)。MNIST 数据由 torchvision 在首次训练时下载；本仓库不重新分发数据集。请在使用数据时遵守其来源的许可和使用条件。
