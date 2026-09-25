# 使用与部署

在仓库根目录安装依赖、训练并评估：

```bash
pip install -r requirements.txt
python train.py --epochs 10
python test.py
streamlit run app.py
```

打开 Streamlit 输出的本地网址（通常是 http://localhost:8501），上传只包含一个手写数字的图片。训练自动下载 MNIST；`weights/mnist_cnn_best.pth` 会在验证集准确率提高时保存。测试集从不参与挑选模型。具体使用方式见 [主 README](../README.md)。

## Docker

先在宿主机运行 `python train.py` 生成权重，再执行 `docker compose up --build`，访问 http://localhost:8501。容器挂载宿主机 `weights/` 目录为只读；首次未训练时网页会显示缺少权重的提示。Docker 镜像不包含训练数据或权重，因此仅运行容器不会自动训练。

这里没有预置实验准确率；运行 `python test.py` 后才能获得当前模型在测试集的分类报告与混淆矩阵。
