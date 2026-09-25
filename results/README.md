# 实验输出

运行 `python train.py` 后，`history.csv` 保存每轮训练损失与验证准确率。运行 `python test.py` 后，`test_metrics.json` 保存独立测试集的准确率、逐类 precision/recall/F1、样本数和 10×10 混淆矩阵。混淆矩阵的行是真实标签，列是预测标签。

实验结果依赖训练权重与环境，默认不提交。运行前没有预设成绩，不能把预期值当作实测结果。
