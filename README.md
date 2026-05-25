# ✍️ MNIST 手写数字识别 Demo

人工智能导论课程演示项目 —— 用一个简单的卷积神经网络 (CNN) 识别手写数字。

## 效果

在网页画板上手写数字，模型实时返回 0-9 各类别的预测概率。

## 项目结构

```
mnist-demo/
├── model.py            # CNN 模型定义 (2层卷积 + 2层全连接)
├── train.py            # 训练脚本 (MNIST 数据集, ~5 epochs)
├── app.py              # Gradio 交互式 Web 界面
├── requirements.txt    # Python 依赖
└── README.md
```

## 快速开始

### 1. 创建 Conda 环境

```bash
conda create -n mnist-demo python=3.11 -y
conda activate mnist-demo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 训练模型

```bash
python train.py
```

在 Mac (Apple Silicon) 上约需 1-2 分钟，训练完成后会生成 `mnist_model.pth`。

### 4. 启动 Web 演示

```bash
python app.py
```

打开浏览器访问 [http://localhost:7860](http://localhost:7860)，在画板上写数字即可看到预测结果。

## 模型简介

| 层 | 说明 |
|---|---|
| Conv2d(1→16, 3×3) + ReLU + MaxPool | 提取低级特征 (边缘、线段) |
| Conv2d(16→32, 3×3) + ReLU + MaxPool | 提取高级特征 (笔画组合) |
| Linear(1568→128) + ReLU + Dropout | 特征映射 |
| Linear(128→10) | 输出 10 个类别的分数 |

总参数量约 **63k**，测试准确率约 **99%**。

## 技术栈

- **PyTorch** — 深度学习框架
- **Gradio** — Web 交互界面
- **MNIST** — 经典手写数字数据集 (60k 训练 / 10k 测试)
