# MNIST 手写数字识别 Demo

人工智能导论课程演示项目 —— 用一个简单的卷积神经网络 (CNN) 识别手写数字。

## 课程说明

本项目是 **《人工智能导论》** 课程的实践环节，旨在通过一个完整的深度学习小项目，帮助大家直观理解：

### 学习目标

1. **理解深度学习基本流程**：数据准备 → 模型构建 → 训练 → 评估 → 部署
2. **掌握 CNN 的核心思想**：卷积层如何提取特征、池化层如何降维、全连接层如何分类
3. **体验完整的模型生命周期**：从训练到 Web 交互式演示，感受深度学习模型的实际应用
4. **动手实践**：修改超参数、调整网络结构、观察对准确率的影响

### 适合人群

- 刚入门深度学习、希望动手跑第一个模型的同学
- 对 CNN 有概念但没见过完整代码的同学
- 想体验"训练 → 部署"全流程的同学

---

## 效果

在网页画板上手写数字，模型实时返回 0-9 各类别的预测概率。

## 项目结构

```
mnist-demo/
├── model.py              # CNN 模型定义 (2层卷积 + 2层全连接)
├── train.py              # 训练脚本 (MNIST 数据集, ~5 epochs)
├── app.py                # Gradio 交互式 Web 界面
├── requirements.txt      # Python 依赖
├── README.md             # 本文件
└── mnist_demo.ipynb      # Jupyter 笔记 (数据集探索 + 模型详解 + 交互演示)
```

## 快速开始

### 1. 创建环境

```bash
# 方式一：使用 Conda（推荐）
conda create -n mnist-demo python=3.11 -y
conda activate mnist-demo

# 方式二：使用 venv（如果已安装 Python 3.11）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 如果下载较慢，可以添加国内镜像源：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 3. 训练模型

```bash
python train.py
```

- 在 CPU 上约需 **3-5 分钟**
- 在 Apple Silicon (MPS) 或 NVIDIA GPU (CUDA) 上约需 **1-2 分钟**
- 训练完成后会生成 `mnist_model.pth`（模型权重文件，约 300KB）

### 4. 启动 Web 演示

```bash
python app.py
```

打开浏览器访问 [http://localhost:7860](http://localhost:7860)，在画板上写数字即可看到预测结果。

---

## 模型架构

| 层 | 输入 → 输出 | 说明 |
|---|---|---|
| Conv2d + ReLU + MaxPool | 1×28×28 → 16×14×14 | 提取低级特征（边缘、线段） |
| Conv2d + ReLU + MaxPool | 16×14×14 → 32×7×7 | 提取高级特征（笔画组合） |
| Dropout(0.25) | — | 随机丢弃 25% 神经元，防止过拟合 |
| Linear + ReLU | 1568 → 128 | 特征映射到 128 维隐层 |
| Dropout(0.25) | — | 再次随机丢弃 |
| Linear | 128 → 10 | 输出 10 个类别的 logits |

- **总参数量**：约 **63k**
- **测试准确率**：约 **99%**
- **损失函数**：CrossEntropyLoss
- **优化器**：Adam（lr=0.001）
- **数据增强**：随机旋转 ±10°、随机平移 ±10%、随机缩放 ±10%、随机剪切 ±5°

---

## 技术栈

- **PyTorch 2.0+** — 深度学习框架
- **Gradio 5+** — Web 交互界面（自动生成前端）
- **MNIST** — 经典手写数字数据集（60k 训练 / 10k 测试）
- **TorchVision** — 数据加载与预处理

---

## 扩展练习

学有余力的同学可以尝试以下方向：

1. **调参实验**：修改 `train.py` 中的学习率、batch size、epochs，观察准确率变化
2. **改网络结构**：增加卷积层或全连接层维度，模型会更好吗？
3. **数据增强**：在 `train_transform` 中添加更多增强方式，看看效果
4. **换数据集**：尝试 Fashion-MNIST 或 EMNIST（修改代码中 `datasets.MNIST` 即可）
5. **部署到云端**：使用 Hugging Face Spaces 免费部署 Gradio 应用

---

## 在线部署

### Jupyter 笔记（GitHub Pages）

本项目的完整分析笔记已部署到 GitHub Pages：
[**https://yuhaolin-79.github.io/TheInstruction2AI/**](https://yuhaolin-79.github.io/TheInstruction2AI/)

### Web 演示（Hugging Face Spaces ⭐ 推荐）

Gradio Web 应用需要后端服务器，可以用 Hugging Face Spaces 免费部署：

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nie2213456/mnist-demo)

在线体验：**[nie2213456/mnist-demo](https://huggingface.co/spaces/nie2213456/mnist-demo)**

> 首次访问时 Space 会从休眠状态唤醒，等待约 30 秒即可使用。

---

## 参考

- [原始项目 GitHub](https://github.com/qiudaoyuu/mnist-demo)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Gradio 文档](https://www.gradio.app/docs/)
- [Hugging Face Spaces 文档](https://huggingface.co/docs/hub/spaces)
