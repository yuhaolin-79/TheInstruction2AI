"""
MNIST 手写数字识别 - 模型定义
一个简单的卷积神经网络 (CNN)，适合教学演示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    简单的 CNN 结构：
    输入 (1x28x28) → Conv1 → Conv2 → FC1 → FC2 → 输出 (10类)
    """

    def __init__(self):
        super().__init__()
        # 第一层卷积: 1个输入通道 → 16个特征图, 3x3卷积核
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 第二层卷积: 16 → 32个特征图
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 第一层: 卷积 → ReLU → 最大池化 (28x28 → 14x14)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # 第二层: 卷积 → ReLU → 最大池化 (14x14 → 7x7)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 展平
        x = x.view(-1, 32 * 7 * 7)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
