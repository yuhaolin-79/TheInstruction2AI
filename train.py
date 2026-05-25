"""
MNIST 手写数字识别 - 训练脚本
运行: python train.py
训练完成后会在项目目录生成 mnist_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTNet
import time
import os


def train():
    # ===================== 超参数 =====================
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001

    # ===================== 设备选择 =====================
    # Mac 上优先使用 MPS (Apple Silicon GPU), 其次 CUDA, 最后 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 使用 Apple Silicon GPU (MPS) 训练")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 使用 NVIDIA GPU (CUDA) 训练")
    else:
        device = torch.device("cpu")
        print("💻 使用 CPU 训练")

    # ===================== 数据准备 =====================
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    print("📦 下载 MNIST 数据集...")
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   训练集: {len(train_dataset)} 张图片")
    print(f"   测试集: {len(test_dataset)} 张图片")

    # ===================== 模型、损失函数、优化器 =====================
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 模型参数量: {total_params:,}")
    print(f"{'='*50}")

    # ===================== 训练循环 =====================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 每 200 个 batch 打印一次进度
            if (batch_idx + 1) % 200 == 0:
                print(
                    f"  Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {running_loss/(batch_idx+1):.4f} | "
                    f"Acc: {100.*correct/total:.1f}%"
                )

        elapsed = time.time() - start_time
        train_acc = 100.0 * correct / total

        # ===================== 测试 =====================
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        test_acc = 100.0 * test_correct / test_total
        print(
            f"✅ Epoch {epoch}/{EPOCHS} 完成 | "
            f"耗时 {elapsed:.1f}s | "
            f"训练准确率 {train_acc:.1f}% | "
            f"测试准确率 {test_acc:.1f}%"
        )
        print(f"{'='*50}")

    # ===================== 保存模型 =====================
    save_path = os.path.join(os.path.dirname(__file__), "mnist_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 模型已保存到 {save_path}")
    print(f"🎉 最终测试准确率: {test_acc:.1f}%")


if __name__ == "__main__":
    train()
