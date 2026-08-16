"""
阶段 6：MNIST 手写数字分类（真实数据集）

这是你第一次用「真实数据集」，和阶段 5 的玩具数据有几点关键不同：
  - 输入是 28×28 灰度图片 = 784 维，要先 flatten 拉平成向量才能进 Linear
  - 10 个类别（数字 0~9），不是 3 类
  - 数据用 torchvision.datasets.MNIST 自动下载，自带 Dataset 接口
  - 真实数据不会 100%，纯 MLP 大约 97% 左右，更接近实际

运行：uv run lessons/06_mnist.py
（首次运行会自动下载 MNIST，约 10MB+；之后用缓存）

⚠️ 若下载报 404 / "File not found or corrupted"（torchvision 新版默认源已失效）：
    先运行 uv run download_mnist.py 从备用镜像拉好数据，再重跑本脚本即可。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 设备自动探测：Mac M4 走 mps，你的 Linux/AMD 走 cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)
print(f"使用设备: {device}")

# =====================================================================
# 1. 数据：自动下载 MNIST（60000 训练 / 10000 测试）
# =====================================================================
# ToTensor() 把 PIL 图片转成 tensor 并归一化到 [0,1]
# Normalize 用 MNIST 官方推荐的均值/标准差，让训练更稳定
transform = transforms.Compose([
    transforms.ToTensor(),                       # (1,28,28) 灰度图 -> tensor，值缩到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),  # 均值0.1307 / 标准差0.3081
])

# root 指定下载位置；train=True 取训练集，download=True 自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(HERE), "data_mnist")  # 仓库内的数据目录

train_ds = torchvision.datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
test_ds  = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

# DataLoader 批量喂数据（和之前完全一致）
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False)

print(f"训练集 {len(train_ds)} 张，测试集 {len(test_ds)} 张，10 个类别")

# =====================================================================
# 2. 模型：两层 MLP（多层感知机）
#    图片 (1,28,28) -> flatten 成 784 -> 隐层 128 -> ReLU -> 输出 10
# =====================================================================
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()              # 把 (B,1,28,28) 压成 (B,784)
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),             # 784 -> 128
            nn.ReLU(),
            nn.Linear(128, 10),                  # 128 -> 10（10 类得分）
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

model = MNISTNet().to(device)
print(model)

# =====================================================================
# 3. 损失 + 优化器（和分类任务完全一致）
# =====================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================================================
# 4. 训练循环（5 个 epoch，每个 batch 五步：前向->loss->zero_grad->backward->step）
# =====================================================================
model.train()
for epoch in range(5):
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)    # 数据搬到设备（cpu 时等于不动）
        logits = model(xb)                        # (batch, 10)
        loss = criterion(logits, yb)              # 直接喂 raw logits
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg = running_loss / len(train_loader)
    print(f"epoch {epoch + 1} | 平均 loss {avg:.4f}")

# =====================================================================
# 5. 评估：测试集准确率
# =====================================================================
model.eval()
correct, total = 0, 0
with torch.no_grad():                            # 评估时关掉梯度，省内存提速
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)           # 取得分最大的类
        correct += (pred == yb).sum().item()
        total += yb.size(0)

acc = correct / total
print(f"\n✅ 测试集准确率: {acc * 100:.2f}%  ({correct}/{total})")
