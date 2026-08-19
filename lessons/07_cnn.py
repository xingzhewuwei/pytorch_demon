"""
阶段 7：用 CNN 做 MNIST（卷积神经网络）

为什么要有这一课？
  阶段 6 的 MLP 把 28×28 图片 Flatten 成 784 维向量，等于把图「拍扁成一条线」，
  模型彻底丢失了「哪些像素相邻」的空间信息。手写数字的关键恰恰是局部形状
  （一横一竖是 7，斜着连是 2），所以 MLP 卡在 ~97%。

  CNN 用「卷积」解决这个问题：
    - Conv2d：用一个小的滑动窗口（卷积核，如 3×3）在图上扫描，
              提取局部特征（边缘 / 角 / 笔画）。
    - 权值共享：同一个卷积核扫完整张图 → 天生对「位置平移」不敏感
                （数字写歪一点也能认出来）。
    - MaxPool2d：把特征图缩小一半 → 降维、抗微小形变。

  本课目标：把上面的 MLP 换成 CNN，看测试集准确率从 ~97% 冲到 99%+。

运行：uv run lessons/07_cnn.py
（数据已在 data_mnist/，不会重新下载）
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 设备自动探测：Mac M4 走 mps，其他走 cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)
print(f"使用设备: {device}")

# =====================================================================
# 1. 数据：和阶段 6 完全相同的预处理管道
#    ToTensor 把图片转成 [0,1]；Normalize 用 MNIST 官方均值/标准差
# =====================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(HERE), "data_mnist")

train_ds = torchvision.datasets.MNIST(root=DATA_ROOT, train=True,  download=False, transform=transform)
test_ds  = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=1000, shuffle=False)
print(f"训练集 {len(train_ds)} 张，测试集 {len(test_ds)} 张，10 个类别")


# =====================================================================
# 2. 模型：CNN
#    输入 (B,1,28,28)
#      → Conv2d(1→16, 3×3, padding=1) + ReLU + MaxPool(2)   → (B,16,14,14)
#      → Conv2d(16→32, 3×3, padding=1) + ReLU + MaxPool(2)  → (B,32,7,7)
#      → Flatten → 32*7*7 = 1568
#      → Linear(1568→128) + ReLU → Linear(128→10)
#
#    维度接力检查（加层时务必保证每一步 in/out 对上）：
#      conv1 不改变尺寸（padding=1），所以 28→28，pool 后 28/2=14
#      conv2 不改变尺寸（padding=1），所以 14→14，pool 后 14/2=7
#      flatten = 32 * 7 * 7 = 1568
# =====================================================================
class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,1,28,28) -> (B,16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B,16,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (B,32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B,32,7,7)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                 # -> (B, 32*7*7 = 1568)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.fc(x)


model = CNNNet().to(device)
print(model)

# =====================================================================
# 3. 损失 + 优化器（和分类任务完全一致）
# =====================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================================================
# 4. 训练循环（CNN 收敛很快，5 个 epoch 就到 99%+）
# =====================================================================
model.train()
for epoch in range(5):
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg = running_loss / len(train_loader)
    print(f"epoch {epoch + 1} | 平均 loss {avg:.4f}")

# =====================================================================
# 5. 评估
# =====================================================================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

acc = correct / total
print(f"\n✅ CNN 测试集准确率: {acc * 100:.2f}%  ({correct}/{total})")
print("（对比阶段6的 MLP：约 97.35%）")
