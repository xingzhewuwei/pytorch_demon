"""
阶段 5：分类任务入门（玩具数据集，零下载依赖）

目标：理解「分类」和之前「回归」在框架里改了哪几处。
  - 标签从连续值   -> 类别编号 (0/1/2)
  - 输出层维度     -> 类别数（不是 1）
  - 损失用 CrossEntropyLoss（内部已含 softmax，模型输出直接给 raw logits）
  - 评估用「准确率」而非 MSE / R²
  - 引入隐藏层 + ReLU 激活，让网络能学到非线性决策边界

运行：uv run python lessons/05_classification.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示器环境也能存图
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. 造玩具分类数据（3 类，2D 特征，零下载依赖）
#    三类中心拉开距离 + 高斯噪声 -> 天然可分，网络轻松学好
# =====================================================================
N_PER_CLASS = 200
centers = {
    0: torch.tensor([ 2.0,  2.0]),   # 右上
    1: torch.tensor([-2.0, -2.0]),   # 左下
    2: torch.tensor([ 2.0, -2.0]),   # 右下
}
X_parts, y_parts = [], []
for cls, c in centers.items():
    pts = c + torch.randn(N_PER_CLASS, 2) * 0.6      # 围绕中心撒点
    X_parts.append(pts)
    y_parts.append(torch.full((N_PER_CLASS,), cls))   # 标签 = 类别编号

X = torch.cat(X_parts, dim=0).float()      # (600, 2)  特征
y = torch.cat(y_parts, dim=0).long()       # (600,)   类别编号，必须是 long 类型！

# 标准化（和回归一样，让训练更快更稳）
mu, sigma = X.mean(0), X.std(0)
X = (X - mu) / sigma

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"数据：{len(X)} 条，{X.shape[1]} 维特征，{y.max().item() + 1} 个类别")

# =====================================================================
# 2. 模型：含隐藏层 + ReLU 的小神经网络
# =====================================================================
class ToyClassifier(nn.Module):
    def __init__(self, n_feat=2, n_hidden=16, n_class=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, n_hidden),     # 输入 -> 隐藏层 (2 -> 16)
            nn.ReLU(),
            nn.Linear(n_hidden, n_class),    # 隐藏层 -> 输出（= 类别数，raw logits）
        )
    def forward(self, x):
        return self.net(x)

model = ToyClassifier()
print(model)
# =====================================================================
# 3. 损失 + 优化器
#    CrossEntropyLoss 内部已含 softmax，所以模型输出【不要】先 softmax
# =====================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =====================================================================
# 4. 训练循环（和回归模板完全一致：前向 -> loss -> zero_grad -> backward -> step）
# =====================================================================
model.train()
for epoch in range(200):
    for xb, yb in loader:
        logits = model(xb)              # (batch, 3)  raw logits
        loss = criterion(logits, yb)    # 直接喂 logits，不要 softmax
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")

# =====================================================================
# 5. 评估：准确率（分类的标配指标）
# =====================================================================
model.eval()
with torch.no_grad():
    logits = model(X)
    pred = logits.argmax(dim=1)         # 取分数最大的类别作为预测
    acc = (pred == y).float().mean().item()
print(f"\n✅ 训练集准确率: {acc * 100:.2f}%")

# =====================================================================
# 6. 可视化决策边界（存 png，headless 也能出图）
# =====================================================================
xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
grid = np.c_[xx.ravel(), yy.ravel()].astype("float32")
grid_t = (torch.tensor(grid) - mu) / sigma     # 用训练时同样的均值/标准差标准化
with torch.no_grad():
    z = model(grid_t).argmax(dim=1).numpy().reshape(xx.shape)

HERE = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots(figsize=(7, 6))
ax.contourf(xx, yy, z, alpha=0.3, cmap="tab10")
X_show = (X * sigma + mu).numpy()              # 还原到原始尺度画散点
colors = ["red", "green", "blue"]
for cls in range(3):
    m = (y == cls).numpy()
    ax.scatter(X_show[m, 0], X_show[m, 1], c=colors[cls], label=f"class {cls}", s=12)
ax.set_title(f"Classification Decision Boundary (acc {acc * 100:.1f}%)")
ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2"); ax.legend()
fig.savefig(os.path.join(HERE, "classification_result.png"), dpi=120)
print("📊 已保存 classification_result.png")

