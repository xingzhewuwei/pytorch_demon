"""
=====================================================================
阶段 4 · 工业级模板：用 nn.Module + DataLoader 重写房价回归
=====================================================================
本文件把前面「手写 w、b、整批喂数据」的回归，
升级成 PyTorch 真实项目的标准写法。学完这一课，你写的就是
「工业级骨架」——以后写 CNN / RNN / Transformer 都套同一个模板。

模板结构（记住这 6 块，顺序固定）：
    1. 造数据 + 标准化（X 和 y 都标准化）
    2. 划分 训练集 / 测试集
    3. 用 Dataset + DataLoader 封装成「一批批」的数据
    4. 用 nn.Module 定义模型（nn.Linear 自带权重）
    5. 选 损失函数 + 优化器
    6. 标准训练循环：for epoch -> for batch -> 前向/损失/反向/更新
    7. 保存/加载权重 state_dict
    8. 在测试集上评估（这是之前没做的「看不见的数据」）

运行：
    python lessons/04_template.py
=====================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

torch.manual_seed(42)  # 固定随机种子，结果可复现

# =====================================================================
# 1. 准备数据
#    真实项目里数据都存在文件里，训练时从磁盘加载（而不是在脚本里现造）。
#    这里优先读 data/house_price.pt；找不到时再临时生成一份（保证脚本独立也能跑）。
#    想重新固化数据：python data/generate_house_price_data.py
# =====================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "house_price.pt")

if os.path.exists(DATA_PATH):
    blob = torch.load(DATA_PATH)
    X_raw  = blob["X_raw"]    # (100, 3) 原始特征（面积/房间数/地铁距离）
    y_data = blob["y_data"]   # (100,)   真实房价
    n = len(y_data)
    print(f"✅ 已从 {DATA_PATH} 加载 {n} 条样本")
else:
    print("⚠️ 未找到 data/house_price.pt，临时生成数据（运行 data/generate_house_price_data.py 可固化）")
    n = 100
    x_area   = torch.rand(n) * 100 + 50          # 面积 50~150
    x_rooms  = torch.randint(1, 6, (n,)).float()  # 房间数 1~5
    x_subway = torch.rand(n) * 5                 # 地铁距离 0~5
    X_raw = torch.stack([x_area, x_rooms, x_subway], dim=1).float()
    # 🌟 标签用【原始 X】生成（别用归一化后的 X，否则空间错位）
    y_data = (0.5 * X_raw[:, 0]
              + 2.0 * X_raw[:, 1]
              - 1.5 * X_raw[:, 2]
              + 10
              + torch.randn(n) * 2)              # 加一点噪声

# 🌟 输入 X 和标签 y 都标准化，让所有量变成 O(1) —— 训练收敛快、稳
mu_X, std_X = X_raw.mean(0), X_raw.std(0)
X = (X_raw - mu_X) / std_X
mu_y, std_y = y_data.mean(), y_data.std()
y = (y_data - mu_y) / std_y


# =====================================================================
# 2. 划分训练集 / 测试集（80% 训练，20% 测试）
#    训练集：模型从这些样本里学规律
#    测试集：模型「从没见过」的数据，用来检验泛化能力
# =====================================================================
n_train = int(0.8 * n)
perm = torch.randperm(n)                 # 随机打乱索引
train_idx, test_idx = perm[:n_train], perm[n_train:]
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]


# =====================================================================
# 3. Dataset + DataLoader（把数据变成「一批批」）
#    - TensorDataset：把 (X, y) 两个张量打包成「一条样本 = (x, y)」
#    - DataLoader：每次吐出 batch_size 条，自动 shuffle、自动分批
#    这就是 mini-batch 训练：每步只看一小批，比「整批算」更快更稳
# =====================================================================
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # 训练要打乱

test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=len(test_ds))        # 测试一次全上


# =====================================================================
# 4. 用 nn.Module 定义模型（告别手写 w、b）
#    nn.Linear(3, 1) 内部自动创建：权重 w(shape 1x3) + 偏置 b(标量)
#    forward() 定义「数据怎么流过网络」——这里就是一次线性变换
# =====================================================================
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)   # 3 个输入特征 -> 1 个输出（房价）

        # 💡 想升级成深度网络？在中间加几层即可，例如：
        # self.net = nn.Sequential(
        #     nn.Linear(3, 16), nn.ReLU(),
        #     nn.Linear(16, 8),  nn.ReLU(),
        #     nn.Linear(8, 1),
        # )
        # 回归任务的最后一层通常不用激活函数。

    def forward(self, x):
        return self.linear(x)           # 输出 shape: (batch, 1)


model = HousePriceModel()
print("模型结构：")
print(model)
print(f"初始权重（随机初始化）：{model.linear.weight.data.squeeze().tolist()}")


# =====================================================================
# 5. 损失函数 + 优化器
#    回归用 MSELoss（均方误差）；优化器用 Adam
# =====================================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # model.parameters() 自动收齐 w、b


# =====================================================================
# 6. 标准训练循环
#    外层 epoch：把所有训练数据过一遍叫一个 epoch
#    内层 batch：从 DataLoader 里一批批取 (xb, yb)
# =====================================================================
n_epochs = 300
model.train()  # 🌟 进入训练模式（Dropout/BatchNorm 才生效，这里虽无但养成习惯）

for epoch in range(n_epochs):
    for xb, yb in train_loader:
        pred = model(xb).squeeze()      # (batch,1) -> (batch,)
        loss = criterion(pred, yb)      # 算损失
        optimizer.zero_grad()           # 🌟 清零上一步梯度（否则会累加）
        loss.backward()                 # 反向传播，算梯度
        optimizer.step()                # 用梯度更新参数

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d} | 训练 loss: {loss.item():.6f}")

print(f"epoch {n_epochs-1:3d} | 训练 loss: {loss.item():.6f}")


# =====================================================================
# 7. 保存 / 加载权重 state_dict
#    state_dict 是「模型参数（w、b）的字典」，体积很小，便于部署
# =====================================================================
torch.save(model.state_dict(), "house_price_model.pth")
print("\n✅ 权重已保存为 house_price_model.pth")

# 模拟「下次重新加载模型」：新建一个同结构模型，灌入权重
model2 = HousePriceModel()
model2.load_state_dict(torch.load("house_price_model.pth"))
model2.eval()  # 🌟 进入评估模式


# =====================================================================
# 8. 在测试集上评估（模型从没见过这些数据）
#    torch.no_grad()：评估时不建计算图，省内存、提速
# =====================================================================
model2.eval()
with torch.no_grad():
    pred_test = model2(X_test).squeeze()        # 标准化空间里的预测
    test_loss = criterion(pred_test, y_test)

    # 反标准化，还原成「真实房价」量级
    pred_test_real = pred_test * std_y + mu_y
    y_test_real    = y_test    * std_y + mu_y

    # 算 R²（决定系数）：越接近 1 说明拟合越好
    ss_res = ((y_test_real - pred_test_real) ** 2).sum()
    ss_tot = ((y_test_real - y_test_real.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

print(f"\n=== 测试集表现 ===")
print(f"测试 loss (标准化空间): {test_loss.item():.6f}")
print(f"R² = {r2.item():.4f}  （越接近 1 越好）")

print("\n=== 反标准化：看模型学到了什么规律 ===")
w_norm = model2.linear.weight.data.squeeze().numpy()
b_norm = model2.linear.bias.data.item()
# 🌟 反标准化公式（X、y 都标准化时）：
#    y_norm = w·x_norm + b
#    => y = (w*std_y/std_X)·x_raw + (b*std_y + mu_y - Σ w*std_y*mu_X/std_X)
w_real = w_norm * std_y.numpy() / std_X.numpy()
b_real = b_norm * std_y.item() + mu_y.item() - np.sum(w_norm * std_y.numpy() * mu_X.numpy() / std_X.numpy())

print(f"机器学到: y = {w_real[0]:.3f}*面积 + {w_real[1]:.3f}*房间数 + ({w_real[2]:.3f})*地铁距离 + {b_real:.2f}")
print(f"真实规律: y = 0.500*面积 + 2.000*房间数 + (-1.500)*地铁距离 + 10.00")

print("\n=== 抽查 3 套房（真实 vs 预测）===")
for i in range(20):
    print(f"第 {i+1} 套: 真实 = {y_test_real[i].item():.2f}, 预测 = {pred_test_real[i].item():.2f}")
