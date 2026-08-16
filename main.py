import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# ⚠️ 运行前请先安装: pip install matplotlib numpy

"""
🏠 实战场景：预测房价（多变量线性回归）——【v2 收敛版】
特征：房屋面积(平米) / 房间数量 / 距离地铁距离(公里)
真实规律：房价 = 0.5*面积 + 2.0*房间数 - 1.5*地铁距离 + 10

🌟 相比上一版的关键修复：
   上一版「输入 X 归一化、但标签 y 没归一化」。
   y 的均值高达 ~62，Adam 的有效步长≈lr(0.01)，截距 b 要从 0 爬到 62
   需要 6000+ 步，2000 步根本没收敛（loss 停在 3100）。
   本版把【标签 y 也标准化】，所有量都变成 O(1)，几十步就收敛。
"""

# ---- 中文乱码处理 ----
from matplotlib import font_manager
def _has_font(name):
    return any(name.lower() in f.lower() for f in font_manager.findSystemFonts())
_cands = ['STHeiti', 'SimHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Source Han Sans SC']
_chosen = next((c for c in _cands if _has_font(c)), None)
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen]
else:
    print("⚠️ 未找到中文字体，图中中文可能显示为方框（不影响训练）。")
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 伪造多变量数据（先造原始特征）
# ==========================================
torch.manual_seed(42)
n_samples = 100

x_area = torch.rand(n_samples) * 100 + 50       # 面积 50~150
x_rooms = torch.randint(1, 6, (n_samples,))     # 房间数 1~5
x_subway = torch.rand(n_samples) * 5            # 地铁距离 0~5
X_raw = torch.stack([x_area, x_rooms, x_subway], dim=1).float()

# 🌟 ① 标签用【原始特征】按真实物理规律生成（必须在归一化之前）
y_data = (0.5 * X_raw[:, 0] + 2.0 * X_raw[:, 1] - 1.5 * X_raw[:, 2] + 10
          + torch.randn(n_samples) * 2)

# 🌟 ② 输入 X 归一化（只动输入，不动标签）
mu_X = X_raw.mean(dim=0)
std_X = X_raw.std(dim=0)
X_data = (X_raw - mu_X) / std_X

# 🌟 ③【核心修复】标签 y 也标准化，让目标变成 O(1)
mu_y = y_data.mean()
std_y = y_data.std()
y_data_norm = (y_data - mu_y) / std_y

# ==========================================
# 2. 初始化模型参数与优化器
# ==========================================
w = torch.zeros(3, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.Adam([w, b], lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

# ==========================================
# 3. 早停机制
# ==========================================
best_loss = float('inf')
patience = 100
no_improve_count = 0

print("=== 开始多变量线性回归（v2 收敛版）===\n")
for step in range(2000):
    y_pred_norm = (X_data @ w) + b
    loss = ((y_pred_norm - y_data_norm) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    current_loss = loss.item()
    if current_loss < best_loss - 1e-6:
        best_loss = current_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
    if no_improve_count >= patience:
        print(f"🛑 第 {step} 步：Loss 已连续 {patience} 步未改善，提前停止训练！")
        break

    if step % 20 == 0:
        print(f"第 {step:4d} 步 | Loss: {loss.item():.6f} | w: {w.detach().numpy()} | b: {b.item():.4f}")

# ==========================================
# 4. 把预测还原回真实房价尺度
# ==========================================
y_pred_norm = (X_data @ w + b).detach()
y_pred = y_pred_norm * std_y + mu_y          # 反标准化预测值

# 反归一化系数到真实世界（推导见下方注释）：
# 模型: y_norm = X_norm·w + b,  X_norm = (X_raw - mu_X)/std_X
#      => y = mu_y + std_y·(X_raw·(w/std_X) + b - Σ(w·mu_X/std_X))
w_real = (std_y * w.detach() / std_X).numpy()
b_real = (mu_y + std_y * (b.detach() - (w.detach() * mu_X / std_X).sum())).item()

print("\n=== 终极验证：反推真实规律 ===")
print(f"机器学到(真实世界): y = {w_real[0]:.3f}*面积 + {w_real[1]:.3f}*房间数 + ({w_real[2]:.3f})*地铁距离 + {b_real:.2f}")
print(f"真实物理规律:       y = 0.500*面积 + 2.000*房间数 + (-1.500)*地铁距离 + 10")

print("\n=== 预测结果验证 ===")
for i in range(3):
    print(f"第 {i+1} 套房: 真实房价 = {y_data[i].item():.2f}, 机器预测 = {y_pred[i].item():.2f}")

# ==========================================
# 5. 画图
# ==========================================
y_true_np = y_data.numpy()
y_pred_np = y_pred.numpy()

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(y_true_np, y_pred_np, color='blue', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_true_np.min(), y_true_np.max()], [y_true_np.min(), y_true_np.max()],
         color='red', linewidth=3, linestyle='--', label='完美预测线')
plt.title('预测值 vs 真实值'); plt.xlabel('真实房价'); plt.ylabel('机器预测房价')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 2)
n_show = 10
x_labels = [f'第{i+1}套' for i in range(n_show)]
bar_width = 0.35
plt.bar([i - bar_width/2 for i in range(n_show)], y_true_np[:n_show], bar_width, label='真实房价', color='steelblue')
plt.bar([i + bar_width/2 for i in range(n_show)], y_pred_np[:n_show], bar_width, label='机器预测', color='orange')
plt.title(f'前{n_show}套预测对比'); plt.xlabel('房子编号'); plt.ylabel('房价')
plt.xticks(range(n_show), x_labels); plt.legend(); plt.grid(True, linestyle='--', alpha=0.3, axis='y')

plt.subplot(1, 3, 3)
plt.plot(y_true_np, color='steelblue', linewidth=2, marker='o', markersize=4, label='真实房价')
plt.plot(y_pred_np, color='orange', linewidth=2, marker='s', markersize=4, label='机器预测')
plt.title('整体走势贴合度'); plt.xlabel('样本序号'); plt.ylabel('房价')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('house_price_v2.png', dpi=120)
print("\n📊 图表已保存为 house_price_v2.png")
