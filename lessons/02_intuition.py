# stage2_intuition.py —— 用"轻轻推一下"理解导数 dy/dx（给初学者）
import torch

# 问题：y = x^2 + 3x + 1，当 x=2 时，y 对 x 有多敏感？
# 直觉：把 x 轻轻推一点点，看 y 变了多少 -> 比值就是导数

# ===== 方法A：PyTorch 自动算（backward）=====
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()                       # 自动反向求导
print("PyTorch 算出的 dy/dx =", x.grad.item())

# ===== 方法B：手动"推一下"验证（有限差分）=====
x0 = 2.0
y0 = x0 ** 2 + 3 * x0 + 1          # x=2 时的 y
x1 = 2.001                         # 把 x 轻轻推 0.001
y1 = x1 ** 2 + 3 * x1 + 1
slope = (y1 - y0) / 0.001          # (y 的变化) / (x 的变化)
print("手动推一下得到的斜率 ≈", slope, " (约等于 7)")

print("\n解读：x 从 2 变成 2.001（只动了 0.001），y 从",
      round(y0, 4), "变成", round(y1, 4))
print("说明 x 每动 1 单位，y 大约动 7 单位 -> 这就是导数 = 7")
