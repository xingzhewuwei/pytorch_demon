# stage2_autograd.py —— 自动求导 autograd 入门
# 运行：在 Linux venv 里执行  python stage2_autograd.py
# autograd 是 PyTorch 和 NumPy 最本质的区别：它能自动算梯度（链式求导），
# 这正是训练神经网络"反向传播"的发动机。

import torch

print("PyTorch 版本:", torch.__version__)

# ---------------------------------------------------------------
# 1) requires_grad：告诉 PyTorch "这个张量需要求梯度"
# ---------------------------------------------------------------
print("\n[1] requires_grad 与 grad")
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1          # y = x^2 + 3x + 1
y.backward()                     # 反向传播，求 dy/dx
print("x =", x.item(), " y =", y.item())
print("dy/dx =", x.grad.item(), "  (解析值 2x+3 =", 2 * x.item() + 3, ")")

# ---------------------------------------------------------------
# 2) 计算图：只有"叶子节点"会保留 .grad，中间结果默认不保留
# ---------------------------------------------------------------
print("\n[2] 计算图与叶子节点")
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = a * b                        # 中间节点（非叶子）
d = c + 1
d.backward()
print("a.grad =", a.grad.item(), " (应= b = 4)")
print("b.grad =", b.grad.item(), " (应= a = 3)")
print("c.grad =", c.grad, " (中间节点默认 None，只有叶子保留 grad)")

# ---------------------------------------------------------------
# 3) 梯度会累加 —— 训练循环里必须清零，否则越加越大
# ---------------------------------------------------------------
print("\n[3] 梯度累加与 zero_grad")
w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    out = w * 2                  # d(out)/dw = 2
    out.backward()
    print(f"  第{i+1}次后 w.grad =", w.grad.item(), "(每次 +2)")
    w.grad.zero_()               # 清零，否则下次累加
print("清零后 w.grad =", w.grad.item())

# ---------------------------------------------------------------
# 4) 实战：用梯度下降手动拟合  y = 2x + 3
#    这一步看清"发动机"：算 loss -> backward -> 更新参数
# ---------------------------------------------------------------
print("\n[4] 实战：梯度下降拟合线性函数")
torch.manual_seed(0)
X = torch.linspace(0, 1, 20).reshape(-1, 1)        # 20 个样本
true_w, true_b = 2.0, 3.0
Y = true_w * X + true_b + 0.1 * torch.randn_like(X)  # 加噪声

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.1
for step in range(200):
    pred = w * X + b
    loss = ((pred - Y) ** 2).mean()                 # MSE 损失
    loss.backward()
    with torch.no_grad():                           # 更新参数时不跟踪梯度
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    if step % 50 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}  w={w.item():.3f}  b={b.item():.3f}")

print(f"收敛后：w≈{w.item():.3f} (目标 2.0)，b≈{b.item():.3f} (目标 3.0)")

# ---------------------------------------------------------------
# 5) torch.no_grad()：推理/评估时关掉图跟踪，省内存、提速
# ---------------------------------------------------------------
print("\n[5] 评估模式 no_grad")
with torch.no_grad():
    test = w * torch.tensor([[0.5]]) + b
    print("预测 x=0.5 ->", test.item())

print("\n✅ 阶段2 autograd 完成。记住：loss.backward() 算梯度，.grad 拿梯度，训练要 zero_grad 再更新。")
