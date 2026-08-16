# stage1_tensor.py —— PyTorch 张量(Tensor) 入门
# 运行：在 Linux venv 里执行  python stage1_tensor.py
# 说明：本机是 AMD 显卡 + CUDA 构建的 torch，device 只会是 cpu。
#       换到 Mac M4 Pro(mps) 或 N 卡(cuda) 时，.to(device) 会自动用 GPU。

import torch
import numpy as np

print("PyTorch 版本:", torch.__version__)

# ---------------------------------------------------------------
# 1) 张量是什么：N 维数组，PyTorch 的基本数据单位
#    类比 NumPy 的 ndarray，但能上 GPU、能记录梯度(后续 autograd)。
# ---------------------------------------------------------------
a = torch.tensor([1, 2, 3, 4])          # 从 Python 列表创建 1 维张量
print("\n[1] 基础创建")
print("a =", a, "| shape:", a.shape, "| dtype:", a.dtype)

# ---------------------------------------------------------------
# 2) 常用创建函数
# ---------------------------------------------------------------
print("\n[2] 常用创建")
print("zeros :\n", torch.zeros(2, 3))                 # 2x3 全 0
print("ones  :\n", torch.ones(2, 3))                  # 2x3 全 1
print("randn :\n", torch.randn(2, 3))                 # 标准正态分布
print("arange->reshape :\n", torch.arange(0, 6).reshape(2, 3))  # 0..5 改形状

# ---------------------------------------------------------------
# 3) 形状(shape) —— 最重要的概念
# ---------------------------------------------------------------
print("\n[3] 形状与变形")
x = torch.randn(2, 3, 4)                  # 形状 (2, 3, 4)
print("x.shape =", x.shape, "| 元素总数 numel:", x.numel())
y = x.reshape(6, 4)                       # 改形状，元素总数必须一致
print("reshape(6, 4) ->", y.shape)
z = x.flatten()                           # 展平为一维
print("flatten ->", z.shape)

# ---------------------------------------------------------------
# 4) 数据类型(dtype)
# ---------------------------------------------------------------
print("\n[4] 数据类型")
i = torch.tensor([1, 2, 3])               # 默认 int64
f = i.float()                             # 转 float32
print("int :", i.dtype, "| 转 float:", f.dtype)

# ---------------------------------------------------------------
# 5) 设备(device)：数据放在哪算
# ---------------------------------------------------------------
print("\n[5] 设备")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("当前可用 device:", device)
t = torch.randn(2, 2).to(device)         # 把张量搬到 device 上
print("t.device =", t.device)

# ---------------------------------------------------------------
# 6) 索引与切片（和 NumPy 完全一致）
# ---------------------------------------------------------------
print("\n[6] 索引切片")
m = torch.arange(1, 10).reshape(3, 3)
print("m:\n", m)
print("m[0]      =", m[0])                # 第 0 行
print("m[:, 1]   =", m[:, 1])             # 第 1 列
print("m[0:2, 1] =", m[0:2, 1])           # 前两行第 1 列

# ---------------------------------------------------------------
# 7) 运算：逐元素 vs 矩阵乘 + 广播(broadcasting)
# ---------------------------------------------------------------
print("\n[7] 运算")
p = torch.tensor([1.0, 2.0, 3.0])
q = torch.tensor([10.0, 20.0, 30.0])
print("逐元素加 p+q =", p + q)            # 逐元素
print("逐元素乘 p*q =", p * q)
print("矩阵乘(点积) p @ p =", p @ p)      # 1 维向量点积 -> 标量
# 广播：形状不同的张量按规则自动对齐后运算
c1 = torch.ones(3, 1)                     # 形状 (3, 1)
c2 = torch.tensor([1.0, 2.0, 3.0])        # 形状 (3,)
print("广播 c1(3,1) + c2(3,) 结果形状:", (c1 + c2).shape)  # -> (3, 3)

# ---------------------------------------------------------------
# 8) 与 NumPy 互转
# ---------------------------------------------------------------
print("\n[8] 与 NumPy 互转")
na = np.array([1.0, 2.0, 3.0])
ta = torch.from_numpy(na)
print("numpy -> tensor:", ta)
print("tensor -> numpy:", ta.numpy())

print("\n✅ 阶段1 张量基础完成。核心三件套：shape / dtype / device")
