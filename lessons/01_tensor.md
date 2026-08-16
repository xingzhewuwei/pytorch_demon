# 阶段 1 · 张量 Tensor

## 目标

搞懂 PyTorch 的基本数据单位「张量（Tensor）」，掌握三件套：**shape（形状）/ dtype（类型）/ device（设备）**。
这是一切的地基，后面所有运算都建立在它之上。

## 这个脚本在演示什么

1. 张量是什么（N 维数组，类比 NumPy 的 `ndarray`，但能上 GPU、能记录梯度）
2. 常用创建：`zeros` / `ones` / `randn` / `arange`
3. 形状与变形：`reshape` / `flatten`（**最重要**）
4. 数据类型 `dtype` 与转换（默认 `int64`，模型里几乎都用 `float32`）
5. 设备 `device` 与 `.to()` 迁移（CPU / MPS / CUDA 自动探测）
6. 索引切片（和 NumPy 完全一致）
7. 运算：逐元素 `*` vs 矩阵乘 `@` + 广播（broadcasting）
8. 与 NumPy 互转

## 怎么跑

```bash
uv run lessons/01_tensor.py
```

## 过关自测

- 能不查文档写出 `x.reshape / transpose`，并说出结果形状
- 能预判 `shape(3,1) + shape(3,)` 的结果形状（→ `(3,3)`，靠广播对齐）
- 知道 `torch.from_numpy` 会**原样保留** dtype（`float64` → `torch.float64`），写模型前记得 `.float()` 统一成 `float32`

## 一句话总结

张量 = 带形状、带类型、能搬设备的多维数组。三件套（shape / dtype / device）看懂，阶段 1 过关。
