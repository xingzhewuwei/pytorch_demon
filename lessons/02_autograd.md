# 阶段 2 · 自动求导 autograd

## 目标

理解 PyTorch 和 NumPy 最本质的区别：**autograd 能自动算梯度（链式法则）**，这正是「反向传播」的发动机。
你几乎永远不用手算导数，只要会调用、懂它"在算导数"就够了。

## 这个脚本在演示什么

1. `requires_grad=True` + `backward()` → 自动算出 `dy/dx`
2. 计算图：只有「叶子节点」保留 `.grad`，中间节点默认是 `None`
3. 梯度会累加 → 训练循环里必须 `zero_grad()`，否则越加越大
4. 实战：用梯度下降手动拟合 `y = 2x + 3`，看 `w`、`b` 逐渐收敛
5. `torch.no_grad()`：推理/评估时关掉图跟踪，省内存、提速

> 看不懂 `dy/dx = 7` 怎么来的？先跑 `02_intuition.py`：用「把 x 推 0.001 看 y 变多少」的笨办法验证，结果和 `backward()` 完全一致。

## 怎么跑

```bash
uv run lessons/02_autograd.py
uv run lessons/02_intuition.py     # 导数直觉（给初学者）
```

## 过关自测

- 能口述完整流程：`loss.backward()` 算梯度 → `.grad` 拿梯度 → 更新参数前先 `zero_grad()`
- 手算一个 2 层小网络的梯度，能和 `backward()` 的结果对上

## 一句话总结

`backward()` 干的事 = 自动求导数；训练时反复「算梯度 → 清零 → 更新」，模型就学会了。
