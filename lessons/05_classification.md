# 阶段 5：分类任务入门（Classification）

> 前置：阶段 4（工业模板 `nn.Module` + `DataLoader` + 标准循环）。
> 本阶段只改「任务类型」，训练骨架与阶段 4 完全一致。

## 学习目标
理解「分类」相对「回归」在 PyTorch 框架里改了哪几处，跑通一个 3 类分类网络。

## 运行
```bash
uv run python lessons/05_classification.py
```
预期：训练集准确率 ~100%，并生成 `lessons/classification_result.png`（决策边界）。

## 核心变化（和回归对比）

| 维度 | 回归（阶段3/4） | 分类（阶段5） |
|---|---|---|
| 标签 y | 连续值（如房价 79.66） | **类别编号**（0 / 1 / 2，必须是 `long`） |
| 输出层 | `Linear(特征, 1)` | `Linear(特征, 类别数)` |
| 损失 | `MSELoss` | **`CrossEntropyLoss`**（内部已含 softmax） |
| 评估 | MSE / R² | **准确率**（`logits.argmax(1)` 与 y 比较） |
| 网络 | 单层线性 | 加**隐藏层 + ReLU**（学非线性边界） |

## 演示要点
1. **标签是类别编号**：`y = torch.cat([full((N,), 0), ...]).long()`，注意必须是 `long` 类型，否则 `CrossEntropyLoss` 报错。
2. **模型输出是 raw logits**：`Linear(2, 16) -> ReLU -> Linear(16, 3)`，最后一层输出 3 个数，代表 3 类的「得分」。
3. **损失不用手动 softmax**：`CrossEntropyLoss` 内部已做 softmax，直接 `criterion(logits, y)` 即可。
4. **预测取 argmax**：`pred = logits.argmax(dim=1)`，得分最大的类即预测结果。
5. **准确率评估**：`(pred == y).float().mean()`，这就是分类的标配指标。
6. **ReLU 是关键**：没有激活函数的多层线性网络等价于单层线性，学不了弯曲的决策边界；ReLU 引入非线性。
7. **决策边界可视化**：在 2D 网格上逐点预测，画出三类区域，直观看到网络学到的边界。

## 为什么用玩具数据（而不是 MNIST）
本脚本用 `torch.randn` 造 3 类 2D 数据，**零下载依赖**，立刻能跑通、能画图。
MNIST 需要下载数据集（`torchvision.datasets.MNIST`），在部分网络环境下可能较慢。
理解本阶段概念后，把「数据段」换成 MNIST 即可无缝迁移（详见下方下一步）。

## 过关自测
- [ ] 能说清 `CrossEntropyLoss` 和 `MSELoss` 的输入区别（标签类型、输出维度）
- [ ] 知道为什么模型输出不用先 softmax
- [ ] 能把 `argmax` 预测 + 准确率这段背出来
- [ ] 改 `n_hidden`（如 4 / 32）重跑，观察决策边界变化
- [ ] 把隐藏层删掉（`nn.Linear(2,3)` 直接输出），看边界是否变差

## 下一步
- 换成真实数据集 MNIST：把「造数据」段替换为
  `torchvision.datasets.MNIST(..., download=True)`，输入改 28×28=784 维，输出 10 类。
- 引入 `nn.functional.softmax` 理解概率输出：`probs = torch.softmax(logits, dim=1)`。

## 加深实验：加隐藏层（强烈推荐）

跑通单层基线后，建议立刻做「加层」练习，体会深度学习里「深」字的本意：

- 对照脚本：**`lessons/05b_deeper.py`**（3 隐藏层 `2→64→32→16→3`，2851 参数）
- 讲解：**`lessons/05b_deeper.md`**（如何加层、维度链规则、手算参数量、为何不是越深越好）

要点速记：每层 `Linear(in, out)` 的 `in` 必须等于上一层的 `out`（数字链接力）；输出层后面**不要**加 ReLU；参数量 = 各层 `in×out + out` 之和。
