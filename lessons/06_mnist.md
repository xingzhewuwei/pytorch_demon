# 阶段 6：MNIST 手写数字分类（真实数据集）

> 对应脚本：`lessons/06_mnist.py`
> 运行：`uv run lessons/06_mnist.py`（首次会自动下载 MNIST，约 10MB+，之后用缓存）
> 前置：阶段 5（分类任务）。本阶段第一次用**真实数据集**，训练骨架完全一致，只是数据换成图片。

## 学习目标
用真实数据走完一遍「图片分类」全流程，理解图片输入和之前玩具数据的具体差异。

## 和阶段 5 玩具数据的 4 个关键区别

| 维度 | 阶段 5 玩具数据 | 阶段 6 MNIST |
|---|---|---|
| 输入 | 2D 向量 `(2,)` | **28×28 灰度图**，需 `flatten` 成 784 维向量 |
| 类别 | 3 类 | **10 类**（数字 0~9） |
| 数据来源 | 脚本里 `torch.randn` 现造 | `torchvision.datasets.MNIST(download=True)` 自动下载 |
| 准确率 | 100%（数据太简单） | **~97%**（真实数据，不会满分） |

## 运行预期

```
使用设备: cpu  (或 mps)
训练集 60000 张，测试集 10000 张，10 个类别
MNISTNet(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (net): Sequential(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=10, bias=True)
  )
)
epoch 1 | 平均 loss 0.xxxx
...
epoch 5 | 平均 loss 0.xxxx
✅ 测试集准确率: 97.xx%
```

> 注意：loss 是**平均每个 batch** 的值，不是单样本；5 个 epoch 后测试集准确率约 **95%~98%** 都算正常。

## 演示要点

1. **`transforms.ToTensor()`**：把 PIL 图片转成 tensor，并自动归一化到 `[0,1]`（像素 0~255 → 0~1）。
2. **`transforms.Normalize((0.1307,), (0.3081,))`**：用 MNIST 官方推荐的均值/标准差再做标准化，让训练更稳（照搬即可，不用自己算）。
3. **`nn.Flatten()`**：把 `(B,1,28,28)` 形状压成 `(B,784)`，才能进 `Linear(784, ...)`。图片数据进全连接层前几乎都要 flatten。
4. **`model.to(device)` + `xb.to(device)`**：把模型和数据搬到同一设备（Mac 走 mps、Linux/AMD 走 cpu）。训练和推理都要搬，且必须一致。
5. **数据下载位置**：脚本里 `DATA_ROOT` 指向仓库内的 `data_mnist/`（已被 `.gitignore` 忽略，不会误提交大文件）。

## ⚠️ 下载失败怎么办（torchvision 默认源已失效）

较新版本的 torchvision 里 `MNIST(download=True)` 默认拉取的两个源都已失效：

- `https://ossci-datasets.s3.amazonaws.com/mnist/` → `File not found or corrupted`
- `http://yann.lecun.com/exdb/mnist/` → `HTTP Error 404`

**表现**：终端会先正常显示下载进度（走到 100%），随后抛出
`RuntimeError: Error downloading train-images-idx3-ubyte.gz … HTTP Error 404`。

**解决（已为你写好脚本）**：仓库根目录的 `download_mnist.py` 会从备用镜像
（CVDF Google Storage / GitHub / 国内 ghproxy 加速）直接下载 4 个 `.gz` 文件到
`data_mnist/MNIST/raw/`，torchvision 检测到本地已有就会跳过下载、直接训练。

```bash
uv run download_mnist.py      # 一次性拉好数据
uv run lessons/06_mnist.py   # 重跑，直接训练
```

---

## 为什么是 MLP 而不是 CNN
本阶段先用**最简单的多层感知机（MLP）**跑通流程，证明「之前学的框架直接套真实数据」就行。
CNN（`nn.Conv2d`）能更好地利用图片的局部结构、准确率更高，留到阶段 7 专门讲。

## 过关自测

- [ ] 能说清：为什么图片要先 `Flatten` 才能进 `Linear`？
- [ ] 知道 MNIST 输入是 `28×28=784` 维、输出是 10 维
- [ ] 能口述 `to(device)` 的作用，且知道模型和数据必须同一个 device
- [ ] 跑通后观察：loss 是否每个 epoch 下降？测试集准确率是多少？
- [ ] 改 `batch_size`（如 32 / 128）重跑，看训练速度 / 最终准确率的变化

## 下一步
- 阶段 7：用 **CNN（`nn.Conv2d` + `nn.MaxPool2d`）** 替换 MLP，准确率能提到 99%+，且参数更少——这是图像任务的真正主力。
- 把模型 `state_dict` 保存下来，做「训练一次、多次推理」的推理脚本。
