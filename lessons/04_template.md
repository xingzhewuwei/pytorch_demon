# 阶段 4 · 工业级模板：`nn.Module` + `DataLoader`

> 把「手写 w、b、整批喂数据」的回归，升级成 PyTorch 真实项目的标准写法。  
> 学完这一课，你写的就是**工业级骨架**——以后写 CNN / RNN / Transformer 都套同一个模板。

## 一、本课目标

把前面阶段 3 的房价回归，用「官方标准姿势」重写一遍，掌握 6 个固定模块：

1. **数据标准化**（X 和 y 都标准化，别只做一半）
2. **train / test 划分**（用「没见过的数据」检验泛化能力）
3. **Dataset + DataLoader**（数据变成一批批，支持 mini-batch）
4. **nn.Module 定义模型**（`nn.Linear` 自带权重，不用手写 w、b）
5. **损失 + 优化器**（`MSELoss` / `Adam`）
6. **标准训练循环** + **state_dict 保存加载** + **测试集评估**

## 二、跑起来

```bash
# 在 torch_demon 目录下（已用 uv 建好环境）
uv run python lessons/04_template.py
# 或者你的 pip venv：
python lessons/04_template.py
```

预期输出（关键看这几行）：

```
=== 测试集表现 ===
R² = 0.9865  （越接近 1 越好）

机器学到: y = 0.497*面积 + 2.131*房间数 + (-1.456)*地铁距离 + 10.22
真实规律: y = 0.500*面积 + 2.000*房间数 + (-1.500)*地铁距离 + 10.00
```

R² ≈ 0.99 说明模型几乎完美复现了真实规律。

> 数据从 `data/house_price.pt` 读取（真实项目都从文件加载数据）；该文件不存在时会自动临时生成一份，脚本独立也能跑。想重新固化数据：`python data/generate_house_price_data.py`。

## 三、逐块讲解

### 1. 模型：`nn.Module` 与 `nn.Linear`

```python
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)   # 3 输入 -> 1 输出，自带 w、b
    def forward(self, x):
        return self.linear(x)
```

- `nn.Linear(3, 1)` 内部自动创建权重 `w`（shape 1×3）和偏置 `b`（标量），并帮你初始化好。
- `forward()` 定义「数据怎么流过网络」。线性回归就是一次线性变换 `y = x·wᵀ + b`。
- **想升级成深度网络？** 把 `self.linear` 换成 `nn.Sequential(Linear -> ReLU -> Linear -> ...)` 即可，模板其余部分不用动。

### 2. 数据：`TensorDataset` + `DataLoader`

```python
train_ds    = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
```

- `TensorDataset` 把 `(X, y)` 两个张量打包成「一条样本 = (x, y)」。
- `DataLoader` 每次吐出 `batch_size` 条，并自动打乱（`shuffle=True`，训练时务必开）。
- 这就是 **mini-batch 训练**：每步只看一小批，比「整批算一次梯度」更快、更稳。

### 3. 训练循环（标准四步）

```python
model.train()                       # 进入训练模式
for epoch in range(n_epochs):       # epoch：把所有训练数据过一遍
    for xb, yb in train_loader:     # batch：一批批取
        pred  = model(xb).squeeze() # 前向
        loss  = criterion(pred, yb) # 算损失
        optimizer.zero_grad()       # 🌟 清零旧梯度（否则会累加！）
        loss.backward()             # 反向传播，算梯度
        optimizer.step()            # 用梯度更新参数
```

> **铁律**：`optimizer.zero_grad()` 必须在 `backward()` 之前。漏掉会让梯度跨步累加，训练崩掉。

### 4. 保存 / 加载：`state_dict`

```python
torch.save(model.state_dict(), "house_price_model.pth")   # 只存参数，体积小
model2 = HousePriceModel()
model2.load_state_dict(torch.load("house_price_model.pth"))
model2.eval()   # 🌟 进入评估模式
```

- `state_dict` 是「模型参数字典」，不包含结构，便于部署和版本管理。
- 加载前必须先 `HousePriceModel()` 造一个**同结构**的空模型。
- `model.eval()` 在推理/评估时调用（关闭 Dropout、固定 BatchNorm）。

### 5. 评估：测试集 + `torch.no_grad()`

```python
with torch.no_grad():              # 评估不建计算图，省内存提速
    pred_test = model2(X_test).squeeze()
    test_loss = criterion(pred_test, y_test)
    r2 = 1 - ss_res / ss_tot      # 决定系数，越接近 1 越好
```

- 测试集是模型**训练时从没见过**的数据，用来检验「泛化能力」。
- `torch.no_grad()` 关掉梯度追踪，评估阶段不需要反向传播。

## 四、上节课的坑，这课全规避了

| 阶段 3 踩过的坑           | 本课怎么处理                     |
| ------------------- | -------------------------- |
| 标签用归一化后的 X 生成，空间错位  | 标签用**原始 X** 生成，再单独标准化      |
| 只标准化 X，不标准化 y → 收敛慢 | **X 和 y 都标准化**，Adam 几十步收敛  |
| 整批喂数据（full-batch）   | 用 **DataLoader 小批量**       |
| 手写 w、b、手动矩阵乘        | 用 **nn.Linear**，框架托管参数     |
| 没有测试集概念             | 划分 **train/test**，算 **R²** |

## 五、过关自测

- [ ] 能默写出训练循环「前向 → 损失 → zero_grad → backward → step」五步顺序
- [ ] 能解释 `nn.Linear(3,1)` 里的 `3` 和 `1` 各代表什么
- [ ] 能说清 `model.train()` / `model.eval()` 分别在什么时候用
- [ ] 能说清 `state_dict` 存的是什么、加载前为什么要先建空模型
- [ ] 改 `batch_size` 跑一遍，观察 loss 曲线变化（batch 越小越抖但更省显存）

## 六、下一步

你已经掌握了 PyTorch 监督学习的**完整标准流程**。后续可以：

- **分类任务**：把 `MSELoss` 换成 `CrossEntropyLoss`，输出改成 `nn.Linear(特征, 类别数)`，跑 MNIST。
- **GPU 加速**：`.to('mps')`（Mac）或 `.to('cuda')`（N 卡），把模型和数据一起搬设备。
- **更复杂的网络**：把 `self.linear` 换成多层 `nn.Sequential`，加激活函数。
