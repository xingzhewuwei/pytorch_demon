"""
手写数字识别器（鼠标手写 → 模型实时识别）

把阶段 6 训好的 MNIST MLP 接上一个鼠标画板：在画板上写 0~9，模型实时识别。
用 gradio 提供画板（浏览器打开，Mac 一定可用，无需 tkinter）。

运行：
    uv add gradio            # 首次安装依赖
    uv run handwriting_app.py
然后打开终端打印的 http://127.0.0.1:7860

要点：
- 画板默认黑底白笔，正好和 MNIST（黑底白字）一致；若你用白底黑笔，代码自动反相，无需手动处理。
- 预处理必须和训练时完全一致：灰度 → 居中缩放至 28×28 → ToTensor([0,1]) → Normalize(0.1307,0.3081)。
- 权重：首次运行会训练 5 epoch 并缓存到 mnist_mlp.pth，之后秒开。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "mnist_mlp.pth")
DATA_ROOT = os.path.join(HERE, "data_mnist")          # MNIST 数据所在（已通过 download_mnist.py 准备好）
device = "mps" if torch.backends.mps.is_available() else "cpu"


# 与 lessons/06_mnist.py 完全一致的模型结构
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                   # (B,1,28,28) -> (B,784)
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                       # 10 类得分（logits）
        )
    def forward(self, x):
        return self.net(self.flatten(x))


def train_and_save(model):
    """首次运行：用 MNIST 训练 5 epoch，权重存盘。"""
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # 与 06 完全一致
    ])
    train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    print("✓ 训练完成，权重已保存到", MODEL_PATH)


def load_or_train():
    model = MNISTNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✓ 已加载缓存权重", MODEL_PATH)
    else:
        print("未找到权重，首次训练（MPS 约几十秒）...")
        train_and_save(model)
    model.eval()
    return model


model = load_or_train()


def preprocess(img):
    """把手绘画板图预处理成模型能吃的 28×28 tensor（与训练管道一致）。

    gradio Sketchpad 返回 dict：{"image": 背景层, "mask": 用户笔画层}（也可能是 numpy / PIL）。
    关键：笔画是"与背景不同的少数像素"，不能按"白像素最多"选层（那会选中纯白背景）。
    做法：把所有图层的"墨迹"（与各自背景差异大的像素）取并集，再统一渲染成黑底白字。
    """
    # 1) 取出所有图层（兼容 PIL / numpy / gradio dict）
    layers = []
    if isinstance(img, dict):
        for key in ("mask", "image", "composite", "layers"):
            v = img.get(key)
            if v is not None:
                layers.append(v)
        if not layers:
            layers = list(img.values())
    elif isinstance(img, np.ndarray):
        layers = [img]
    else:  # PIL.Image
        layers = [img]

    # 2) 统一转成灰度 (H,W) float32
    grays = []
    for layer in layers:
        if layer is None:
            continue
        a = np.array(layer, dtype=np.float32)
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]            # 去掉 alpha
        if a.ndim == 3:
            a = a.mean(axis=2)         # 转灰度
        grays.append(a)

    if not grays:
        return None

    shape = grays[0].shape
    # 3) 合并墨迹：与各自背景(中位数)差异大的像素 —— 这才是笔画
    ink = np.zeros(shape, dtype=bool)
    for g in grays:
        if g.shape != shape:
            continue
        bg = np.median(g)
        thr = max(25.0, 0.1 * (g.max() - g.min() + 1e-6))
        ink |= np.abs(g - bg) > thr

    n_ink = int(ink.sum())
    print(f"[debug] 图层数={len(grays)} 墨迹像素={n_ink}")
    if n_ink == 0:
        return None                    # 没画东西，交给 recognize 提示

    # 4) 墨迹 → 黑底白字 → 裁剪到笔画框 → 缩放居中到 28×28
    digit = (ink * 255).astype(np.uint8)
    ys, xs = np.where(ink)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = digit[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)          # 等比缩放到约 20px（对齐 MNIST 数字尺寸）
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    crop = np.array(Image.fromarray(crop).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[(28 - nh) // 2:(28 - nh) // 2 + nh,
           (28 - nw) // 2:(28 - nw) // 2 + nw] = crop

    # 5) 转张量 + 归一化（与训练完全一致：ToTensor→[0,1]→Normalize(0.1307,0.3081)）
    t = torch.from_numpy(canvas.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    return (t - 0.1307) / 0.3081


def recognize(img):
    if img is None:
        return {}, [["", 0.0] for _ in range(10)]
    x = preprocess(img)
    if x is None:
        print("未检测到笔画，请先写一个数字")
        return {str(i): 0.0 for i in range(10)}, [[str(i), 0.0] for i in range(10)]
    x = x.to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    print(f"预测: {pred}  置信度: {probs[pred] * 100:.1f}%")
    label_dict = {str(i): float(probs[i]) for i in range(10)}
    # 明细：每个数字 (0~9) 的概率百分比，保留 2 位小数
    table = [[str(i), round(float(probs[i]) * 100, 2)] for i in range(10)]
    return label_dict, table


with gr.Blocks(title="手写数字识别") as demo:
    gr.Markdown("# ✍️ 手写数字识别（MNIST MLP）\n用鼠标在左侧画板写 0~9，右侧实时显示识别结果，并列出每个数字的概率。")
    with gr.Row():
        sketch = gr.Sketchpad(label="在这里写数字", height=280, width=280)
        with gr.Column():
            label = gr.Label(num_top_classes=10, label="识别结果（概率条形）")
            table = gr.Dataframe(
                headers=["数字", "概率 (%)"],
                datatype=["str", "number"],
                row_count=10,
                column_count=2,
                interactive=False,
                label="每个数字的概率明细",
            )
    sketch.change(recognize, sketch, [label, table])

if __name__ == "__main__":
    demo.launch()
