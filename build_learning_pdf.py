# -*- coding: utf-8 -*-
"""
生成《PyTorch 从零实战学习手册》PDF。

依赖：reportlab + Pillow（已用清华镜像安装到托管 venv）。
中文字体：reportlab 内置 CID 字体 STSong-Light（无需本地 TTF，所有阅读器可渲染中文）。

运行：
    uv run --extra "" python build_learning_pdf.py
或直接用装好依赖的 python 运行本文件。

输出：仓库根目录 PyTorch从零实战学习手册.pdf
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Preformatted,
    Image, Table, TableStyle, ListFlowable, ListItem, PageBreak,
)
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "PyTorch从零实战学习手册.pdf")

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
CJK = "STSong-Light"

# ---------- 样式 ----------
def style(name, **kw):
    base = dict(fontName=CJK, fontSize=10, leading=15, textColor=colors.HexColor("#222222"))
    base.update(kw)
    return ParagraphStyle(name, **base)

S_TITLE = style("title", fontSize=26, leading=32, alignment=TA_CENTER,
                textColor=colors.HexColor("#0b3d91"), spaceAfter=6)
S_SUB = style("sub", fontSize=13, leading=18, alignment=TA_CENTER,
              textColor=colors.HexColor("#555555"), spaceAfter=4)
S_META = style("meta", fontSize=10, leading=14, alignment=TA_CENTER,
               textColor=colors.HexColor("#888888"))
S_H1 = style("h1", fontSize=16, leading=22, spaceBefore=16, spaceAfter=8,
             textColor=colors.HexColor("#0b3d91"))
S_H2 = style("h2", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=4,
             textColor=colors.HexColor("#1a5fb4"))
S_BODY = style("body", spaceAfter=6)
S_BULLET = style("bullet", spaceAfter=2)
S_CODE = ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10.5,
                        backColor=colors.HexColor("#f4f5f7"), borderPadding=6,
                        textColor=colors.HexColor("#1a1a1a"), leftIndent=2)
S_NOTE = style("note", fontSize=9.5, leading=14, textColor=colors.HexColor("#5a3e00"))
S_CAP = style("cap", fontSize=8.5, leading=12, alignment=TA_CENTER,
              textColor=colors.HexColor("#888888"), spaceBefore=2, spaceAfter=10)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(t, st=S_BODY):
    return Paragraph(esc(t), st)


def H1(t):
    return Paragraph(esc(t), S_H1)


def H2(t):
    return Paragraph(esc(t), S_H2)


def CODE(t):
    # 代码块用 Courier，保持 ASCII 对齐；注释用英语以免中文字体缺失
    return Preformatted(t.rstrip("\n"), S_CODE)


def NOTE(t):
    inner = Paragraph(esc(t), S_NOTE)
    tbl = Table([[inner]], colWidths=[170 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7e0")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e0b400")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def BULLETS(items):
    return ListFlowable(
        [ListItem(Paragraph(esc(i), S_BULLET), leftIndent=10, value="•") for i in items],
        bulletType="bullet", start="•", leftIndent=12,
    )


def IMG(path, max_w=150 * mm, caption=None):
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return []
    sw, sh = PILImage.open(p).size
    w = max_w
    h = w * sh / sw
    if h > 110 * mm:
        h = 110 * mm
        w = h * sw / sh
    flow = [Image(p, width=w, height=h)]
    if caption:
        flow.append(Paragraph(esc(caption), S_CAP))
    else:
        flow.append(Spacer(1, 8))
    return flow


def TABLE(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    ts = [
        ("FONTNAME", (0, 0), (-1, -1), CJK),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        ts += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), CJK),
        ]
    t.setStyle(TableStyle(ts))
    return t


# ---------- 正文 ----------
story = []

# 封面
story += [Spacer(1, 40 * mm),
          Paragraph("PyTorch 从零实战学习手册", S_TITLE),
          Paragraph("从张量到手写数字识别 · 小白陪练笔记", S_SUB),
          Spacer(1, 6 * mm),
          Paragraph("行者  ·  2026-08-16", S_META),
          Paragraph("配套仓库：torch_demon（uv 管理）", S_META),
          PageBreak()]

# 目录
story += [H1("目录")]
toc = [
    "1. 如何使用这本手册",
    "2. 环境搭建（uv + 镜像源 + Mac MPS）",
    "3. 阶段1：张量 Tensor",
    "4. 阶段2：自动求导 autograd",
    "5. 阶段3：线性回归实战（房价预测）",
    "6. 阶段4：工业级模板（nn.Module + DataLoader）",
    "7. 阶段5：分类任务 Classification",
    "8. 阶段5b：加深对照（加层 / 参数量）",
    "9. 阶段6：MNIST 手写数字分类",
    "10. 应用：手写数字识别器",
    "11. 踩坑合集（环境 / 数据 / 代码）",
    "12. 下一步：CNN 路线图",
    "附录：命令速查表",
]
story += [BULLETS(toc), PageBreak()]

# 1
story += [H1("1. 如何使用这本手册")]
story += [P("这是一份面向零基础 / 工程背景读者的 PyTorch 上手笔记。主线是「房价预测」：从张量 → 自动求导 → 手写回归 → 标准化 → 收敛，再过渡到分类、MNIST，最后落到一个能用的手写数字识别应用。")]
story += [P("学习原则（也是本手册的写法）：概念先行、代码在后、小步验证、不堆公式。每一章都给了「目标 / 核心概念 / 关键代码 / 踩坑 / 过关标准」，跟着跑一遍比读十篇博客有用。")]
story += [NOTE("手册内容与仓库 lessons/ 各课一一对应，但更连贯，适合从头读一遍。所有代码都能在仓库里直接 uv run 复现。")]

# 2
story += [H1("2. 环境搭建（uv + 镜像源 + Mac MPS）")]
story += [P("用 uv 管理环境与依赖（比 pip+venv 更快、可复现，自动生成 uv.lock）。")]
story += [CODE(
"""# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh   # 或 brew install uv

# 进入本仓库后，一键复原环境
uv sync

# 验证 PyTorch 与可用后端
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
""")]
story += [H2("中国大陆：配置 PyPI 镜像源（必看）")]
story += [P("PyPI 官方源在国内经常连不上，uv add 装 gradio/torch 这类大依赖会反复超时。把默认源换成清华镜像，一劳永逸：")]
story += [CODE(
"""mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml <<'EOF'
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
EOF
""")]
story += [NOTE("uv 配置文件顶层直接写 index-url = ... 才会替换默认源；写成 [install] 表或只设 UV_DEFAULT_INDEX_URL 环境变量都不生效（前者会让 uv 启动直接报错，后者只当额外源、PyPI 仍是首选）。")]
story += [H2("GPU 后端说明")]
story += [TABLE([
    ["硬件", "后端", "备注"],
    ["Apple Silicon（Mac M4 等）", "MPS", "PyTorch 自带，mps.is_available() 直接为 True，无需额外驱动"],
    ["NVIDIA 显卡", "CUDA", "cuda.is_available()"],
    ["AMD 显卡（如 RX 6600）", "ROCm", "消费级卡常不在官方白名单，需 HSA_OVERRIDE_GFX_VERSION 等额外配置；学习阶段建议先用 CPU"],
], [55 * mm, 25 * mm, 90 * mm])]
story += [P("本仓库默认 device = 'cpu' / 'mps' / 'cuda' 自动探测，不强依赖 GPU，CPU 即可跑完所有教程。本书示例主力用 Mac M4 Pro 的 MPS。")]

# 3
story += [H1("3. 阶段1：张量 Tensor")]
story += [P("张量（Tensor）就是带维度的数组，是 PyTorch 里一切数据的容器。三个最该记住的属性：")]
story += [BULLETS([
    "shape：形状，如 (2,2) 表示 2 行 2 列；",
    "dtype：数据类型，默认 float32；",
    "device：存在哪（cpu / mps / cuda），模型和数据必须在同一 device。",
])]
story += [P("关键操作：reshape 变形、广播（broadcast，不同形状自动对齐再运算）。")]
story += [CODE(
"""import torch

x = torch.tensor([[1., 2.], [3., 4.]])
print(x.shape)    # torch.Size([2, 2])
print(x.dtype)    # torch.float32
print(x.device)   # cpu

y = x.reshape(4)                 # 变形：2x2 -> 1x4
z = torch.zeros(2, 3)            # 全 0
r = torch.rand(3, 3)             # 均匀随机 [0,1)

# 广播：不同形状自动对齐
a = torch.tensor([1., 2., 3.])   # shape (3,)
b = torch.tensor([[1.], [2.]])   # shape (2,1)
print(a + b)                     # -> (2,3)：b 被广播成 2 行
""")]
story += [H2("过关标准")]
story += [P("能不查文档写出 reshape / 广播，并说清一个张量的 shape / dtype / device。")]

# 4
story += [H1("4. 阶段2：自动求导 autograd")]
story += [P("autograd 是 PyTorch 的核心魔法：你只写「前向计算」，它自动算出每个参数的梯度（导数），反向传播全帮你做了。")]
story += [BULLETS([
    "requires_grad=True：标记「这个量需要求梯度」；",
    "loss.backward()：自动沿计算图反向，把梯度累加进 .grad；",
    "optimizer.zero_grad()：每个 batch 更新前清掉上一次的梯度（否则会累加）；",
    "梯度用于更新参数：w = w - lr * w.grad。",
])]
story += [P("直觉例子：原函数 y = x² + 3x + 1，导数 dy/dx = 2x + 3。在 x=2 处应为 7：")]
story += [CODE(
"""x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1        # 前向：y = x^2 + 3x + 1
y.backward()              # 自动求 dy/dx
print(x.grad)             # -> 7.0  (= 2*2 + 3)

# 多变量
w = torch.tensor([1., 2.], requires_grad=True)
loss = (w**2).sum()       # = w1^2 + w2^2
loss.backward()
print(w.grad)             # -> [2., 4.]  (= 2*w)
""")]
story += [NOTE("backward() 默认会对标量（单个 loss 值）求梯度。如果 loss 是个向量，需要 backward(torch.ones_like(loss)) 指定权重——但训练时 loss 一般都是标量，放心用。")]
story += [H2("过关标准")]
story += [P("能口述 backward() → .grad → zero_grad 的完整流程，并解释「为什么每个 batch 前要 zero_grad」。")]

# 5
story += [H1("5. 阶段3：线性回归实战（房价预测）")]
story += [P("目标：用「面积 / 房间数 / 地铁距离」预测房价，真实规律是 y = 0.5*面积 + 2.0*房间 - 1.5*地铁 + 10。我们让模型自己把系数学出来。")]
story += [H2("两个真实的坑")]
story += [BULLETS([
    "坑A：用「归一化后的 X」去算 y，弱信号特征（面积）会被噪声淹没，系数直接学成 0。正确做法：用原始 X_raw 生成 y，X 单独做归一化。",
    "坑B：只标准化 X、不标准化 y。当 y 均值高达 ~62 时，Adam 的有效步长约 lr，截距要从 0 爬到 62，需 6000+ 步；2000 步根本没收敛（loss 停在 3100）。把 y 也标准化，几十步就收敛。",
])]
story += [P("标准化（Standardization）铁律：输入 X 和标签 y 都要标准化。")]
story += [CODE(
"""# 1) 输入 X 与标签 y 分别标准化（各自减均值除标准差）
X = (X_raw - X_mean) / X_std
y = (y_data - y_mean) / y_std

model = torch.nn.Linear(3, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(2000):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()   # MSE
    opt.zero_grad(); loss.backward(); opt.step()

# 2) 反标准化还原成真实尺度的系数
w_true_scale = model.weight.detach() * (y_std / X_std)
b_true_scale = y_mean - (w_true_scale * X_mean).sum() + model.bias.detach()
# 学出约 0.49/2.07/-1.53/10.9，贴合真实 0.5/2.0/-1.5/10
""")]
story += IMG("house_price_v2.png", caption="图1：房价回归收敛过程与学出的规律")
story += [H2("过关标准")]
story += [P("学出的系数接近真实规律，并能解释「为什么标签 y 也要标准化」。")]

# 6
story += [H1("6. 阶段4：工业级模板（nn.Module + DataLoader）")]
story += [P("把阶段3 的手写写法，重写成 PyTorch 标准工业模板。这是后续写任何网络（CNN/RNN/Transformer）都通用的骨架。")]
story += [BULLETS([
    "nn.Module：用类定义网络，前向写在 forward() 里；",
    "TensorDataset + DataLoader：把数据打包成小批量（batch），shuffle 打乱；",
    "nn.MSELoss / CrossEntropyLoss：损失函数；",
    "optim.Adam：优化器；",
    "state_dict：保存 / 加载权重；",
    "train / test 划分：用测试集评估泛化，报告 R²≈0.99。",
])]
story += [CODE(
"""class HouseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 1)
    def forward(self, x):
        return self.net(x)

model = HouseModel()
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

model.train()
for epoch in range(50):
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()

# 保存与加载
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
""")]
story += [H2("过关标准")]
story += [P("能用 nn.Module + DataLoader 重写阶段3，训练循环「取 batch → 前向 → 损失 → 清零 → 反向 → 更新」五步顺序口述无误，测试集 R²≈0.99。")]

# 7
story += [H1("7. 阶段5：分类任务 Classification")]
story += [P("从「预测连续值」切到「预测类别」。用纯 torch 造 3 类 2D 玩具数据（零下载依赖，先避开 MNIST 的下载坑）。")]
story += [BULLETS([
    "标签是类别编号（整数 long），输出维度 = 类别数（本例 3）；",
    "CrossEntropyLoss 内部已含 softmax，模型输出直接喂 raw logits，不要自己再 softmax；",
    "评估用 logits.argmax(1) 与真实 y 比较，算准确率。",
])]
story += [CODE(
"""model = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 3),
)
loss_fn = torch.nn.CrossEntropyLoss()      # 内含 softmax
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for xb, yb in loader:
    logits = model(xb)
    loss = loss_fn(logits, yb)
    opt.zero_grad(); loss.backward(); opt.step()

# 评估：取概率最大的类别
pred = logits.argmax(1)
acc = (pred == yb).float().mean()
""")]
story += IMG("classification_result.png", caption="图2：分类决策边界（玩具数据，3 类）")
story += [H2("过关标准")]
story += [P("能说清 CrossEntropyLoss 与 MSELoss 的区别、输出维度为何等于类别数、如何用 argmax + 准确率评估。")]

# 8
story += [H1("8. 阶段5b：加深对照（加层 / 参数量）")]
story += [P("把网络从 1 个隐藏层加到 3 个（2→64→32→16→3），对比基线。加深能拟合更复杂的边界。")]
story += [BULLETS([
    "加层规则：nn.Sequential 中每层 Linear(in,out) 的 in/out 维度必须接力；",
    "输出层不加 ReLU（要保留原始 logits 给 CrossEntropyLoss）；",
    "参数量手算：第一层 2*64+64=192，第二层 64*32+32=2080，第三层 32*16+16=528，第四层 16*3+3=51，合计 2851；",
    "不是越深越好：过深会过拟合、梯度消失，玩具数据上 100% 准确率两者都能到，但真实任务要权衡。",
])]
story += [H2("过关标准")]
story += [P("能动手加隐藏层、保证维度链 in/out 接力、手算参数量（2→64→32→16→3 = 2851）。")]

# 9
story += [H1("9. 阶段6：MNIST 手写数字分类")]
story += [P("终于上真实数据集：MNIST（6 万张 28×28 手写数字）。模型是一个 MLP。")]
story += [BULLETS([
    "图片是 28×28 = 784 个像素，必须先 Flatten 拉平成 784 维向量，才能进 Linear；",
    "Flatten 只是「 reshaping」，不改变数值，只是把空间结构拍平（这也是 MLP 的短板来源）；",
    "模型与数据必须在同一 device（mps / cpu）；",
    "预处理：ToTensor([0,1]) + Normalize(0.1307, 0.3081)；",
    "结果：测试集准确率约 97.35%（纯 MLP 的教科书区间 95%~98%）。",
])]
story += [CODE(
"""class MNISTNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()              # (B,1,28,28) -> (B,784)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),                 # 10 类
        )
    def forward(self, x):
        return self.net(self.flatten(x))
""")]
story += [NOTE("为什么不是 100%？因为 Flatten 丢掉了空间结构（不知道哪两个像素相邻），且对位置/形变不鲁棒。这正是下一阶段 CNN 要解决的——CNN 在 MNIST 上轻松到 99%+。")]
story += [H2("过关标准")]
story += [P("能说清图片为何要先 Flatten 才能进 Linear、模型与数据需同 device、测试集准确率约 95%~98%。")]

# 10
story += [H1("10. 应用：手写数字识别器")]
story += [P("把刚训好的模型接上一个鼠标画板（gradio Sketchpad），写个数字实时识别，右侧输出概率条形 + 0~9 概率明细表。")]
story += [BULLETS([
    "复用 MNISTNet(784→128→ReLU→10)；",
    "首次运行自动训练 5~10 epoch 并缓存 mnist_mlp.pth，之后秒开；",
    "预处理铁律：推理必须与训练完全一致（灰度 → 取笔画 → 缩放居中 28×28 → Normalize(0.1307,0.3081)），否则识别失真；",
    "取笔画：把所有图层的「墨迹」（与各自背景差异大的像素）取并集，再渲染成黑底白字——这样白底黑笔 / 黑底白笔都能正确识别。",
])]
story += [CODE(
"""def preprocess(img):
    # gradio Sketchpad 返回 dict: {"image": 背景, "mask": 笔画}
    layers = [img[k] for k in ("mask","image") if k in img] or list(img.values())
    grays = [np.array(l, float).mean(2) if np.array(l).ndim==3 else np.array(l,float)
             for l in layers]
    ink = np.zeros_like(grays[0], bool)
    for g in grays:
        bg = np.median(g)                       # 背景估计
        ink |= np.abs(g - bg) > 25             # 墨迹 = 与背景差异大的像素
    digit = (ink * 255).astype("uint8")         # 黑底白字
    # 裁剪到笔画框 -> 缩放居中到 28x28 -> ToTensor -> Normalize
    ...
    return (t - 0.1307) / 0.3081
""")]
story += [P("运行：uv add gradio && uv run handwriting_app.py，打开终端打印的 http://127.0.0.1:7860 。")]
story += [NOTE("两个曾踩的坑：① Sketchpad 回调传的是 dict 不是 PIL，直接 .convert() 会报错；② 若按「白像素最多的图层」选笔画，会选中纯白背景，导致恒输出同一个数字（如恒为 5）。已用「墨迹并集」修复。")]

# 11
story += [H1("11. 踩坑合集（环境 / 数据 / 代码）")]
story += [H2("环境")]
story += [BULLETS([
    "uv add 装大依赖超时 → 配 ~/.config/uv/uv.toml 顶层 index-url 清华镜像（不要写 [install] 表，也不要只设环境变量）；",
    "Mac M4 走 MPS（自带支持）；AMD 显卡 ROCm 坑多，学习阶段用 CPU 即可。",
])]
story += [H2("数据")]
story += [BULLETS([
    "标签要用原始特征生成、归一化只作用于输入 X；",
    "输入 X 和标签 y 都要标准化，否则 Adam 步长太小不收敛；",
    "梯度会累加，每个 batch 前必须 zero_grad()；",
    "MNIST 下载源不稳定：torchvision 对 .gz 做 MD5 校验，不同镜像重新压缩后 MD5 会变 → 死循环。正确做法是用 download_mnist.py 拉官方 .gz → 校验 → 解压出 .ubyte → 校验内容 → 落盘到 data_mnist/MNIST/raw/。",
])]
story += [H2("代码")]
story += [BULLETS([
    "CrossEntropyLoss 内部含 softmax，模型输出别再 softmax；",
    "加层时维度链 in/out 必须接力，输出层不加 ReLU；",
    "模型和数据必须同一 device；",
    "应用：Sketchpad 返回 dict、且不能按白像素选笔画层（见第 10 章）。",
])]

# 12
story += [H1("12. 下一步：CNN 路线图")]
story += [P("MLP 把图片拍扁、丢了空间结构。卷积网络（CNN）用一个小窗口在图上滑动提取局部特征（权值共享），天生对位置/形变鲁棒，MNIST 上能到 99%+。")]
story += [CODE(
"""class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32*7*7, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)
""")]
story += [BULLETS([
    "Conv2d(1→16, kernel=3)：用 3×3 窗口提取 16 种局部特征（边缘/角/笔画）；",
    "MaxPool2d(2)：把图缩小一半，降维且让特征对位置不敏感；",
    "两次卷积+池化后特征图变为 32×7×7，展平接入全连接；",
    "核心一句话：卷积=局部特征提取（权值共享）；池化=缩小+抗平移。",
])]
story += [P("学完 CNN，就可以把应用 handwriting_app.py 的底层换成 CNN，识别率从 97% 冲到 99%+。")]

# 附录
story += [H1("附录：命令速查表")]
story += [TABLE([
    ["目的", "命令"],
    ["复原环境", "uv sync"],
    ["逐课运行", "uv run lessons/01_tensor.py"],
    ["房价预测", "uv run main.py"],
    ["工业模板", "uv run lessons/04_template.py"],
    ["分类/加深", "uv run lessons/05_classification.py / 05b_deeper.py"],
    ["MNIST", "uv run lessons/06_mnist.py（源失效先跑 download_mnist.py）"],
    ["手写识别应用", "uv add gradio && uv run handwriting_app.py"],
    ["生成本手册", "uv run --extra \"\" python build_learning_pdf.py"],
], [55 * mm, 115 * mm])]
story += [Spacer(1, 6 * mm),
          P("—— 手册完。回到代码里多跑几遍，比反复读更重要。")]


# ---------- 页脚页码 ----------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(CJK, 8)
    canvas.setFillColor(colors.HexColor("#999999"))
    canvas.drawCentredString(A4[0] / 2.0, 12 * mm,
                             "PyTorch 从零实战学习手册  ·  第 %d 页" % doc.page)
    canvas.restoreState()


frame = Frame(20 * mm, 18 * mm, A4[0] - 40 * mm, A4[1] - 36 * mm, id="main")
doc = BaseDocTemplate(OUT, pagesize=A4, title="PyTorch 从零实战学习手册",
                      author="行者")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
doc.build(story)
print("✓ 已生成:", OUT, "(%d 字节)" % os.path.getsize(OUT))
