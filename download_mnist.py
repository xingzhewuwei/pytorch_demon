"""
下载并解压 MNIST 到 data_mnist/MNIST/raw/，绕过 torchvision 失效的默认下载源。

要点（踩坑总结）：
- torchvision 的 MNIST 默认下载源（yann.lecun / 部分 oss 源）会中途断流或 404，
  导致 `uv run lessons/06_mnist.py` 卡在进度后报
  RuntimeError: ... File not found or corrupted
- torchvision 实际**只认解压后的 .ubyte 文件**（train-images-idx3-ubyte 等），
  只要这些文件存在，它就跳过下载、也跳过 .gz 的 MD5 校验。
- 所以本脚本：从可用镜像拉官方 .gz → 校验 gzip 完整性 → 解压出 .ubyte →
  校验内容（magic 字节 + 维度 + 尺寸）后落盘。不依赖 .gz 的 MD5
  （不同镜像重新压缩后 MD5 会变，但解压内容一致）。

运行：uv run download_mnist.py
之后：uv run lessons/06_mnist.py  （检测到本地 .ubyte，直接训练）
"""

import gzip
import os
import struct
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_mnist", "MNIST", "raw")
os.makedirs(RAW, exist_ok=True)

# (gz 名, 解压后的 ubyte 名, 类型, 样本数)
FILES = [
    ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte", "image", 60000),
    ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte", "label", 60000),
    ("t10k-images-idx3-ubyte.gz",  "t10k-images-idx3-ubyte",  "image", 10000),
    ("t10k-labels-idx1-ubyte.gz",  "t10k-labels-idx1-ubyte",  "label", 10000),
]

# 候选镜像（按顺序尝试）。{f} 会被文件名替换。
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/{f}",       # torchvision 官方兜底源
    "https://storage.googleapis.com/cvdf-datasets/mnist/{f}",  # CVDF 官方源
    "https://huggingface.co/datasets/ylecun/mnist/resolve/main/{f}",
]

TIMEOUT = 120


def verify_ubyte(path, kind, n):
    """校验解压后的 .ubyte 是否为官方原版（不看 .gz 的 MD5）。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return False
    if len(data) < 8:
        return False
    magic = struct.unpack(">I", data[:4])[0]
    if magic != (0x0803 if kind == "image" else 0x0801):
        return False
    ndim = data[3]
    dims = struct.unpack(">%dI" % ndim, data[4:4 + 4 * ndim])
    header = 4 + 4 * ndim
    data_size = n * (28 * 28 if kind == "image" else 1)
    return len(data) == header + data_size and dims[0] == n


def fetch(url, dest):
    """下载 url 到 dest。优先 urllib，失败则尝试系统 curl（更抗断流）。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dest, "wb") as w:
            w.write(r.read())
        return True
    except Exception as e:
        print(f"      urllib 失败: {e}")
    # 回退到 curl
    try:
        subprocess.run(
            ["curl", "-fsSL", "--retry", "5", "--retry-all-errors",
             "--retry-delay", "2", "--max-time", str(TIMEOUT), "-o", dest, url],
            check=True,
        )
        return True
    except Exception as e:
        print(f"      curl 失败: {e}")
    return False


def download_one(gz, out, kind, n):
    out_path = os.path.join(RAW, out)
    if verify_ubyte(out_path, kind, n):
        print(f"  ✓ {out} 已存在且校验通过，跳过")
        return True

    gz_path = os.path.join(RAW, gz)
    for base in MIRRORS:
        url = base.format(f=gz)
        for attempt in range(1, 4):
            try:
                print(f"  ↓ {url}" + (f" (重试 {attempt})" if attempt > 1 else ""))
                if not fetch(url, gz_path):
                    continue
                # 解压 + 内容校验
                with gzip.open(gz_path, "rb") as f:
                    blob = f.read()
                with open(out_path, "wb") as f:
                    f.write(blob)
                if verify_ubyte(out_path, kind, n):
                    print(f"  ✓ {out} 下载并校验完成 ({os.path.getsize(out_path):,} 字节)")
                    return True
                print("    ✗ 内容校验失败（疑似损坏），丢弃重试")
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                break
            except Exception as e:
                print(f"    ✗ 失败: {e}")
                if attempt < 3:
                    time.sleep(2)
    print(f"  ✗✗ {out} 所有镜像均失败")
    return False


if __name__ == "__main__":
    print(f"目标目录: {RAW}\n")
    ok = True
    for gz, out, kind, n in FILES:
        if not download_one(gz, out, kind, n):
            ok = False
    if ok:
        print("\n✅ 全部就绪（.ubyte 已校验）。现在运行:")
        print("   uv run lessons/06_mnist.py")
        print("（torchvision 检测到本地 .ubyte 文件，跳过下载直接训练）")
    else:
        print(f"\n❌ 部分文件失败，请检查网络或手动放置到: {RAW}")
        sys.exit(1)
