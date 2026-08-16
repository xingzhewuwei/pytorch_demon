"""
生成「房价预测」学习数据集，并落盘到 data/ 目录。
- house_price.pt  ：PyTorch 原生格式（dict，含 X_raw / y_data / 特征名），供 lessons 直接加载
- house_price.csv ：人可读格式，方便在 Excel / 编辑器里检视

数据生成逻辑与 lessons/04_template.py 完全一致（固定 seed=42），保证可复现。
运行：python data/generate_house_price_data.py
"""

import os
import csv
import torch

# 固定随机种子，保证每次生成的数据一致
torch.manual_seed(42)

n = 100
x_area   = torch.rand(n) * 100 + 50          # 面积 50~150 平米
x_rooms  = torch.randint(1, 6, (n,)).float()  # 房间数 1~5
x_subway = torch.rand(n) * 5                 # 距地铁 0~5 公里
X_raw = torch.stack([x_area, x_rooms, x_subway], dim=1).float()  # (100, 3)

# 真实规律：y = 0.5*面积 + 2.0*房间数 - 1.5*地铁距离 + 10 + 噪声
y_data = (0.5 * X_raw[:, 0]
          + 2.0 * X_raw[:, 1]
          - 1.5 * X_raw[:, 2]
          + 10
          + torch.randn(n) * 2)

HERE = os.path.dirname(os.path.abspath(__file__))

# 1) 保存为 PyTorch 原生格式
pt_path = os.path.join(HERE, "house_price.pt")
torch.save({
    "X_raw": X_raw,
    "y_data": y_data,
    "feature_names": ["area", "rooms", "subway_distance"],
    "target": "price",
    "desc": "y = 0.5*area + 2.0*rooms - 1.5*subway_distance + 10 + N(0,2)",
}, pt_path)

# 2) 保存为 CSV，方便人检视
csv_path = os.path.join(HERE, "house_price.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["area", "rooms", "subway_distance", "price"])
    for i in range(n):
        writer.writerow([
            f"{X_raw[i, 0].item():.4f}",
            int(X_raw[i, 1].item()),
            f"{X_raw[i, 2].item():.4f}",
            f"{y_data[i].item():.4f}",
        ])

print(f"✅ 已生成 {n} 条样本")
print(f"   - {pt_path}")
print(f"   - {csv_path}")
