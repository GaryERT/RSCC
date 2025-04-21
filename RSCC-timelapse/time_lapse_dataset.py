import pandas as pd
import json
import random
from collections import defaultdict

# 读取 CSV 文件
csv_path = "/root/autodl-tmp/data/RSCC-timelapse/qwen_change_detection_results.csv"
df = pd.read_csv(csv_path)

# 分类函数（时间跨度标签）
def classify_label(row):
    if not row["change_detected"] or str(row["change_detected"]).lower() == "false":
        return "E"
    days = int(str(row["day_diff"]).split()[0])
    if days < 14:
        return "A"
    elif days < 60:
        return "B"
    elif days < 365:
        return "C"
    else:
        return "D"

# 分桶
buckets = defaultdict(list)
for _, row in df.iterrows():
    label = classify_label(row)
    buckets[label].append(row)

# 平衡采样数量
min_count = min(len(bucket) for bucket in buckets.values())
print("每类采样数量为：", min_count)

# 固定选项文本
option_text = "A. days\nB. weeks\nC. months\nD. years\nE. no change"

# 构造样本
samples = []
for label, rows in buckets.items():
    selected = random.sample(rows, min_count)
    for row in selected:
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Picture 1: <image> Picture 2: <image>\nYou are given two pictures taken at different times. Please select the most appropriate option describing the time span of visible changes between them.\n\n{option_text}\n\nOnly answer with the letter of the correct option (e.g., \"A\", \"B\", etc.)."
                },
                {
                    "role": "assistant",
                    "content": label
                }
            ],
            "images": [row["image_1"], row["image_2"]]
        }
        samples.append(item)

# 🔀 打乱样本
random.shuffle(samples)

# 保存为 JSON
output_path = "change_span_balanced_shuffled.json"
with open(output_path, "w") as f:
    json.dump(samples, f, indent=2)

print(f"✅ 已完成：共生成 {len(samples)} 条数据，结果已打乱，保存至 {output_path}")
