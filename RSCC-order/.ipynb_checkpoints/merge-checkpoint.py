import json
import random

# === 输入文件路径 ===
input_paths = [
    "./finetune_dataset/remote_sensing_1k_pairwise_order_dataset.json",
    "./finetune_dataset/remote_sensing_1k_sequence_order_variable_len.json",
    "./finetune_dataset/reorder_dataset_train_only.json"
]

# === 输出文件路径 ===
output_path = "./finetune_dataset/RSCC-order.json"

# === 合并数据 ===
merged_data = []

for path in input_paths:
    with open(path, 'r') as f:
        part = json.load(f)
        merged_data.extend(part)

# === 打乱样本顺序 ===
random.seed(42)  # 可复现性（可选）
random.shuffle(merged_data)

# === 保存合并后的文件 ===
with open(output_path, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f"✅ Merged {len(merged_data)} samples into {output_path}")
