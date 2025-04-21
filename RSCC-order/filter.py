import os
import json

# 首先读取测试集数据路径
test_json_path = "/root/remote_sensing_1k_test_dataset.json"
reorder_json_path = "/root/autodl-tmp/data/RSCC-order/remote_sensing_reorder_dataset.json"
filtered_output_path = "/root/autodl-tmp/data/RSCC-order/finetune_dataset/reorder_dataset_test_only.json"

# 读取测试集 JSON，提取所有子文件夹编号（从图片路径中解析）
with open(test_json_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# 提取文件夹编号
test_folder_ids = set()
for item in test_data:
    if "images" in item:
        for path in item["images"]:
            parts = path.split("/")
            if "all_1k_dataset" in parts:
                idx = parts.index("all_1k_dataset")
                if idx + 1 < len(parts):
                    test_folder_ids.add(parts[idx + 1])

# 读取排序数据集
with open(reorder_json_path, "r", encoding="utf-8") as f:
    reorder_data = json.load(f)

# 筛选只保留在测试集中的样本
filtered_data = []
for entry in reorder_data:
    if "images" in entry and entry["images"]:
        parts = entry["images"][0].split("/")
        if "all_1k_dataset" in parts:
            idx = parts.index("all_1k_dataset")
            if idx + 1 < len(parts) and parts[idx + 1] in test_folder_ids:
                filtered_data.append(entry)

# 保存结果
with open(filtered_output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)
