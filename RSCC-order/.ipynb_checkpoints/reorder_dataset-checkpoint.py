import os
import json
import random

# 原始数据路径
dataset_path = "/root/autodl-tmp/data/remote_sensing_data/all_1k_dataset"
# 输出文件路径
output_json_path = "/root/autodl-tmp/data/RSCC-order/remote_sensing_reorder_dataset.json"

# 提示模板
prompt = """
You are given 5 pictures in random order: Picture 1, Picture 2, Picture 3, Picture 4, and Picture 5.

Your task is to determine their correct chronological order based on visual content.

Please output five integers representing the correct temporal order of the pictures.

Each integer should be between 1 and 5, indicating the index of the picture in the original input list (1 = Picture 1, 2 = Picture 2, ..., 5 = Picture 5).

The output should be a permutation of 1 to 5, indicating the order in which the input pictures occurred in time.

For example, if the correct chronological order is: Picture 4, Picture 2, Picture 3, Picture 1, Picture 5, you should output: 4 2 3 1 5
"""

data_list = []

# 遍历子文件夹
folders = sorted(
    [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))],
    key=lambda x: int(x) if x.isdigit() else float('inf')
)

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    # 获取 PNG 图像
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    png_files.sort()

    if len(png_files) < 5:
        continue  # 跳过图像不足的样本

    # 采样 5 张图像（这些图是时间有序的）
    sampled_files = random.sample(png_files, 5)

    # 打乱顺序作为模型输入
    shuffled_files = random.sample(sampled_files, len(sampled_files))

    # 构造时间顺序（按文件名排序）
    time_sorted = sorted(sampled_files)

    # 得到正确顺序（时间最早 → 最晚）在输入中的索引（1-based）
    correct_order = [str(shuffled_files.index(f) + 1) for f in time_sorted]

    # 构造带 Picture 标签的 <image> prompt
    image_prompt = " ".join([f"Picture {i+1}: <image>" for i in range(5)])

    # 构造最终对话内容
    user_content = image_prompt + "\n" + prompt
    assistant_content = " ".join(correct_order)

    # 构造图像路径
    image_paths = [os.path.join(dataset_path, folder, img) for img in shuffled_files]

    # 构造样本结构
    data_entry = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": image_paths
    }

    data_list.append(data_entry)

# 打乱最终数据顺序
random.shuffle(data_list)

# 保存为 JSON 文件
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=4, ensure_ascii=False)

print(f"✅ 生成完成，数据保存于：{output_json_path}，共计样本数：{len(data_list)}")
