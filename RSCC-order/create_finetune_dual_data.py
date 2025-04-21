import json
import os
import random
from itertools import combinations

# === 配置 ===
input_path = "remote_sensing_1k_train_dataset.json"
output_path = "./finetune_dataset/remote_sensing_1k_pairwise_order_dataset.json"
pairs_per_folder = 3  # 每个文件夹抽取几个图像对
random.seed(42)

prompt_template = (
    "You are given two pictures: Picture 1 and Picture 2.\n\n"
    "Please determine whether their temporal order is correct.\n"
    "Picture 1 appears earlier than Picture 2 in time.\n"
    "Answer with 'True' if the order is correct, otherwise answer with 'False'."
)

# === 加载原始数据 ===
with open(input_path, 'r') as f:
    data = json.load(f)

new_dataset = []

for sample in data:
    image_paths = sample["images"]
    if len(image_paths) < 2:
        continue

    # 提取文件夹编号和图像字母
    folder_to_images = {}
    for img_path in image_paths:
        parts = img_path.split("/")
        folder = parts[-2]
        letter = os.path.splitext(parts[-1])[0]
        folder_to_images[letter] = img_path

    letters = sorted(folder_to_images.keys())
    if len(letters) < 2:
        continue

    # 所有两两组合
    all_pairs = list(combinations(letters, 2))
    sampled_pairs = random.sample(all_pairs, min(pairs_per_folder, len(all_pairs)))

    for l1, l2 in sampled_pairs:
        p1 = folder_to_images[l1]
        p2 = folder_to_images[l2]

        # 随机决定是否打乱
        if random.random() < 0.5:
            images = [p1, p2]
            label = True
        else:
            images = [p2, p1]
            label = False

        new_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Picture 1: <image>Picture 2: <image>\n{prompt_template}"
                },
                {
                    "role": "assistant",
                    "content": str(label)
                }
            ],
            "images": images
        }

        new_dataset.append(new_sample)

# === 保存 ===
with open(output_path, 'w') as f:
    json.dump(new_dataset, f, indent=2)

print(f"Saved {len(new_dataset)} samples to {output_path}")
