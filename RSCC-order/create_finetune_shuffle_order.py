import json
import random

# 参数配置
input_path = "remote_sensing_1k_train_dataset.json"
output_path = "remote_sensing_1k_sequence_order_variable_len.json"
shuffle_prob = 0.5
min_images = 3
max_images = 7
random.seed(42)

# prompt 模板
prompt_prefix = (
    "You are given several pictures in a sequence: Picture 1, Picture 2, ..., Picture N.\n\n"
    "Please determine whether these pictures are presented in the correct temporal order.\n\n"
    "Answer with \"True\" if the sequence is in the correct chronological order, otherwise answer with \"False\"."
)

# 加载原始数据
with open(input_path, 'r') as f:
    data = json.load(f)

result = []

for sample in data:
    all_images = sample["images"]
    if len(all_images) < min_images:
        continue

    # 从样本中随机选取 3~7 张图像作为一个子序列
    k = random.randint(min_images, min(max_images, len(all_images)))
    selected = sorted(random.sample(all_images, k), key=lambda p: p.split("/")[-1])  # 正确顺序
    original = selected.copy()

    if random.random() < shuffle_prob:
        # 打乱顺序直到和原始不同
        while True:
            random.shuffle(selected)
            if selected != original:
                break
        label = False
    else:
        label = True

    # 构建带编号的 <image> prompt
    image_prompt = ""
    for i in range(len(selected)):
        image_prompt += f"Picture {i+1}: <image> "

    result.append({
        "messages": [
            {
                "role": "user",
                "content": image_prompt.strip() + "\n" + prompt_prefix
            },
            {
                "role": "assistant",
                "content": str(label)
            }
        ],
        "images": selected
    })

# 保存文件
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"✅ Saved {len(result)} samples to {output_path}")
