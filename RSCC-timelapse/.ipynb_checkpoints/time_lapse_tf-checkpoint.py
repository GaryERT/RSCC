import json
import random

# === 文件路径 ===
input_path = "./dataset/change_span_finetune_raw.json"
output_path = "./dataset/RSCC-timelapse.json"

# === 加载原始数据 ===
with open(input_path, "r", encoding="utf-8") as f:
    original_data = json.load(f)

# === 标签对应自然语言 ===
label_to_text = {
    "A": "days",
    "B": "weeks",
    "C": "months",
    "D": "years",
    "E": "no change"
}

# === 构造增强数据（包含选择题和是非题） ===
augmented_data = []

for entry in original_data:
    image_paths = entry["images"]
    true_label = entry["messages"][1]["content"].strip().upper()
    true_text = label_to_text[true_label]

    # === 添加原始选择题 ===
    augmented_data.append(entry)

    # === 构造是非题 ===
    if random.random() < 0.5:
        # 正确说法
        if true_label == "E":
            hypothesis = "There is no change between the two pictures."
        else:
            hypothesis = f"The change between the two pictures happened over {true_text}."
        answer = "True"
    else:
        # 错误说法：换一个标签
        wrong_labels = [k for k in label_to_text if k != true_label]
        wrong_label = random.choice(wrong_labels)
        wrong_text = label_to_text[wrong_label]

        if wrong_label == "E":
            hypothesis = "There is no change between the two pictures."
        else:
            hypothesis = f"The change between the two pictures happened over {wrong_text}."
        answer = "False"

    # 构造完整是非 entry
    yesno_entry = {
        "messages": [
            {
                "role": "user",
                "content": f"Picture 1: <image> Picture 2: <image>\n{hypothesis}\n\nDo you think this description is correct? Answer with 'True' or 'False'."
            },
            {
                "role": "assistant",
                "content": answer
            }
        ],
        "images": image_paths
    }

    augmented_data.append(yesno_entry)

# === 打乱所有样本顺序 ===
random.shuffle(augmented_data)

# === 保存输出 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(augmented_data, f, indent=2)

print(f"✅ 增强完成，处理 no change 情况，生成样本总数 {len(augmented_data)}，输出保存至：{output_path}")
