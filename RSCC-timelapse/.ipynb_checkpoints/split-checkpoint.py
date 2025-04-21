import json
import random

# 加载数据
input_path = "change_span_balanced_shuffled.json"
with open(input_path, "r") as f:
    data = json.load(f)

# 随机打乱
random.shuffle(data)

# 分割
eval_size = 400
eval_data = data[:eval_size]
finetune_data = data[eval_size:]

# 保存评估集
with open("change_span_eval_400.json", "w") as f:
    json.dump(eval_data, f, indent=2)

# 保存微调集
with open("change_span_finetune_rest.json", "w") as f:
    json.dump(finetune_data, f, indent=2)

print(f"✅ 已完成数据划分：评估集 400 条，微调集 {len(finetune_data)} 条。")
