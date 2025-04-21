## order
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# === 路径配置 ===
test_json_path = "/root/autodl-tmp/data/RSCC-order/reorder_dataset_test_only.json"
tsv_output_dir = "/root/autodl-tmp/data/RSCC-order/eval_result"
lora_ckpt_path = "/root/autodl-tmp/finetuned_qwen_2_5_ckpt/RSCC-mix600_order_0_timelapse_600_caption/checkpoint-225"
model_base_path = "/root/autodl-tmp/Qwen/Qwen2_5-VL-7B-Instruct"

# === 加载模型 & 处理器 ===
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_base_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model = PeftModel.from_pretrained(model, lora_ckpt_path)
model = model.merge_and_unload()

processor = AutoProcessor.from_pretrained(model_base_path, min_pixels=256*28*28, max_pixels=1280*28*28)

# === 加载测试数据 ===
with open(test_json_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# === 准备结果列表 ===
results = []
timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# === 推理主循环 + tqdm ===
for entry in tqdm(test_data, desc="Evaluating test set"):
    folder = entry["images"][0].split("/")[-2]
    image_paths = entry["images"]
    gt_order = entry["messages"][1]["content"].strip()

    prompt = (
        "You are given 5 pictures in random order: Picture 1, Picture 2, Picture 3, Picture 4, and Picture 5. "
        "Your task is to determine their correct chronological order based on visual content. "
        "Please output five integers representing the correct temporal order of the pictures. "
        "Each integer should be between 1 and 5, indicating the index of the picture in the original input list "
        "(1 = Picture 1, 2 = Picture 2, ..., 5 = Picture 5). "
        "The output should be a permutation of 1 to 5, indicating the order in which the input pictures occurred in time. "
        "For example, if the correct chronological order is: Picture 4, Picture 2, Picture 3, Picture 1, Picture 5, "
        "you should output: 4 2 3 1 5"
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": p} for p in image_paths] +
                       [{"type": "text", "text": prompt}]
        }
    ]

    # 处理输入
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to("cuda")

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # 记录结果
    results.append({
        "folder": folder,
        "ground_truth": gt_order,
        "prediction": generated_text.strip(),
        "timestamp": timestamp_now,
        "images": "|".join(image_paths)
    })

# === 保存 TSV ===
df = pd.DataFrame(results)
tsv_save_path = os.path.join(tsv_output_dir, f"reorder_eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
df.to_csv(tsv_save_path, sep="\t", index=False)

# === 全局 PNR 统计（与 ground_truth 对比，完全保留 1-based 编号）===
total_pos, total_neg = 0, 0

for gt_str, pred_str in zip(df["ground_truth"], df["prediction"]):
    try:
        gt_order = [int(x) for x in gt_str.strip().split()]     # 每个是图片编号（1-based）
        pred_order = [int(x) for x in pred_str.strip().split()] # 每个是图片编号（1-based）

        # 构建预测 rank：图编号 → 预测中排第几
        pred_rank = {img_id: rank for rank, img_id in enumerate(pred_order)}

        # 遍历 ground truth 顺序中的所有图对
        for i in range(len(gt_order)):
            for j in range(i + 1, len(gt_order)):
                a = gt_order[i]
                b = gt_order[j]
                if pred_rank[a] < pred_rank[b]:
                    total_pos += 1
                else:
                    total_neg += 1
    except Exception as e:
        continue


# === 输出全局 PNR ===
if total_neg == 0:
    global_pnr = float('inf') if total_pos > 0 else 0.0
else:
    global_pnr = total_pos / total_neg

print(f"✅ 推理完成，保存至：{tsv_save_path}")
print(f"📊 全局 PNR（正序对 / 逆序对）: {global_pnr:.4f}    正序对数: {total_pos}, 逆序对数: {total_neg}")

