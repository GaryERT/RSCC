## timelapse
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
eval_json_path = "/root/autodl-tmp/data/RSCC-timelapse/dataset/change_span_eval_400.json"
tsv_output_dir = "/root/autodl-tmp/data/RSCC-timelapse/eval_result"
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
model.eval()

processor = AutoProcessor.from_pretrained(model_base_path)

# === 加载评估数据 ===
with open(eval_json_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

results = []
timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
correct_count = 0

# === 评估循环 ===
for entry in tqdm(eval_data, desc="Evaluating Multiple Choice QA"):
    image_paths = entry["images"]
    gt_answer = entry["messages"][1]["content"].strip().upper()

    # 构建输入 prompt
    user_msg = """You are given two pictures taken at different times. Please select the most appropriate option describing the time span of visible changes between them. A. days B. weeks C. months D. years E. no change Only answer with the letter of the correct option (e.g., "A", "B", etc.)."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": p} for p in image_paths] +
                       [{"type": "text", "text": user_msg}]
        }
    ]
    # print(messages)

    # === 构造输入 ===
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to("cuda")

    # === 推理 ===
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16)
    output = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    prediction = output[0].upper() if output and output[0].upper() in ["A", "B", "C", "D", "E"] else "?"

    # 判断正确性
    is_correct = (prediction == gt_answer)
    if is_correct:
        correct_count += 1

    results.append({
        "ground_truth": gt_answer,
        "prediction": prediction,
        "raw_output": output,
        "correct": is_correct,
        "images": "|".join(image_paths),
        "timestamp": timestamp_now
    })

# === 保存结果 TSV ===
df = pd.DataFrame(results)
os.makedirs(tsv_output_dir, exist_ok=True)
tsv_path = os.path.join(tsv_output_dir, f"change_eval_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
df.to_csv(tsv_path, sep="\t", index=False)

# === 打印准确率 ===
total = len(results)
accuracy = correct_count / total * 100
print(f"\n✅ 推理完成，保存至：{tsv_path}")
print(f"📊 多选题准确率：{accuracy:.2f}% ({correct_count}/{total})")
