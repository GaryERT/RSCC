import os
import json
import base64
from datetime import datetime
from itertools import combinations
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

write_lock = Lock()  # 用于写 CSV 的锁

def write_result_to_csv(result):
    df = pd.DataFrame([result])
    with write_lock:  # 保证写入过程线程安全
        if not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0:
            df.to_csv(output_csv, mode='a', header=True, index=False)
        else:
            df.to_csv(output_csv, mode='a', header=False, index=False)


# ========== 参数设置 ==========
# api_key = "sk-ZgDFRCmm1b5b2aEAe4E2T3BlbKFJDb13bdebCfa840aaAa75"  # <<< 修改为你自己的 API Key
api_key = "sk-1d0c4ad63cca426ea45c2b1027425673"
root_dir = "/root/autodl-tmp/data/RSCC-timelapse/TAMMs"
output_csv = "/root/autodl-tmp/data/RSCC-timelapse/qwen_change_detection_results_content.csv"
max_workers = 5  # 并发线程数
# ==============================

# client = OpenAI(api_key=api_key, base_url="https://www.aigptx.top/v1")
client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 任务函数（给线程池调用）
# 任务函数（给线程池调用）
def process_pair(img1, img2):
    try:
        date1 = datetime.strptime(os.path.basename(img1).split(".")[0], "%Y-%m-%d")
        date2 = datetime.strptime(os.path.basename(img2).split(".")[0], "%Y-%m-%d")
        day_diff = abs((date2 - date1).days)

        b64_img1 = encode_image(img1)
        b64_img2 = encode_image(img2)

        # Prompt 增强：要求返回是否变化 + 简短一句话描述（不超过25词）
        response = client.chat.completions.create(
            model="qwen-vl-plus-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are given two satellite images of the same location taken at different times.

Your task has two parts:

1. Determine whether there are any **real-world changes** in the physical scene between the two images.
2. If yes, describe the change in a single English sentence under **25 words**.

Focus only on **meaningful structural or land-use changes** such as buildings, roads, water bodies, or vegetation.

Ignore differences caused by:
- Seasonal effects (e.g., snow vs no snow)
- Lighting, shadows, clouds
- Color or brightness variations

Format your answer like this (without quotes):
"True: A new building was constructed in the center."
Or just:
"False"
"""
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img1}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img2}"}},
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=100
        )

        answer = response.choices[0].message.content.strip()
        is_changed = answer.startswith("True")
        change_desc = answer[5:].strip() if is_changed else ""

        return {
            "image_1": img1,
            "image_2": img2,
            "day_diff": f"{day_diff} days",
            "change_detected": is_changed,
            "change_description": change_desc
        }

    except Exception as e:
        print(f"\n❌ Error comparing {img1} and {img2}: {e}")
        return None


# ✅ Step 0: 读取已有结果，构建跳过对照表
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
    processed_pairs = set(
        tuple(sorted((row["image_1"], row["image_2"]))) for _, row in existing_df.iterrows()
    )
else:
    processed_pairs = set()

# ✅ Step 1: 构建所有未处理的任务
all_tasks = []
for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    if not os.path.isdir(category_path):
        continue

    for sequence in os.listdir(category_path):
        sequence_path = os.path.join(category_path, sequence)
        if not os.path.isdir(sequence_path):
            continue

        image_files = [f for f in os.listdir(sequence_path) if f.endswith(".jpg")]
        if len(image_files) < 2:
            continue

        image_paths = [
            os.path.join(sequence_path, f)
            for f in sorted(image_files, key=lambda name: datetime.strptime(name.split(".")[0], "%Y-%m-%d"))
        ]
        for pair in combinations(image_paths, 2):
            pair_sorted = tuple(sorted(pair))
            if pair_sorted not in processed_pairs:
                all_tasks.append(pair)

# ✅ Step 2: 实时写入版本（并发推理 + 即时存储）
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_pair, img1, img2) for img1, img2 in all_tasks]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing image pairs (realtime write)"):
        result = future.result()
        if result:
            write_result_to_csv(result)

# # ✅ Step 3: 保存追加结果
# if results:
#     new_df = pd.DataFrame(results)
#     if os.path.exists(output_csv):
#         new_df.to_csv(output_csv, mode='a', header=False, index=False)
#     else:
#         new_df.to_csv(output_csv, index=False)
#     print(f"\n📄 Appended {len(results)} new results to: {output_csv}")
# else:
#     print("\n✅ No new pairs to process. Everything is up to date.")
