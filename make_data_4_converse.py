import json
import os
import random

# ================= 配置区域 =================
# 1. 输入文件 (你刚才生成的包含 prompt/chosen/rejected 的文件)
INPUT_FILE = "dpo_train_data_final.jsonl"

# 2. 输出目录 (对应你训练脚本里的 --train_file_dir 和 --validation_file_dir)
OUTPUT_DIR_TRAIN = "./dataset/DPO/train"
OUTPUT_DIR_VAL = "./dataset/DPO/val"

# 3. 验证集比例 (0.05 表示 5% 的数据做验证集)
VAL_RATIO = 0.05

# ================= 代码逻辑 =================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📂 创建目录: {directory}")

def main():
    print(f"🚀 开始转换数据格式...")
    
    # 1. 读取原始数据
    data_list = []
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print(f"📊 读取到原始数据: {len(data_list)} 条")

    # 2. 转换字段名 (Mapping)
    formatted_data = []
    for item in data_list:
        new_record = {
            "question": item.get("prompt", ""),
            "response_chosen": item.get("chosen", ""),
            "response_rejected": item.get("rejected", "")
        }
        # 简单过滤：确保数据不为空
        if new_record["question"] and new_record["response_chosen"] and new_record["response_rejected"]:
            formatted_data.append(new_record)

    print(f"✅ 格式转换完成，有效数据: {len(formatted_data)} 条")

    # 3. 打乱并划分训练/验证集
    random.shuffle(formatted_data)
    
    split_index = int(len(formatted_data) * (1 - VAL_RATIO))
    train_data = formatted_data[:split_index]
    val_data = formatted_data[split_index:]

    # 4. 确保输出目录存在
    ensure_dir(OUTPUT_DIR_TRAIN)
    ensure_dir(OUTPUT_DIR_VAL)

    # 5. 保存文件
    train_file = os.path.join(OUTPUT_DIR_TRAIN, "medical_reward_dpo.jsonl")
    val_file = os.path.join(OUTPUT_DIR_VAL, "medical_reward_dpo.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 大功告成！")
    print(f"   - 训练集 ({len(train_data)}条): {train_file}")
    print(f"   - 验证集 ({len(val_data)}条): {val_file}")
    print(f"\n💡 下一步：直接运行你的 dpo_training.py 即可！")

if __name__ == "__main__":
    main()
