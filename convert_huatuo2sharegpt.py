import json
import os
from datasets import load_dataset
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 保存路径 (会自动创建)
OUTPUT_DIR = "./dataset/SFT_huotuo"
TRAIN_FILE_NAME = "train/huatuo_train.jsonl"
VAL_FILE_NAME = "val/huatuo_val.jsonl"

# 2. 验证集比例
# 设置为 0.01 表示切分 1% 做验证集，设置为整数(如 5000)表示固定切出 5000 条
VAL_SIZE = 0.01

# 3. 随机种子 (保证每次切分的数据是一样的)
SEED = 42
# ===============================================

def ensure_dir(directory):
    """如果目录不存在，则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📂 已创建目录: {directory}")
    else:
        print(f"📂 目录已存在: {directory}")

def convert_to_sharegpt(hf_dataset, output_path):
    """将 HuggingFace 数据集转换为 ShareGPT 格式并保存"""
    print(f"🔄 正在处理并写入: {output_path} ...")
    
    count = 0
    skipped = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 tqdm 显示进度
        for item in tqdm(hf_dataset, desc="转换进度", unit="条"):
            
            # 1. 提取字段 (兼容不同命名习惯)
            q = item.get('question', item.get('input', '')).strip()
            a = item.get('answer', item.get('output', '')).strip()
            
            # 2. 简单清洗
            if not q or not a:
                skipped += 1
                continue
                
            # 3. 构建 ShareGPT 格式
            sharegpt_entry = {
                "conversations": [
                    {
                        "from": "human",
                        "value": q
                    },
                    {
                        "from": "gpt",
                        "value": a
                    }
                ]
            }
            
            # 4. 写入
            f.write(json.dumps(sharegpt_entry, ensure_ascii=False) + '\n')
            count += 1
            
    print(f"   ✅ 完成！有效数据: {count}, 跳过空数据: {skipped}")
    return count

def main():
    # 1. 准备目录
    ensure_dir(OUTPUT_DIR)
    
    # 2. 加载数据集
    print("🚀 正在加载 FreedomIntelligence/Huatuo26M-Lite 数据集...")
    try:
        # 加载全量数据
        ds = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split="train")
        print(f"📦 原始数据总量: {len(ds)} 条")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # 3. 切分训练集和验证集 (Train/Test Split)
    print(f"✂️ 正在切分数据集 (验证集大小: {VAL_SIZE}, 随机种子: {SEED})...")
    
    # 使用 HuggingFace 自带的切分功能，非常高效
    split_ds = ds.train_test_split(test_size=VAL_SIZE, seed=SEED)
    
    train_ds = split_ds['train']
    val_ds = split_ds['test']
    
    print(f"   📊 训练集大小: {len(train_ds)}")
    print(f"   📊 验证集大小: {len(val_ds)}")

    # 4. 分别转换并保存
    train_output_path = os.path.join(OUTPUT_DIR, TRAIN_FILE_NAME)
    val_output_path = os.path.join(OUTPUT_DIR, VAL_FILE_NAME)
    
    convert_to_sharegpt(train_ds, train_output_path)
    convert_to_sharegpt(val_ds, val_output_path)

    print("\n" + "="*40)
    print("🎉 所有任务已完成！")
    print(f"1️⃣ 训练集: {train_output_path}")
    print(f"2️⃣ 验证集: {val_output_path}")
    print("="*40)

if __name__ == "__main__":
    main()