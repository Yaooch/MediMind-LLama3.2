import json
import os
import torch
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 模型与数据
MODEL_NAME = 'BAAI/bge-base-zh-v1.5'
TARGET_SIZE = 20000  # 筛选出的总数据量
TOP_K_AVG = 3       # Top-3 均值策略

# 2. 输出目录配置
BASE_DIR = "./dataset/SFT_huatuo_filter_test_q"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# 3. 切分比例 (例如 90% 训练, 10% 验证)
SPLIT_RATIO = 0.99 
# ===========================================

def save_json(data, folder, filename="data.json"):
    """辅助函数：创建目录并保存JSON"""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"   已保存 {len(data)} 条数据到 -> {path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 设备: {device} | 目标: {TARGET_SIZE}条 | 策略: Top-{TOP_K_AVG} Avg")

    # 1. 加载 C-Eval (Anchors)
    print("📂 1. 加载 C-Eval 验证集...")
    subsets = ["basic_medicine", "clinical_medicine", "physician"]
    ceval_queries = []
    for sub in subsets:
        try:
            # ds = load_dataset("ceval/ceval-exam", sub, split="val", trust_remote_code=True)
            ds = load_dataset("ceval/ceval-exam", sub, split="test", trust_remote_code=True)
            for item in ds:
                text = f"{item['question']} A.{item['A']} B.{item['B']} C.{item['C']} D.{item['D']}"
                ceval_queries.append(text)
        except Exception as e:
            print(f"⚠️ 跳过 {sub}: {e}")
    print(f"✅ C-Eval 锚点: {len(ceval_queries)} 条")

    # 2. 加载 Huatuo (Candidates)
    print("📂 2. 加载 Huatuo 数据集...")
    try:
        ds_huatuo = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split="train", trust_remote_code=True)
        huatuo_list = ds_huatuo
        # huatuo_list = ds_huatuo.select(range(50000)) # 调试用
    except Exception as e:
        print(f"❌ 失败: {e}")
        return

    # 3. 编码
    print("🧠 3. 向量编码中...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    query_embeddings = model.encode(ceval_queries, convert_to_tensor=True, normalize_embeddings=True)
    
    # corpus_sentences = [f"{item['question']} {item['answer']}" for item in tqdm(huatuo_list, desc="准备文本")]
    corpus_sentences = [f"{item['question']}" for item in tqdm(huatuo_list, desc="准备文本")]

    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)

    # 4. 计算 Top-3 均值并排序
    print("🔍 4. 计算相似度并筛选...")
    cos_scores = util.cos_sim(corpus_embeddings, query_embeddings)
    top_k_vals, _ = torch.topk(cos_scores, k=TOP_K_AVG, dim=1)
    final_scores = torch.mean(top_k_vals, dim=1)
    
    # 选取 Top-N 索引
    _, top_indices = torch.topk(final_scores, k=min(TARGET_SIZE, len(huatuo_list)))
    selected_indices = top_indices.cpu().numpy().tolist()

    # 5. 格式转换 (ShareGPT)
    print("🔄 5. 格式转换 (Huatuo -> ShareGPT)...")
    filtered_data = []
    for idx in selected_indices:
        item = huatuo_list[idx]
        filtered_data.append({
            "conversations": [
                {"from": "human", "value": item['question']},
                {"from": "gpt", "value": item['answer']}
            ]
        })

    # 6. 打乱并切分 (Train/Val Split)
    print(f"✂️  6. 切分数据集 (比例 {SPLIT_RATIO})...")
    random.seed(42) # 固定种子，保证复现
    random.shuffle(filtered_data) # 【关键】打乱顺序，防止验证集偏差

    split_point = int(len(filtered_data) * SPLIT_RATIO)
    train_data = filtered_data[:split_point]
    val_data = filtered_data[split_point:]

    # 7. 保存到指定目录
    print("💾 7. 保存文件...")
    # 保存训练集
    save_json(train_data, TRAIN_DIR, "train.json")
    # 保存验证集
    save_json(val_data, VAL_DIR, "val.json")

    print("\n✅ 全部完成！")
    print(f"   训练集路径: {os.path.abspath(TRAIN_DIR)}/train.json")
    print(f"   验证集路径: {os.path.abspath(VAL_DIR)}/val.json")

if __name__ == "__main__":
    main()