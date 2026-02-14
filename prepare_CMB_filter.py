import json
import torch
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 模型与参数
# 建议使用 large 模型效果更好，如果显存不够改回 base
MODEL_NAME = 'BAAI/bge-base-zh-v1.5' 
TARGET_SIZE = 20000  # 筛选保留 2w 条
TOP_K_AVG = 3       # 评分策略：取最相似的 3 个分数求平均

# 2. 数据集 URL
CMB_TRAIN_URL = "https://hf-mirror.com/datasets/FreedomIntelligence/CMB/resolve/main/CMB-Exam/CMB-train/CMB-train-merge.json"
CMB_VAL_URL = "https://hf-mirror.com/datasets/FreedomIntelligence/CMB/resolve/main/CMB-Exam/CMB-val/CMB-val-merge.json"

# 3. 输出路径 (会自动创建子目录)
OUTPUT_DIR = "./dataset/SFT_CMB_filter"
OUTPUT_TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train', "cmb_train_20k.json")
OUTPUT_VAL_FILE = os.path.join(OUTPUT_DIR, 'val', "cmb_val.json")
# ===========================================

def format_option_text(question, options_dict):
    """
    将题目和选项拼成 Human 输入文本
    """
    text = f"{question}____\n"
    valid_keys = ['A', 'B', 'C', 'D', 'E', 'F']
    for key in valid_keys:
        val = options_dict.get(key)
        if val:
            text += f"{key}. {val}\n"
    text += "\n答案："
    return text.strip()

def convert_to_sharegpt(item, include_explanation=False):
    """
    转换为 ShareGPT 格式
    参数 include_explanation: 
       - False: 仅输出 "答案：A" (用于训练集，强调格式约束)
       - True: 输出 "答案：A\n\n解析：..." (用于验证集，提供推理逻辑)
    """
    # 1. 构造输入
    human_input = format_option_text(item['question'], item['option'])
    
    # 2. 构造输出
    gpt_output = f"{item['answer']}"
    
    # 3. 根据策略追加解析
    if include_explanation:
        expl = item.get('explanation')
        # 过滤掉 None 或过短的无效解析
        if expl and len(str(expl).strip()) > 5:
            gpt_output += f"\n\n解析：{expl}"
    
    return {
        "conversations": [
            {"from": "human", "value": human_input},
            {"from": "gpt", "value": gpt_output}
        ]
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 自动创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_VAL_FILE), exist_ok=True)
    
    print(f"🚀 设备: {device} | 目标筛选: {TARGET_SIZE} 条 | 模型: {MODEL_NAME}")

    # ---------------------------------------------------------
    # 1. 加载 C-Eval (Anchors)
    # ---------------------------------------------------------
    print("📂 1. 加载 C-Eval 验证集 (Anchors)...")
    ceval_queries = []
    subsets = ["basic_medicine", "clinical_medicine", "physician"]
    for sub in subsets:
        try:
            ds = load_dataset("ceval/ceval-exam", sub, split="val", trust_remote_code=True)
            for item in ds:
                text = f"{item['question']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
                ceval_queries.append(text)
        except Exception as e:
            print(f"⚠️ 警告: 加载 C-Eval {sub} 失败")
    
    print(f"✅ C-Eval Anchor 数量: {len(ceval_queries)}")

    # ---------------------------------------------------------
    # 2. 加载 CMB 数据
    # ---------------------------------------------------------
    print("📂 2. 加载 CMB 数据集...")
    ds_cmb_train = load_dataset("json", data_files=CMB_TRAIN_URL, split="train")
    ds_cmb_val = load_dataset("json", data_files=CMB_VAL_URL, split="train")
    
    print(f"   CMB Train 原始: {len(ds_cmb_train)}")
    print(f"   CMB Val 原始: {len(ds_cmb_val)}")

    # ---------------------------------------------------------
    # 3. 过滤与编码 (核心修改)
    # ---------------------------------------------------------
    print("🧠 3. 加载 Embedding 模型并准备数据...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 3.1 编码 C-Eval
    query_embeddings = model.encode(ceval_queries, convert_to_tensor=True, normalize_embeddings=True)

    # 3.2 准备 CMB Train (只保留单项选择题)
    print("   正在过滤CMB的单项选择题并准备 CMB Train 文本...")
    train_sentences = []
    valid_indices = [] # 【关键】用于记录保留下来的数据在原始数据集中的下标

    for idx, item in enumerate(tqdm(ds_cmb_train, desc="过滤单选题")):
        # 1. 必须是单选题
        if item.get('question_type') != '单项选择题':
            continue
        # 2. 答案必须是单个字母 (排除答案是 'ABCD' 或空的脏数据)
        ans = str(item.get('answer', '')).strip()
        if len(ans) != 1 or ans not in ['A', 'B', 'C', 'D', 'E', 'F']:
            continue
            
        text = format_option_text(item['question'], item['option'])
        train_sentences.append(text)
        valid_indices.append(idx) # 记录原始 ID

    print(f"   过滤后剩余: {len(train_sentences)} 条 (原始 {len(ds_cmb_train)} 条)")
    print(f"   正在编码 CMB Train...")
    
    # 批量编码
    corpus_embeddings = model.encode(train_sentences, batch_size=64, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)

    # ---------------------------------------------------------
    # 4. 相似度计算与筛选
    # ---------------------------------------------------------
    print(f"🔍 4. 计算相似度并筛选 Top-{TARGET_SIZE}...")
    
    cos_scores = util.cos_sim(corpus_embeddings, query_embeddings)
    top_k_vals, _ = torch.topk(cos_scores, k=TOP_K_AVG, dim=1)
    final_scores = torch.mean(top_k_vals, dim=1)
    
    # 选取 Top-N (这里的 indices 是 train_sentences 列表的下标，不是原始数据集的)
    _, top_indices_local = torch.topk(final_scores, k=min(TARGET_SIZE, len(train_sentences)))
    selected_indices_local = top_indices_local.cpu().numpy().tolist()

    # ---------------------------------------------------------
    # 6. 转换格式并保存
    # ---------------------------------------------------------
    print("💾 6. 转换格式并保存...")

    # 6.1 处理筛选后的 Train (无解析)
    filtered_train_data = []
    for local_idx in selected_indices_local:
        original_idx = valid_indices[local_idx] # 【关键】通过映射找回原始数据
        item = ds_cmb_train[original_idx]
        
        # 训练集：不带解析
        filtered_train_data.append(convert_to_sharegpt(item, include_explanation=False))
    
    with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_train_data, f, indent=4, ensure_ascii=False)
    print(f"✅ 训练集已保存: {OUTPUT_TRAIN_FILE} (共 {len(filtered_train_data)} 条)")

    # 6.2 处理全部 Val (有解析，不筛选)
    # 验证集通常不做严格过滤，保留多样性，或者也可以加上单选题过滤，这里默认保留全部
    val_data = []
    for item in ds_cmb_val:
        # 验证集：带解析
        val_data.append(convert_to_sharegpt(item, include_explanation=True))
        
    with open(OUTPUT_VAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)
    print(f"✅ 验证集已保存: {OUTPUT_VAL_FILE} (共 {len(val_data)} 条)")

if __name__ == "__main__":
    main()