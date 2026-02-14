import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# ================= 配置区域 =================
# 您的 SFT 模型路径
# MODEL_PATH = "./merged-llama3-3b-sft-CMB-v1"
MODEL_PATH = "./models/LLM-Research/Llama-3___2-3B-Instruct"

# C-Eval 的三个医学科目
TASKS = [
    "basic_medicine",    # 基础医学
    "clinical_medicine", # 临床医学
    "physician"          # 医师资格
]
# ===========================================

def load_model_and_tokenizer(model_path):
    print(f"🚀 Loading model from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16, # 3B模型推荐用BF16，如果报错显卡不支持则改为float16
        trust_remote_code=True
    )
    return model, tokenizer, device

def format_example(item):
    """
    将 C-Eval 的一行数据格式化为模型训练时的样子
    格式：题目 + 选项 + 答案
    """
    question = item['question']
    options = f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}"
    answer = item['answer']
    
    # 这里构造一个符合人类阅读逻辑的完整文本
    # 如果您的 SFT 训练数据有特定的 prompt 模板（比如 ChatML），这里最好保持一致
    # 这里使用通用的“完形填空”风格，这对 PPL 计算最公平
    text = f"题目：{question}\n{options}\n答案：{answer}"
    return text

def calculate_perplexity(model, tokenizer, device, dataset):
    model.eval()
    nlls = [] # 存储每个样本的 Negative Log Likelihood
    total_tokens = 0
    
    with torch.no_grad():
        for item in tqdm(dataset, desc="Calculating PPL"):
            # 1. 格式化文本
            text = format_example(item)
            
            # 2. 编码
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            target_ids = input_ids.clone()
            
            # 3. 计算 Loss
            # HuggingFace 的 loss 默认是 CrossEntropyLoss
            outputs = model(input_ids, labels=target_ids)
            
            # outputs.loss 是这个样本所有 token loss 的平均值
            # 我们需要还原成 sum，因为不同样本长度不同，不能直接对 loss 求平均
            neg_log_likelihood = outputs.loss * input_ids.shape[1]
            
            nlls.append(neg_log_likelihood)
            total_tokens += input_ids.shape[1]

    # 4. 计算整个数据集的 PPL
    # PPL = exp( Sum(Loss) / Total_Tokens )
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_tokens)
    
    return ppl.item()

def main():
    model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
    
    results = {}
    print("\nStarting PPL Evaluation on C-Eval Medicine subsets...")
    
    for task in TASKS:
        print(f"\n🧪 Evaluating task: {task}")
        try:
            # 加载验证集 (split='val')
            dataset = load_dataset("ceval/ceval-exam", task, split="val", trust_remote_code=True)
            
            ppl = calculate_perplexity(model, tokenizer, device, dataset)
            results[task] = ppl
            print(f"   -> {task} PPL: {ppl:.4f}")
            
        except Exception as e:
            print(f"   -> Error evaluating {task}: {e}")
            results[task] = "Error"

    print("\n" + "="*40)
    print("📊 Final Perplexity Results")
    print("="*40)
    avg_ppl = 0
    count = 0
    for task, score in results.items():
        print(f"{task:<25}: {score:.4f}")
        if isinstance(score, (int, float)):
            avg_ppl += score
            count += 1
    
    if count > 0:
        print("-" * 40)
        print(f"Average Medical PPL    : {avg_ppl/count:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()