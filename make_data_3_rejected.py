import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
# 1. 你的 SFT 模型路径 (必须是合并后的完整模型)
MODEL_PATH = "./merged-llama3-3b-dpo-v2" 

# 2. 文件路径
# INPUT_FILE = "dpo_data_with_chosen.jsonl"  # 上一步生成的包含 prompt 和 chosen 的文件
# OUTPUT_FILE = "./dataset/DPO/dpo_train_data_final.jsonl" # 最终用于 DPO 训练的文件
INPUT_FILE = "dpo_test_answers.jsonl"  # 上一步生成的包含 prompt 和 chosen 的文件
OUTPUT_FILE = "dpo_test_answers_2.jsonl" # 最终用于 DPO 训练的文件


# 3. 生成参数
BATCH_SIZE = 16           # 显存够大(24G)可以开到 8 或 16，Mac 或 显存小就 4
MAX_NEW_TOKENS = 512     # 让它多说点，越长越容易暴露错误
TEMPERATURE = 0.9        # 稍微高点，诱导它产生幻觉 (Temperature越高，越容易胡说八道)

# ================= 代码逻辑 =================

def load_model():
    print(f"🚀 正在加载模型: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 自动处理 padding token (Llama3 需要特别注意)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # 生成任务必须 left padding

    # 加载模型
    # 如果是 Mac，torch_dtype=torch.float16, device_map="mps" (或者 auto)
    # 如果是 Nvidia 显卡，torch_dtype=torch.bfloat16, device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model.eval() # 开启评估模式
    print("✅ 模型加载完成！")
    return model, tokenizer

def format_prompt_llama3(prompt, tokenizer):
    """
    构造 Llama-3 的对话模板。
    必须要和 SFT 训练时的一模一样！
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    # apply_chat_template 会自动添加 <|begin_of_text|>, <|start_header_id|> 等
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    model, tokenizer = load_model()

    # 1. 读取数据
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f if line.strip()]

    print(f"📊 待处理数据: {len(data_list)} 条")

    # 2. 准备输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        pass # 清空文件

    # 3. 批量生成
    # 每次处理 BATCH_SIZE 条数据
    for i in tqdm(range(0, len(data_list), BATCH_SIZE), desc="生成 Rejected"):
        batch_data = data_list[i : i + BATCH_SIZE]
        
        # 3.1 提取 Prompt 并应用模板
        raw_prompts = [item["prompt"] for item in batch_data]
        formatted_prompts = [format_prompt_llama3(p, tokenizer) for p in raw_prompts]

        # 3.2 Tokenize
        inputs = tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(model.device)

        # 3.3 Generate (推理)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,        # 开启采样，让它产生多样性(容易出错)
                temperature=TEMPERATURE,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )

        # 3.4 Decode (解码)
        # 只保留新生成的 token (去掉输入的 prompt)
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 3.5 保存结果
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
            for j, item in enumerate(batch_data):
                # 组合最终的 DPO 数据项
                dpo_item = {
                    "prompt": item["prompt"],    # 问题
                    # "chosen": item["chosen"],    # 专家回答 (GPT-4)
                    "answer_sft": item["answer_sft"],    # 专家回答 (GPT-4)
                    "answer_dpo": decoded_responses[j].strip() # 你的模型生成的烂回答
                }
                f_out.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")

    print(f"\n🎉 恭喜！DPO 数据集构建完成：{OUTPUT_FILE}")
    print("现在你可以去跑 dpo_training.py 了！")

if __name__ == "__main__":
    main()