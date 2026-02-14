import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# ================= 配置区域 =================
# 1. API 配置 (建议用 GPT-4 或 DeepSeek-V3 等强模型)
API_BASE_URL = ""
API_KEY = ""       # 你的 API Key
MODEL_NAME = "MiniMaxAI/MiniMax-M2.1"        

# 2. 文件路径
INPUT_FILE = "dpo_prompts_2k.jsonl"           # 上一步生成的 Prompt 文件
OUTPUT_FILE = "dpo_data_with_chosen.jsonl"    # 结果保存文件

# 🚀 并发设置 (关键参数)
MAX_WORKERS = 20  # 同时跑多少个请求 (建议 5-10，太多容易被封/限流)


SYSTEM_PROMPT = """
你是一位三甲医院的主治医师，正在进行在线问诊。你的回答需要专业、安全、有逻辑，同时保持自然流畅的对话风格。

# 核心要求
1. **去标签化**：不要使用【核心结论】、【详细解析】这类僵硬的标题。请通过自然的分段、粗体 (**Text**) 和列表来组织内容。
2. **黄金回复结构**：
   - **第一段**：直接给出明确的医学结论或建议（直击痛点）。
   - **中间部分**：解释原因、病理机制或纠正患者的认知误区（科普教育）。
   - **后续建议**：使用数字列表 (1. 2. 3.) 给出具体的护理、用药或就医指导（行动指南）。
   - **结尾**：必要的禁忌提醒或安抚（安全兜底）。

# 安全红线 (最高优先级)
- 如果用户涉及**急救错误**（如昏迷喂水）、**高危行为**（如自杀、配毒药）或**严重误区**，必须在**第一句话**用严肃的语气进行警告和纠正。
- 如果涉及**不存在的病名**（幻觉），请自然地指出：“医学上没有这个概念，您是否指……”

# 语气风格
- 专业但不高冷，严谨但有温度。
- 像一位耐心的老专家在面对面叮嘱患者。

# 示例 (仅供参考风格，不要照抄内容)
User: 感冒嗓子疼能吃头孢吗？
Assistant: **不建议直接吃头孢。** 头孢是抗生素，只对细菌有效，而绝大多数感冒和嗓子疼是由病毒引起的，吃头孢不仅无效，还可能导致耐药性。
建议您采取以下措施：
1. **多喝温水**：保持咽喉湿润。
2. **对症用药**：如果疼痛剧烈，可使用润喉片或布洛芬缓解。
3. **观察症状**：如果出现高烧不退或扁桃体化脓，请及时就医查血象。
特别提醒：**服用头孢期间严禁饮酒**。
"""

# ================= 代码逻辑 =================

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def process_single_item(line):
    """处理单条数据的函数"""
    try:
        record = json.loads(line)
        prompt = record.get("promp t", "")
        if not prompt:
            return None

        # 调用 API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            timeout=60 # 设置超时防止卡死
        )
        
        chosen_response = response.choices[0].message.content.strip()
        
        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "source": "synthetic_expert"
        }
    except Exception as e:
        # print(f"⚠️ Error: {e}") # 报错太多会刷屏，先注释掉
        return None

def main():
    # 1. 读取输入
    if not os.path.exists(INPUT_FILE):
        print("❌ 找不到输入文件")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_lines = f.readlines()

    # 2. 检查已处理 (断点续传)
    processed_prompts = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "prompt" in data:
                        processed_prompts.add(data["prompt"])
                except:
                    pass
    
    # 过滤掉已经跑过的
    todos = [line for line in input_lines if json.loads(line).get("prompt") not in processed_prompts]
    print(f"🚀 总数据: {len(input_lines)}, 已处理: {len(processed_prompts)}, 剩余任务: {len(todos)}")

    if not todos:
        print("✅ 所有数据已处理完毕！")
        return

    # 3. 多线程处理
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_line = {executor.submit(process_single_item, line): line for line in todos}
            
            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(future_to_line), total=len(todos), desc="🚀 多线程生成中"):
                result = future.result()
                if result:
                    # 写入结果 (加锁写入是更好的习惯，但在 Python GIL 下简单的 append write 问题不大，或者直接单线程写)
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                else:
                    # 如果失败了，可以选择不处理，下次跑脚本会自动重试
                    pass

    print(f"\n🎉 全部完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()