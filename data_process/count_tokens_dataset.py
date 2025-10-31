import json
import os
import re
from typing import Callable, Optional
import contextlib, io, sys

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


def openai_token_counter(model: Optional[str] = None) -> Callable[[str], int]:
    """返回一个基于 OpenAI/tiktoken 的 token 计数函数。
    """
    if not TIKTOKEN_AVAILABLE or tiktoken is None:
        raise RuntimeError(
            "tiktoken 未安装或不可用。"
        )

    # 尝试为给定模型获取编码器，否则使用通用 cl100k_base 编码
    try:
        if model:
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        raise RuntimeError(f"无法创建 tiktoken 编码器: {e}")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text))

    return count_tokens


def autotokenizer_token_counter(model: str) -> Callable[[str], int]:
    """使用 Hugging Face 的 Tokenizer 来为开源模型计数 token。
    """
    if not TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
        raise RuntimeError("transformers 未安装或不可用。")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"无法加载 autotokenizer: {e}")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        # 暂时屏蔽 tokenizer 内部输出（包括 stdout/stderr）
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, 'w')
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        finally:
            # 恢复正常输出
            sys.stdout.close()
            sys.stdout, sys.stderr = old_stdout, old_stderr
        return len(ids)

    return count_tokens


def token_counter_for_model(model: Optional[str]):
    """根据 model 名称返回合适的 token 计数函数。
    """
    if not model:
        raise ValueError("model 参数不能为空")
    
    if model and model.lower().startswith('gpt'):
        return openai_token_counter(model)
    
    return autotokenizer_token_counter(model)


def build_default_prompts(record: dict) -> tuple:
    """构建默认的 system_content 和按要求拼接的用户文本（user_content）。
    返回 (system_content, user_content)。
    """
    system_content = (
        "你是一名技术预测专家，任务是预测企业未来1-2年的研究方向。"
        "要求："
        "1. 方向具体、前瞻，符合企业所属行业逻辑；"
        "2. 与提供的学术研究趋势和研究成果紧密相关；"
        "3. 仅返回条目列表，用 1., 2., 3. 编号；每条为单行，包含“方向，关键技术/措施”；"
        "请用简体中文，不要任何解释、标题或前后缀。生成 2-3 条。"
    )

    company_info = record.get('company_info')
    research_trends = record.get('research_field')
    achievements = record.get('achievement')

    user_content = (
        f"请根据以下企业信息、学术研究趋势和研究成果，给出未来1-2年的主要研究方向。"
        f"企业信息：{company_info}"
        f"学术研究趋势：{research_trends}。"
        f"相关研究成果：{achievements}。"
    )
    return system_content, user_content


def process_file(input_path: str, output_json: str, model: Optional[str], output_tokens: int, limit: Optional[int] = None):
    """处理单个输入文件，按记录计算 input tokens（system+user）与假定的 output tokens，并写出 JSONL 和摘要。
    """
    counter = token_counter_for_model(model)

    results = []
    # 三个并行的累加器，用于摘要统计
    input_tokens_list = []
    output_tokens_list = []
    total_tokens_list = []

    with open(input_path, 'r', encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            system_content, user_content = build_default_prompts(rec)

            sys_tokens = counter(system_content)
            user_tokens = counter(user_content)

            input_tokens = sys_tokens + user_tokens
            output_tokens = int(output_tokens)
            total_tokens_output = input_tokens + output_tokens

            per_record = {
                'index': idx,
                'sys_tokens': sys_tokens,
                'user_tokens': user_tokens,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens_output': total_tokens_output,
            }
            results.append(per_record)

            input_tokens_list.append(input_tokens)
            output_tokens_list.append(output_tokens)
            total_tokens_list.append(total_tokens_output)

    # 写 JSONL（每条记录为一行 JSON）
    os.makedirs(os.path.dirname(output_json), exist_ok=True) if os.path.dirname(output_json) else None
    with open(output_json, 'w', encoding='utf-8') as jf:
        for r in results:
            jf.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 生成 JSON 概要并打印到终端
    summary = {
        'count': len(results),
        'input_tokens_min': min(input_tokens_list) if input_tokens_list else 0,
        'input_tokens_max': max(input_tokens_list) if input_tokens_list else 0,
        'input_tokens_sum': sum(input_tokens_list) if input_tokens_list else 0,
        'output_tokens_min': min(output_tokens_list) if output_tokens_list else 0,
        'output_tokens_max': max(output_tokens_list) if output_tokens_list else 0,
        'output_tokens_sum': sum(output_tokens_list) if output_tokens_list else 0,
        'total_tokens_min': min(total_tokens_list) if total_tokens_list else 0,
        'total_tokens_max': max(total_tokens_list) if total_tokens_list else 0,
        'total_tokens_sum': sum(total_tokens_list) if total_tokens_list else 0,
        'model': model,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"模型 {model} 已处理 {summary['count']} 条记录；"
        f"input_tokens_sum={summary['input_tokens_sum']}，output_tokens_sum={summary['output_tokens_sum']}，total_tokens_sum={summary['total_tokens_sum']}"
    )

def main():
    INPUT_FILE = 'data_process_outputs/enterprises_prompt_inputs_with_matches.jsonl'
    OUTPUT_DIR = 'data_process_outputs'
    BASE_NAME = 'token_counts'  # 输出文件基础名，会生成 BASE_NAME_{model}.jsonl
    MODEL = ["gpt-5", "deepseek-ai/DeepSeek-R1", "moonshotai/Kimi-K2-Instruct-0905", "Qwen/Qwen3-235B-A22B-Instruct-2507"]
    OUTPUT_TOKENS = 128
    LIMIT = None  # 若仅处理前 N 条记录，设置为整数；否则为 None

    def sanitize_model_name(name: str) -> str:
        # 把常见的分隔符和特殊字符替换为下划线，保证文件名安全
        return re.sub(r"[^0-9A-Za-z-]+", "_", name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model in MODEL:
        safe = sanitize_model_name(model)
        per_model_output = os.path.join(OUTPUT_DIR, f"{BASE_NAME}_{safe}.jsonl")
        process_file(INPUT_FILE, per_model_output, model, OUTPUT_TOKENS, LIMIT)


if __name__ == '__main__':
    main()