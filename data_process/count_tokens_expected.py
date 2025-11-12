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
            f"你是一名顶级的企业技术战略顾问，专注于前沿技术预测和创新方向规划，核心能力为根据外部的前沿学术趋势和技术研究成果，预测出企业未来2-3年内可以落地的、具有商业潜力的研发方向。\n"
            
            f"你的核心任务是精准分析并融合“企业信息”、“学术研究趋势”和“相关研究成果”三方面信息，生成两个版本的预测结果：一个是正确的预测（可采纳），一个是错误的预测（不可采纳）。其中，正确的预测需要包含 2-3 个符合该企业发展逻辑的、具有前瞻性的研究方向，并为每个方向提供清晰的战略解释和可执行的技术攻关路线。而错误的预测只需要根据直觉或常见趋势泛泛地列出一些方向，不需要进行深入分析，也不需要考虑与企业实际业务、学术研究趋势或外部研究成果的联。\n"

            f"对于正确的预测，你的分析和推理必须深度融合所有输入信息："
            f"1. 以【企业信息】明确的业务领域和行业定位，作为技术转化的“应用场景”和“战略需求”；"
            f"2. 以【学术研究趋势】作为判断技术价值的“宏观指引”；"
            f"3. 以【相关学术成果】（非企业自身的、外部的成果）作为“具体技术抓手”，识别那些最有可能被该企业吸收、应用并产品化的新兴技术点；"
            f"4. 强相关性: 每个研究方向都必须是所提供三类信息的直接产物，能明确体现三者之间的逻辑联系；"
            f"5. 技术可行性：研究方向必须与该企业的业务领域和行业定位方向相符，并且在企业现有能力基础上是可以实现突破的；"
            f"6. 市场前瞻性: 方向必须明确、不空泛，必须是未来2-3年内最具有商业价值的方向。\n"
            
            f"对于错误的预测，你的分析和推理可以："
            f"1. 使用宽泛、模糊的词汇（如“智能化”、“绿色发展”、“数字转型”等）；"
            f"2. 仅基于单一信息（例如只根据企业信息或研究趋势）；"
            f"3. 可以忽略【战略价值】或【技术路线】部分；"
            f"4. 可以包含简单描述、通用建议或非研究性方向（如“加强人才培养”或“拓展国际市场”等）。\n"

            f"输出要求："
            f"1. 你需要分别先后生成“可采纳预测”和“不可采纳预测”两部分内容，并以一下格式明显区分：【可采纳预测】、【不可采纳预测】；"
            f"2. 对于可采纳预测，必须按【方向一】、【方向二】、【方向三】逐条返回，直接从“【方向一】”开始输出，禁止添加其余任何非条目内容的标题、前言或总结，并且每个方向都必须严格按照以下输出结构的格式生成："
            f"【方向一】研究方向名称（用一句话凝练地概括R&D方向）。【战略价值】用2-3句话详细解释为何提出此方向，要求结合企业信息、学术趋势和前沿成果，说明其前瞻性和商业价值。【技术路线】需说明3-5个实现该方向需要攻关的核心技术点，或具体的实施步骤/措施，每个技术点必须用“1.”, “2.”, “3.”这样的编号进行标记。"
            f"3. 对于不可采纳预测，同样必须按【方向一】、【方向二】、【方向三】逐条返回，直接从“【方向一】”开始输出，禁止添加其余任何非条目内容的标题、前言或总结，并且每个方向都必须严格按照以下输出结构的格式生成："
            f"【方向一】研究方向名称（用一句话凝练地概括R&D方向）（简要说明为何该方向不可采纳，指出其缺乏前瞻性、技术可行性或与企业业务的关联性）。"
            f"4. 使用简体中文，术语准确，逻辑清晰，分别为【可采纳预测】和【不可采纳预测】生成2-3个研究方向。\n"

            f"输出格式示例如下：\n"
            f"【可采纳预测】\n"
            f"【方向一】基于多模态大模型的智能客服系统。【战略价值】该方向承接学术界在多模态预训练模型的最新突破，将文本、语音、图像理解能力整合到企业客服场景。符合行业向智能化服务转型的趋势，可显著提升客户体验并降低人力成本。【技术路线】1.多模态数据融合：构建统一的嵌入空间，采用对比学习方法对齐文本-语音-图像特征；2.领域知识注入：基于RAG架构整合企业知识库，使用LoRA微调适配垂直领域；3.实时推理优化：模型量化压缩与KV-Cache优化，实现毫秒级响应；4.多轮对话管理：设计基于状态机的上下文追踪机制，处理复杂业务流程；5.效果评估体系：建立包含准确率、响应时间、用户满意度的综合评价指标。"
            f"【不可采纳预测】\n"
            f"【方向一】基于区块链的火星殖民物联网供应链溯源（过于宽泛）。【方向二】基于Windows XP内核的工业互联网实时操作系统研发（过时技术方向）。【方向三】面向传统手摇纺车的5G边缘计算智能纱锭改造（与企业行业无关）。"
    )

    company_info = record.get('company_info')
    research_trends = record.get('research_field')
    achievements = record.get('achievement')

    user_content = (
        f"作为一名企业技术战略顾问，请根据以下信息（企业信息、学术研究趋势、相关研究成果），为该公司制定未来2-3年【可采纳预测】和【不可采纳预测】的研究方向。"
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
    # 计算费用（以人民币元计）：输入 0.0020 元 / 千 tokens，输出 0.0080 元 / 千 tokens
    try:
        input_price_per_k = 0.0020  # 元 / 千 tokens
        output_price_per_k = 0.0080  # 元 / 千 tokens
        input_tokens_total = summary.get('input_tokens_sum', 0)
        output_tokens_total = summary.get('output_tokens_sum', 0)

        cost_input = (input_tokens_total / 1000.0) * input_price_per_k
        cost_output = (output_tokens_total / 1000.0) * output_price_per_k
        cost_total = cost_input + cost_output

        # 每条记录平均费用
        count = summary.get('count', 0) or 0
        avg_cost_per_record = (cost_total / count) if count else 0.0

        print("\n估算费用：")
        print(f"  输入 token 总数: {input_tokens_total}，单价 {input_price_per_k} 元/千 tokens，费用: {cost_input:.4f} 元")
        print(f"  输出 token 总数: {output_tokens_total}，单价 {output_price_per_k} 元/千 tokens，费用: {cost_output:.4f} 元")
        print(f"  预计总费用: {cost_total:.4f} 元，平均每条记录: {avg_cost_per_record:.6f} 元")
    except Exception as e:
        print(f"计算费用时出错: {e}")

def main():
    INPUT_FILE = 'data_process_outputs/sample_inputs_with_matches.jsonl'
    OUTPUT_DIR = 'data_process_outputs'
    BASE_NAME = 'token_counts'  # 输出文件基础名，会生成 BASE_NAME_{model}.jsonl
    MODEL = ["Qwen/Qwen3-235B-A22B-Instruct-2507"]
    OUTPUT_TOKENS = 1024
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