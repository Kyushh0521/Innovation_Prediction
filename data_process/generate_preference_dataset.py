import json
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time
import re

# 加载环境变量
load_dotenv()

# 配置 OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# -----------------------------------------------------------------------------
# 少样本 (few-shot) 示例（注释版）
# 下面的代码示例展示如何将 few-shot 示例注入到正例 prompt 中以引导模型输出格式和风格。
# 目前为注释状态；如需启用，请移除注释并根据需要调整示例内容与位置。
#
# # 示例 few-shot messages：每个示例由 user/assistant 对组成，可包含多对
# FEW_SHOT_EXAMPLES = [
#     {"role": "user", "content": "企业信息：示例公司A；学术研究趋势：多模态融合；相关研究成果：跨模态对齐方法。"},
#     {"role": "assistant", "content": (
#         "方向一：基于跨模态对齐的智能检索。\n"
#         "【战略价值】结合公司A的检索业务和学术界跨模态对齐成果，可提升检索准确率并扩展到图像/音频检索。\n"
#         "【技术路线】1.构建统一嵌入空间；2.使用对比学习进行跨模态对齐；3.集成RAG检索模块。"
#     )},
#     # 可继续添加更多示例对
# ]
#
# # 将 few-shot 示例合并到构建 prompt 的函数（示例实现，注释）
# def build_positive_prompt(company_info, research_trends, achievements, few_shot_examples=None):
#     base = [
#         {"role": "system", "content": "...（系统角色提示，保持原有内容）..."},
#         {"role": "user", "content": f"企业信息：{company_info} 学术研究趋势：{research_trends} 相关研究成果：{achievements}"}
#     ]
#     if few_shot_examples:
#         # 将示例放在前面，作为上下文引导
#         return few_shot_examples + base
#     return base
#
# # 调用示例（注释）
# pos_answer_raw = call_llm(build_positive_prompt(company_info, research_trends, achievements, few_shot_examples=FEW_SHOT_EXAMPLES), temperature=0.5)
# -----------------------------------------------------------------------------

# Prompt 模板
def build_prompt(company_info, research_trends, achievements):
    return [
        {"role": "system", "content":
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

        },
        {"role": "user", "content":
            f"作为一名企业技术战略顾问，请根据以下信息（企业信息、学术研究趋势、相关研究成果），为该公司制定未来2-3年【可采纳预测】和【不可采纳预测】的研究方向。"
            f"企业信息：{company_info}"
            f"学术研究趋势：{research_trends}。"
            f"相关研究成果：{achievements}。"
        }
    ]


def call_llm(messages, model="qwen3-235b-a22b-instruct-2507", temperature=0.5):
    """
    调用 LLM 并返回文本。
    temperature: 控制输出的随机性，较低值（如 0.0-0.4）产出更确定性的文本，较高值（如 0.7-1.0）产出更发散的文本。
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048
        )
        text = response.choices[0].message.content
        # 等待一段时间以防速率限制（可通过脚本常量调整）
        time.sleep(1)
        return text
    except Exception as e:
        print(f"调用模型失败: {e}")
        time.sleep(3)
        return ""

def split_joint_output(response_text):
    accepted_match = re.search(r"【可采纳预测】(.*?)【不可采纳预测】", response_text, re.S)
    rejected_match = re.search(r"【不可采纳预测】(.*)", response_text, re.S)

    accepted = accepted_match.group(1).strip() if accepted_match else ""
    rejected = rejected_match.group(1).strip() if rejected_match else ""

    return accepted, rejected

# 构建 偏好数据
def build_preference_example(company_info, research_trends, achievements):
    # 正例使用较低温度以保证输出稳定、可重复；反例使用较高温度以增加多样性
    messages = build_prompt(company_info, research_trends, achievements)
    raw = call_llm(messages, temperature=0.6)

    chosen_raw, rejected_raw = "", ""

    if raw:
        # 首先尝试用已有的分割函数
        try:
            chosen_raw, rejected_raw = split_joint_output(raw)
        except Exception:
            chosen_raw, rejected_raw = "", ""

        if not chosen_raw and not rejected_raw:
            if "【不可采纳预测】" in raw:
                parts = raw.split("【不可采纳预测】", 1)
                chosen_raw = parts[0].strip()
                rejected_raw = parts[1].strip() if len(parts) > 1 else ""
            else:
                m1 = re.search(r"【可采纳预测】(.*?)(?:【不可采纳预测】|$)", raw, re.S)
                m2 = re.search(r"【不可采纳预测】(.*)", raw, re.S)
                chosen_raw = m1.group(1).strip() if m1 else ""
                rejected_raw = m2.group(1).strip() if m2 else ""

    return {
        "instruction": "根据企业信息、学术研究趋势和相关研究成果，预测该企业未来2-3年的研究方向。",
        "input": f"企业信息：{company_info}研究趋势：{research_trends}。研究成果：{achievements}",
        "chosen": chosen_raw,
        "rejected": rejected_raw
    }

def main():
    input_file = "data_process_outputs/sample_inputs_with_matches.jsonl"
    output_file = "data_process_outputs/sample_preference_dataset.json"

    results = []
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    limit = None  # 只取前几条用于验证
    if limit is not None and limit > 0:
        print(f"只处理前 {limit} 条记录（用于检查验证）")
        lines = lines[:limit]

    for item in tqdm(lines, desc="生成偏好数据"):
        company_info = item["company_info"]
        research_trends = item["research_field"]
        achievements = item["achievement"]
        pref_example = build_preference_example(company_info, research_trends, achievements)
        results.append(pref_example)

    # 保存 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已生成偏好数据集: {output_file}，共 {len(results)} 条")

if __name__ == "__main__":
    main()
