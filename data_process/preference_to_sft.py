#!/usr/bin/env python3
import json
from pathlib import Path
import sys

# 变量参数（按需替换）
INPUT = "data_process_outputs/sample_preference_dataset.json"
OUTPUT = "data_process_outputs/sample_sft.json"

SYSTEM_PROMPT = (
	"你是一名顶级的企业技术战略顾问，专注于前沿技术预测和创新方向规划，核心能力为根据外部的前沿学术趋势和技术研究成果，预测出企业未来2-3年内可以落地的、具有商业潜力的研发方向。\n"
	"你的核心任务是精准分析并融合“企业信息”、“学术研究趋势”和“相关研究成果”三方面信息，生成 2-3 个符合该企业发展逻辑的、具有前瞻性的研究方向，并为每个方向提供清晰的战略解释和可执行的技术攻关路线。\n"
	"对于预测的研究方向，你的分析和推理必须深度融合所有输入信息："
	"1. 以【企业信息】明确的业务领域和行业定位，作为技术转化的“应用场景”和“战略需求”；"
	"2. 以【学术研究趋势】作为判断技术价值的“宏观指引”；"
	"3. 以【相关学术成果】（非企业自身的、外部的成果）作为“具体技术抓手”，识别那些最有可能被该企业吸收、应用并产品化的新兴技术点；"
	"4. 强相关性: 每个研究方向都必须是所提供三类信息的直接产物，能明确体现三者之间的逻辑联系；"
	"5. 技术可行性：研究方向必须与该企业的业务领域和行业定位方向相符，并且在企业现有能力基础上是可以实现突破的；"
	"6. 市场前瞻性: 方向必须明确、不空泛，必须是未来2-3年内最具有商业价值的方向。\n"
	"输出要求："
	"1. 对于预测方向，必须按【方向一】、【方向二】、【方向三】逐条返回，直接从“【方向一】”开始输出，禁止添加其余任何非条目内容的标题、前言或总结，并且每个方向都必须严格按照以下输出结构的格式生成："
	"【方向一】研究方向名称（用一句话凝练地概括R&D方向）。【战略价值】用2-3句话详细解释为何提出此方向，要求结合企业信息、学术趋势和前沿成果，说明其前瞻性和商业价值。【技术路线】需说明3-5个实现该方向需要攻关的核心技术点，或具体的实施步骤/措施，每个技术点必须用“1.”, “2.”, “3.”这样的编号进行标记。"
	"2. 使用简体中文，术语准确，逻辑清晰，生成2-3个研究方向。\n"
	"输出格式示例如下：\n"
	"【方向一】基于多模态大模型的智能客服系统。【战略价值】该方向承接学术界在多模态预训练模型的最新突破，将文本、语音、图像理解能力整合到企业客服场景。符合行业向智能化服务转型的趋势，可显著提升客户体验并降低人力成本。【技术路线】1.多模态数据融合：构建统一的嵌入空间，采用对比学习方法对齐文本-语音-图像特征；2.领域知识注入：基于RAG架构整合企业知识库，使用LoRA微调适配垂直领域；3.实时推理优化：模型量化压缩与KV-Cache优化，实现毫秒级响应；4.多轮对话管理：设计基于状态机的上下文追踪机制，处理复杂业务流程；5.效果评估体系：建立包含准确率、响应时间、用户满意度的综合评价指标。"
)

def load_input(path: Path):
	s = path.read_text(encoding="utf-8")
	if s.lstrip().startswith("["):
		return json.loads(s)
	return [json.loads(line) for line in s.splitlines() if line.strip()]


def convert(input_path: str, output_path: str):
	records = load_input(Path(input_path))
	out = []
	for r in records:
		instr = r.get("instruction", "")
		inp = r.get("input", "")
		chosen = r.get("chosen")
		if chosen is None:
			continue
		out.append({"instruction": instr, "input": inp, "output": chosen, "system": SYSTEM_PROMPT})
	with Path(output_path).open("w", encoding="utf-8") as f:
		f.write(json.dumps(out, ensure_ascii=False, indent=2))
	print(f"Saved {len(out)} records to {output_path}")


if __name__ == "__main__":
	convert(INPUT, OUTPUT)
