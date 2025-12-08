import argparse
import yaml
import json
import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------

def init_logging(out_dir: str):
    log_path = os.path.join(out_dir, "ppl_eval.log")

    # 防止重复 handler
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("PPL 日志初始化完成")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 模型加载（支持 LoRA）
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg_model):
    model_path = cfg_model["model_name_or_path"]
    adapter_path = cfg_model.get("adapter_name_or_path")
    cache_dir = cfg_model.get("model_cache")

    logging.info(f"加载基座模型: {model_path}")
    if adapter_path:
        logging.info(f"检测到 LoRA 适配器: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, device_map="auto")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 计算困惑度
# ---------------------------------------------------------------------------

import torch

def compute_ppl_for_sample(model, tokenizer, system_prompt: str, instruction: str, input_text: str, output_text: str, cutoff_len: int):
    """
    计算样本的困惑度 (Perplexity).
    逻辑：构建 [System + User + Assistant(Target)] 的完整对话，
    Mask 掉 System 和 User 部分，只计算 Assistant 回复部分的 Loss。
    """

    # 0. 基础校验
    if not output_text or not output_text.strip():
        return None  # 没有 Ground Truth，无法计算 PPL

    # 1. 构造消息列表 (与 generate 保持一致)
    if input_text:
        user_msg = f"{instruction}\n\n输入：\n{input_text}"
    else:
        user_msg = instruction

    # 构造 "Prompt" 部分的消息 (用于计算长度，以便掩盖)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    # 构造 "Full" 部分的消息 (包含我们要评估的 Ground Truth)
    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output_text}, # 这里的 content 是我们期望模型生成的真实标签
    ]

    # 2. 应用 Chat Template
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Current tokenizer does not support chat templates.")

    # A. 获取纯 Prompt 的文本 (用于计算 mask 长度)
    # 注意：add_generation_prompt=True 会添加 <|assistant|> 等引导符
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # B. 获取完整文本 (Prompt + Answer)
    # 注意：这里不需要 add_generation_prompt，因为最后一条已经是 assistant
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    # 3. 编码 (Tokenize)
    # 先对 Prompt 进行编码，仅为了获取长度
    prompt_enc = tokenizer(
        prompt_text, 
        return_tensors="pt", 
        add_special_tokens=False # 模板通常已经包含了 special tokens
    )
    prompt_len = prompt_enc.input_ids.shape[1]

    # 对 Full Text 进行编码
    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=cutoff_len
    )
    
    input_ids = full_enc.input_ids
    
    # 边界检查：如果截断太厉害，导致连 Prompt 都不完整，则该样本无效
    if input_ids.shape[1] < prompt_len:
        return None 

    # 4. 构建 Labels 并进行 Masking
    labels = input_ids.clone()
    
    # 核心步骤：将 Prompt 部分的 Label 设为 -100 (PyTorch Loss 会忽略这些位置)
    labels[:, :prompt_len] = -100

    # 5. 设备处理 (复用 reference 代码的逻辑)
    try:
        first_param = next(model.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = full_enc.attention_mask.to(device)

    # 6. 计算 Loss
    with torch.no_grad():
        # 注意：不要手动 shift labels，CausalLM 模型内部会自动 shift
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

    # 7. 计算 PPL
    ppl = torch.exp(loss).item()
    return ppl


# ---------------------------------------------------------------------------
# 数据集整体评估
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset_path: str, model, tokenizer, cutoff_len: int, out_path: str):
    data = load_json(dataset_path)

    results = []
    ppl_list = []

    for sample in tqdm(data, desc="Evaluating PPL"):
        instruction = sample.get("instruction", "")
        input_data = sample.get("input", "")
        label = sample.get("output", "")
        sys_prompt = sample.get("system", "")

        try:
            ppl = compute_ppl_for_sample(model, tokenizer, sys_prompt, instruction, input_data, label, cutoff_len)
        except Exception as e:
            logging.exception(f"PPL 计算失败: {str(e)}")
            ppl = None

        results.append({
            "instruction": instruction,
            "input": input_data,
            "label": label,
            "ppl": ppl
        })

        if ppl is not None:
            ppl_list.append(ppl)

    # 写入文件
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if ppl_list:
        mean_ppl = sum(ppl_list) / len(ppl_list)
        max_ppl = max(ppl_list)
        min_ppl = min(ppl_list)
    else:
        mean_ppl = max_ppl = min_ppl = None

    return results, {
        "mean_ppl": mean_ppl,
        "max_ppl": max_ppl,
        "min_ppl": min_ppl,
        "count": len(ppl_list)
    }



# ---------------------------------------------------------------------------
# 主函数入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPL Evaluation with YAML Config")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg.get("model_name_or_path")
    adapter_name = cfg.get("adapter_name_or_path")
    model_cache = cfg.get("model_cache")

    dataset_path = cfg.get("dataset_path")
    cutoff_len = cfg.get("cutoff_len", 2048)
    output_dir = cfg.get("output_dir", "eval_outputs_ppl")
    run_label = cfg.get("run_label", "ppl_eval")

    if not dataset_path:
        raise ValueError("配置文件中缺少 dataset_path")

    os.makedirs(output_dir, exist_ok=True)
    init_logging(output_dir)

    logging.info(f"加载配置文件: {args.config}")
    logging.info(f"数据集路径: {dataset_path}")

    model_cfg = {
        "model_name_or_path": model_name,
        "adapter_name_or_path": adapter_name,
        "model_cache": model_cache
    }

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # 评估
    out_path = os.path.join(output_dir, f"{run_label}_results.json")

    results, metrics = evaluate_dataset(
        dataset_path,
        model,
        tokenizer,
        cutoff_len,
        out_path
    )

    logging.info("====== PPL 评估完成 ======")
    logging.info(f"平均困惑度（mean_ppl）: {metrics['mean_ppl']}")
    logging.info(f"最高困惑度 max_ppl: {metrics['max_ppl']}")
    logging.info(f"最低困惑度 min_ppl: {metrics['min_ppl']}")


if __name__ == "__main__":
    main()
