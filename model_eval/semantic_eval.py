import argparse
from functools import cache
import yaml
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------

def init_logging(out_dir: str):
    log_path = os.path.join(out_dir, "semantic_eval.log")
    # 清除旧的 handlers 避免重复日志
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("日志系统初始化完成。")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_directions(text: str) -> List[str]:
    if not text or not text.strip():
        return []
        
    lines = text.split("【方向")
    results = []
    for block in lines:
        if '】' in block:
            _, rest = block.split('】', 1)
            rest = rest.strip()
            if rest:
                results.append(rest)
    if not results:
        return [text.strip()]
    return results


def load_model_and_tokenizer(cfg_model):
    model_path = cfg_model['model_name_or_path']
    adapter_path = cfg_model.get('adapter_name_or_path')
    cache_dir = cfg_model.get('model_cache')
    
    logging.info(f"正在加载基座模型: {model_path}")
    if cache_dir:
        logging.info(f"使用缓存目录: {cache_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
    
    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True, 
        cache_dir=cache_dir
    )

    if adapter_path:
        logging.info(f"检测到 LoRA 配置，正在挂载适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.float16, cache_dir=cache_dir, device_map="auto")
        # 如果是推理模式，通常会合并权重以加快速度（可选）
        # model = model.merge_and_unload() 
    
    model.eval() # 开启评估模式
    return model, tokenizer


def get_generate_fn(model, tokenizer, cutoff_len: int):
    """
    构建模型推理函数（企业版增强）
    支持：
    - ChatTemplate
    - 输入截断
    - 安全的输出切片
    - 全自动 device map
    """

    def generate(system_prompt: str, instruction: str, input_text: str):

        # 1. 标准消息格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "user", "content": input_text}
        ]

        # 2. 应用模型 Chat Template
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Current tokenizer does not support chat templates.")

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 3. 编码 + 截断
        model_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=cutoff_len
        ).to(model.device)

        # 4. 生成（强制 deterministic）
        with torch.no_grad():
            generated = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=False,                   # 确保评测一致性
                pad_token_id=tokenizer.eos_token_id
            )

        # 5. 安全提取输出（切 token）
        input_len = model_inputs.input_ids.shape[1]

        if generated.shape[1] <= input_len:
            return ""  # 防止模型没有产生新 token

        output_ids = generated[0][input_len:]

        # 6. 解码
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        return response.strip()

    return generate


# ---------------------------------------------------------------------------
# 评估流程
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset_path: str, model_generate, embedder_name: str, out_path: str, threshold: float = 0.5, embedding_cache: Optional[str] = None):
    data = load_json(dataset_path)
    embedder = SentenceTransformer(embedder_name, device="auto",cache_folder=embedding_cache)
    results = []

    # 全局累加器 (用于 Micro 计算)
    # 分子：匹配程度的总和
    total_pred_matches_sum = 0.0  
    total_gt_matches_sum = 0.0    
    # 分母：项目总数
    total_pred_count = 0          
    total_gt_count = 0            

    for sample in tqdm(data, desc="Evaluating"):
        # 提取数据
        system_prompt = sample.get("system", "")
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        label = sample.get("output", "")
        # 模型推理
        model_output = model_generate(system_prompt, instruction, input_text)

        # 提取方向列表
        gt_dirs = extract_directions(label)
        pred_dirs = extract_directions(model_output)

        # 处理空真值/空预测的边界情况，避免对空 tensor 调用 max()
        if len(gt_dirs) == 0 or len(pred_dirs) == 0:
            # 当任一方为空时，视为没有匹配
            sample_recall_sum = 0.0
            sample_precision_sum = 0.0
        else:
            # 编码向量（确保在 CUDA 上生成 tensor）
            gt_embeds = embedder.encode(gt_dirs, convert_to_tensor=True)
            pred_embeds = embedder.encode(pred_dirs, convert_to_tensor=True)

            # 计算相似度矩阵
            sim_matrix = util.cos_sim(pred_embeds, gt_embeds)
            
            # A. 计算 Recall (针对 GT: 每一个GT是否被召回?)
            recall_vals = sim_matrix.max(dim=0).values
            recall_vals[recall_vals < threshold] = 0.0 # 阈值过滤
            sample_recall_sum = recall_vals.sum().item()
            
            # B. 计算 Precision (针对 Pred: 每一个Pred是否准确?)
            precision_vals = sim_matrix.max(dim=1).values
            precision_vals[precision_vals < threshold] = 0.0 # 阈值过滤
            sample_precision_sum = precision_vals.sum().item()

        # 单样本指标
        s_recall = sample_recall_sum / len(gt_dirs) if len(gt_dirs) > 0 else 0.0
        s_precision = sample_precision_sum / len(pred_dirs) if len(pred_dirs) > 0 else 0.0
        
        if s_precision + s_recall == 0:
            s_f1 = 0.0
        else:
            s_f1 = 2 * s_precision * s_recall / (s_precision + s_recall)

        # 累加全局统计量
        total_gt_matches_sum += sample_recall_sum
        total_gt_count += len(gt_dirs)
        
        total_pred_matches_sum += sample_precision_sum
        total_pred_count += len(pred_dirs)

        # 保存单条数据的详细指标
        results.append({
            "instruction": instruction,
            "input": input_text,
            "label": label,
            "prediction": model_output,
            "score": s_f1,              # F1
            "precision": s_precision,   # P
            "recall": s_recall          # R
        })

    # 保存详细结果文件
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 计算整体指标
    # Macro Average
    if len(results) > 0:
        macro_precision = sum(r['precision'] for r in results) / len(results)
        macro_recall = sum(r['recall'] for r in results) / len(results)
        macro_f1 = sum(r['score'] for r in results) / len(results)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    # Micro Average
    micro_precision = total_pred_matches_sum / total_pred_count if total_pred_count > 0 else 0.0
    micro_recall = total_gt_matches_sum / total_gt_count if total_gt_count > 0 else 0.0
    
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    # 整体指标汇总
    metrics = {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall
    }

    logging.info(f"详细结果已保存至: {out_path}")

    return results, metrics


# ---------------------------------------------------------------------------
# YAML 加载器
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    # 仅通过命令行指定配置文件路径
    parser = argparse.ArgumentParser(description="使用 YAML 配置运行评估")
    parser.add_argument('--config', type=str, required=True, help='YAML 配置文件路径')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    model_name=cfg.get('model_name_or_path', 'Qwen/Qwen2.5-0.5B-Instruct')
    model_cache=cfg.get('model_cache', None)
    adapter_name =cfg.get('adapter_name_or_path', None)
    embedding_name=cfg.get('embedding_name_or_path', 'BAAI/bge-large-zh-v1.5')
    embedding_cache=cfg.get('embedding_cache', None)
    dataset_path=cfg.get('dataset_path', './data/test_dataset.json')
    cutoff_len=cfg.get('cutoff_len', 3072)
    output_dir=cfg.get('output_dir', "eval_outputs/qwen2.5-0.5B-Instruct/original")
    run_label=cfg.get('run_label', "original_test")
    threshold=cfg.get('threshold', 0.6)

    if not dataset_path:
        raise ValueError("配置文件中未找到 dataset.path，请在 config.yaml 中设置。")

    # 创建输出目录并初始化日志
    os.makedirs(output_dir, exist_ok=True)
    init_logging(output_dir)

    logging.info(f"使用配置文件: {args.config}")
    logging.info(f"数据集: {dataset_path}")
    logging.info(f"模型: {model_name}，适配器: {adapter_name if adapter_name else '无'}")
    logging.info(f"嵌入模型: {embedding_name}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"运行 ID: {run_label}")

    model_cfg = {
        'model_name_or_path': model_name,
        'adapter_name_or_path': adapter_name,
        'model_cache': model_cache,
    }

    # 加载模型与 tokenizer（支持 LoRA）
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # 构建生成函数
    model_generate = get_generate_fn(model, tokenizer, cutoff_len)

    # 运行评估
    out_path = os.path.join(output_dir, f"{run_label}_results.json")
    results, metrics = evaluate_dataset(
        dataset_path=dataset_path,
        model_generate=model_generate,
        embedder_name=embedding_name,
        out_path=out_path,
        threshold=threshold,
        embedding_cache=embedding_cache
    )

    # 输出摘要
    logging.info("=== 评估摘要 ===")
    logging.info(f"宏平均 精确率 (Macro): {metrics['macro_precision']:.4f}")
    logging.info(f"宏平均 召回率 (Macro): {metrics['macro_recall']:.4f}")
    logging.info(f"宏平均 F1 (Macro): {metrics['macro_f1']:.4f}")
    logging.info(f"微平均 精确率 (Micro): {metrics['micro_precision']:.4f}")
    logging.info(f"微平均 召回率 (Micro): {metrics['micro_recall']:.4f}")
    logging.info(f"微平均 F1 (Micro): {metrics['micro_f1']:.4f}")


if __name__ == '__main__':
    main()
