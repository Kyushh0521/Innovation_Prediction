import argparse
import yaml
import json
import os
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
    
# 初始化 jieba，避免并在运行时打印日志干扰
jieba.setLogLevel(logging.CRITICAL)
jieba.initialize()


# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------

def init_logging(out_dir: str):
    log_path = os.path.join(out_dir, "bleu_rouge_eval.log")
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

def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# 模型加载与生成函数
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg_model):
    model_path = cfg_model['model_name_or_path']
    adapter_path = cfg_model.get('adapter_name_or_path')
    cache_dir = cfg_model.get('model_cache')
    
    logging.info(f"正在加载基座模型: {model_path}")
    if cache_dir:
        logging.info(f"使用缓存目录: {cache_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True, 
        cache_dir=cache_dir
    )

    if adapter_path:
        logging.info(f"检测到 LoRA 配置，正在挂载适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir, device_map="auto")
    
    model.eval()
    return model, tokenizer

def get_generate_fn(model, tokenizer, cutoff_len: int):
    """
    构建模型推理函数 (保持原逻辑不变)
    """
    def generate(system_prompt: str, instruction: str, input_text: str):
        if input_text:
            user_msg = f"{instruction}\n\n输入：\n{input_text}"
        else:
            user_msg = instruction

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Current tokenizer does not support chat templates.")

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=cutoff_len
        )

        try:
            first_param = next(model.parameters())
            device_for_inputs = first_param.device
        except StopIteration:
            device_for_inputs = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_inputs = model_inputs.to(device_for_inputs)

        with torch.no_grad():
            generated = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = model_inputs.input_ids.shape[1]
        if generated.shape[1] <= input_len:
            return ""

        output_ids = generated[0][input_len:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        return response.strip()

    return generate

# ---------------------------------------------------------------------------
# 评估逻辑 (BLEU & ROUGE)
# ---------------------------------------------------------------------------

def compute_metrics(prediction: str, label: str) -> Dict[str, float]:
    """
    计算单条样本的 BLEU-4 和 ROUGE 分数
    """
    # 使用 jieba 分词
    hypothesis = list(jieba.cut(prediction))
    reference = list(jieba.cut(label))

    # 计算 BLEU-4
    # sentence_bleu 期望 reference 是 list of lists (因为可以有多个参考答案)
    bleu_score = sentence_bleu(
        [reference],
        hypothesis,
        smoothing_function=SmoothingFunction().method3,
    )

    # 计算 ROUGE
    # 如果预测为空或标签为空，rouge计算会报错或无意义
    if len(hypothesis) == 0 or len(reference) == 0:
        rouge_result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        # rouge 需要以空格分隔的字符串
        try:
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            rouge_result = scores[0]
        except Exception:
            # 极少数情况下（如只包含标点），rouge可能无法计算
            rouge_result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

    def safe_get_f(scores_dict, key):
        metric_data = scores_dict.get(key, {})
        if not isinstance(metric_data, dict):
            return 0.0
        return metric_data.get("f", 0.0)

    return {
        "bleu-4": round(float(bleu_score) * 100.0, 4), # type: ignore
        "rouge-1": round(float(safe_get_f(rouge_result, "rouge-1")) * 100.0, 4),
        "rouge-2": round(float(safe_get_f(rouge_result, "rouge-2")) * 100.0, 4),
        "rouge-l": round(float(safe_get_f(rouge_result, "rouge-l")) * 100.0, 4),
    }

def evaluate_dataset(dataset_path: str, model_generate, out_path: str):
    data = load_json(dataset_path)
    
    results = []
    # 累加器用于计算平均分
    total_metrics = {"bleu-4": 0.0, "rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    valid_count = 0

    for sample in tqdm(data, desc="Evaluating"):
        system_prompt = sample.get("system", "")
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        label = sample.get("output", "")

        # 1. 模型推理
        try:
            model_output = model_generate(system_prompt, instruction, input_text)
            gen_error = None
        except Exception as e:
            logging.exception(f"模型生成失败: {instruction[:50]}...")
            model_output = ""
            gen_error = str(e)

        # 2. 计算指标
        # 即使生成为空，也需要计算指标（此时分数为0），除非发生程序错误跳过
        metrics = compute_metrics(model_output, label)

        # 累加
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        valid_count += 1

        results.append({
            "instruction": instruction,
            "input": input_text,
            "label": label,
            "prediction": model_output,
            "gen_error": gen_error,
            "metrics": metrics
        })

    # 3. 计算平均值
    avg_metrics = {k: (v / valid_count if valid_count > 0 else 0.0) for k, v in total_metrics.items()}

    # 保存详细结果
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"详细结果已保存至: {out_path}")
    return avg_metrics

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="运行 BLEU/ROUGE 评估")
    parser.add_argument('--config', type=str, required=True, help='YAML 配置文件路径')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    # 读取配置
    model_name = cfg.get('model_name_or_path', 'Qwen/Qwen2.5-0.5B-Instruct')
    model_cache = cfg.get('model_cache', None)
    adapter_name = cfg.get('adapter_name_or_path', None)
    dataset_path = cfg.get('dataset_path', 'model_eval/sample_sft_test.json')
    cutoff_len = cfg.get('cutoff_len', 2048)
    output_dir = cfg.get('output_dir', "eval_outputs")
    run_label = cfg.get('run_label', "bleu_rouge_eval")

    if not dataset_path:
        raise ValueError("配置文件中未找到 dataset_path")

    # 初始化环境
    os.makedirs(output_dir, exist_ok=True)
    init_logging(output_dir)

    logging.info(f"配置: {args.config}")
    logging.info(f"模型: {model_name}")
    logging.info(f"适配器: {adapter_name}")

    # 加载模型
    model_cfg = {
        'model_name_or_path': model_name,
        'adapter_name_or_path': adapter_name,
        'model_cache': model_cache,
    }
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    model_generate = get_generate_fn(model, tokenizer, cutoff_len)

    # 运行评估
    out_path = os.path.join(output_dir, f"{run_label}_results.json")
    avg_metrics = evaluate_dataset(
        dataset_path=dataset_path,
        model_generate=model_generate,
        out_path=out_path
    )

    # 输出摘要
    logging.info("=== 评估摘要 (平均值) ===")
    logging.info(f"BLEU-4:  {avg_metrics['bleu-4']:.4f}")
    logging.info(f"ROUGE-1: {avg_metrics['rouge-1']:.4f}")
    logging.info(f"ROUGE-2: {avg_metrics['rouge-2']:.4f}")
    logging.info(f"ROUGE-L: {avg_metrics['rouge-l']:.4f}")

if __name__ == '__main__':
    main()