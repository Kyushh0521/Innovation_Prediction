# metrics_lib.py
import torch
import jieba
import logging
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional

# 初始化 jieba
jieba.setLogLevel(logging.CRITICAL)
jieba.initialize()

# ===========================
# 1. PPL（困惑度）计算
# ===========================

def compute_ppl(model, tokenizer, system_prompt: str, instruction: str, input_text: str, output_text: str, cutoff_len: int):
    """
    计算单个样本的困惑度（Perplexity）。
    如果没有参考输出（output_text 为空），返回 None。
    """
    if not output_text or not output_text.strip():
        return None

    if input_text:
        user_msg = f"{instruction}\n\n输入：\n{input_text}"
    else:
        user_msg = instruction

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output_text},
    ]

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("当前 tokenizer 不支持对话模板（apply_chat_template）。")

    # 构造只包含 prompt 的文本（用于确定 prompt 长度）
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    # 构造包含参考答案的完整文本
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = prompt_enc.input_ids.shape[1]

    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=cutoff_len)
    input_ids = full_enc.input_ids

    # 如果编码后长度比 prompt 短，说明异常，返回 None
    if input_ids.shape[1] < prompt_len:
        return None

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # 屏蔽 prompt 部分，不计入 loss

    try:
        device = next(model.parameters()).device
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = full_enc.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    return torch.exp(loss).item()



# ===========================
# 2. BLEU / ROUGE 计算
# ===========================

def compute_bleu_rouge(prediction: str, label: str) -> Dict[str, float]:
    """
    计算单条的 BLEU-4 与 ROUGE（返回百分制值，保留 4 位小数）。
    """
    hypothesis = list(jieba.cut(prediction))
    reference = list(jieba.cut(label))

    # 计算 BLEU-4
    bleu_score = sentence_bleu(
        [reference],
        hypothesis,
        smoothing_function=SmoothingFunction().method3,
    )

    # 计算 ROUGE
    if len(hypothesis) == 0 or len(reference) == 0:
        rouge_result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        try:
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            rouge_result = scores[0]
        except:
            rouge_result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

    def safe_get_f(d, k):
        return d.get(k, {}).get("f", 0.0)

    return {
        "bleu-4": round(float(bleu_score) * 100.0, 4),  # type: ignore
        "rouge-1": round(float(safe_get_f(rouge_result, "rouge-1")) * 100.0, 4),
        "rouge-2": round(float(safe_get_f(rouge_result, "rouge-2")) * 100.0, 4),
        "rouge-l": round(float(safe_get_f(rouge_result, "rouge-l")) * 100.0, 4),
    }



# ===========================
# 3. 语义相似度计算（Semantic）
# ===========================

def extract_directions(text: str) -> List[str]:
    """
    从文本中提取“方向”列表，若无明确分段则返回全文作为单个方向。
    """
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


class SemanticEvaluator:
    def __init__(self, embedding_name: str, cache_dir: Optional[str] = None, batch_size: int = 32):
        logging.info(f"加载语义嵌入模型: {embedding_name}")
        # SentenceTransformer 自动选择设备（device="auto"）
        self.embedder = SentenceTransformer(embedding_name, device="cuda", cache_folder=cache_dir)
        self.batch_size = batch_size

    def compute_batch_scores(self, records: List[Dict], threshold: float = 0.5):
        """
        对一批记录计算语义匹配指标（Macro 与 Micro），并在每条记录上添加相关语义字段。
        返回聚合指标字典。
        """
        all_gt_texts = []
        all_pred_texts = []
        gt_segments = []
        pred_segments = []

        # 1. 预处理：将每条记录拆分为 direction 列表并记录索引区间
        for rec in records:
            gt_dirs = extract_directions(rec["label"])
            pred_dirs = extract_directions(rec["prediction"])

            gt_segments.append((len(all_gt_texts), len(gt_dirs)))
            pred_segments.append((len(all_pred_texts), len(pred_dirs)))

            all_gt_texts.extend(gt_dirs)
            all_pred_texts.extend(pred_dirs)

            # 暂存该样本的方向数量，便于调试与追溯
            rec["semantic_meta"] = {"gt_count": len(gt_dirs), "pred_count": len(pred_dirs)}

        # 2. 批量编码（如果列表为空则返回 None）
        gt_embeddings = self.embedder.encode(all_gt_texts, convert_to_tensor=True, batch_size=self.batch_size) if all_gt_texts else None
        pred_embeddings = self.embedder.encode(all_pred_texts, convert_to_tensor=True, batch_size=self.batch_size) if all_pred_texts else None

        # 3. 计算逐样本指标并累加用于 Micro 计算
        total_pred_matches = 0.0
        total_gt_matches = 0.0
        total_pred_count = 0
        total_gt_count = 0

        scores_list = []

        for idx, rec in enumerate(records):
            gt_start, gt_len = gt_segments[idx]
            pred_start, pred_len = pred_segments[idx]

            if gt_len == 0 or pred_len == 0:
                s_recall = 0.0
                s_precision = 0.0
                sample_recall_sum = 0.0
                sample_precision_sum = 0.0
            else:
                if gt_embeddings is None or pred_embeddings is None:
                    raise ValueError("嵌入张量为空（None），但对应方向数量不为零。请检查编码阶段。")
                gt_embeds = gt_embeddings[gt_start: gt_start + gt_len]
                pred_embeds = pred_embeddings[pred_start: pred_start + pred_len]

                sim_matrix = util.cos_sim(pred_embeds, gt_embeds)

                recall_vals = sim_matrix.max(dim=0).values
                recall_vals[recall_vals < threshold] = 0.0
                sample_recall_sum = recall_vals.sum().item()

                precision_vals = sim_matrix.max(dim=1).values
                precision_vals[precision_vals < threshold] = 0.0
                sample_precision_sum = precision_vals.sum().item()

                s_recall = sample_recall_sum / gt_len
                s_precision = sample_precision_sum / pred_len

            s_f1 = 0.0 if (s_precision + s_recall) == 0 else (2 * s_precision * s_recall / (s_precision + s_recall))

            rec["semantic_score"] = s_f1
            rec["semantic_precision"] = s_precision
            rec["semantic_recall"] = s_recall

            scores_list.append(s_f1)

            # 累加用于 Micro 指标计算
            total_gt_matches += sample_recall_sum
            total_gt_count += gt_len
            total_pred_matches += sample_precision_sum
            total_pred_count += pred_len

        # 计算 Macro 与 Micro 聚合指标
        macro_prec = sum([rec["semantic_precision"] for rec in records]) / len(records) if records else 0.0
        macro_rec = sum([rec["semantic_recall"] for rec in records]) / len(records) if records else 0.0
        macro_f1 = sum(scores_list) / len(scores_list) if scores_list else 0.0

        micro_prec = total_pred_matches / total_pred_count if total_pred_count > 0 else 0.0
        micro_rec = total_gt_matches / total_gt_count if total_gt_count > 0 else 0.0
        micro_f1 = 0.0 if (micro_prec + micro_rec) == 0 else (2 * micro_prec * micro_rec / (micro_prec + micro_rec))

        return {
            "semantic_macro_precision": macro_prec,
            "semantic_macro_recall": macro_rec,
            "semantic_macro_f1": macro_f1,
            "semantic_micro_f1": micro_f1,
            "semantic_micro_precision": micro_prec,
            "semantic_micro_recall": micro_rec
        }

def compute_semantic(records: List[Dict], embedding_name: Optional[str] = None, cache_dir: Optional[str] = None, batch_size: int = 32, threshold: float = 0.5):
    """
    语义指标调用接口
    """
    if embedding_name is None:
        embedding_name = "Qwen/Qwen3-Embedding-0.6B"
    evaluator = SemanticEvaluator(embedding_name, cache_dir, batch_size)
    return evaluator.compute_batch_scores(records, threshold=threshold)