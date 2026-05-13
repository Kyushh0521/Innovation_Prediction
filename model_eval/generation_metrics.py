# metrics_lib.py
import torch
import logging
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Optional, Tuple, cast
from bert_score import score as bs_score
import nltk
import torch.nn.functional as F
from torch import Tensor


def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("正在下载 NLTK punkt 分词数据...")
        try:
            nltk.download('punkt')
            nltk.download('punkt_tab')
        except Exception as e:
            logging.warning(f"NLTK punkt 下载失败: {e}。将回退到空格分词。")

def tokenize_text(text: str) -> List[str]:
    """
    强制使用 NLTK 进行分词
    """
    text = text.strip()
    if not text:
        return []
    try:
        return nltk.word_tokenize(text)
    except:
        # 万一 NLTK 失败，回退到 split
        return text.split()

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
    hypothesis = tokenize_text(prediction)
    reference = tokenize_text(label)

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
# 3. BERTScore 计算
# ===========================
def compute_bertscore(records: List[Dict], lang: str = "en", batch_size: int = 32, model_type: str = "/mnt/share/wwt/models/microsoft/deberta-v2-xlarge-mnli") -> Dict[str, float]:
    if bs_score is None:
        return {}
    
    predictions = [r.get("prediction", "") for r in records]
    references = [r.get("label", "") for r in records]

    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    logging.info(f"开始计算 BERTScore (device={device}, lang={lang})...")
    
    try:
        P, R, F1 = cast(Tuple[Tensor, Tensor, Tensor], bs_score(
            predictions, 
            references, 
            lang=lang, 
            verbose=True, 
            batch_size=batch_size, 
            device=device,
            model_type=model_type
        ))
        
        p_list = P.detach().cpu().tolist()
        r_list = R.detach().cpu().tolist()
        f1_list = F1.detach().cpu().tolist()
        
        for i, rec in enumerate(records):
            rec["bert_score"] = {
                "precision": p_list[i],
                "recall": r_list[i],
                "f1": f1_list[i]
            }
            
        return {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item()
        }
    except Exception as e:
        logging.error(f"BERTScore 计算失败: {e}")
        return {}


# ===========================
# 4. 语义相似度计算（Semantic）
# ===========================

class SemanticEvaluator:
    def __init__(self, embedding_name: str, cache_dir: Optional[str] = None, batch_size: int = 32):
        logging.info(f"加载语义嵌入模型: {embedding_name}")
        # SentenceTransformer 自动选择设备（device="auto"）
        self.embedder = SentenceTransformer(embedding_name, device="cuda", cache_folder=cache_dir)
        self.batch_size = batch_size

    def compute_batch_scores(self, records: List[Dict]):
        predictions = [r.get("prediction", "").strip() for r in records]
        labels = [r.get("label", "").strip() for r in records]

        pred_embeddings = self.embedder.encode(predictions, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=True)
        label_embeddings = self.embedder.encode(labels, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=True)

        cosine_scores = F.cosine_similarity(pred_embeddings, label_embeddings, dim=1)
        scores_list = cosine_scores.cpu().tolist()

        for i, r in enumerate(records):
            r["semantic_similarity"] = scores_list[i]

        mean_similarity = sum(scores_list) / len(scores_list) if scores_list else 0.0

        return {
            "semantic_similarity": mean_similarity
        }

def compute_semantic(records: List[Dict], embedding_name: Optional[str] = None, cache_dir: Optional[str] = None, batch_size: int = 32, threshold: float = 0.5):
    """
    语义指标调用接口
    """
    if embedding_name is None:
        embedding_name = "Qwen/Qwen3-Embedding-0.6B"
    evaluator = SemanticEvaluator(embedding_name, cache_dir, batch_size)
    return evaluator.compute_batch_scores(records)