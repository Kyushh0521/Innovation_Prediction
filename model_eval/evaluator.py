# evaluator.py
import argparse
import os
import json
import logging
from tqdm import tqdm
import torch
import gc

# 引入自定义模块
from utils import init_logging, load_json, load_config, load_model_and_tokenizer, get_generate_fn
import model_eval.generation_metrics as generation_metrics

def main():
    parser = argparse.ArgumentParser(description="统一模型评估工具")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = load_config(args.config)
    
    # 路径与基础设置
    dataset_path = cfg.get("dataset_path", "model_eval/sample_sft_test.json")
    output_dir = cfg.get("output_dir", "eval_results")
    run_label = cfg.get("run_label", "combined_eval")
    # 获取语言配置 (仅用于 BERTScore)
    eval_lang = cfg.get("language", "en")
    os.makedirs(output_dir, exist_ok=True)
    
    init_logging(output_dir, log_name=f"{run_label}.log")
    logging.info(f"配置加载完成: {args.config}")
    logging.info(f"配置摘要: 数据集={dataset_path}, 输出目录={output_dir}, 运行标识={run_label}")
    
    # 开关
    do_ppl = cfg.get("eval_ppl", False)
    do_bleu_rouge = cfg.get("eval_bleu_rouge", False)
    do_semantic = cfg.get("eval_semantic", False)
    do_bertscore = cfg.get("eval_bertscore", False)

    # 根据配置打印将要执行的评估项和简要配置摘要，便于在日志中快速查看
    enabled = []
    if do_ppl: enabled.append("PPL")
    if do_bleu_rouge: enabled.append("BLEU/ROUGE")
    if do_semantic: enabled.append("Semantic")
    if do_bertscore: enabled.append("BERTScore")

    logging.info(f"将要运行的评估: {', '.join(enabled) if enabled else '无'}")
    
    
    # 2. 数据准备
    raw_data = load_json(dataset_path)
    logging.info(f"加载数据集: {len(raw_data)} 条样本")

    results = []
    # 将原始数据转入结果列表，保持结构
    for item in raw_data:
        results.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "label": item.get("output", ""),
            "system": item.get("system", "")
        })

    # 3. 加载 LLM 模型 (如果任何评估需要用到 LLM)
    # 注意：PPL 和 生成 都需要 LLM
    if do_ppl or do_bleu_rouge or do_semantic or do_bertscore:
        model_cfg = {
            "model_name_or_path": cfg.get("model_name_or_path"),
            "adapter_name_or_path": cfg.get("adapter_name_or_path"),
            "model_cache": cfg.get("model_cache")
        }
        model, tokenizer = load_model_and_tokenizer(model_cfg)
        cutoff_len = cfg.get("cutoff_len", 2048)
    else:
        logging.warning("未开启任何需要模型的评估任务，程序结束。")
        return

    # ==========================================
    # Phase 1: PPL 评估 (不需要生成，直接算Loss)
    # ==========================================
    ppl_scores = []
    if do_ppl:
        logging.info(">>> 开始 PPL 评估...")
        for res in tqdm(results, desc="计算 PPL"):
            ppl = generation_metrics.compute_ppl(
                model, tokenizer,
                res["system"], res["instruction"], res["input"], res["label"],
                cutoff_len
            )
            res["ppl"] = ppl
            if ppl is not None:
                ppl_scores.append(ppl)
        
        mean_ppl = sum(ppl_scores)/len(ppl_scores) if ppl_scores else 0
        logging.info(f"平均 PPL: {mean_ppl:.4f}")

    # ==========================================
    # Phase 2: 文本生成 (如果需要 BLEU/ROUGE 或 Semantic)
    # ==========================================
    if do_bleu_rouge or do_semantic:
        logging.info(">>> 开始模型生成...")
        generate_fn = get_generate_fn(model, tokenizer, cutoff_len)
        
        for res in tqdm(results, desc="生成响应"):
            try:
                pred = generate_fn(res["system"], res["instruction"], res["input"])
                res["prediction"] = pred
                res["gen_error"] = None
            except Exception as e:
                logging.error(f"生成失败: {e}")
                res["prediction"] = ""
                res["gen_error"] = str(e)
    
    # ==========================================
    # Phase 3: 基于生成的指标计算 (BLEU/ROUGE)
    # ==========================================
    bleu_rouge_agg = {}
    if do_bleu_rouge:
        logging.info(">>> 开始 BLEU/ROUGE 计算...")
        total_metrics = {"bleu-4": 0.0, "rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        valid_count = 0
        
        for res in tqdm(results, desc="文本评分"):
            pred = res.get("prediction", "")
            label = res.get("label", "")
            
            scores = generation_metrics.compute_bleu_rouge(pred, label)
            res["text_metrics"] = scores
            
            for k in total_metrics:
                total_metrics[k] += scores[k]
            valid_count += 1
            
        bleu_rouge_agg = {k: v/valid_count for k, v in total_metrics.items()} if valid_count > 0 else total_metrics
        for k, v in bleu_rouge_agg.items():
            logging.info(f"{k.upper()}: {v:.4f}")


    if do_semantic or do_bertscore:
        try:
            logging.info("释放 LLM 显存：删除 model 和 tokenizer，并触发 GC 与 CUDA cache 清理")
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.warning(f"释放显存时发生异常: {e}")


    # ==========================================
    # Phase 4: BERTScore
    # ==========================================
    bertscore_agg = {}
    if do_bertscore:
        logging.info(f">>> 开始 BERTScore 评估 (Language={eval_lang})...")
        bertscore_model = cfg.get("bertscore_model", "microsoft/deberta-xlarge-mnli")
        bertscore_agg = generation_metrics.compute_bertscore(
            results,
            lang=eval_lang, 
            batch_size=cfg.get("embed_batch_size", 32),
            model_type=bertscore_model
        )
        for k, v in bertscore_agg.items():
            logging.info(f"{k}: {v:.4f}")

    # ==========================================
    # Phase 5: 语义相似度计算 (Semantic)
    # ==========================================
    semantic_agg = {}
    if do_semantic:
        logging.info(">>> 开始语义评估 (Semantic)...")
        embed_model_name = cfg.get("embedding_name_or_path", "Qwen/Qwen3-Embedding-0.6B")
        embed_cache = cfg.get("embedding_cache")
        embed_bs = cfg.get("embed_batch_size", 32)

        semantic_agg = generation_metrics.compute_semantic(results, embedding_name=embed_model_name, cache_dir=embed_cache, batch_size=embed_bs)
        
        logging.info(f"平均语义相似度: {semantic_agg['semantic_similarity']:.4f}")

    # ==========================================
    # Phase 5: 结果汇总与保存
    # ==========================================
    final_summary = {
        "run_label": run_label,
        "config": args.config,
        "language": eval_lang,
        "metrics_summary": {
            "ppl_mean": sum(ppl_scores)/len(ppl_scores) if ppl_scores else None,
            **bleu_rouge_agg,
            **bertscore_agg,
            **semantic_agg
        }
    }
    
    # 保存详细结果
    out_json_path = os.path.join(output_dir, f"{run_label}_details.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # 保存摘要结果
    out_summary_path = os.path.join(output_dir, f"{run_label}_summary.json")
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
        
    logging.info(f"====== 评估全部完成 ======")
    logging.info(f"详细结果: {out_json_path}")
    logging.info(f"摘要结果: {out_summary_path}")

if __name__ == "__main__":
    main()