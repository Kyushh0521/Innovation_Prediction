# adv_evaluator.py
import argparse
import os
import json
import logging
import torch
import gc
from tqdm import tqdm
from utils import init_logging, load_json, load_config, load_model_and_tokenizer, get_generate_fn
import generation_metrics as generation_metrics
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019, DeepWordBugGao2018
from textattack.goal_functions import MinimizeBleu

class BatchHuggingFaceModelWrapper(HuggingFaceModelWrapper):
    def __init__(self, model, tokenizer, batch_size=1):
        super().__init__(model, tokenizer)
        self.batch_size = batch_size

    def __call__(self, text_input_list):
        # 显存保护：手动分批推理
        outputs = []
        for i in range(0, len(text_input_list), self.batch_size):
            batch = text_input_list[i : i + self.batch_size]
            with torch.no_grad():
                batch_output = super().__call__(batch)
            outputs.append(batch_output)
            torch.cuda.empty_cache() # 关键：每批次清理
        return torch.cat(outputs, dim=0)

class MedAdvPipeline:
    def __init__(self, model, tokenizer, cfg: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        # 包装模型以适配 TextAttack
        self.model_wrapper = BatchHuggingFaceModelWrapper(model, tokenizer, batch_size=1)
        
        # 初始化攻击算法
        self.recipes = {
            "word_level": TextFoolerJin2019.build(self.model_wrapper),
            "char_level": DeepWordBugGao2018.build(self.model_wrapper)
        }

        for name, recipe in self.recipes.items():
            recipe.goal_function = MinimizeBleu(self.model_wrapper, target_bleu=0.0)

    def run_adv_test(self, clean_data: list, output_dir: str):
        """
        执行完整的对抗评估流程：攻击生成 -> 模型预测 -> 语义比对 -> ASR 统计
        """
        summary_report = {}
        cutoff_len = self.cfg.get("cutoff_len", 2048)
        generate_fn = get_generate_fn(self.model, self.tokenizer, cutoff_len)
        threshold = self.cfg.get("semantic_threshold", 0.5)

        for name, recipe in self.recipes.items():
            logging.info(f"\n>>> 正在针对算法 [{name}] 进行对抗评估...")
            adv_results = []
            
            for item in tqdm(clean_data, desc=f"对抗攻击与推理 ({name})"):
                try:
                    # 1. 寻找对抗干扰项
                    # 注意：TextAttack 需要 (input, output) 进行目标导向搜索
                    attack_res = recipe.attack(item['input'], item['output'])
                    adv_input = attack_res.perturbed_result.attacked_text.text
                    
                    # 2. 模型实时预测响应
                    prediction = generate_fn(item.get("system", ""), item.get("instruction", ""), adv_input)
                    
                    adv_results.append({
                        "instruction": item.get("instruction", ""),
                        "original_input": item['input'],
                        "adversarial_input": adv_input,
                        "label": item['output'],
                        "prediction": prediction,
                        "attack_type": name,
                        "queries": attack_res.num_queries # 记录攻击代价
                    })

                    torch.cuda.empty_cache()
                    gc.collect()

                except Exception as e:
                    logging.warning(f"样本评估跳过: {e}")
                    continue

            # 3. 计算该对抗集下的语义指标
            # 为了节省显存，如果数据量大，建议在此处调用 compute_semantic
            logging.info(f"正在计算 [{name}] 的语义鲁棒性指标...")
            embed_model = self.cfg.get("embedding_name_or_path", "Qwen/Qwen3-Embedding-0.6B")
            metrics = generation_metrics.compute_semantic(
                adv_results, 
                embedding_name=embed_model,
                batch_size=self.cfg.get("embed_batch_size", 32)
            )

            # 4. 计算 ASR (攻击成功率)
            # 定义：语义相似度低于阈值的样本占比
            success_count = sum(1 for r in adv_results if r["semantic_similarity"] < threshold)
            asr = success_count / len(adv_results) if adv_results else 0
            
            summary_report[f"{name}_asr"] = asr
            summary_report[f"{name}_avg_sim"] = metrics["semantic_similarity"]
            summary_report[f"{name}_avg_queries"] = sum(r["queries"] for r in adv_results) / len(adv_results)

            # 保存当前算法的详细 JSON
            out_path = os.path.join(output_dir, f"adv_details_{name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(adv_results, f, ensure_ascii=False, indent=2)

        return summary_report

def main():
    parser = argparse.ArgumentParser(description="独立对抗鲁棒性评估工具")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = load_config(args.config)
    output_dir = cfg.get("output_dir", "eval_outputs/adv_eval")
    os.makedirs(output_dir, exist_ok=True)
    
    init_logging(output_dir, log_name="adv_eval.log")
    
    # 2. 加载模型 (必须合并 LoRA 权重)
    model_cfg = {
        "model_name_or_path": cfg.get("model_name_or_path"),
        "adapter_name_or_path": cfg.get("adapter_name_or_path"),
        "model_cache": cfg.get("model_cache")
    }
    # 强制进行 merge_and_unload，因为对抗搜索需要完整的参数矩阵
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    
    # 注意：此处需确保 utils.py 中的 load_model_and_tokenizer 支持 merge 逻辑
    # 如果暂未修改 utils.py，建议在此处手动执行合并：
    if hasattr(model, "merge_and_unload"):
        logging.info("正在执行 LoRA 权重合并...")
        model = model.merge_and_unload() #type: ignore

    # 3. 加载原始干净数据集
    data_path = cfg.get("dataset_path")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("配置项 'dataset_path' 未设置或类型不为字符串，请在配置文件中指定数据集路径")
    clean_data = load_json(data_path)
    logging.info(f"加载原始数据集共 {len(clean_data)} 条样本")

    # 4. 运行对抗流水线
    pipeline = MedAdvPipeline(model, tokenizer, cfg)
    summary = pipeline.run_adv_test(clean_data, output_dir)

    # 5. 保存摘要
    summary_path = os.path.join(output_dir, "adv_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(f"====== 对抗评估完成 ======")
    for k, v in summary.items():
        logging.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()