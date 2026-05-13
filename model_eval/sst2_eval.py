import os
import yaml
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import argparse

# 引用您在 trainer.py 中定义的类
from sst2dataset import SST2ClassificationDataset, get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="SST2 模型评估脚本")
    parser.add_argument("--config", "-c", type=str, help="配置文件路径 (YAML 格式)")
    return parser.parse_args()

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def plot_heatmap(cm, output_dir, img_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - SST2')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    path = os.path.join(output_dir, img_name)
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def format_report_percent(report_dict):
    """
    将原始指标字典转换为百分制并保留两位小数的字符串表格
    """
    header = f"{'':>15} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>10}"
    rows = []
    
    # 需要处理的类别标签
    labels = ['Negative', 'Positive', 'macro avg', 'weighted avg']
    
    for label in labels:
        if label not in report_dict: continue
        metrics = report_dict[label]
        # 数值 * 100 并保留两位小数
        row = (f"{label:>15} {metrics['precision']*100:>12.2f} "
               f"{metrics['recall']*100:>12.2f} {metrics['f1-score']*100:>12.2f} "
               f"{int(metrics['support']):>10}")
        rows.append(row)
        if label == 'Positive': rows.append("") # 类别与平均值之间留空行
    
    # 准确率行
    acc = report_dict['accuracy']
    rows.insert(3, f"{'accuracy':>15} {'':>12} {'':>12} {acc*100:>12.2f} {int(report_dict['macro avg']['support']):>10}")
    
    return header + "\n" + "\n".join(rows)

def main():
    # 1. 加载配置
    args = parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.get("output_dir"), exist_ok=True)

    # 2. 打印加载信息
    print(f"正在加载分词器: {cfg.get('model_name_or_path')} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("model_name_or_path"), cache_dir=cfg.get("model_cache"))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    print(f"正在加载基座模型并合并 LoRA 权重 ...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.get("model_name_or_path"),
        num_labels=cfg.get("num_labels"),
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cfg.get("model_cache")
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.get("adapter_name_or_path"):
        model = PeftModel.from_pretrained(base_model, cfg["adapter_name_or_path"])
    else:
        model = base_model
    model.eval()

    # 3. 加载数据
    print(f"\n正在对 {cfg.get('dataset_path')} 进行评估...")
    test_ds = SST2ClassificationDataset(cfg.get("dataset_path"), tokenizer, cfg)
    test_dl = get_dataloader(test_ds, tokenizer, cfg, shuffle=False)

    all_preds, all_labels = [], []

    # 4. 推理循环
    with torch.no_grad():
        for batch in tqdm(test_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 生成报告与矩阵
    target_labels = [0, 1]
    target_names = ['Negative', 'Positive']
    report_dict = classification_report(all_labels, all_preds, labels=target_labels, target_names=target_names, output_dict=True)
    # 格式化输出文本
    percent_report = format_report_percent(report_dict)
    cm = confusion_matrix(all_labels, all_preds, labels=target_labels)
    acc = accuracy_score(all_labels, all_preds)
    
    # --- 格式化终端输出 ---
    print("\n==================== 评估报告 ====================")
    print(percent_report)
    print(f"\nOverall Accuracy: {acc*100:.2f}%") # 保留两位小数
    print("\n混淆矩阵:")
    print(cm)
    print("================================================\n")
    
    # 6. 可视化与保存
    run_label = cfg.get("run_label", "pre")
    output_dir = cfg.get("output_dir", "eval_outputs")
    os.makedirs(output_dir, exist_ok=True)
    img_name = f"confusion_matrix_{run_label}.png"
    json_name = f"test_evaluation_{run_label}.json"
    img_path = plot_heatmap(cm, output_dir, img_name)
    json_path = os.path.join(output_dir, json_name)
    
    # 保存结果时使用百分制字符串
    save_data = {
        "run_label": run_label,
        "accuracy_percent": f"{acc*100:.2f}%",
        "confusion_matrix": cm.tolist(),
        "report": percent_report
    }
    
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=4)
        
    print(f"混淆矩阵热力图已保存至: {img_path}")
    print(f"评估结果 JSON 已保存至: {json_path}")

if __name__ == "__main__":
    main()