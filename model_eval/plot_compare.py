# plot_compare.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import load_config

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

SEMANTIC_KEYS = ["semantic_similarity"]
BLEU_KEYS = ["bleu-4", "rouge-1", "rouge-2", "rouge-l"]
BERTSCORE_KEYS = ["bert_precision", "bert_recall", "bert_f1"]
PPL_KEY = "ppl_mean"

def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics_summary", {})

def plot_semantic(summaryA, summaryB, labelA, labelB, outpath):
    valA = summaryA.get("semantic_similarity", 0.0)
    valB = summaryB.get("semantic_similarity", 0.0)
    
    valA = valA * 100 if valA is not None else 0
    valB = valB * 100 if valB is not None else 0
    
    vals = [valA, valB]
    labels = [labelA, labelB]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(2)
    bars = ax.bar(x, vals, color=["tab:blue", "tab:orange"])
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Semantic Similarity (%)")
    ax.set_title("语义相似度对比")
    ax.set_ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
                
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_bleu_rouge(summaryA, summaryB, labelA, labelB, outpath):
    valsA = [summaryA.get(k, 0) for k in BLEU_KEYS]
    valsB = [summaryB.get(k, 0) for k in BLEU_KEYS]

    x = np.arange(len(BLEU_KEYS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, valsA, width, label=labelA)
    ax.bar(x + width/2, valsB, width, label=labelB)
    ax.set_xticks(x)
    ax.set_xticklabels(["BLEU-4", "R-1", "R-2", "R-L"])
    ax.set_ylabel("Score")
    ax.set_title("BLEU / ROUGE 对比")
    ax.legend()

    for i, (a, b) in enumerate(zip(valsA, valsB)):
        diff = b - a
        ax.text(i, max(a, b) + 1, f"{diff:+.1f}", ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_bertscore(summaryA, summaryB, labelA, labelB, outpath):
    valsA = [summaryA.get(k, 0) for k in BERTSCORE_KEYS]
    valsB = [summaryB.get(k, 0) for k in BERTSCORE_KEYS]
    
    # 转换为百分比
    valsA = [v * 100 for v in valsA]
    valsB = [v * 100 for v in valsB]

    x = np.arange(len(BERTSCORE_KEYS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, valsA, width, label=labelA)
    ax.bar(x + width/2, valsB, width, label=labelB)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylabel("BERTScore (%)")
    ax.set_title("BERTScore 对比")
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)

    for i, (a, b) in enumerate(zip(valsA, valsB)):
        diff = b - a
        ax.text(i, max(a, b) + 1, f"{diff:+.2f}", ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_ppl(summaryA, summaryB, labelA, labelB, outpath):
    a = summaryA.get(PPL_KEY, None)
    b = summaryB.get(PPL_KEY, None)
    vals = [a if a else 0, b if b else 0]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(2)
    ax.bar(x, vals, color=["tab:blue", "tab:orange"])
    ax.set_xticks(x)
    ax.set_xticklabels([labelA, labelB])
    ax.set_ylabel("PPL")
    ax.set_title("PPL 对比 (越低越好)")

    for i, v in enumerate(vals):
        if v > 0:
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    pre_path = cfg.get("pre_path")
    post_path = cfg.get("post_path")
    outdir = cfg.get("output_path", "eval_results/plots")
    os.makedirs(outdir, exist_ok=True)

    summaryA = load_summary(pre_path)
    summaryB = load_summary(post_path)

    plot_semantic(summaryA, summaryB, "Pre", "Post", os.path.join(outdir, "semantic_compare.png"))
    plot_bleu_rouge(summaryA, summaryB, "Pre", "Post", os.path.join(outdir, "bleu_rouge_compare.png"))
    plot_bertscore(summaryA, summaryB, "Pre", "Post", os.path.join(outdir, "bertscore_compare.png"))
    plot_ppl(summaryA, summaryB, "Pre", "Post", os.path.join(outdir, "ppl_compare.png"))

    print(f"图表已保存至: {outdir}")

if __name__ == "__main__":
    main()