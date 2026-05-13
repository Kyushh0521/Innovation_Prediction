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
ADV_ASR_KEYS = ["word_level_asr", "char_level_asr"]
ADV_QUERY_KEYS = ["word_level_avg_queries", "char_level_avg_queries"]

def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics_summary", {})

def plot_adv_asr(summaryA, summaryB, labelA, labelB, outpath):
    """
    绘制攻击成功率 (ASR) 对比图。ASR 越低，证明模型越鲁棒。
    """
    valsA = [summaryA.get(k, 0) * 100 for k in ADV_ASR_KEYS]
    valsB = [summaryB.get(k, 0) * 100 for k in ADV_ASR_KEYS]
    
    if sum(valsA) == 0 and sum(valsB) == 0:
        return

    x = np.arange(len(ADV_ASR_KEYS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, valsA, width, label=labelA, color="tab:gray")
    ax.bar(x + width/2, valsB, width, label=labelB, color="tab:red")
    
    ax.set_xticks(x)
    ax.set_xticklabels(["词级攻击 (TextFooler)", "字符级攻击 (DeepWordBug)"])
    ax.set_ylabel("攻击成功率 ASR (%) - 越低越安全")
    ax.set_title("对抗鲁棒性对比 (ASR)")
    ax.legend()
    ax.set_ylim(0, 105)
    
    for i, (a, b) in enumerate(zip(valsA, valsB)):
        diff = b - a
        ax.text(i, max(a, b) + 1, f"差值: {diff:+.1f}%", ha='center', fontsize=9, fontweight='bold')
                
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_adv_queries(summaryA, summaryB, labelA, labelB, outpath):
    """
    绘制平均查询次数对比图。查询次数越多，证明攻击难度越大，模型安全性越高。
    """
    valsA = [summaryA.get(k, 0) for k in ADV_QUERY_KEYS]
    valsB = [summaryB.get(k, 0) for k in ADV_QUERY_KEYS]

    if sum(valsA) == 0 and sum(valsB) == 0:
        return

    x = np.arange(len(ADV_QUERY_KEYS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, valsA, width, label=labelA, color="tab:gray")
    ax.bar(x + width/2, valsB, width, label=labelB, color="tab:blue")
    
    ax.set_xticks(x)
    ax.set_xticklabels(["词级平均查询", "字符级平均查询"])
    ax.set_ylabel("平均查询次数 - 越高难度越大")
    ax.set_title("攻击复杂度对比 (Avg Queries)")
    ax.legend()
    
    for i, (a, b) in enumerate(zip(valsA, valsB)):
        diff = b - a
        ax.text(i, max(a, b) + 1, f"{diff:+.1f}", ha='center', fontsize=9)
                
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

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
    plot_adv_asr(summaryA, summaryB, "Pre", "Post", os.path.join(outdir, "adv_asr_compare.png"))
    plot_adv_queries(summaryA, summaryB, "Pre", "Post", os.path.join(outdir, "adv_queries_compare.png"))

    print(f"图表已保存至: {outdir}")

if __name__ == "__main__":
    main()