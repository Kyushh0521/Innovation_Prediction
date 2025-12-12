# plot_compare.py
"""
模型评估比较图脚本（中文界面）
功能：
1) 绘制语义六项指标的雷达对比图
2) 绘制 BLEU / ROUGE 四项指标的柱状对比图
3) 绘制 PPL 的柱状对比图

用法示例：
python plot_compare.py --config plot_config.yaml

输出为静态 PNG 图片，保存到指定目录。
"""
import os
import json
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import load_config

# 统一使用微软雅黑字体显示中文，并确保负号正常显示
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

SEMANTIC_KEYS = [
    "semantic_macro_precision",
    "semantic_macro_recall",
    "semantic_macro_f1",
    "semantic_micro_precision",
    "semantic_micro_recall",
    "semantic_micro_f1",
]
BLEU_KEYS = ["bleu-4", "rouge-1", "rouge-2", "rouge-l"]
PPL_KEY = "ppl_mean"


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics_summary", {})


def to_percent_if_needed(v):
    """如果 v 在 0..1 之间，按百分比转换到 0..100；否则原样返回。"""
    try:
        if v is None:
            return np.nan
        v = float(v)
        if 0.0 <= v <= 1.0:
            return v * 100.0
        return v
    except Exception:
        return np.nan


def make_radar(ax, categories, values_list, labels, colors=None):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # 闭合
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    # 找最大值用于统一刻度
    max_val = 0
    for vals in values_list:
        max_val = max(max_val, np.nanmax(vals))
    if max_val <= 0:
        max_val = 1.0

    ax.set_ylim(0, max_val * 1.1)

    for i, vals in enumerate(values_list):
        vals = list(vals) + [vals[0]]
        color = None if colors is None else colors[i % len(colors)]
        ax.plot(angles, vals, label=labels[i], color=color)
        ax.fill(angles, vals, alpha=0.2, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))


def plot_semantic(summaryA, summaryB, labelA, labelB, outpath):
    valsA = [to_percent_if_needed(summaryA.get(k)) for k in SEMANTIC_KEYS]
    valsB = [to_percent_if_needed(summaryB.get(k)) for k in SEMANTIC_KEYS]

    categories = [k.replace("semantic_", "") for k in SEMANTIC_KEYS]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    make_radar(ax, categories, [valsA, valsB], [labelA, labelB], colors=["tab:blue", "tab:orange"])
    plt.title("语义指标对比")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_bleu_rouge(summaryA, summaryB, labelA, labelB, outpath):
    valsA = [summaryA.get(k, np.nan) for k in BLEU_KEYS]
    valsB = [summaryB.get(k, np.nan) for k in BLEU_KEYS]

    x = np.arange(len(BLEU_KEYS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, valsA, width, label=labelA)
    ax.bar(x + width/2, valsB, width, label=labelB)
    ax.set_xticks(x)
    ax.set_xticklabels(["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"])  # 保持缩写以便识别
    ax.set_ylabel("得分")
    ax.set_title("BLEU / ROUGE 指标对比")
    ax.legend()

    # 在每个柱上标注差值
    for i,(a,b) in enumerate(zip(valsA, valsB)):
        try:
            if np.isnan(a) or np.isnan(b):
                continue
            diff = b - a
            ax.text(i, max(a,b) + 0.5, f"{diff:+.2f}", ha='center')
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_ppl(summaryA, summaryB, labelA, labelB, outpath):
    a = summaryA.get(PPL_KEY, None)
    b = summaryB.get(PPL_KEY, None)
    vals = [a if a is not None else np.nan, b if b is not None else np.nan]
    labels = [labelA, labelB]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(2)
    ax.bar(x, vals, color=["tab:blue", "tab:orange"] )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PPL")
    ax.set_title("PPL 对比")

    for i,v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + max(0.01, 0.01 * v), f"{v:.4f}", ha='center')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="模型评估可视化（使用 YAML 配置）")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pre_path = cfg.get("pre_path")
    post_path = cfg.get("post_path")
    outdir = cfg.get("output_path", "eval_results/plots")

    os.makedirs(outdir, exist_ok=True)

    summaryA = load_summary(pre_path)
    summaryB = load_summary(post_path)

    # semantic 雷达
    out_sem = os.path.join(outdir, f"Pre_vs_Post_semantic_radar.png")
    plot_semantic(summaryA, summaryB, "Pre", "Post", out_sem)

    # BLEU/ROUGE 柱状
    out_bleu = os.path.join(outdir, f"Pre_vs_Post_bleu_rouge.png")
    plot_bleu_rouge(summaryA, summaryB, "Pre", "Post", out_bleu)

    # PPL 柱状
    out_ppl = os.path.join(outdir, f"Pre_vs_Post_ppl.png")
    plot_ppl(summaryA, summaryB, "Pre", "Post", out_ppl)

    print(f"已保存图表到目录: {outdir}")

if __name__ == "__main__":
    main()
