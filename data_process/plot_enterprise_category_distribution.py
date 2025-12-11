# -*- coding: utf-8 -*-
"""
Plot distribution of enterprise industries (by `category` field) from the
cleaned enterprises Excel file (`Dataset/enterprises_full_cleaned.xlsx`).

Outputs:
 - data_process_outputs/enterprise_category_distribution.png  (bar chart)

Usage:
    python data_process/plot_enterprise_category_distribution.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys


def _ensure_chinese_font():
    """确保 matplotlib 使用可显示中文的字体。
    尝试常见的 Windows 字体: Microsoft YaHei, SimHei, SimSun；若找不到则使用 matplotlib 默认并打印警告。
    """
    # 常见中文字体优先列表（Windows 优先）
    candidates = [
        "Microsoft YaHei",  # 微软雅黑
        "Microsoft YaHei UI",
        "SimHei",           # 黑体
        "SimSun",           # 宋体
        "PingFang SC",      # macOS 常见
        "Heiti SC",
    ]

    for name in candidates:
        try:
            font_path = fm.findfont(name, fallback_to_default=False)
            if os.path.exists(font_path):
                plt.rcParams['font.family'] = name
                plt.rcParams['axes.unicode_minus'] = False
                return name
        except Exception:
            # findfont 可能在某些系统上抛出异常，忽略并继续尝试其他名字
            continue

    # fallback: 使用系统默认字体，但关闭 unicode_minus 以避免负号显示为方块
    plt.rcParams['axes.unicode_minus'] = False
    print("警告: 未检测到常见中文字体，图表可能仍显示乱码。可在系统上安装 'Microsoft YaHei' 或 'SimHei'，或设置环境变量 MATPLOTLIBRC 指向包含中文字体的 rc 文件。", file=sys.stderr)
    return None


def _search_and_add_font():
    """更激进地搜索系统字体目录并尝试加载已知中文字体文件（按常见 Windows 文件名）。
    返回实际设置的字体名称或 None。
    """
    # 允许通过环境变量指定字体文件路径（优先）
    env_fp = os.environ.get('PLOT_CHINESE_FONT_PATH') or os.environ.get('MATPLOTLIB_CHINESE_FONT')
    if env_fp and os.path.exists(env_fp):
        try:
            fm.fontManager.addfont(env_fp)
            prop = fm.FontProperties(fname=env_fp)
            name = prop.get_name()
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用环境变量指定字体: {env_fp} -> {name}")
            return name
        except Exception as e:
            print(f"尝试加载环境字体失败: {env_fp} -> {e}")

    # 常见 Windows 字体文件名
    candidate_files = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\msyhbd.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
    ]

    for fp in candidate_files:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                prop = fm.FontProperties(fname=fp)
                name = prop.get_name()
                plt.rcParams['font.family'] = name
                plt.rcParams['axes.unicode_minus'] = False
                print(f"检测到字体文件: {fp} -> 使用字体: {name}")
                return name
            except Exception as e:
                print(f"加载字体文件失败: {fp} -> {e}")
                continue

    # 尝试列出系统字体并寻找中文名字体
    try:
        sys_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        for fp in sys_fonts:
            try:
                prop = fm.FontProperties(fname=fp)
                name = prop.get_name()
                # 常见中文字体名关键词
                if any(k in name for k in ['Hei', 'Sim', 'YaHei', 'Song', 'Kai']):
                    fm.fontManager.addfont(fp)
                    plt.rcParams['font.family'] = name
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"在系统字体中找到合适字体: {fp} -> {name}")
                    return name
            except Exception:
                continue
    except Exception:
        pass

    return None


def load_enterprises(xlsx_path: Path):
    """Load enterprises from a single xlsx path. Raises FileNotFoundError if missing.

    This script assumes the cleaned enterprises table is provided as an Excel
    file at `Dataset/enterprises_full_cleaned.xlsx`.
    """
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_excel(p)
    return df, p


def plot_category_distribution(df: pd.DataFrame, category_col: str = 'category', out_dir: Path = Path('data_process_outputs')):
    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not found in dataframe")

    # Normalize category values: strip, convert to str, treat empty as NaN
    cats = df[category_col].astype(str).str.strip()
    cats = cats.replace({'nan': None, 'None': None})
    cats = cats.dropna()

    counts = cats.value_counts()
    if counts.empty:
        raise ValueError('No categories found to plot')

    out_dir.mkdir(parents=True, exist_ok=True)

    # 聚合为前5 + 其余为“其他”，并绘制饼状图
    top_n = 5
    if len(counts) > top_n:
        top = counts.iloc[:top_n].copy()
        other_sum = counts.iloc[top_n:].sum()
        # 使用中文“其他”标签以便在中文环境下更直观
        top['其他'] = other_sum
        plot_counts = top
    else:
        plot_counts = counts.copy()

    # 字体与样式设置（更适合缩小后阅读的 PPT）
    title_fs = 18
    label_fs = 14
    tick_fs = 12
    value_fs = 12

    labels = [str(s) for s in plot_counts.index]
    sizes = plot_counts.values.tolist()

    # 为饼图选择方形画布，大小可适当调整
    fig_size = (8, 8)
    fig, ax = plt.subplots(figsize=fig_size)

    # 将标签与占比直接显示在饼图上，删除右侧图例以节省空间（便于 PPT 使用）
    def make_autopct(values):
        def autopct(pct):
            total = sum(values)
            # 四舍五入为整数计数
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({val})"
        return autopct

    pie_result = ax.pie(
        sizes,
        labels=labels,
        startangle=140,
        autopct=make_autopct(sizes),
        pctdistance=0.6,
        labeldistance=1.05,
        textprops={'fontsize': label_fs}
    )

    # matplotlib 的不同版本返回长度为2或3的 tuple，统一兼容处理
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
    else:
        wedges, texts = pie_result
        autotexts = []

    # 设置文本大小（标签 + 百分比）
    for t in texts:
        try:
            t.set_fontsize(label_fs)
        except Exception:
            pass
    for at in autotexts:
        try:
            at.set_fontsize(value_fs)
        except Exception:
            pass

    # 保持饼图为圆形
    ax.axis('equal')
    # 不在切片处单独放标题，而是在图像底部居中显示中文标题，便于 PPT 布局
    title_text = '企业类别分布（前5类 + 其他）'
    # y=0.02 放在图下方，若需要可以调整为更低的数值
    fig.text(0.5, 0.02, title_text, ha='center', va='center', fontsize=title_fs)

    out_png = out_dir / 'enterprise_category_distribution_pie.png'
    # 高 DPI 保存以便在 PPT 中缩放仍然清晰
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close()

    return out_png


def main():
    src = Path('Dataset/enterprises_full_cleaned.xlsx')
    try:
        df, used = load_enterprises(src)
    except FileNotFoundError:
        print(f'Missing input file: {src}. Please provide cleaned enterprises Excel at this path.')
        return

    print(f'Loaded enterprises from: {used} (rows={len(df)})')

    try:
        # 尝试设置中文字体以避免图表中文乱码
        font_used = _ensure_chinese_font()
        if not font_used:
            font_used = _search_and_add_font()
        if font_used:
            print(f"使用字体: {font_used}")
        else:
            print("未能自动配置中文字体，图像可能仍显示乱码。可设置环境变量 PLOT_CHINESE_FONT_PATH 指向字体文件。")

        out_png = plot_category_distribution(df, category_col='category')
        print(f'Category distribution saved: {out_png}')
    except Exception as e:
        print('Error while plotting category distribution:', e)


if __name__ == '__main__':
    main()
