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

    # Plot top categories (show all if <=20 else top 20 + rest aggregated)
    top_n = 20
    if len(counts) > top_n:
        top = counts.iloc[:top_n]
        other_sum = counts.iloc[top_n:].sum()
        top['Other'] = other_sum
        plot_counts = top
    else:
        plot_counts = counts

    plt.figure(figsize=(10, 6))
    plot_counts.sort_values().plot(kind='barh', color='#66b3ff')
    plt.xlabel('Number of enterprises')
    plt.title('Enterprise distribution by category')
    plt.tight_layout()
    out_png = out_dir / 'enterprise_category_distribution.png'
    plt.savefig(out_png, dpi=300)
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
        out_png = plot_category_distribution(df, category_col='category')
        print(f'Category distribution saved: {out_png}')
    except Exception as e:
        print('Error while plotting category distribution:', e)


if __name__ == '__main__':
    main()
