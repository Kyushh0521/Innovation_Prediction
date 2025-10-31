import os
import sys
import pandas as pd

# 临时脚本：按 contect_type 将数据拆分为成果与需求两张表
# 约定：
# - 输入：Dataset/needs_achievements_full.xlsx（sheet=0）
# - 字段：contect_type，值为字符串 '1' 代表成果，'2' 代表需求（其他值忽略）
# - 输出：Dataset/achievements_full.xlsx 与 Dataset/needs_full.xlsx

INPUT_PATH = "Dataset/needs_achievements_full.xlsx"
SHEET = 0
OUT_ACHIEVEMENTS = "Dataset/achievements_full.xlsx"
OUT_NEEDS = "Dataset/needs_full.xlsx"


def normalize_contect_type(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # 有些数据可能是 float，如 1.0/2.0
    if s.endswith('.0'):
        s = s[:-2]
    return s


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"输入文件不存在: {INPUT_PATH}")
        sys.exit(2)

    try:
        df = pd.read_excel(INPUT_PATH, sheet_name=SHEET, dtype=str, keep_default_na=False)
    except Exception as e:
        print("读取 Excel 失败:", e)
        sys.exit(3)

    if 'contect_type' not in df.columns:
        print("错误: 未找到列 'contect_type'")
        sys.exit(4)

    ct = df['contect_type'].map(normalize_contect_type)
    ach_mask = ct == '1'
    needs_mask = ct == '2'

    df_ach = df.loc[ach_mask].copy()
    df_needs = df.loc[needs_mask].copy()

    # 输出目录确保存在
    for out in (OUT_ACHIEVEMENTS, OUT_NEEDS):
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    try:
        df_ach.to_excel(OUT_ACHIEVEMENTS, index=False)
        df_needs.to_excel(OUT_NEEDS, index=False)
    except Exception as e:
        print("保存输出失败:", e)
        sys.exit(5)

    print("拆分完成")
    print(f"总行数: {len(df)}")
    print(f"成果(contect_type=1): {len(df_ach)} -> {OUT_ACHIEVEMENTS}")
    print(f"需求(contect_type=2): {len(df_needs)} -> {OUT_NEEDS}")
    other = len(df) - len(df_ach) - len(df_needs)
    if other:
        print(f"其余(contect_type 不是 1/2): {other} 行，未写出")


if __name__ == '__main__':
    main()
