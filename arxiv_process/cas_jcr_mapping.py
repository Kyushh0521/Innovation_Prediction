import pandas as pd
import json
import sys
import os
import re
from sklearn.preprocessing import MinMaxScaler

# 配置区：按需填写
FILES = [
    {
    "xlsx_path": "Dataset\\JCR.xlsx",
    "json_path": "arxiv_process_outputs\\jcr_mapping.json",
        "journal_col": "Journal",
        "rank_col": "Quartile",
        "prefix": "JCR ",
        "suffix": "",
    },
    {
    "xlsx_path": "Dataset\\CAS.xlsx",
    "json_path": "arxiv_process_outputs\\cas_mapping.json",
        "journal_col": "Journal",
        "rank_col": "Rank",
        "prefix": "中科院 ",
        "suffix": " 区",
    },
]

def normalize_string(value: str) -> str:
    """去除首尾空格、换行符并将字符串转换为大写"""
    value = value.strip().replace("\n", "").replace("\r", "")  # 去除首尾空格和换行符
    return value.upper()  # 转换为大写

def build_mapping(xlsx_path: str, json_path: str,
                  journal_col: str, quartile_col: str,
                  prefix: str, suffix: str) -> None:
    """读取 Excel，提取指定列并添加前后缀，写入 JSON 文件。"""
    try:
        df = pd.read_excel(xlsx_path, usecols=[journal_col, quartile_col])
    except ValueError as e:
        raise ValueError(f"读取 {xlsx_path} 时出错，请检查列名：{e}")
    
    # 去除首尾空格、换行符并将字符串转换为大写
    df[journal_col] = df[journal_col].apply(normalize_string)
    df = df.dropna(subset=[journal_col, quartile_col])

    mapping = {
        str(row[journal_col]): f"{prefix}{row[quartile_col]}{suffix}"
        for _, row in df.iterrows()
    }

    out_dir = os.path.dirname(json_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    print(f"已生成 {json_path}（共 {len(mapping)} 条记录）")

# def query_interactive():
#     """
#     交互式查询：每次从两个 JSON 文件读取映射并检索。
#     """
#     print("进入交互式查询，输入期刊名后回车查询，输入 exit 退出。")
#     while True:
#         name = input("期刊名> ").strip()
#         if name.lower() == "exit":
#             break
#         found = False
#         for cfg in FILES:
#             try:
#                 with open(cfg["json_path"], "r", encoding="utf-8") as f:
#                     mapping = json.load(f)
#             except Exception as e:
#                 print(f"无法加载 {cfg['json_path']}: {e}", file=sys.stderr)
#                 continue
#             # 归一化处理查询名称，确保查询时不受大小写影响
#             normalized_name = normalize_string(name)
#             if normalized_name in mapping:
#                 print(f"{mapping[normalized_name]}")
#                 found = True
#         if not found:
#             print(f"未在任何映射中找到期刊 “{name}” 的等级信息")

def main():
    # 生成或更新 JSON 文件
    for cfg in FILES:
        try:
            build_mapping(
                cfg["xlsx_path"],
                cfg["json_path"],
                cfg["journal_col"],
                cfg["rank_col"],
                cfg.get("prefix", ""),
                cfg.get("suffix", ""),
            )
        except Exception as e:
            print(f"处理 {cfg['xlsx_path']} 时出错: {e}", file=sys.stderr)
            sys.exit(1)

    # # 进入查询模式
    # query_interactive()

if __name__ == "__main__":
    main()