import os
import re
import sys
from typing import List

import pandas as pd

# === 配置区域：将所有输入/输出和要删除的列在这里硬编码，按需修改 ===
# 输入文件（Workbook）
INPUT_PATH = "Dataset/experts_full.xlsx"
# 读取的 sheet（名称或索引）
SHEET = 0
# 输出文件路径
OUTPUT_PATH = "Dataset/experts_full_cleaned.xlsx"
# 要删除的无效列列表（编辑此处以指定要删除的列）
DROP_COLS: List[str] = ['id', 'user_id','image','btype_code','industry_id','sort','status','del_flag','remark','create_by','create_time','update_by','update_time','erweima','erweima2','technical_category','collect_num','biz_object_id','qrcode','rec','special','opened','wx_no','dock_man','customer','ncustomer','global_id','office_phone','achievements_honors','linkman','follow_people','is_activate','professional_title','college','label','email', 'phone','brief_intr']


def normalize_duty(duty: str) -> str:
    """将 duty 中用来分隔职位的字符统一为中文分号 '；'，并去重连续分号与首尾分号。"""
    if duty is None:
        return ""
    s = str(duty)
    # 将常见的分隔或连接符统一替换为分号；包括英文/中文逗号、分号、括号、以及任意空白
    s = re.sub(r"[\s,，;；()（）、/|]+", "；", s)
    # 合并连续的分号
    s = re.sub(r";+", "；", s)
    # 去掉首尾的分号
    s = s.strip('；')
    return s


def normalize_application(application: str) -> str:
    """归一化 application 字段：
    - 如果为空返回空字符串
    - 去除首尾中括号 []（如果存在）
    - 将双引号外的逗号替换为分号
    - 去掉所有双引号

    例: ["a,b","c,d"] -> a,b;c,d
    """
    if application is None:
        return ""
    s = str(application).strip()
    if s == "":
        return ""
    # 先将全角逗号统一为半角逗号，便于后续按双引号外逗号分隔处理
    s = s.replace('，', ',')
    # 去掉首尾方括号
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    out_chars = []
    in_quote = False
    for ch in s:
        if ch == '"':
            in_quote = not in_quote
            # 丢弃双引号
            continue
        if ch == ',' and not in_quote:
            out_chars.append(';')
        else:
            out_chars.append(ch)

    res = ''.join(out_chars)
    # 清理空白和重复分隔符
    res = re.sub(r"\s*;\s*", ";", res)
    res = re.sub(r";+", ";", res)
    res = res.strip(' ;')
    return res


def normalize_research_field(field: str) -> str:
    r"""
    研究方向字段（research_field）标准化说明：

    目的：移除字段中常见的序号标记并统一分隔符，使后续文本匹配/分割更可靠。

    处理步骤（按顺序）：
    1. 空值处理：若为 None 或空字符串，返回空字符串（""）。
    2. 移除各种序号格式：对字段中出现的序号进行全局删除（对任意位置出现的序号都会被移除）：
       - 半角阿拉伯数字后跟句点：例如 `1.`、`2.` 等（正则：`\d+\.`）
       - 全角句点：例如 `1．`（正则：`\d+．`）
       - 中文顿号后的数字：例如 `1、`（正则：`\d+、`）
       - 中括号格式：例如 `[1]`（正则：`\[\d+\]`）
       - 中文或半角括号：例如 `（1）`、`(1)`（正则：`\（\d+\）`、`\(\d+\)`）
       - 数字加右括号：例如 `1)` 或 `1）`（正则：`\d+\)`、`\d+\）`）
       - 数字后跟逗号：例如 `1,`（正则：`\d+,`）
       说明：这些替换使用全局替换（re.sub），会删除字段中所有出现的此类序号。

    3. 符号替换与清理：
       - 将全角逗号 `，` 替换为半角逗号 `,`，将全角分号 `；` 替换为半角分号 `;`，便于统一后续处理。
       - 删除常见的带序号的圆周/括号数字（如 ①、②、Ⅰ 等 unicode 编号块），使用 unicode 范围过滤。
       - 去除所有空白（包括空格/换行/制表符），因为研究方向通常以短语形式紧凑存储。

    4. 分隔符归一化：将连续的逗号/分号/句号等替换为单个半角逗号 `,`，然后把多个连续逗号压缩为一个，最后去掉首尾逗号。
       这样方便之后按逗号或分号切分成多个研究方向项。

    注意与边界情况：
    - 此逻辑会移除任何位置的序号，而不仅是行首。例如字符串 `算法1, 机器学习2` 中的 `1` 和 `2` 会被删除，只保留文本部分。
    - 为避免误删真正含数字的短语（例如 `3D打印`），可以根据需要在正则中限定序号必须跟随特定分隔符或位于词边界；当前实现较激进，会删除数字后跟标点的形式。
    - 如果希望保留像 `3D` 这样的合法数字组合，后续可以加入额外规则（例如只删除后面紧跟标点的纯数字序号）。

    返回值：清洗并归一化后的字符串（不包含首尾逗号），若输入为空则返回空字符串。
    """
    if field is None:
        return ""
    s = str(field).strip()
    if s == "":
        return ""

    t = s

    t = re.sub(r'\d+\.', '', t)
    t = re.sub(r'\d+．', '', t)
    t = re.sub(r'\d+、', '', t)
    t = re.sub(r'\[\d+\]', '', t)
    t = re.sub(r'\（\d+\）', '', t)
    t = re.sub(r'\(\d+\)', '', t)
    t = re.sub(r'\d+\）', '', t)
    t = re.sub(r'\d+\)', '', t)
    t = re.sub(r'\d+,', '', t)
    t = re.sub(r'[•♦▪▫◆◇●○►▶◄◢◣◤◥★☆✦✧✸✹✶✷✺✻✽✾✿❖❂❃❁❀✪✫✬✭✮✯✰]+', '', t)
    t = t.replace('，', ',')
    t = t.replace('；', ';')
    t = re.sub(r'[\u2460-\u24FF\u2776-\u2793\u3250-\u32FF]+', '', t)
    t = re.sub(r'\s+', '', t)
    t = re.sub(r'[,.;]+', ',', t)
    t = re.sub(r',+', ',', t)
    t = t.strip(',')


    return t


def append_if_missing(base: str, addition: str, sep: str = "") -> str:
    """将 addition 追加到 base：
    - 如果 addition 为空，返回 base
    - 如果 base 已包含 addition（做简单子串判断），则不重复追加
    - 否则追加：如果 sep 非空则用 sep 连接，否则直接拼接（用于 duty 的要求）
    """
    base = "" if base is None else str(base)
    addition = "" if addition is None else str(addition)
    if addition == "":
        return base
    if addition in base:
        return base
    if base == "":
        return addition
    if sep:
        return base + sep + addition
    return base + addition


def clean_dataframe(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    # 确保字段存在，缺失则创建为空字符串，避免 KeyError
    for col in ["research_field", "professional_title", "duty", "college", "btype_name"]:
        if col not in df.columns:
            df[col] = ""

    # 将 None/NaN 统一为空字符串，便于字符串操作
    df = df.fillna("")

    # 按顺序依次删除以下列为空的行，并打印每一步删除的行数
    seq_remove_cols = ["user_name", "research_field", "brief_intr", "introduction"]
    for col in seq_remove_cols:
        if col not in df.columns:
            # 如果列不存在，跳过
            continue
        before_len = len(df)
        df = df[df[col].astype(str).str.strip() != ""].copy()
        removed = before_len - len(df)
        print(f"删除 '{col}' 为空的行: {removed} 行")

    # 删除 research_field 中仅包含一个中文汉字的行
    before_len = len(df)
    df = df[~df["research_field"].astype(str).str.match(r"^[\u4e00-\u9fa5]$")].copy()
    removed = before_len - len(df)
    print(f"删除 research_field 中仅包含一个中文汉字的行: {removed} 行")

    # 逐行追加并规范化 duty、btype_name、application、research_field
    def process_row(r):
        duty = str(r.get("duty", ""))
        prof = str(r.get("professional_title", ""))
        college = str(r.get("college", ""))
        btype = str(r.get("btype_name", ""))
        application = r.get("application", "")
        research_field = r.get("research_field", "")

        # professional_title 追加到 duty，追加时用 ';' 分隔，但若已包含则不追加
        duty = append_if_missing(duty, prof, sep=";")

        # college 追加到 btype_name（直接追加，不加分隔符），若已包含则不追加
        btype = append_if_missing(btype, college, sep="")

        # 归一化 duty 的分隔符
        duty = normalize_duty(duty)

        # 归一化 application 字段并写回
        r["application"] = normalize_application(application)

        # 归一化 research_field 字段并写回（移除各种序号并统一分隔）
        r["research_field"] = normalize_research_field(research_field)

        r["duty"] = duty
        r["btype_name"] = btype
        return r

    # 逐行处理：使用显式迭代以避免 apply 返回 DataFrame 导致没有 tolist() 的情况
    processed_rows = [process_row(row) for _, row in df.iterrows()]
    df = pd.DataFrame(processed_rows)
    # 删除用户指定的无效列（如果存在）
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

    return df


def main():
    # 使用脚本内硬编码配置
    input_path = INPUT_PATH
    sheet = SHEET
    output_path = OUTPUT_PATH
    drop_cols = DROP_COLS

    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        sys.exit(2)

    # 读取 Excel，尽量将空值保持为空字符串
    try:
        df = pd.read_excel(input_path, sheet_name=sheet, dtype=str, keep_default_na=False)
    except Exception as e:
        print("读取 Excel 失败:", e)
        sys.exit(3)

    before_count = len(df)
    cleaned = clean_dataframe(df, drop_cols=drop_cols)
    after_count = len(cleaned)

    # 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        cleaned.to_excel(output_path, index=False)
    except Exception as e:
        print("保存文件失败:", e)
        sys.exit(4)

    print(f"原始行数: {before_count}")
    print(f"清洗后行数: {after_count}")


if __name__ == "__main__":
    # 运行清洗；若需更改输入/输出/删除列，请修改脚本顶部的配置常量
    main()
