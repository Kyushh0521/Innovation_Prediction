import os
import sys
import pandas as pd
import re
from typing import List

# === 配置区域：将所有输入/输出和要删除的列在这里硬编码，按需修改 ===
# 输入文件（Workbook）
INPUT_PATH = "Dataset/needs_full.xlsx"
# 读取的 sheet（名称或索引）
SHEET = 0
# 输出文件路径
OUTPUT_PATH = "Dataset/needs_full_cleaned.xlsx"
# 要删除的无效列列表（编辑此处以指定要删除的列）
DROP_COLS: List[str] = ['id','pub_code','service_type','tech_type','root_id','industry_id','product','first_img_url','suoluetu','step','cooperation_level','pro_cost','contect','remark','is_knowledge','knowledge_code','is_cert','cert_code','invention_name','link_name','link_tel','hot_num','contect_type','status','del_flag','deal_status','read_num','pub_recommend_status','pub_user','pub_tel','pub_version','dock_name','create_by','create_time','update_by','update_time','pub_status','pub_show_status','pub_date','res_begin_date','res_end_date','erweima','collect_num','grade','enterprise_id','hits','biz_object_id','qrcode','rec','recs','charger','ncharger','display','global_id','demand_status','tmp_create_time','innovation_type','self_maturity','required_support','aplication_field_scenario','main_function','main_advantage','scene_lable','chain_lable','budget','delivery_place','validity_time','undertaker','marker','technical_broker','product_names','label_modified','dock_name_tel','undertaker_tel','cooperation','level','category']

def normalize_title(title: str) -> str:
    """
    标准化 title 字段
    """
    if title is None:
        return ""
    s = str(title)
    # 删除指定的关键词
    s = s.replace("【品牌整机厂创新需求】", "")
    s = s.replace("需求：", "")
    s = s.replace("需求:", "")
    s = s.replace("寻找：", "")
    s = s.replace("寻找:", "")
    s = s.replace("寻找", "")
    s = s.replace("寻求：", "")
    s = s.replace("寻求:", "")
    s = s.replace("寻求", "")
    s = s.replace("寻：", "")
    s = s.replace("寻:", "")
    s = s.replace("寻", "")
    return s

def normalize_application(application: str) -> str:
    """
    标准化 application 字段：
    - 将所有的 '/' 替换为英文分号 ';'
    """
    if application is None:
        return ""
    s = str(application)
    s = s.replace("/", ";")
    return s

def normalize_analyse_contect(analyse_contect: str) -> str:
    if analyse_contect is None:
        return ""
    s = str(analyse_contect)

    # 判断是否包含 HTML 标签
    if re.search(r'<[^>]+>', s):
        # 为章节标题添加冒号，正确位置为找到章节标题后直到出现第一个 <...>，在 < 前加一个冒号
        s = re.sub(r'(（[一二三四五六七八九十]+）[^<]+)(?=<)', r'\1：', s)

        # 修改数字小标题的逻辑，确保在数字小标题后直到出现第一个 <...> 前添加冒号
        s = re.sub(r'(\d+\.[^<]+)(?=<)', r'\1：', s)

        # 去掉所有HTML标签
        s = re.sub(r'<[^>]+>', '', s)

        # 去掉多余的空格
        s = re.sub(r'\s+', '', s)
    
    # 删除所有的“寻：”
    s = s.replace("寻：", "")

    # 删除所有的①这一类的格式，以及将所有的“数字，”“数字.”“数字）”和“（数字）”改为“数字、”
    s = re.sub(r'[\u2460-\u24FF]', lambda m: str(ord(m.group()) - 9311) + '、', s)

    # 单独处理“1.”这种序号，判断后面是否为数字或字母
    s = re.sub(r'(\d+)\.(?![\da-zA-Z])', r'\1、', s)

    # 处理其他序号格式
    s = re.sub(r'（(\d+)）|(?<=\d)[，,）)]', r'\1、', s)

    # 去除换行和空格（提前，避免“。 2、”等情况）
    s = re.sub(r'\s+', '', s)

    # 仅当序号为 2 或更大数字时，如果前面没有标点符号，则在前面插入中文分号
    s = re.sub(r'(?<![；;，,。！？、])(?=(?:[2-9]\d*|1\d+)、)', r'；', s)

    # 删除句首的单个标点符号（如果句首只有一个标点，且后面不是标点，则删除）
    s = re.sub(r'^[，,；;。！？、:：\-—](?=[^，,；;。！？、:：\-—])', '', s)

    # 将句尾不成对的和单个的标点符号替换为句号
    s = re.sub(r'[，,；;。.！？、]+$', '。', s)

    # 若内容不为空且句尾没有标点符号，则补充句号
    if s and not re.search(r'[。！？]$', s):
        s += '。'
    return s

def clean_dataframe(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    # 确保字段存在，缺失则创建为空字符串，避免 KeyError
    df = df.fillna("")

    # 标准化 title 字段
    if "title" in df.columns:
        df["title"] = df["title"].apply(normalize_title)

    # 标准化 application 字段
    if "application" in df.columns:
        df["application"] = df["application"].apply(normalize_application)

    # 标准化 analyse_contect 字段
    if "analyse_contect" in df.columns:
        df["analyse_contect"] = df["analyse_contect"].apply(normalize_analyse_contect)

    # 删除重复的 title 数据，只保留第一个出现的数据
    if "title" in df.columns:
        before_len = len(df)
        df = df.drop_duplicates(subset="title", keep="first")
        removed = before_len - len(df)
        print(f"删除重复的 'title' 数据: {removed} 行")

    # 删除 title 字段中内容只为 "1" 的数据
    if "title" in df.columns:
        before_len = len(df)
        df = df[df["title"].astype(str).str.strip() != "1"].copy()
        removed = before_len - len(df)
        print(f"删除 'title' 字段中内容为 '1' 的行: {removed} 行")

    # 按顺序依次删除以下列为空的行，并打印每一步删除的行数
    seq_remove_cols = ["title", "analyse_contect", "pub_dept"]  # 替换为实际的列名
    for col in seq_remove_cols:
        if col not in df.columns:
            # 如果列不存在，跳过
            continue
        before_len = len(df)
        df = df[df[col].astype(str).str.strip() != ""].copy()
        removed = before_len - len(df)
        print(f"删除 '{col}' 为空的行: {removed} 行")

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