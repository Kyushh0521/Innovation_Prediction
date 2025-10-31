import os
import sys
from typing import List
import re

import pandas as pd

# === 配置区域：将所有输入/输出和要删除的列在这里硬编码，按需修改 ===
# 输入文件（Workbook）
INPUT_PATH = "Dataset/enterprises_full.xlsx"
# 读取的 sheet（名称或索引）
SHEET = 0
# 输出文件路径
OUTPUT_PATH = "Dataset/enterprises_full_cleaned.xlsx"
# 要删除的无效列列表（编辑此处以指定要删除的列）
DROP_COLS: List[str] = [
    'id', 'logo_url', 'org_id', 'name_simple', 'industry', 'introduction', 'nature_type', 'needs', 'province_code',
    'city_code', 'area_code', 'label_keyword', 'add_date', 'service_name', 'status', 'create_by', 'create_time',
    'update_by', 'update_time', 'remark', 'org_type', 'is_alliance', 'linkman_dept', 'linkman_duty', 'linkman_email',
    'industry_field', 'high_tech_enterprise', 'science_middle_and_small_enterprise', 'workers_total',
    'R_D_personnel_total', 'first_half_revenue', 'first_revenue', 'first_half_profit', 'technological_innovation',
    'technological_innovation_plan', 'university_cooperation', 'university_cooperation_plan', 'patent', 'patent_json',
    'enterprise_cooperation', 'enterprise_cooperation_plan', 'r_d_place', 'r_d_place_json', 'r_d_personnel',
    'r_d_personnel_json', 'r_d_equipment', 'r_d_equipment_json', 'r_d_project', 'r_d_project_json', 'r_d_investment',
    'r_d_investment_json', 'management_system', 'construction_type', 'management_system_json', 'num', 'biz_object_id',
    'total_score', 'customer', 'ncustomer', 'authentication_image', 'qrcode', 'type', 'province_name', 'city_name',
    'area_name', 'street_name', 'reg_location', 'social_staff_num', 'category_small', 'reg_status', 'tags', 'board',
    'boardtime', 'chain', 'baseinfo', 'baseuser', 'basetime', 'devbase', 'devuser', 'devtime', 'devtop', 'basetax',
    'topuser', 'toptime', 'reward', 'qualify', 'is_yunyan_award', 'industry_label', 'rangese', 'main_product',
    'purchase_product', 'prosupply', 'record_supply_chain', 'record_reduction', 'record_reduce_material', 'record_capacity',
    'record_Capacity_surplus', 'record_Capacity_lack', 'record_production_equipment', 'big_scale', 'org_code'
]


def clean_dataframe(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    # 根据 org_name 字段去重，只保留第一次出现的数据
    if 'org_name' in df.columns:
        df = df.drop_duplicates(subset='org_name', keep='first')
    # 将 None/NaN 统一为空字符串，便于字符串操作
    df = df.fillna("")

    # 辅助：删除单元格内容首尾的标点符号（中英文常见）
    def strip_edge_punct(val: object) -> object:
        if val is None:
            return val
        s = str(val)
        # 定义要移除的首尾标点集合（包括中英文常用标点和括号）
        punct_class = r"\(\)\[\]【】（）,，。\.．!！\?？;；:：…\-—/\\\"'“”‘’<>《》、#*&^$%@』『·" 
        # 先去掉开头的连续标点
        s = re.sub(rf'^[{punct_class}]+', '', s)
        # 再去掉结尾的连续标点
        s = re.sub(rf'[{punct_class}]+$', '', s)
        return s

    # 删除用户指定的无效列（如果存在）
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

    # 兼容拼写错误的列名：estiblish_time -> establish_time
    if 'estiblish_time' in df.columns:
        # 若目标列不存在，则直接重命名；若存在则用原列补充后删除错拼列
        if 'establish_time' not in df.columns:
            df = df.rename(columns={'estiblish_time': 'establish_time'})
        else:
            df['establish_time'] = df['establish_time'].astype(str).fillna('')
            df['establish_time'] = df['establish_time'].where(df['establish_time'].astype(str).str.strip() != '', df['estiblish_time'])
            df = df.drop(columns=['estiblish_time'])

    # 将 scale 字段的数字映射为中文标签（仅在 scale 列存在时）
    if 'scale' in df.columns:
        mapping = {'1': '小型', '2': '中型', '3': '大型'}
        df['scale'] = df['scale'].astype(str).apply(lambda v: mapping.get(v.strip(), v.strip()))

    # 将 establish_time 时间字段截断为日期（只保留 YYYY-MM-DD）
    if 'establish_time' in df.columns:
        df['establish_time'] = df['establish_time'].astype(str).apply(
            lambda v: v.strip().split()[0] if v and str(v).strip() != '' else v
        )

    # 标准化 company_org_type：删除所有 '(' 或 '（' 及之后的内容，并去除首尾空白
    if 'company_org_type' in df.columns:
        def normalize_company_org_type(v: object) -> str:
            s = '' if v is None else str(v)
            # 删除任意形式的括号及其后的内容
            s = re.sub(r'[（(].*$', '', s).strip()
            return s

        df['company_org_type'] = df['company_org_type'].apply(normalize_company_org_type)

    # 对 business 字段做清洗：去除乱码/控制字符
    if 'business' in df.columns:
        patterns = [
            r'〓',
            r'许可经营项目是：无',
            r'许可经营项目是：',
            r'一般经营项目是：',
            r'许可经营项目是',
            r'一般经营项目是',
            r'许可经营项目：',
            r'一般经营项目：',
            r'许可经营项目',
            r'一般经营项目',
            r'许可项目：',
            r'一般项目：',
            r'许可项目',
            r'一般项目',
        ]

        def clean_business_cell(x: object) -> str:
            s = '' if x is None else str(x)
            # 先移除已知样板文本
            for p in patterns:
                s = re.sub(p, '', s)
            # 删除指定的乱码字符（仅移除 # * & ^ $）
            s = re.sub(r'[#*&^$]', '', s)
            # 删除常见控制字符
            s = re.sub(r'[\r\n\t\x0b\x0c]', '', s)
            # 删除替代字符（如 U+FFFD）
            s = s.replace('\uFFFD', '')

            # 先统一所有括号为中文小括号，以便后续一致处理（支持中英文及其它样式）
            s = s.replace('(', '（').replace('[', '（').replace('{', '（')
            s = s.replace('<', '（').replace('〈', '（').replace('【', '（')
            s = s.replace(')', '）').replace(']', '）').replace('}', '）')
            s = s.replace('>', '）').replace('〉', '）').replace('】', '）')

            # 删除所有括号及括号内的内容（已统一为中文括号），使用循环处理嵌套
            while True:
                new_s = re.sub(r'（[^（）]*）', '', s)
                if new_s == s:
                    break
                s = new_s
            # 移除可能残留的孤立括号字符
            s = s.replace('（', '').replace('）', '')

            # 先把连续的标点序列压缩为一个统一的句号 '。'，以去除重复或混合的错误标点
            s = re.sub(r'[，。,.!?；;:：]{2,}', '。', s)

            # 合并多个空白并去两端空白
            s = re.sub(r'\s+', ' ', s).strip()

            # 句首处理：若句首有标点符号则删除（包括中英文常见标点和括号符号）
            # 注意：这里只删除开头连续的标点字符，不会影响中间内容
            s = re.sub(r'^[\(\)（）\[\]【】,，。\.．!！\?？;；:：…\-—/\\]+', '', s)

            # 句末处理：统一使句子以中文句号 '。' 结尾
            # 如果末尾存在标点（中/英文常见），替换为一个中文句号；若末尾没有标点则追加一个句号
            if s:
                s = re.sub(r'[，,。.．.!！?？;；:：…]+$', '。', s)
                if not s.endswith('。'):
                    s = s + '。'

            return s

        df['business'] = df['business'].apply(clean_business_cell)

    # 对所有字段的字符串单元格删除首尾标点（保证一致性）
    # 仅对字符串/对象列逐列处理，避免 applymap 在含有复杂类型时出错
    from pandas.api.types import is_string_dtype
    text_cols = [c for c in df.columns if is_string_dtype(df[c]) or df[c].dtype == object]
    for col in text_cols:
        df[col] = df[col].map(lambda v: strip_edge_punct(v) if isinstance(v, str) else v)

    # 统一标准化 category 相关三列，并删除其中任一列为纯数字的行
    def normalize_category_value(v: object) -> str:
        if v is None:
            return ''
        s = str(v).strip()
        # 全角空格和特殊空白替换
        s = s.replace('\u3000', ' ').replace('\xa0', ' ')
        # 合并多空格
        s = re.sub(r'\s+', ' ', s)
        # 去掉末尾标点
        s = re.sub(r'[，,。\.；;：:\s]+$', '', s)
        return s

    for c in ('category', 'category_big', 'category_middle'):
        if c in df.columns:
            df[c] = df[c].astype(str).apply(normalize_category_value)

    # 删除任一 category 列为纯数字的行（例如 '123'）
    def is_pure_number(s: str) -> bool:
        if s is None:
            return False
        s = str(s).strip()
        return bool(re.fullmatch(r"\d+", s))

    if any(col in df.columns for col in ('category', 'category_big', 'category_middle')):
        mask_numeric = df.apply(lambda row: any(is_pure_number(row.get(col, '')) for col in ('category', 'category_big', 'category_middle')), axis=1)
        if mask_numeric.any():
            df = df[~mask_numeric].copy()
    # 删除在任一保留字段中存在空值（仅由空白或空字符串构成）或仅为 '-' 的行
    if not df.empty:
        non_empty_mask = df.apply(lambda col: ~col.astype(str).str.strip().isin(["", "-"])).all(axis=1)
        df = df[non_empty_mask].copy()

    return df


def main():
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
    main()