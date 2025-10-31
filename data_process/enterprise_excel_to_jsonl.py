import pandas as pd
import json

# 读取 Excel 文件
file_path = "Dataset/enterprises_full_cleaned.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# 选择关键字段
key_fields = [
     "org_name", "scale", "business", "address", "reg_capital", "company_org_type", "establish_time", "category", "category_big", "category_middle"
]

# 如果缺少某些字段，自动忽略
fields_in_data = [f for f in key_fields if f in df.columns]
df_selected = df[fields_in_data]

def build_prompt_input(row, index=None):
    """将单行企业数据拼接成提示词所需 JSON"""
    company_info = {
        "企业名称": str(row.get("org_name", "")),
        "地址": str(row.get("address", "")),
        "主营业务": str(row.get("business", "")),
        "行业领域": " | ".join(
            [str(row.get("category", "")), str(row.get("category_big", "")), str(row.get("category_middle", ""))]
        ),
        "企业规模": str(row.get("scale", "")),
        "企业性质": str(row.get("company_org_type", "")),
        "成立时间": str(row.get("establish_time", "")),
        "注册资金": str(row.get("reg_capital", ""))
    }
    
    # 拼接成一个描述文本（可直接放进 Prompt）
    text_description = (
        f"{company_info['企业名称']}，成立于{company_info['成立时间']}，位于{company_info['地址']}，"
        f"是一家{company_info['企业性质']}，规模为{company_info['企业规模']}企业。"
        f"主营业务包括：{company_info['主营业务']}。"
        f"所属行业领域为：{company_info['行业领域']}。"
        f"注册资金：{company_info['注册资金']}。"
    )
    
    if index is not None:
        obj = {"enterprise_index": int(index), "company_info": text_description.strip()}
    else:
        obj = {"company_info": text_description.strip()}
    return obj

# 构建所有企业的提示词输入
prompt_inputs = [build_prompt_input(row, i) for i, (_, row) in enumerate(df_selected.iterrows())]

# 保存为 JSONL 格式
output_file = "data_process_outputs/enterprises_inputs.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in prompt_inputs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"已生成输入文件: {output_file}，共 {len(prompt_inputs)} 条数据")