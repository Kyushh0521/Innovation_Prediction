import pandas as pd
import json
import os
import time
from typing import List, Set
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 本地 LLM 地址 (例如 Ollama 默认地址)
LOCAL_LLM_URL = "http://localhost:11434/v1" 
# 本地模型名称 (需与你本地部署的一致，如 'qwen2.5:7b', 'deepseek-r1', 'llama3' 等)
MODEL_NAME = "qwen2.5:7b" 
API_KEY = "lm-studio" # 本地服务通常不需要，随便填

# 输入文件路径
FILES_TO_PROCESS = [
    'Dataset/enterprises_full_cleaned.xlsx',
    'Dataset/experts_full_cleaned_filled.xlsx',
    'Dataset/achievements_full_cleaned.xlsx'
]
# 输出术语表路径
OUTPUT_TERMS_FILE = 'data_process_outputs/domain_terms_llm.txt'
# ===========================================

client = OpenAI(base_url=LOCAL_LLM_URL, api_key=API_KEY)

def load_data(file_path: str) -> List[str]:
    """读取 Excel 并拼接关键文本字段"""
    if not os.path.exists(file_path):
        print(f"文件不存在跳过: {file_path}")
        return []
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取 Excel 失败 {file_path}: {e}")
        return []

    texts = []
    
    # 根据不同文件类型选择拼接字段
    cols = []
    if 'enterprises' in file_path:
        cols = ['business', 'category_middle']
    elif 'experts' in file_path:
        cols = ['research_field', 'application']
    elif 'achievements' in file_path:
        cols = ['title', 'analyse_contect', 'application_field_scenario']
        
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            if c in df.columns and pd.notna(row[c]):
                parts.append(str(row[c]).strip())
        if parts:
            texts.append("，".join(parts))
    return texts

def extract_terms_with_llm(text: str) -> List[str]:
    """调用 LLM 提取全行业专业术语（优化版）"""
    
    # === 优化后的 Prompt ===
    prompt = f"""
    [角色设定]
    你是一名跨领域的专业术语标注专家，能够识别各行各业（包括但不限于IT、制造、医疗、金融、能源、化工等）中的关键技术概念和行业黑话。

    [任务]
    分析下方的【输入文本】，提取其中具有高检索价值的“专业技术名词”、“行业术语”或“特定实体”。

    [严格约束 - 禁止提取]
    1. ❌ 禁止动词和动作描述（如：研发、生产、提供、建立、促进、加工、销售）。
    2. ❌ 禁止泛指的通用商业词汇（如：服务、解决方案、平台、体系、项目、业务、我们、客户、需求、应用）。
    3. ❌ 禁止纯形容词或修饰语（如：高效的、智能的、先进的、全面的）。
    4. ❌ 禁止过于宽泛的单词（如：技术、数据、系统、设备），除非它是专有名词的一部分（如“分布式系统”是可接受的）。

    [提取标准 - 应该提取]
    1. ✅ 具体的专有名词（如：算法名称、化学分子式、特定机械型号、法规标准）。
    2. ✅ 行业特定的概念（如：供应链金融、光伏逆变器、单克隆抗体、边缘计算、数控机床）。
    3. ✅ 尽量保持复合名词的完整性（例如提取“自然语言处理”而不是拆分为“自然语言”和“处理”）。

    [输出格式要求]
    1. 仅输出一个纯 JSON 字符串列表（Array of Strings）。
    2. 严禁包含 Markdown 标记（如 ```json ... ```）。
    3. 严禁包含任何“好的”、“如下所示”等解释性文字，只返回列表本身。

    [输入文本]
    {text}

    [输出]
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, # 低温度确保结果稳定
            max_tokens=1024  # 增加 token 数以应对可能的长列表
        )
        message = response.choices[0].message
        # 安全检查：如果 content 是 None，直接返回空列表，跳过 strip()
        if message.content is None:
            # 这种情况通常是模型拒绝回答或出错了
            return []
        content = message.content.strip()
        
        # === 增强的清洗逻辑 ===
        # 1. 去除可能存在的 Markdown 代码块标记
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
        
        # 2. 寻找 JSON 列表的起止位置，防止前后有废话
        start = content.find('[')
        end = content.rfind(']') + 1
        
        if start != -1 and end != -1:
            json_str = content[start:end]
            terms = json.loads(json_str)
            
            # 3. 数据清洗：必须是字符串，长度>1，且不是纯数字
            valid_terms = []
            for t in terms:
                if isinstance(t, str):
                    t = t.strip()
                    # 过滤单字 (如 "云") 和 纯数字 (如 "2025")
                    if len(t) > 1 and not t.isdigit():
                        valid_terms.append(t)
            return valid_terms
        else:
            return []
            
    except json.JSONDecodeError:
        # 如果模型输出的不是合法 JSON，记录但不中断
        # print(f"JSON 解析失败: {content[:20]}...") 
        return []
    except Exception as e:
        print(f"LLM 调用出错: {e}")
        return []

def main():
    all_terms: Set[str] = set()
    
    # 1. 读取所有待处理文本
    all_texts = []
    print("开始读取数据文件...")
    for fp in FILES_TO_PROCESS:
        print(f"正在读取: {fp}")
        file_texts = load_data(fp)
        all_texts.extend(file_texts)
        print(f"  - 提取到 {len(file_texts)} 条记录")
    
    print(f"原始文本总数: {len(all_texts)}")
    
    # 2. 文本去重
    # 很多企业或专家可能有重复记录，去重能大幅节省 LLM 调用时间
    unique_texts = list(set(all_texts)) 
    unique_texts = [t for t in unique_texts if len(t) >= 5] # 再次过滤过短文本
    print(f"去重后需处理文本数: {len(unique_texts)}")
    
    if not unique_texts:
        print("没有可处理的文本，程序退出。")
        return

    print(f"开始调用本地 LLM ({MODEL_NAME}) 提取术语...")

    # 3. 批量处理
    # 使用 tqdm 显示进度条
    for i, text in enumerate(tqdm(unique_texts, desc="Processing")):
        terms = extract_terms_with_llm(text)
        for t in terms:
            all_terms.add(t)
            
        # 实时保存（防止程序崩溃或中断丢失数据）
        # 每处理 50 条保存一次
        if i > 0 and i % 50 == 0:
            _save_terms(all_terms)
                
    # 4. 最终保存
    _save_terms(all_terms)
    print(f"处理完成！")

def _save_terms(terms_set: Set[str]):
    """辅助函数：将集合写入文件"""
    os.makedirs(os.path.dirname(OUTPUT_TERMS_FILE) or '.', exist_ok=True)
    
    # 排序，方便人类查看
    sorted_terms = sorted(list(terms_set))
    
    try:
        with open(OUTPUT_TERMS_FILE, 'w', encoding='utf-8') as f:
            for term in sorted_terms:
                f.write(term + '\n')
    except Exception as e:
        print(f"写入文件失败: {e}")
    else:
        # 可选：打印当前提取的数量
        # print(f"已保存 {len(sorted_terms)} 个术语")
        pass

if __name__ == '__main__':
    main()