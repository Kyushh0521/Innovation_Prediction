import os
import sys
from typing import List
import re
import json

import pandas as pd

# === 配置区域：仿照 enterprises_full_clean.py 的最小删列脚本 ===
# 输入文件（Workbook）
INPUT_PATH = "Dataset/achievements_full.xlsx"
# 读取的 sheet（名称或索引）
SHEET = 0
# 输出文件路径
OUTPUT_PATH = "Dataset/achievements_full_cleaned.xlsx"

# 可配置：要按 title 精确匹配删除的记录（把需要删除的 title 字符串放在此列表中）
# 例如: DELETE_TITLES = ["测试记录A", "测试记录B", "测试记录C"]
DELETE_TITLES: List[str] = ['测试供给', '测试 发成果', '涵养山矿泉水']

# 要删除的无效列列表（按需调整）
DROP_COLS: List[str] = [
    'id', 'pub_code', 'service_type', 'tech_type', 'root_id', 'industry_id', 'first_img_url', 'suoluetu',
    'step', 'cooperation', 'level', 'pro_cost', 'contect', 'remark', 'is_knowledge', 'knowledge_code', 'is_cert',
    'cert_code', 'invention_name', 'link_name', 'link_tel', 'hot_num', 'contect_type', 'status', 'del_flag',
    'deal_status', 'read_num', 'pub_recommend_status', 'pub_user', 'pub_tel', 'pub_version', 'dock_name',
    'create_by', 'create_time', 'update_by', 'update_time', 'pub_status', 'pub_show_status', 'pub_date',
    'res_begin_date', 'res_end_date', 'erweima', 'collect_num', 'grade', 'enterprise_id', 'hits', 'biz_object_id',
    'qrcode', 'rec', 'recs', 'charger', 'ncharger', 'display', 'global_id', 'demand_status', 'tmp_create_time',
    'innovation_type', 'self_maturity', 'required_support', 'budget', 'delivery_place', 'validity_time', 'undertaker',
    'marker', 'technical_broker', 'product_names', 'label_modified', 'dock_name_tel', 'undertaker_tel','product','category',
    'application','aplication_field_scenario','main_function','main_advantage','scene_lable','chain_lable'
]

# 若需要仅校验特定必填字段，将其列名放入该列表；留空则对所有保留列进行校验
REQUIRED_NONEMPTY_COLS: List[str] = ['analyse_contect']


def drop_invalid_columns(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """仅删除指定的无效列（存在时）。不做额外清洗。"""
    if not drop_cols:
        return df
    exist = [c for c in drop_cols if c in df.columns]
    if exist:
        df = df.drop(columns=exist)
    return df


def drop_rows_with_empty_fields(df: pd.DataFrame, required_cols: List[str] | None = None) -> pd.DataFrame:
    """删除在必填字段中出现空值（空白或 '-'）的行。

    - required_cols 为空或 None 时，默认对所有列进行检查。
    - 仅将值为 ''、仅空白、或 '-' 的视为“空”。
    """
    if df.empty:
        return df
    cols = list(required_cols) if required_cols else list(df.columns)
    # 仅保留存在于 DataFrame 的列
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    non_empty_mask = df[cols].apply(lambda col: ~col.astype(str).str.strip().isin(["", "-"])).all(axis=1)
    return df[non_empty_mask].copy()


def standardize_application_cell(val: object) -> object:
    """标准化 application 单元格：
    1) 若包含方括号 []：
       - 先将【方括号外】以及【引号（" 或 “ ”）内】的英文逗号 , 替换为中文分号 ；
       - 然后移除所有方括号 [ ] 与双引号（"、“、”）
    2) 否则（不含 []）：
       - 将连续的分隔符折叠为一个中文分号 ；
       - 并将单个分隔符也统一替换为中文分号 ；
    注：空值（空串/仅空白/“-”）不处理。
    """
    s = "" if val is None else str(val)
    s_strip = s.strip()
    if s_strip == "":
        return val

    has_brackets = ('[' in s_strip) and (']' in s_strip)
    # 先删除可能存在的 HTML 标签标记（保留标签内部文本）
    s_strip = remove_html_tags(s_strip)

    if has_brackets:
        # 一次扫描：跟踪方括号层级与引号状态
        out = []
        bracket_depth = 0
        in_quotes = False  # 仅识别双引号：英文 " 与中文 “ ”
        for ch in s_strip:
            if ch == '"' or ch == '“' or ch == '”':
                # 仅切换状态，不输出引号
                in_quotes = not in_quotes
                continue
            if ch == '[':
                bracket_depth += 1
                out.append(ch)  # 暂时保留，后续统一移除
                continue
            if ch == ']':
                bracket_depth = max(0, bracket_depth - 1)
                out.append(ch)  # 暂时保留，后续统一移除
                continue
            if ch == ',':
                # 仅在【方括号内】且【不在引号内】时替换为中文分号
                if bracket_depth > 0 and not in_quotes:
                    out.append('；')
                else:
                    out.append(',')
                continue
            out.append(ch)
        mid = ''.join(out)
        # 删除 [] 与双引号（"、“、”）
        mid = mid.replace('[', '').replace(']', '')
        mid = mid.replace('"', '').replace('“', '').replace('”', '')
        return mid.strip()
    else:
        # 非 []：把所有常见分隔符折叠为中文分号
        # 分隔符集合：中英文逗号/分号/顿号/斜杠/反斜杠/竖线
        return re.sub(r'[，,;；/／\\\|｜、]+', '；', s_strip)


def standardize_analyse_contect_cell(val: object) -> object:
    s = "" if val is None else str(val)
    if s == "":
        return ""
    # 删除 HTML 标签
    # 先移除常见的非断行空格实体（例如 &nbsp; / &nbsp / &#160;）
    s = re.sub(r'&nbsp;?|&#160;?', '', s, flags=re.IGNORECASE)
    # 删除 HTML 标签
    s = remove_html_tags(s)
    # 规范化换行：把 CRLF/CR -> LF，再按行处理
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    raw_lines = [ln for ln in s.split('\n')]
    # 去除全角空格与不间断空格
    # 标点集合用于判断是否在两段之间插入分号
    punct_re = re.compile(r"[。！？；，、：:,.!?;]")

    cleaned_lines: list[tuple[str, bool]] = []  # (line_text, had_index_or_bullet)
    index_or_bullet_re = re.compile(r"^\s*(?:[◆•●・◼]+|[（(]?\s*\d+\s*[\.、\)）:\：])\s*")
    for ln in raw_lines:
        ln_stripped = ln.strip()
        if ln_stripped == "":
            continue
        # 判断该行是否原始包含序号或指定无序符号
        had_index_or_bullet = bool(index_or_bullet_re.match(ln_stripped))
        # 剥离开头的序号样式或无序符号（如上）以得到干净的正文
        ln_clean = index_or_bullet_re.sub('', ln_stripped)
        # 去除全角空格与不间断空格
        ln_clean = ln_clean.replace('\u3000', '').replace('\xa0', '')
        cleaned_lines.append((ln_clean, had_index_or_bullet))

    if not cleaned_lines:
        return ""

    # 如果整组行中至少有一行带序号/无序符号，则对所有带标记的行按出现顺序编号
    any_indexed = any(flag for _, flag in cleaned_lines)

    result_parts: list[str] = []
    numbering_counter = 1
    for i, (part, had_flag) in enumerate(cleaned_lines):
        if any_indexed and had_flag:
            # 为带标记的行添加递增序号
            numbered = f"{numbering_counter}.{part}" if part else f"{numbering_counter}."
            numbering_counter += 1
        else:
            # 不带标记的行保留原文（不加序号）
            numbered = part

        if not result_parts:
            result_parts.append(numbered)
        else:
            prev_raw = cleaned_lines[i - 1][0] if i - 1 >= 0 else ''
            prev_last_char = prev_raw[-1] if prev_raw else ''
            if punct_re.match(prev_last_char):
                # 若上一段以标点结束，直接连接
                result_parts.append(numbered)
            else:
                # 否则在前面插入中文分号再连接
                result_parts.append('；' + numbered)

    joined = ''.join(result_parts)

    # 把所有连续空白先折叠为单个空格（便于后续判断）
    joined = re.sub(r"\s+", " ", joined)

    # 当相邻两个 token 都是纯 ASCII 单词/数字时，保留单空格；否则删除空格
    tokens = joined.split(' ')
    if len(tokens) == 1:
        return tokens[0]

    def is_ascii_word(tok: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9]+", tok) is not None

    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == '':
            continue
        out.append(tok)
        if i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok == '':
                continue
            if is_ascii_word(tok) and is_ascii_word(next_tok):
                out.append(' ')
            else:
                # 不保留空格
                pass

    result_text = ''.join(out)

    # 如果最终输出非空且末尾没有常见句末标点，则追加中文句号 '。'
    # 常见句末标点包括：。！？；，、： 以及英文的 . , ! ? ;
    if result_text and not re.search(r'[。]\s*$', result_text):
        result_text = result_text.rstrip() + '。'

    return result_text


def standardize_aplication_field_scenario_cell(val: object) -> object:
    """标准化 aplication_field_scenario：
    - 输入通常为 JSON 字符串，如: [{"title":"生产","describe":"区域。\n"},{"title":"技术","describe":"资源。\n"}]
    - 字段取舍规则：若 describe 不为空，则只保留 describe；否则保留非空的 title；两者都空则忽略该项。
        - 为输出项按顺序添加序号前缀（1., 2., ...），并统一为 “n.” 的格式：
            无论原文本是否已带序号（如 1. / 1、 / (1) / （1）/ 1：/ 1) 等），都先剥离原有序号样式，再按顺序添加标准格式 “n.”。
    - 去掉 title/describe 中的所有换行/制表/全角空格/不间断空格，并去首尾空白。
    - 若 JSON 解析失败，回退用正则尽力提取配对的 title/describe。
    """
    s = "" if val is None else str(val)
    if not s or s.strip() == "":
        return val

    def clean_txt(x: str) -> str:
        if x is None:
            return ""
        x = str(x)
        # 去除全角空格与不间断空格
        x = x.replace('\u3000', '').replace('\xa0', '')
        # 去除换行/制表等空白
        x = re.sub(r"[\r\n\t]", "", x)
        # 去首尾空白
        x = x.strip()
        return x

    pairs = []  # List[Tuple[title, describe]]
    parsed = None
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        parsed = [parsed]

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            title = clean_txt(item.get('title', ''))
            desc = clean_txt(item.get('describe', ''))
            if title or desc:
                pairs.append((title, desc))
    else:
        # 回退：正则提取 "title":"..." 与 "describe":"..."（尽力）
        # 允许中间有其它内容，采用 DOTALL
        for m in re.finditer(r'"title"\s*:\s*"(.*?)".*?"describe"\s*:\s*"(.*?)"', s, flags=re.DOTALL):
            title = clean_txt(m.group(1))
            desc = clean_txt(m.group(2))
            if title or desc:
                pairs.append((title, desc))

    if not pairs:
        # 即便回退为原始文本，也先清理 HTML 标签
        return clean_txt(remove_html_tags(s))

    def strip_leading_index(text: str) -> str:
        # 剥离开头的序号样式：1. / 1、 / 1) / (1) / （1）/ 1：/ 1: 等
        return re.sub(r"^\s*[（(]?\s*\d+\s*[\.、)）：:]\s*", "", text, count=1)

    out_parts: list[str] = []
    idx = 1
    for t, d in pairs:
        t_empty = (t is None or str(t).strip() == "")
        d_empty = (d is None or str(d).strip() == "")
        if d_empty and t_empty:
            continue
        # 选择输出文本：优先 describe，其次 title
        chosen = (str(d).strip() if not d_empty else str(t).strip())
        # 统一序号：先剥离原有样式，再按标准格式添加，如 “1.”、“2.”
        body = strip_leading_index(chosen).strip()
        chosen = f"{idx}.{body}" if body else f"{idx}."
        # 递增序号，保持顺序
        idx += 1
        out_parts.append(chosen)

    # 直接无分隔符拼接
    # 对最终拼接结果也去除可能残留的 HTML 标签（保险起见）
    return remove_html_tags(''.join(out_parts))


def standardize_scene_lable_cell(val: object) -> str:
    """标准化 scene_lable 字段：
    - 把常见中英文标点统一替换为中文逗号 '，'
    - 将连续多个标点折叠为一个中文逗号
    - 删除首尾的逗号
    返回始终为字符串（空值 -> 空串）。
    """
    s = "" if val is None else str(val)
    s = s.strip()
    if s == "":
        return ""

    # 把一系列可能的标点符号替换为中文逗号
    # 包含常见的中英文逗号/分号/句号/顿号/冒号/问号/感叹号/斜杠/括号/引号/中划线等
    punct_pattern = r"[，,;；:：\.。!！\?？、/／\\\|\-—()（）\[\]【】{}<>\"'“”‘’·•…]+"
    s = re.sub(punct_pattern, '，', s)
    # 折叠连续中文逗号为一个
    s = re.sub(r'，+', '，', s)
    # 删除首尾的逗号
    s = s.strip('，')
    return s


def standardize_title_cell(val: object) -> str:
    """标准化 title 字段：
    - 删除所有出现的前缀 '成果：' 或 '成果:'（不区分空格），会去除首部的该前缀并返回干净的字符串。
    - 返回字符串（空值 -> 空串）。
    """
    s = "" if val is None else str(val)
    s = s.strip()
    if s == "":
        return ""
    # 删除所有出现的 '成果：' 或 '成果:'（任意位置），允许周围可选空白
    s = re.sub(r'成果\s*[:：]\s*', '', s)

    # 折叠连续空白为单个空格，便于后续判定
    s = re.sub(r"\s+", " ", s)

    # 删除单个不成对的 ASCII 双引号：保留成对的双引号，删除未配对的 " 字符
    # 实现：扫描字符串，使用布尔开关跟踪当前是否在配对状态；如果遇到一个双引号且该位置无法配对，则跳过（删除）
    if '"' in s:
        out_chars: list[str] = []
        stack = []  # 用栈记录未配对的双引号位置（用于更严格的配对判断）
        for ch in s:
            if ch == '"':
                # 如果栈为空，尝试查找后续是否存在配对（更昂贵），这里采用逐步配对策略：
                # 若下一个双引号会形成配对，则保留当前双引号并把位置记录到栈中；否则视为不成对，删除。
                if stack:
                    # 当前为配对的闭合引号，保留并弹出栈顶
                    out_chars.append(ch)
                    stack.pop()
                else:
                    # 尝试检查剩余字符串中是否还有双引号可配：
                    # 若存在，则把当前视为开引号，保留并记录；否则跳过（删除）
                    # 这里使用简单的存在检查以避免额外复杂性
                    # 注意：这种策略在极少数复杂场景下与语义解析不同，但对去除孤立引号很实用
                    if '"' in s[s.index(ch)+1:]:
                        out_chars.append(ch)
                        stack.append(len(out_chars)-1)
                    else:
                        # 不保留该孤立引号（删除）
                        pass
            else:
                out_chars.append(ch)
        s = ''.join(out_chars)

    # 仅在相邻两个 token 都为纯 ASCII 单词/数字时保留单空格；否则删除空格
    def is_ascii_word(tok: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9]+", tok) is not None

    tokens = s.split(' ')
    if len(tokens) == 1:
        return tokens[0]

    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == '':
            continue
        out.append(tok)
        if i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok == '':
                continue
            if is_ascii_word(tok) and is_ascii_word(next_tok):
                out.append(' ')
            else:
                # 不保留空格
                pass

    return ''.join(out)


def remove_html_tags(s: object) -> str:
    """移除字符串中的 HTML 标签标记（如 <tag> 与 </tag>），但保留标签内部的文本。

    说明：
    - 若输入为 None，返回空字符串；否则将输入转换为字符串并移除标签标记与孤立的尖括号。
    - 本函数不会删除标签内的文本内容；对复杂 HTML 或实体解码有局限，必要时请使用
      更专业的解析器（例如 BeautifulSoup）或先用 html.unescape 解码实体。
    """
    if s is None:
        return ""
    text = str(s)
    # 将 HTML 标签本身移除，但保留标签内的文本内容。
    # 匹配开标签如 <tag ...> 和闭标签 </tag>（允许属性），并替换为空字符串。
    try:
        # 删除所有开标签，例如 <div class="x"> -> ''
        text = re.sub(r"<\s*[a-zA-Z0-9:_-]+(?:\s+[^<>]*?)?>", "", text)
        # 删除所有闭标签，例如 </div> -> ''
        text = re.sub(r"<\s*/\s*[a-zA-Z0-9:_-]+\s*>", "", text)
        # 如果还有孤立的尖括号或不完整标签，移除尖括号但保留其中内容
        text = text.replace('<', '').replace('>', '')
        # 同时删除特定需要剔除的符号，如实心箭头 ▶
        text = text.replace('▶', '')
    except re.error:
        # 发生正则错误时，回退为删除尖括号的简单处理
        text = text.replace('<', '').replace('>', '')
    return text.strip()


def main():
    input_path = INPUT_PATH
    sheet = SHEET
    output_path = OUTPUT_PATH

    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        sys.exit(2)

    # 读取 Excel（全部以字符串读入，防止类型丢失）
    try:
        df = pd.read_excel(input_path, sheet_name=sheet, dtype=str, keep_default_na=False)
    except Exception as e:
        print("读取 Excel 失败:", e)
        sys.exit(3)

    before_cols = len(df.columns)
    before_rows = len(df)

    # 删列
    df_clean = drop_invalid_columns(df, DROP_COLS)
    after_cols = len(df_clean.columns)

    # 标准化 application 字段（若存在）
    if 'application' in df_clean.columns:
        df_clean['application'] = df_clean['application'].map(standardize_application_cell)

    # 标准化 aplication_field_scenario 字段（若存在）
    if 'aplication_field_scenario' in df_clean.columns:
        df_clean['aplication_field_scenario'] = df_clean['aplication_field_scenario'].map(standardize_aplication_field_scenario_cell)

    # 标准化 analyse_contect 字段（若存在）：删除所有空格与换行
    if 'analyse_contect' in df_clean.columns:
        df_clean['analyse_contect'] = df_clean['analyse_contect'].map(standardize_analyse_contect_cell)
        # 删除字段值为 "无" 的行（去首尾空格后匹配）
        df_clean = df_clean[~df_clean['analyse_contect'].astype(str).str.strip().isin(['无'])]

    # 删除 title 和 analyse_contect 完全一致的行
    if 'title' in df_clean.columns and 'analyse_contect' in df_clean.columns:
        before_len = len(df_clean)
        df_clean = df_clean[df_clean['title'] != df_clean['analyse_contect']].copy()
        removed = before_len - len(df_clean)
        if removed:
            print(f"删除 title 和 analyse_contect 完全一致的行数: {removed}")

    # 若存在 main_function 字段，剥离其中的 HTML 标签标记并去首尾空白
    if 'main_function' in df_clean.columns:
        # 先去除常见的 non-breaking-space 实体，然后剥离 HTML 标签标记并去首尾空白
        def clean_main_function(v: object) -> str:
            if v is None:
                return ""
            s = str(v)
            # 删除 HTML 空白实体 &nbsp; / &nbsp / &#160;
            s = re.sub(r'&nbsp;?|&#160;?', '', s, flags=re.IGNORECASE)
            return remove_html_tags(s).strip()

        df_clean['main_function'] = df_clean['main_function'].map(clean_main_function)

    # 若存在 main_advantage 字段，按同样规则先删除 HTML 空白实体，再剥离 HTML 标签并去首尾空白
    if 'main_advantage' in df_clean.columns:
        def clean_main_advantage(v: object) -> str:
            if v is None:
                return ""
            s = str(v)
            # 删除 HTML 空白实体 &nbsp; / &nbsp / &#160;
            s = re.sub(r'&nbsp;?|&#160;?', '', s, flags=re.IGNORECASE)
            return remove_html_tags(s).strip()

        df_clean['main_advantage'] = df_clean['main_advantage'].map(clean_main_advantage)

    # 若存在 scene_lable 字段，统一标点为中文逗号并合并多重标点
    if 'scene_lable' in df_clean.columns:
        df_clean['scene_lable'] = df_clean['scene_lable'].map(standardize_scene_lable_cell)

    # 若存在 chain_lable 字段，按与 scene_lable 相同的规则进行标准化
    if 'chain_lable' in df_clean.columns:
        df_clean['chain_lable'] = df_clean['chain_lable'].map(standardize_scene_lable_cell)

    # 若存在 title 字段，去除所有出现的 '成果：' / '成果:' 标签
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].map(standardize_title_cell)
        # 如果配置了需要删除的 title 列表，则删除匹配的行（对 DELETE_TITLES 中的项也先做 standardize_title_cell 规范化）
        if DELETE_TITLES:
            # 允许用户直接在 DELETE_TITLES 中写入带有 '成果：' 的原始标题；这里把它们规范化后再比较
            normalized_to_remove = [standardize_title_cell(t) for t in DELETE_TITLES if t and str(t).strip() != ""]
            titles_to_remove = set([t.strip() for t in normalized_to_remove if t and str(t).strip() != ""])
            if titles_to_remove:
                df_clean = df_clean[~df_clean['title'].astype(str).str.strip().isin(titles_to_remove)]

    # 按 title 去重：在 title 标准化与用户指定删除之后，去除 title 完全重复的记录，保留首次出现的那一行
    if 'title' in df_clean.columns:
        before_dedup = len(df_clean)
        # 确保 title 列为字符串并去除首尾空白，再按 title 去重
        df_clean['title'] = df_clean['title'].astype(str).str.strip()
        df_clean = df_clean.drop_duplicates(subset=['title'], keep='first')
        dedup_removed = before_dedup - len(df_clean)
        if dedup_removed:
            print(f"按 title 去重，删除重复行数: {dedup_removed}")

    # 删空值行（默认对所有保留列；若配置 REQUIRED_NONEMPTY_COLS 则仅对指定列）
    req_cols = REQUIRED_NONEMPTY_COLS if REQUIRED_NONEMPTY_COLS else list(df_clean.columns)
    df_clean = drop_rows_with_empty_fields(df_clean, req_cols)
    after_rows = len(df_clean)

    # 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        df_clean.to_excel(output_path, index=False)
    except Exception as e:
        print("保存文件失败:", e)
        sys.exit(4)

    print(f"原始列数: {before_cols}")
    print(f"清洗后列数: {after_cols}")
    removed = [c for c in DROP_COLS if c in df.columns]
    print(f"删除列数量: {len(removed)}")
    if removed:
        print("已删除列: ", ", ".join(removed))
    print(f"原始行数: {before_rows}")
    print(f"清洗后行数: {after_rows}")


if __name__ == "__main__":
    main()