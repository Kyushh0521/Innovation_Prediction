
import json
from collections import defaultdict
import os
import re
import sys

def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"文件 {path} 的第 {i} 行包含无效的 JSON：{e}")
    return items


def append_matches_to_prompts(prompts, matches):
    """从 matches 中提取所有专家的 application，并合并追加到 prompts 中。

    返回: (updated_prompts, total_matches_count)
    每个更新后的 prompt 包含键 'research_field'，值为去重后的字符串（可能为空）。
    """

    def clean_text(s):
        """清理字符串：将非字符串转为字符串，strip 并返回空字符串表示无效。"""
        if not s:
            return ''
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return ''
        return s.strip()

    def split_and_normalize(s):
        """仅按英文分号 ';' 将 application 字符串拆分并清理每一项，返回非空列表"""
        if not s:
            return []
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return []
        parts = s.split(';')
        return [p.strip() for p in parts if p and p.strip()]

    # 为每个 enterprise_index 收集 application（使用 set 去重）
    grouped_fields = defaultdict(set)
    total_matches = 0

    for m in matches:
        if 'enterprise_index' not in m:
            continue
        try:
            idx = int(m['enterprise_index'])
        except Exception:
            continue

        # 获取匹配载荷（可能是单项或列表）
        if 'match' in m:
            payload = m['match']
        elif 'matches' in m:
            payload = m['matches']
        else:
            payload = {k: v for k, v in m.items() if k != 'enterprise_index'}

        # payload 可以是 list 或 dict 或字符串
        items = []
        if isinstance(payload, list):
            items = payload
        else:
            items = [payload]

        for it in items:
            # 如果条目本身是字符串，直接作为一个 research_field 文本保留
            if isinstance(it, str):
                for piece in split_and_normalize(it):
                    txt = clean_text(piece)
                    if txt:
                        grouped_fields[idx].add(txt)
            elif isinstance(it, dict):
                # 优先从 match 的 expert 字段提取 application（保留原字符串）
                if 'expert' in it and isinstance(it['expert'], dict):
                    expert = it['expert']
                    if 'application' in expert and expert['application']:
                        val = expert['application']
                        if isinstance(val, list):
                            for v in val:
                                for piece in split_and_normalize(v):
                                    txt = clean_text(piece)
                                    if txt:
                                        grouped_fields[idx].add(txt)
                        else:
                            for piece in split_and_normalize(val):
                                txt = clean_text(piece)
                                if txt:
                                    grouped_fields[idx].add(txt)
                        continue

                # 回退到遍历 dict 的所有值以寻找可能的 application 描述（保留原字符串）
                for v in it.values():
                    if isinstance(v, list):
                        for vv in v:
                            for piece in split_and_normalize(vv):
                                txt = clean_text(piece)
                                if txt:
                                    grouped_fields[idx].add(txt)
                    elif isinstance(v, str):
                        for piece in split_and_normalize(v):
                            txt = clean_text(piece)
                            if txt:
                                grouped_fields[idx].add(txt)

        total_matches += 1

    # 构建最终 updated 列表，将合并后的 application 字符串写入每个 prompt
    updated = []
    # 显式使用每个 prompt 的 enterprise_index 字段来映射聚合结果
    for p in prompts:
        p_copy = dict(p)
        p_idx = None
        if 'enterprise_index' in p:
            try:
                p_idx = int(p['enterprise_index'])
            except Exception:
                p_idx = None

        parts = sorted(grouped_fields.get(p_idx, [])) if p_idx is not None else []
        if parts:
            # 使用中文分号合并为一个字符串
            merged = '；'.join(parts)
            # 写入键名为 research_field
            p_copy['research_field'] = merged
        else:
            p_copy['research_field'] = ''
        updated.append(p_copy)

    return updated, total_matches


def append_achievements_to_prompts(prompts, matches):
    """从 matches 中提取所有成果的 title:analyse_contect，并合并追加到 prompts 中。

        规则与 append_matches_to_prompts 对称：
        - 对每个 enterprise_index 收集成果描述，优先从显式的 'achievement' 字段读取（可为字符串或 dict）。
        - 对同一 enterprise_index 下的条目去重（使用 set），并按稳定顺序排序后按序号拼接为单个字符串。
            拼接格式为：
                成果1：title————analyse_contect 成果2：title2————analyse2
            各项直接拼接（不使用分号），序号从 1 开始依次递增，最终写入键名 'achievement'（字符串，可能为空）。

        返回: (updated_prompts, total_matches)
    """

    def clean_text(s):
        if not s:
            return ''
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return ''
        return s.strip()

    def make_achievement_text(title, analyse):
        t = clean_text(title)
        a = clean_text(analyse)
        if t and a:
            return f"{t}：{a}"
        if t:
            return t
        if a:
            return a
        return ''

    def format_achievements_concat(parts):
        """将去重后的 achievements 列表按序号拼接为字符串，格式示例：
        成果1：title——analyse_contect 成果2：title2——analyse2
        每项之间以一个空格连接，若只有 title 或 analyse 则只保留存在的部分。
        """
        if not parts:
            return ''
        out = []
        for i, p in enumerate(parts, start=1):
            if not p:
                continue
            # 优先按中文冒号拆分，否则按英文冒号拆分
            if '：' in p:
                title, analyse = p.split('：', 1)
            elif ':' in p:
                title, analyse = p.split(':', 1)
            else:
                title, analyse = p, ''
            title = title.strip()
            analyse = analyse.strip()
            if title and analyse:
                out.append(f"成果{i}：{title}————{analyse}")
            elif title:
                out.append(f"成果{i}：{title}")
            elif analyse:
                out.append(f"成果{i}：{analyse}")
        # 直接拼接并以空格分隔各项，避免使用分号
        return ''.join(out)

    grouped = defaultdict(set)
    total_matches = 0

    for m in matches:
        if 'enterprise_index' not in m:
            continue
        try:
            idx = int(m['enterprise_index'])
        except Exception:
            continue

        if 'match' in m:
            payload = m['match']
        elif 'matches' in m:
            payload = m['matches']
        else:
            payload = {k: v for k, v in m.items() if k != 'enterprise_index'}

        items = []
        if isinstance(payload, list):
            items = payload
        else:
            items = [payload]

        for it in items:
            # 字符串直接尝试作为 achievement 文本（保留原字符串）
            if isinstance(it, str):
                txt = clean_text(it)
                if txt:
                    grouped[idx].add(txt)
            elif isinstance(it, dict):
                # 优先使用 explicit achievement 字段
                if 'achievement' in it and it['achievement']:
                    ach = it['achievement']
                    if isinstance(ach, dict):
                        txt = make_achievement_text(ach.get('title'), ach.get('analyse_contect'))
                        if txt:
                            grouped[idx].add(txt)
                        else:
                            # 如果 dict 中有其他字符串值也尝试追加
                            for v in ach.values():
                                if isinstance(v, str):
                                    vv = clean_text(v)
                                    if vv:
                                        grouped[idx].add(vv)
                    elif isinstance(ach, str):
                        vv = clean_text(ach)
                        if vv:
                            grouped[idx].add(vv)
                    continue

                # 如果当前 dict 本身可能是一个 achievement 对象
                if 'title' in it or 'analyse_contect' in it:
                    txt = make_achievement_text(it.get('title'), it.get('analyse_contect'))
                    if txt:
                        grouped[idx].add(txt)
                        continue

                # 回退到遍历 dict 的所有值以寻找可能的成果描述
                for v in it.values():
                    if isinstance(v, dict):
                        txt = make_achievement_text(v.get('title'), v.get('analyse_contect'))
                        if txt:
                            grouped[idx].add(txt)
                    elif isinstance(v, list):
                        for vv in v:
                            if isinstance(vv, dict):
                                txt = make_achievement_text(vv.get('title'), vv.get('analyse_contect'))
                                if txt:
                                    grouped[idx].add(txt)
                            elif isinstance(vv, str):
                                s = clean_text(vv)
                                if s:
                                    grouped[idx].add(s)
                    elif isinstance(v, str):
                        s = clean_text(v)
                        if s:
                            grouped[idx].add(s)

        total_matches += 1

    updated = []
    for p in prompts:
        p_copy = dict(p)
        p_idx = None
        if 'enterprise_index' in p:
            try:
                p_idx = int(p['enterprise_index'])
            except Exception:
                p_idx = None

        parts = sorted(grouped.get(p_idx, [])) if p_idx is not None else []
        if parts:
            # 按指定的新格式拼接（成果序号依次往后推）
            merged = format_achievements_concat(parts)
            p_copy['achievement'] = merged
        else:
            p_copy['achievement'] = ''
        updated.append(p_copy)

    return updated, total_matches


def main():
    enterprise_prompt_path = 'data_process_outputs/enterprises_inputs.jsonl'
    experts_matches_path = 'data_process_outputs/enterprises_experts_matches.jsonl'
    achievements_matches_path = 'data_process_outputs/enterprises_achievements_matches.jsonl'
    out_path = 'data_process_outputs/enterprises_inputs_with_matches.jsonl'

    # 检查必需的文件存在性
    if not os.path.exists(enterprise_prompt_path):
        raise SystemExit(f"缺少文件：{enterprise_prompt_path}")
    if not os.path.exists(experts_matches_path):
        raise SystemExit(f"缺少文件：{experts_matches_path}")
    if not os.path.exists(achievements_matches_path):
        raise SystemExit(f"缺少文件：{achievements_matches_path}")

    prompts = load_jsonl(enterprise_prompt_path)
    experts_matches = load_jsonl(experts_matches_path)
    achievements_matches = load_jsonl(achievements_matches_path)

    # 先追加 experts 的 research_field
    prompts_with_research, total_expert_matches = append_matches_to_prompts(prompts, experts_matches)

    # 再追加 achievements 的 achievement 字段（在已有 prompts 基础上）
    prompts_with_both, total_achievement_matches = append_achievements_to_prompts(prompts_with_research, achievements_matches)

    # 仅写出 research_field 和 achievement 都非空的条目
    written = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in prompts_with_both:
            if item.get('research_field') and item.get('achievement'):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                written += 1

    print(f"专家 matches 计数：{total_expert_matches}")
    print(f"成果 matches 计数：{total_achievement_matches}")
    print(f"已写出 {written} 条（research_field 和 achievement 均非空）数据到：{out_path}")


if __name__ == '__main__':
    main()