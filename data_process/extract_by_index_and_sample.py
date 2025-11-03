import json
import os
import pandas as pd


INPUT_JSONL = os.path.join('data_process_outputs', 'enterprises_inputs_with_matches.jsonl')
INPUT_XLSX = os.path.join('Dataset', 'enterprises_full_cleaned.xlsx')
OUTPUT_XLSX = os.path.join('data_process_outputs', 'extracted_enterprises_by_index.xlsx')


def read_indices(jsonl_path):
    """从 JSONL 读取 enterprise_index 列表（假设每行包含键 'enterprise_index'）。"""
    with open(jsonl_path, 'r', encoding='utf-8') as fin:
        return [int(json.loads(line)['enterprise_index']) for line in fin if line.strip()]


def load_excel(xlsx_path):
    """加载 Excel 为 DataFrame（直接使用 pandas 读取）。"""
    return pd.read_excel(xlsx_path)


def extract_rows_by_indices(df, indices):
    """根据给定的整数索引列表按顺序从 df 中抽取行并返回新的 DataFrame。"""
    rows = []
    for idx in indices:
        # 使用 copy 防止对原始 df 的视图修改
        row = df.iloc[idx].copy()
        # 记录原始索引到新列 original_index
        row['original_index'] = int(idx)
        rows.append(row)
    return pd.DataFrame(rows)


def write_excel(df, path):
    df.to_excel(path, index=False)


def write_counts_txt(df_out, result_txt, cols):
    """对指定的列写入计数统计（按出现次数降序），每列用分隔线分隔。"""
    with open(result_txt, 'w', encoding='utf-8') as fout:
        for col in cols:
            fout.write('#' * 40 + '\n')
            fout.write(f'COLUMN: {col}\n')
            fout.write('#' * 40 + '\n')
            if col in df_out.columns:
                vc = df_out[col].dropna().astype(str).value_counts()
                fout.write(f'TOTAL UNIQUE: {len(vc)}\n')
                for label, cnt in vc.items():
                    fout.write(f'{label}\t{cnt}\n')
            else:
                fout.write(f'[MISSING COLUMN: {col}]\n')
            fout.write('\n')


def allocate_and_sample_by_category(df_out):
    """按 category 占比分配抽样并返回 sample_df 以及目标 target 行数。

    规则：总量 = 10%（四舍五入、至少1），且每个标签至少1条；如果类别数 > 初始 target，则把 target 扩大为类别数。
    """
    df_for_sample = df_out
    total_rows = len(df_for_sample)
    target = max(1, int(round(total_rows * 0.05)))

    cat_series = df_for_sample['category'].fillna('').astype(str)
    counts = cat_series.value_counts().to_dict()
    categories = list(counts.keys())

    if len(categories) > target:
        target = len(categories)

    def allocate_counts_simple(counts_dict, total_target):
        cats = list(counts_dict.keys())
        total_available = sum(counts_dict.values())
        alloc = {c: 0 for c in cats}
        if total_available <= total_target:
            for c in cats:
                alloc[c] = counts_dict[c]
            return alloc

        for c in cats:
            prop = counts_dict[c] / total_available
            alloc[c] = max(1, int(round(prop * total_target)))
            if alloc[c] > counts_dict[c]:
                alloc[c] = counts_dict[c]

        s = sum(alloc.values())
        if s > total_target:
            over = s - total_target
            for c in sorted(cats, key=lambda x: alloc[x], reverse=True):
                if over <= 0:
                    break
                can = alloc[c] - 1
                if can <= 0:
                    continue
                sub = min(can, over)
                alloc[c] -= sub
                over -= sub
        elif s < total_target:
            need = total_target - s
            for c, cap in sorted(((c, counts_dict[c] - alloc[c]) for c in cats), key=lambda x: x[1], reverse=True):
                if need <= 0:
                    break
                if cap <= 0:
                    continue
                add = min(cap, need)
                alloc[c] += add
                need -= add

        return alloc

    alloc = allocate_counts_simple(counts, target)

    sampled_rows = []
    for c, need in alloc.items():
        if need <= 0:
            continue
        group = df_for_sample[cat_series == c]
        if len(group) <= need:
            sampled = group
        else:
            sampled = group.sample(n=need, random_state=42)
        sampled_rows.append(sampled)

    if sampled_rows:
        sample_df = pd.concat(sampled_rows, ignore_index=True)
    else:
        sample_df = df_for_sample.head(0).copy()

    return sample_df, target


def write_sampled_jsonl_from_sample_df(sample_df, input_jsonl_path, output_jsonl_path):
    """更精简的实现：按 sample_df 中 unique original_index 输出每个索引在 JSONL 中的第一条匹配记录。

    说明：输出顺序为 JSONL 中匹配行的出现顺序。重复采样在此处被禁止（只输出每个索引的第一条匹配）。
    """
    if 'original_index' not in sample_df.columns:
        raise ValueError('sample_df must contain column original_index')

    # 只保留需要写出的唯一索引集合
    needed = set(int(x) for x in sample_df['original_index'].astype(int).tolist())
    written_set = set()

    with open(input_jsonl_path, 'r', encoding='utf-8') as fin, open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            idx = None
            if 'enterprise_index' in obj:
                try:
                    idx = int(obj['enterprise_index'])
                except Exception:
                    idx = None

            if idx is not None and idx in needed and idx not in written_set:
                fout.write(line.rstrip('\n') + '\n')
                written_set.add(idx)
                if written_set == needed:
                    break

    return len(written_set)


def main():
    indices = read_indices(INPUT_JSONL)
    df = load_excel(INPUT_XLSX)
    out_df = extract_rows_by_indices(df, indices)
    write_excel(out_df, OUTPUT_XLSX)
    print(f'写出 {len(out_df)} 行到: {OUTPUT_XLSX}')

    # 统计并写入 TXT
    result_txt = os.path.join('data_process_outputs', 'extracted_enterprises_category_counts.txt')
    write_counts_txt(out_df, result_txt, ['category', 'category_big', 'category_middle'])
    print(f'统计已写出: {os.path.abspath(result_txt)}')

    # 按 category 抽样并写出
    sample_xlsx = os.path.join('data_process_outputs', 'sample_by_category.xlsx')
    sample_df, target = allocate_and_sample_by_category(out_df)
    sample_df.to_excel(sample_xlsx, index=False)
    # 根据 sample_df 中的 original_index 抽取原始 JSONL 中对应的记录并写出
    output_sample_jsonl = os.path.join('data_process_outputs', 'sample_inputs_with_matches.jsonl')
    
    # 统计并写入 TXT
    sample_txt = os.path.join('data_process_outputs', 'sample_category_counts.txt')
    write_counts_txt(sample_df, sample_txt, ['category', 'category_big', 'category_middle'])
    print(f'统计已写出: {os.path.abspath(sample_txt)}')

    print(f'按 category 占比分配抽样完成，目标 {target} 行，实际抽取 {len(sample_df)} 行，文件: {sample_xlsx}')
    try:
        written = write_sampled_jsonl_from_sample_df(sample_df, INPUT_JSONL, output_sample_jsonl)
        print(f'已从 JSONL 中按采样索引写出 {written} 条记录到: {output_sample_jsonl}')
    except Exception as e:
        print(f'写出采样 JSONL 时出错: {e}')

if __name__ == '__main__':
    main()
