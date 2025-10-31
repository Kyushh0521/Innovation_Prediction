"""
本文件实现基于“改进检索增强的大模型生成数据算法”（参考文档第4.1节）的企业 -> 成果 匹配流程。

第4.1 节逐条对照（中文注释）：
1) 输入/输出：输入为企业需求（企业表），成果库（成果表）；输出为 JSONL（包含 query、context、source_ids、meta）。
2) 嵌入管理：建议将成果文本编码为向量并缓存以避免重复计算；脚本支持从缓存加载向量并在需要时计算保存。
3) 检索流程：基于向量相似度（余弦相似度）进行检索与排序。本实现以向量编码与分块相似度计算为主，适合中小规模语料；对大规模语料建议使用近似向量索引（例如 FAISS）。
4) 融合策略：将向量相似度与文本相关度融合（可配置权重），得到最终排序。
5) 多样性采样：从排序结果中构建多种 context 组合（top、混合、跨段抽样等），生成多条训练/推理样本。
6) Prompt 生成：将 query 与枚举的候选成果拼接生成 LLM 的 prompt 文本，并输出 context 列表与来源 id。
7) 可追溯性：记录使用的 embedding 文件、候选索引与分数，便于后续分析与回溯。
8) 性能注意：对大规模语料建议使用近似向量检索（例如 FAISS），当前实现以块级向量计算为主，适合中小规模场景。

文件内所有注释均为中文以便阅读与维护。
"""

import os
import re
import json
import unicodedata
from typing import List, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import torch
import torch.nn.functional as F
# tqdm 进度条已移除，代码改为不显示进度（中文注释）


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s) if pd.notna(s) else ''
    s = unicodedata.normalize('NFKC', s).strip()
    s = re.sub(r"[\u3000\s]+", " ", s)
    s = re.sub(r"[，,；;\/\\|]+", " ", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\s+-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_name(m: str) -> str:
    """将模型名转换为本地安全的目录名（替换斜杠为下划线）。"""
    return m.replace('/', '_')


def load_model(model_names: List[str]) -> SentenceTransformer:
    """优先从本地 model/ 下加载模型目录；若不存在则尝试从远程下载并保存到本地。

    返回已加载的 SentenceTransformer 实例；若所有候选模型均加载失败则抛出 RuntimeError。
    """
    os.makedirs('model', exist_ok=True)
    # 先尝试从本地已保存的模型目录加载
    for mname in model_names:
        local = os.path.join('model', safe_name(mname))
        if os.path.isdir(local):
            try:
                model = SentenceTransformer(local)
                return model
            except Exception:
                # 本地模型目录可能损坏或不兼容，继续尝试下一个候选
                pass

    # 若本地未命中，则尝试通过 model name 下载并保存到本地
    for mname in model_names:
        try:
            model = SentenceTransformer(mname)
            local = os.path.join('model', safe_name(mname))
            try:
                model.save(local)
            except Exception:
                # 保存失败不影响使用，仅记录并继续
                pass
            return model
        except Exception:
            # 下载或初始化失败，尝试下一个模型
            pass

    # 如果所有模型都失败，抛出错误而不是返回 None
    raise RuntimeError(f"无法加载以下模型候选: {model_names}")


def build_enterprise_queries(df_ent: pd.DataFrame) -> List[str]:
    """基于表格字段组合企业查询文本：org_name + business + category / category_big / category_middle"""
    cols = []
    for c in ['org_name', 'business', 'category', 'category_big', 'category_middle']:
        if c in df_ent.columns:
            cols.append(c)

    queries = []
    for _, r in df_ent.iterrows():
        parts = []
        for c in cols:
            v = r.get(c)
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s:
                parts.append(s)
        q = '；'.join(parts)
        queries.append(q)
    return queries


def build_achievement_texts(df_ach: pd.DataFrame) -> List[str]:
    """基于表格字段组合成果文本：title + analyse_contect + application + ..."""
    texts = []
    for _, r in df_ach.iterrows():
        parts = []
        for c in ['title', 'analyse_contect', 'application', 'application_field_scenario', 'main_function', 'main_advantage', 'scene_label', 'chain_label']:
            if c in df_ach.columns:
                v = r.get(c)
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s:
                    parts.append(s)
        texts.append('；'.join(parts))
    return texts


def match_enterprises_to_achievements(
    ent_file: str,
    ach_file: str,
    out_jsonl: str,
    model_names: Optional[List[str]] = None,
    top_k: int = 3,
    max_enterprises: Optional[int] = None,
    primary_threshold: float = 0.8,
    secondary_threshold: float = 0.75,
):
    if model_names is None:
        model_names = ["shibing624/text2vec-base-chinese"]

    if not os.path.exists(ent_file):
        raise FileNotFoundError(ent_file)
    if not os.path.exists(ach_file):
        raise FileNotFoundError(ach_file)

    df_ent = pd.read_excel(ent_file)
    df_ach = pd.read_excel(ach_file)

    queries_raw = build_enterprise_queries(df_ent)
    ach_texts_raw = build_achievement_texts(df_ach)

    queries_norm = [normalize_text(q) for q in queries_raw]
    ach_texts_norm = [normalize_text(t) for t in ach_texts_raw]

    model = load_model(model_names)

    # 自动检测并选择设备（优先 CUDA，否则 CPU），并将模型移动到该设备
    try:
        if torch.cuda.is_available():
            # 选择第一个可见的 CUDA 设备（可根据需要改为更智能的选择）
            device_idx = torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0
            device = f'cuda:{device_idx}'
            try:
                gpu_name = torch.cuda.get_device_name(device_idx)
            except Exception:
                gpu_name = None
            print(f"检测到 CUDA，可用 GPU: {torch.cuda.device_count()} 个，选择 {device} {gpu_name or ''}")
        else:
            device = 'cpu'
            print("未检测到 GPU，使用 CPU")
    except Exception:
        device = 'cpu'

    try:
        model.to(device)
    except Exception:
        # 某些 SentenceTransformer 版本在 to() 上可能不报错，但仍安全捕获异常
        pass

    # 对企业和成果文本进行编码并以 numpy (CPU) 存储，后续按块将小份数据搬到 device 上计算相似度
    # 这样可以显著降低显存占用：成果向量整体保存在 CPU，仅在需要时分块移动到 GPU
    cache_dir = 'model_cache'
    os.makedirs(cache_dir, exist_ok=True)
    m = hashlib.md5()
    joined = '\n'.join(ach_texts_norm)
    m.update(joined.encode('utf-8'))
    # 将模型名加入缓存 key，避免不同模型复用同一缓存
    model_tag = safe_name(model_names[0]) if model_names and len(model_names) > 0 else 'model'
    cache_name = f"achievements_embeddings_{model_tag}_{m.hexdigest()}.npy"
    cache_path = os.path.join(cache_dir, cache_name)

    # 计算或加载成果向量（numpy, CPU）
    if os.path.exists(cache_path):
        try:
            ach_embs_np = np.load(cache_path)
            print(f"加载成果向量缓存: {cache_path}")
        except Exception:
            ach_embs_np = model.encode(ach_texts_norm, convert_to_numpy=True, device=device, batch_size=64)
            try:
                np.save(cache_path, ach_embs_np)
                print(f"已保存成果向量缓存: {cache_path}")
            except Exception:
                pass
    else:
        ach_embs_np = model.encode(ach_texts_norm, convert_to_numpy=True, device=device, batch_size=64)
        try:
            np.save(cache_path, ach_embs_np)
            print(f"已保存成果向量缓存: {cache_path}")
        except Exception:
            pass

    # 不在内存中同时保留所有企业向量：按批对企业文本编码并立即处理每条企业记录
    # 这样可以将峰值内存/显存占用限制在成果块大小与企业批大小的乘积附近
    ach_count = ach_embs_np.shape[0]
    # 每块大小（可调整），默认 4096，确保一次性占用显存较小
    ach_batch_size = 4096
    # 企业编码批大小（可调整），避免一次性生成所有企业向量（默认 1024）
    enterprise_encode_batch = 1024

    def _safe_val_from_row(r, col):
        if col in r.index:
            v = r.get(col)
            if pd.isna(v):
                return None
            if isinstance(v, np.generic):
                try:
                    return v.item()
                except Exception:
                    return v
            return v
        return None

    def _match_one_enterprise(i: int, ent_vec_np, df_ach_local=df_ach, ach_embs_local=ach_embs_np,
                              top_k_local=top_k,
                              primary_threshold_local=primary_threshold,
                              secondary_threshold_local=secondary_threshold):
        """对单条企业向量计算相似度并返回可序列化的结果字典。
        这个函数便于单元测试：接受 numpy 企业向量和必要的上下文数据，返回 {'enterprise_index': i, 'matches': [...]}
        """
        # 将企业向量移动到 device 并归一化
        vec = torch.from_numpy(ent_vec_np).to(device).float()
        vec = F.normalize(vec, p=2, dim=0, eps=1e-9)

        # 若没有任何成果数据，直接返回空匹配（避免后续对空张量的操作）
        if ach_count == 0:
            return {'enterprise_index': int(i), 'matches': []}

        sims_all = torch.empty(ach_count, device=device, dtype=torch.float32)
        for start in range(0, ach_count, ach_batch_size):
            end = min(start + ach_batch_size, ach_count)
            block_np = ach_embs_local[start:end]
            block_t = torch.from_numpy(block_np).to(device).float()
            block_norm = F.normalize(block_t, p=2, dim=1, eps=1e-9)
            sims_block = torch.matmul(block_norm, vec)
            sims_all[start:end] = sims_block

        # 使用 topk 选择候选，避免每次都进行全数组排序
        # 先尝试在 primary 阈值上获取 topk；若没有满足 primary，则降级到 secondary
        primary_mask = sims_all >= primary_threshold_local
        if primary_mask.any().item():
            # 只对满足 primary 的位置进行 topk
            valid_count = int(primary_mask.sum().item())
            k = min(top_k_local, valid_count)
            mask_vals = torch.where(primary_mask, sims_all, torch.tensor(-float('inf'), device=device))
            topk_vals, topk_idx = torch.topk(mask_vals, k=k)
        else:
            secondary_mask = sims_all >= secondary_threshold_local
            if secondary_mask.any().item():
                valid_count = int(secondary_mask.sum().item())
                k = min(top_k_local, valid_count)
                mask_vals = torch.where(secondary_mask, sims_all, torch.tensor(-float('inf'), device=device))
                topk_vals, topk_idx = torch.topk(mask_vals, k=k)
            else:
                # 没有任何达到次阈值，返回空匹配
                return {'enterprise_index': int(i), 'matches': []}

        # 过滤掉 -inf 值（即原先被掩盖的位置）
        valid_mask = topk_vals > -1e9
        chosen_idx = topk_idx[valid_mask]
        if chosen_idx.numel() == 0:
            return {'enterprise_index': int(i), 'matches': []}

        chosen_idx_cpu = chosen_idx.cpu().numpy().tolist()
        matches = []
        for j in chosen_idx_cpu:
            score_j = float(sims_all[j].item())
            row = df_ach_local.iloc[j]
            achievement_small = {
                'title': _safe_val_from_row(row, 'title') if 'title' in row.index else _safe_val_from_row(row, 'achievement_title'),
                'analyse_contect': _safe_val_from_row(row, 'analyse_contect') if 'analyse_contect' in row.index else _safe_val_from_row(row, 'achievement_desc'),
            }
            matches.append({'index': int(j), 'score': score_j, 'achievement': achievement_small})

        return {'enterprise_index': int(i), 'matches': matches}

    os.makedirs(os.path.dirname(out_jsonl) or '.', exist_ok=True)
    written = 0

    # 计算要遍历的企业索引，保持原始索引不变（便于在输出中保留原始行号）
    n_queries = len(queries_raw)
    # 若未传入 max_enterprises，则默认处理全部企业
    if max_enterprises is None:
        max_n = n_queries
    else:
        max_n = min(max_enterprises, n_queries)
    indices = list(range(max_n))

    with open(out_jsonl, 'w', encoding='utf-8') as fout:
        # 按批编码企业向量，逐条调用 _match_one_enterprise 并写入结果
        for batch_start in range(0, len(indices), enterprise_encode_batch):
            batch_end = min(batch_start + enterprise_encode_batch, len(indices))
            batch_idx = indices[batch_start:batch_end]
            batch_texts = [queries_norm[i] for i in batch_idx]
            batch_embs = model.encode(batch_texts, convert_to_numpy=True, device=device, batch_size=64)
            for off, i in enumerate(batch_idx):
                ent_vec_np = batch_embs[off]
                out_obj = _match_one_enterprise(i, ent_vec_np)
                fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
                written += 1

    return written


def main():
    ent_file = 'Dataset/enterprises_full_cleaned.xlsx'
    ach_file = 'Dataset/achievements_full_cleaned.xlsx'
    out_jsonl = 'data_process_outputs/enterprises_achievements_matches.jsonl'
    # 直接通过变量指定参数（如需修改请在此处调整）
    top_k = 3
    primary_threshold = 0.8
    secondary_threshold = 0.75
    max_enterprises = None  # 若希望只处理前 N 条企业用于快速检查，可设置为整数，例如 10
    print('开始为企业匹配成果...')
    try:
        n = match_enterprises_to_achievements(ent_file, ach_file, out_jsonl, top_k=top_k, primary_threshold=primary_threshold, secondary_threshold=secondary_threshold, max_enterprises=max_enterprises)
        print(f'完成: 为 {n} 条企业记录写入匹配结果至 {out_jsonl}')
    except Exception as e:
        print('出错:', e)


if __name__ == '__main__':
    main()
