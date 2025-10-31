import os
import re
import json
import unicodedata
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import torch
import torch.nn.functional as F
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


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
    for c in ['business', 'category', 'category_big', 'category_middle']:
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


def build_expert_texts(df_exp: pd.DataFrame) -> List[str]:
    """基于表格字段组合专家文本：user_name + research_field + application"""
    texts = []
    for _, r in df_exp.iterrows():
        parts = []
        for c in ['research_field', 'application']:
            if c in df_exp.columns:
                v = r.get(c)
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s:
                    parts.append(s)
        texts.append('；'.join(parts))
    return texts


def match_enterprises_to_experts(
    ent_file: str,
    exp_file: str,
    out_jsonl: str,
    model_names: Optional[List[str]] = None,
    top_k: int = 3,
    max_enterprises: Optional[int] = None,
    primary_threshold: float = 0.8,
    keyword_alpha: float = 0.2,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 1),
    save_terms_path: Optional[str] = None,
):
    if model_names is None:
        model_names = ["shibing624/text2vec-base-chinese"]

    if not os.path.exists(ent_file):
        raise FileNotFoundError(ent_file)
    if not os.path.exists(exp_file):
        raise FileNotFoundError(exp_file)

    df_ent = pd.read_excel(ent_file)
    df_exp = pd.read_excel(exp_file)

    queries_raw = build_enterprise_queries(df_ent)
    exp_texts_raw = build_expert_texts(df_exp)

    queries_norm = [normalize_text(q) for q in queries_raw]
    exp_texts_norm = [normalize_text(t) for t in exp_texts_raw]

    model = load_model(model_names)

    # --- 关键词提取与 TF-IDF 构建（用于加权向量检索）
    # Blacklist: 可根据需要扩展
    TERM_BLACKLIST = set(["的", "和", "与", "及", "或", "等", "在", "为", "与", "对", "基于", "利用", "生产","加工","销售","产销","零售","制造", "服务", "批发", "开发", "产品", "活动", "工程", "应用", "其他", "有关", "相关", "支持", "提供", "技术", "及其", "研究", "应用", "行业", "专业", "非专业", "专用", "专门", "企业", "运行", "公开", "内容","平台","产业","从事","系统","方案","项目","管理","服务","服务平台","服务体系","服务网络"])

    def _tokenize_text(s: str):
        # 使用 jieba 分词并带词性标注，保留名词/动名词等可能的技术术语
        try:
            words = []
            for w, flag in pseg.cut(s):
                # 保留以 n (名词) 或 vn/v (动名词/动词) 开头的词，以及英文字母/数字混合项
                if not w or w.strip() == '':
                    continue
                if w in TERM_BLACKLIST:
                    continue
                if len(w) == 1:
                    # 跳过单字一般词
                    continue
                if flag and (flag.startswith('n') or flag.startswith('v') or flag == 'eng'):
                    words.append(w)
                else:
                    # 也尝试保留包含字母或数字的词（如技术名词）
                    if any(ch.isalnum() for ch in w):
                        words.append(w)
            return words
        except Exception:
            return []

    def build_term_tfidf(docs_exp: List[str], docs_ent: List[str], min_df: int = 2, ngram_range: Tuple[int,int]=(1,1)):
        # 使用 sklearn 的 TfidfVectorizer 来构建 TF-IDF 矩阵（稀疏矩阵）
        all_docs = docs_exp + docs_ent
        if len(all_docs) == 0:
            return [], {}, None, None, None

        def _jieba_tokenizer(s: str):
            # 返回 tokens 列表，不去重，供 TfidfVectorizer 使用
            try:
                toks = _tokenize_text(s)
                return toks
            except Exception:
                return []

        vectorizer = TfidfVectorizer(tokenizer=_jieba_tokenizer, min_df=min_df, ngram_range=ngram_range)
        X = vectorizer.fit_transform(all_docs)  # shape: (N_docs, n_features), csr_matrix

        feature_names = vectorizer.get_feature_names_out()
        term2idx = {t: i for i, t in enumerate(feature_names)}

        if X.shape[1] == 0:
            # 没有特征时仍返回 vectorizer 以便可能获取 idf_（为空）
            return list(feature_names), term2idx, None, None, vectorizer

        # 为避免对大矩阵切片的不兼容性，分别对专家文档和企业文档做 transform
        expert_tfidf = vectorizer.transform(docs_exp)
        query_tfidf = vectorizer.transform(docs_ent)

        return list(feature_names), term2idx, expert_tfidf, query_tfidf, vectorizer

    # 为 TF-IDF 构建包含专家的 research_field 与 application 字段，以及企业 business 字段的文本集合
    exp_app_texts = []
    for _, r in df_exp.iterrows():
        parts = []
        for c in ['application', 'research_field']:
            if c in df_exp.columns:
                vv = r.get(c)
                if pd.isna(vv):
                    continue
                s = str(vv).strip()
                if s:
                    parts.append(s)
        exp_app_texts.append(normalize_text('；'.join(parts)))

    ent_cat_texts = []
    for _, r in df_ent.iterrows():
        parts = []
        for c in ['business', 'category_middle']:
            if c in df_ent.columns:
                vv = r.get(c)
                if pd.isna(vv):
                    continue
                s = str(vv).strip()
                if s:
                    parts.append(s)
        ent_cat_texts.append(normalize_text('；'.join(parts)))

    terms, term2idx, expert_tfidf_matrix, query_tfidf_matrix, tfidf_vectorizer = build_term_tfidf(exp_app_texts, ent_cat_texts, min_df=min_df, ngram_range=ngram_range)

    # 若用户指定了保存路径，则将分词后的技术术语逐行写入文件（UTF-8）
    if save_terms_path:
        try:
            os.makedirs(os.path.dirname(save_terms_path) or '.', exist_ok=True)
            # 尝试获取 idf 权重；若不可用则使用 1.0
            idf_arr = None
            try:
                idf_arr = getattr(tfidf_vectorizer, 'idf_', None)
            except Exception:
                idf_arr = None
            with open(save_terms_path, 'w', encoding='utf-8') as tf:
                if terms:
                    # 只保留完全由中文汉字组成的词条（Unicode 区间 \u4e00-\u9fff）
                    zh_re = re.compile(r'^[\u4e00-\u9fff]+$')
                    term_weights = []
                    for idx, t in enumerate(terms):
                        if not t or not zh_re.match(t):
                            continue
                        weight = 1.0
                        try:
                            if idf_arr is not None and idx < len(idf_arr):
                                weight = float(idf_arr[idx])
                        except Exception:
                            weight = 1.0
                        term_weights.append((t, weight))
                    # 按权重降序排序后写入
                    term_weights.sort(key=lambda x: x[1], reverse=True)
                    for t, w in term_weights:
                        tf.write(f"{t}\t{w}\n")
        except Exception:
            # 保存失败不应阻塞主流程，仅打印信息
            try:
                print(f"警告: 无法将术语写入 {save_terms_path}")
            except Exception:
                pass

    # 仅使用 TF-IDF 矩阵用于关键词点乘评分（Jaccard/Top-N 已移除）


    # 自动检测并选择设备（优先 CUDA，否则 CPU），并将模型移动到该设备
    try:
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device() if torch.cuda.device_count() > 0 else 0
            device = f'cuda:{device_idx}'
            print(f"检测到 CUDA，可用 GPU: {torch.cuda.device_count()} 个，选择 {device}")
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

    # 对企业和专家文本进行编码并以 numpy (CPU) 存储，后续按块将小份数据搬到 device 上计算相似度
    # 这样可以显著降低显存占用：专家向量整体保存在 CPU，仅在需要时分块移动到 GPU
    cache_dir = 'model_cache'
    os.makedirs(cache_dir, exist_ok=True)
    m = hashlib.md5()
    joined = '\n'.join(exp_texts_norm)
    m.update(joined.encode('utf-8'))
    # 将模型名加入缓存 key，避免不同模型复用同一缓存
    model_tag = safe_name(model_names[0]) if model_names and len(model_names) > 0 else 'model'
    cache_name = f"experts_embeddings_{model_tag}_{m.hexdigest()}.npy"
    cache_path = os.path.join(cache_dir, cache_name)

    # 计算或加载专家向量（numpy, CPU）
    if os.path.exists(cache_path):
        try:
            exp_embs_np = np.load(cache_path)
            print(f"加载专家向量缓存: {cache_path}")
        except Exception:
            exp_embs_np = model.encode(exp_texts_norm, convert_to_numpy=True, device=device, batch_size=64)
            try:
                np.save(cache_path, exp_embs_np)
                print(f"已保存专家向量缓存: {cache_path}")
            except Exception:
                pass
    else:
        exp_embs_np = model.encode(exp_texts_norm, convert_to_numpy=True, device=device, batch_size=64)
        try:
            np.save(cache_path, exp_embs_np)
            print(f"已保存专家向量缓存: {cache_path}")
        except Exception:
            pass

    # 不在内存中同时保留所有企业向量：按批对企业文本编码并立即处理每条企业记录
    # 这样可以将峰值内存/显存占用限制在专家块大小与企业批大小的乘积附近
    exp_count = exp_embs_np.shape[0]
    # 每块大小（可调整），默认 4096，确保一次性占用显存较小
    exp_batch_size = 4096
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

    def _match_one_enterprise(i: int, ent_vec_np, df_exp_local=df_exp, exp_embs_local=exp_embs_np,
                              top_k_local=top_k,
                              primary_threshold_local=primary_threshold):
        """对单条企业向量计算相似度并返回可序列化的结果字典。
        这个函数便于单元测试：接受 numpy 企业向量和必要的上下文数据，返回 {'enterprise_index': i, 'matches': [...]}
        """
        # 将企业向量移动到 device 并归一化
        vec = torch.from_numpy(ent_vec_np).to(device).float()
        vec = F.normalize(vec, p=2, dim=0, eps=1e-9)

        # 若没有任何专家数据，直接返回空匹配（避免后续对空张量的操作）
        if exp_count == 0:
            return {'enterprise_index': int(i), 'matches': []}

        sims_all = torch.empty(exp_count, device=device, dtype=torch.float32)
        # 计算关键词相似度分数：使用 TF-IDF 矩阵点乘
        keyword_scores_np = None
        if expert_tfidf_matrix is not None and query_tfidf_matrix is not None and i < query_tfidf_matrix.shape[0]:
            try:
                qvec = query_tfidf_matrix.getrow(i)
                if qvec.nnz == 0:
                    keyword_scores_np = np.zeros(exp_count, dtype=np.float32)
                else:
                    try:
                        prod = expert_tfidf_matrix @ qvec.T
                        try:
                            keyword_scores_np = np.asarray(prod.todense()).reshape(-1).astype(np.float32)
                        except Exception:
                            keyword_scores_np = np.asarray(prod).reshape(-1).astype(np.float32)
                    except Exception:
                        try:
                            q_dense = qvec.toarray().reshape(-1).astype(np.float32)
                            prod_dense = expert_tfidf_matrix @ q_dense
                            keyword_scores_np = np.asarray(prod_dense).reshape(-1).astype(np.float32)
                        except Exception:
                            keyword_scores_np = None
            except Exception:
                keyword_scores_np = None
        for start in range(0, exp_count, exp_batch_size):
            end = min(start + exp_batch_size, exp_count)
            block_np = exp_embs_local[start:end]
            block_t = torch.from_numpy(block_np).to(device).float()
            block_norm = F.normalize(block_t, p=2, dim=1, eps=1e-9)
            sims_block = torch.matmul(block_norm, vec)
            sims_all[start:end] = sims_block

        # 将向量相似度从 [-1, 1] 映射到 [0, 1]
        sims_norm = (sims_all + 1.0) / 2.0

        # 若有关键词分数，则将其标准化到 [0,1] 并与向量相似度融合
        if keyword_scores_np is not None:
            # keyword_scores_np 已经是点乘结果，范围为 [-1,1] 或 [0,1]（因归一化）。确保在 [0,1]
            ks = keyword_scores_np.astype(np.float32)
            # 截取可能的异常值
            ks = np.clip(ks, 0.0, 1.0)
            # 转为 tensor
            ks_t = torch.from_numpy(ks).to(device).float()
            final_scores = (1.0 - keyword_alpha) * sims_norm + keyword_alpha * ks_t
        else:
            final_scores = sims_norm

        # 使用 final_scores 作为后续选择依据
        sims_all = final_scores

        # 使用 topk 选择候选，避免每次都进行全数组排序
        # 只在 primary 阈值上获取 topk，未达到则直接返回空匹配
        primary_mask = sims_all >= primary_threshold_local
        if primary_mask.any().item():
            # 只对满足 primary 的位置进行 topk
            valid_count = int(primary_mask.sum().item())
            k = min(top_k_local, valid_count)
            mask_vals = torch.where(primary_mask, sims_all, torch.tensor(-float('inf'), device=device))
            topk_vals, topk_idx = torch.topk(mask_vals, k=k)
        else:
            # 没有达到阈值，返回空匹配
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
            row = df_exp_local.iloc[j]
            expert_small = {
                'user_name': _safe_val_from_row(row, 'user_name'),
                'research_field': _safe_val_from_row(row, 'research_field'),
                'application': _safe_val_from_row(row, 'application'),
            }
            matches.append({'index': int(j), 'score': score_j, 'expert': expert_small})

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
    exp_file = 'Dataset/experts_full_cleaned_filled.xlsx'
    out_jsonl = 'data_process_outputs/enterprises_experts_matches.jsonl' 
    top_k = 3
    primary_threshold = 0.70
    max_enterprises = None  # 若希望只处理前 N 条企业用于快速检查，可设置为整数，例如 10
    model_names = ["shibing624/text2vec-base-chinese"]
    keyword_alpha = 0.2
    min_df = 2
    ngram_range = (1, 1)
    save_terms_path = 'data_process_outputs/terms.txt'

    print('开始为企业匹配专家...')
    try:
        n = match_enterprises_to_experts(
            ent_file,
            exp_file,
            out_jsonl,
            model_names=model_names,
            top_k=top_k,
            max_enterprises=max_enterprises,
            primary_threshold=primary_threshold,
            keyword_alpha=keyword_alpha,
            min_df=min_df,
            ngram_range=ngram_range,
            save_terms_path=save_terms_path,
        )
        print(f'完成: 为 {n} 条企业记录写入匹配结果至 {out_jsonl}')
    except Exception as e:
        print('出错:', e)


if __name__ == '__main__':
    main()