import os
import re
import unicodedata
from typing import Any
import pandas as pd
import numpy
from sentence_transformers import SentenceTransformer

def find_application_column(df: pd.DataFrame) -> str:
	"""不区分大小写地查找 'application' 列。"""
	for col in df.columns:
		if col == 'application':
			return col
	lower_map = {col.lower(): col for col in df.columns}
	if 'application' in lower_map:
		return lower_map['application']
	raise KeyError("在 DataFrame 中找不到 'application' 列。")

def build_mapping(infile: str) -> list:
	"""从 'application' 列构建一个唯一的应用领域术语列表。"""
	df = pd.read_excel(infile)
	app_col = find_application_column(df)

	terms = set()
	for val in df[app_col].dropna():
		if not isinstance(val, str):
			val = str(val)
		# 按分号分割，并将非空、去除首尾空格的部分添加到集合中
		parts = [p.strip() for p in re.split(r"[;；]+", val) if p and p.strip()]
		terms.update(parts)

	return sorted(list(terms))

def find_research_field_column(df: pd.DataFrame) -> str:
	"""查找 'research_field' 列。"""
	if 'research_field' in df.columns:
		return 'research_field'
	raise KeyError("在 DataFrame 中找不到 'research_field' 列。")


def map_empty_applications(df: pd.DataFrame, mapping_terms: list) -> tuple:
	"""
	仅使用本地 sentence-transformers 模型（位于 `model/`）对空的 `application` 进行匹配并填充。

	行为：
	- 从 `mapping_terms` 或已有非空 `application` 中提取以 '；' 分隔的标签库。
	- 优先载入本地模型目录 `model/{safe_name}`，若不存在则尝试下载首选模型并保存到 `model/` 下。
	- 使用模型计算句向量并通过余弦相似度进行匹配。

	返回 (df_copy, filled_count)
	"""
	df_copy = df.copy()
	try:
		app_col = find_application_column(df_copy)
		if 'research_field' in df_copy.columns:
			rf_col = 'research_field'
		else:
			raise KeyError("在 DataFrame 中找不到 'research_field' 列。")
	except KeyError as e:
		print(f"错误: {e}")
		return df_copy, 0

	# 提取空 application 行
	empty_mask = df_copy[app_col].isna() | (df_copy[app_col].astype(str).str.strip() == '')
	rows_to_fill = df_copy[empty_mask]
	if rows_to_fill.empty:
		return df_copy, 0

	# 辅助：按分号拆分单元格内的标签
	def extract_tags_from_cell(cell: str):
		if not isinstance(cell, str) or not cell.strip():
			return []
		parts = [p.strip() for p in re.split(r"[;；]+", cell) if p and p.strip()]
		return parts

	# 构建标签集合（保留出现顺序并去重），优先使用传入的 mapping_terms，否则从 DataFrame 的 application 列中提取
	terms_set = []
	if mapping_terms:
		for t in mapping_terms:
			if isinstance(t, str) and t.strip():
				for p in extract_tags_from_cell(t):
					if p not in terms_set:
						terms_set.append(p)
	else:
		for v in df_copy[app_col].dropna().astype(str):
			for p in extract_tags_from_cell(v):
				if p not in terms_set:
					terms_set.append(p)

	if not terms_set:
		return df_copy, 0

	# 文本规范化函数：统一全半角、去除多余空白及不必要字符，保留中英文字母与数字
	def normalize_text(s: str) -> str:
		if not isinstance(s, str):
			s = str(s) if pd.notna(s) else ''
		s = unicodedata.normalize('NFKC', s).strip()
		s = re.sub(r"[\u3000\s]+", " ", s)
		s = re.sub(r"[，,；;\/\\|]+", " ", s)
		s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\s+-]+", " ", s)
		s = re.sub(r"\s+", " ", s).strip()
		return s

	terms_orig = terms_set
	terms_norm = [normalize_text(t) for t in terms_orig]

	# 模型加载：优先使用本地 model/{safe_name} 下的模型
	model_names = [
		"shibing624/text2vec-base-chinese"
	]
	model = None
	# ensure model directory exists
	# 确保本地模型目录存在（用于保存从远程下载的模型）
	os.makedirs('model', exist_ok=True)

	def safe_name(m: str) -> str:
		return m.replace('/', '_')

	# 先尝试从本地路径加载模型
	for mname in model_names:
		local = os.path.join('model', safe_name(mname))
		if os.path.isdir(local):
			try:
				model = SentenceTransformer(local)
				break
			except Exception:
				model = None

	# 若本地未找到，则尝试从远程（Hugging Face）下载首个可用模型，并保存至本地以便后续离线使用
	if model is None:
		for mname in model_names:
			try:
				m = SentenceTransformer(mname)
				local = os.path.join('model', safe_name(mname))
				try:
					m.save(local)
				except Exception:
					pass
				model = m
				break
			except Exception:
				model = None

	if model is None:
		print("警告: 无法加载或下载 sentence-transformers 模型，填充将跳过。")
		return df_copy, 0

	# 计算所有标签的向量表示（一次性编码，加速后续相似度计算）
	term_embs = model.encode(terms_norm, convert_to_numpy=True, show_progress_bar=False)
	term_norms = (term_embs / (numpy.linalg.norm(term_embs, axis=1, keepdims=True) + 1e-9))

	# 匹配参数：主阈值、次阈值与候选数量。可以根据精度/召回目标调整。
	filled_count = 0
	MAIN_SIM = 0.78  # 高置信度阈值
	SECOND_SIM = 0.70  # 次高置信度阈值（作为降级策略）
	TOP_K = 3  # 每个查询考虑的最相似标签数

	for index, row in rows_to_fill.iterrows():
		research_field = row[rf_col]
		if not isinstance(research_field, str) or not research_field.strip():
			continue

		# 将 research_field 按常见分隔符拆分为若干候选片段（先尝试短片段匹配，再尝试整体匹配）
		q_orig = research_field
		q_parts = [p.strip() for p in re.split(r"[;；,，/、\\|]+", q_orig) if p and p.strip()]
		candidates = q_parts + [q_orig] if q_parts else [q_orig]

		matched = []
		seen = set()
		for q in candidates:
			q_norm = normalize_text(q)
			if not q_norm:
				continue
			# 计算查询片段的向量并归一化
			q_emb = model.encode([q_norm], convert_to_numpy=True)
			q_emb = q_emb / (numpy.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

			# 由于 term_norms 和 q_emb 都已归一化，点积即为余弦相似度
			sims = (term_norms @ q_emb[0])
			top_idx = numpy.argsort(-sims)[:TOP_K]
			high = [int(i) for i in top_idx if sims[i] >= MAIN_SIM]
			if not high:
				sec = [int(i) for i in top_idx if sims[i] >= SECOND_SIM]
				chosen = sec[:TOP_K]
			else:
				chosen = high

			# 兜底策略：若主/次阈值未触发，但 top-1 的相似度 >= 0.65，则接受 top-1（提高召回）
			if not chosen and sims[top_idx[0]] >= 0.65:
				chosen = [int(top_idx[0])]

			for i in chosen:
				term = terms_orig[i]
				if term not in seen:
					seen.add(term)
					matched.append((term, float(sims[i])))

		if matched:
			matched.sort(key=lambda x: x[1], reverse=True)
			matched_terms = [t for t, s in matched]
			# 保证每个 term 为字符串并过滤空/NaN，避免 join 报错
			matched_terms = [str(t).strip() for t, s in matched if pd.notna(t) and str(t).strip()]
			# 若缺少该列，先创建以避免 KeyError
			if app_col not in df_copy.columns:
				df_copy[app_col] = None
			# 为了避免静态类型检查器（如 Pylance）将 index 的类型推断为 tuple 并报错，
			# 在赋值前把 index 强制为 Any；运行时行为保持不变。
			idx: Any = index
			df_copy.at[idx, app_col] = '；'.join(matched_terms)
			filled_count += 1

	return df_copy, filled_count

def main():
	"""主执行函数。"""
	inputfile = 'Dataset/experts_full_cleaned.xlsx'
	if not os.path.exists(inputfile):
		print(f"输入文件未找到: {inputfile}")
		return

	print("正在从现有的 'application' 术语构建映射表...")
	terms = build_mapping(inputfile)

	# 将唯一的术语保存到文本文件
	txt_out = 'data_process_outputs/applications_mapping.txt'
	out_dir = os.path.dirname(txt_out)
	os.makedirs(out_dir, exist_ok=True)

	with open(txt_out, 'w', encoding='utf-8') as f:
		for term in terms:
			f.write(f"{term}\n")
	print(f"发现并保存了 {len(terms)} 个唯一术语到 {txt_out}")

	# 读取原始表格以填充空的 'application' 行
	try:
		df = pd.read_excel(inputfile)
	except Exception as e:
		print(f"读取输入文件时出错: {e}")
		return

	try:
		app_col = find_application_column(df)
	except KeyError as e:
		print(f"错误: {e}。无法继续填充。")
		return

	total = len(df)
	empty_mask_before = df[app_col].isna() | (df[app_col].astype(str).str.strip() == '')
	empty_before = int(empty_mask_before.sum())
	print(f"\n处理前数据总量: {total}")
	print(f"填充前已有 'application' 数据的行数: {total - empty_before}")
	print(f"填充前 'application' 为空的行数: {empty_before}")

	# 执行填充过程
	print("\n开始使用 embedding 匹配填充空的 'application' 字段...")
	df_filled, filled_count = map_empty_applications(df, terms)

	# 删除所有仍为空的 application 行，并打印删除前后数据总量
	out_filled = 'Dataset/experts_full_cleaned_filled.xlsx'
	# 计算仍为空的行数
	empty_mask_after = df_filled[app_col].isna() | (df_filled[app_col].astype(str).str.strip() == '')
	empty_after = int(empty_mask_after.sum())

	# 保留 application 非空的行
	df_pruned = df_filled.loc[~empty_mask_after].copy()
	total_after_prune = len(df_pruned)

	# 保存修剪后的 DataFrame（只包含有 application 的行）
	try:
		# 将 application 列中的英文逗号和分号替换为中文标点
		if app_col in df_pruned.columns:
			def replace_punct(cell: Any) -> Any:
				# Pylance 有时会对 pd.isna 的重载与 object 类型不匹配报警，使用 Any 可避免类型告警
				if pd.isna(cell):
					return cell
				# 保留原有非字符串值的可读性，先转换为字符串再替换
				s = str(cell)
				# 英文逗号 -> 中文逗号，英文分号 -> 中文分号
				s = s.replace(',', '，').replace(';', '；')
				return s

			df_pruned[app_col] = df_pruned[app_col].map(replace_punct)

		df_pruned.to_excel(out_filled, index=False)
	except Exception as e:
		print(f"保存完整填充文件时出错: {e}")
		return

	print("处理完成")
	print(f"\n实际填充的行数: {filled_count}")
	print(f"填充后已有 'application' 数据的行数: {total - empty_after}")
	print(f"填充后 'application' 仍为空的行数: {empty_after}")
	print(f"删除空 'application' 后的剩余数据总量: {total_after_prune}")
	print(f"完整结果已保存至: {out_filled}")

if __name__ == '__main__':
	main()