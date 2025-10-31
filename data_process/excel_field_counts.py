import os
import pandas as pd

INPUT = 'Dataset/enterprises_full.xlsx'
OUTPUT_FILE = 'data_process_outputs/enterprises_full.txt'

def count_non_empty(series: pd.Series) -> int:
	s = series.dropna()
	if s.empty:
		return 0
	try:
		counts = s.map(lambda x: False if str(x).strip() == '' else True).sum()
	except Exception:
		counts = s.shape[0]
	return int(counts)


def process(input_path: str, output_path: str) -> int:
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	try:
		df = pd.read_excel(input_path, dtype=object)
	except Exception as e:
		print(f"读取文件 '{input_path}' 时出错：{e}")
		return 1

	with open(output_path, 'w', encoding='utf-8') as fout:
		for col in df.columns:
			cnt = count_non_empty(df[col])
			fout.write(f"{col} {cnt}\n")

	print(f"已将统计结果写入 '{output_path}'")
	return 0


if __name__ == '__main__':
	rc1 = process(INPUT, OUTPUT_FILE)