from collections import Counter
import sys
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

INPUT_PATH = "Dataset/experts_full_cleaned_filled.xlsx"
OUTPUT_PATH = "data_process_outputs/application_tag_counts.txt"
IMAGES_DIR = "data_process_outputs/application_tag_counts_plots"

def count_application_tags(excel_path: str) -> Counter:
	# pandas 能直接接受文件路径字符串
	df = pd.read_excel(excel_path)

	# 找到名为 application 的列（大小写不敏感）
	col_name = next((c for c in df.columns if str(c).strip().lower() == "application"), None)
	if col_name is None:
		raise KeyError("未找到名为 'application' 的列。请检查 Excel 表头。")

	counter = Counter()
	for cell in df[col_name].astype(str).fillna(""):
		text = cell.strip()
		if not text or text.lower() in {"nan", "none"}:
			continue
		# 按分号拆分
		parts = []
		if ";" in text:
			parts = [p.strip() for p in text.split(";")]
		elif "；" in text:
			parts = [p.strip() for p in text.split("；")]
		else:
			# 没有分号，整体作为一个标签
			parts = [text]

		for tag in parts:
			if tag:
				counter[tag] += 1

	return counter

def write_counts(output_path: str, counter: Counter) -> None:
	# 使用内置 open 写入，不使用 Path
	with open(output_path, "w", encoding="utf-8") as f:
		for tag, cnt in counter.most_common():
			f.write(f"{tag} {cnt}\n")

def plot_counts(output_path: str, counter: Counter, max_per_plot: int = 50) -> None:
	"""将标签词频绘制为水平柱状图，超过 max_per_plot 时拆分为多个图并保存为 PNG。

	输出目录为输出文件所在目录下的 application_tag_counts_plots。
	如果 matplotlib 不可用则跳过绘图并打印提示。
	"""
	if not counter:
		print("没有标签数据，跳过绘图。", file=sys.stderr)
		return

	labels, counts = zip(*counter.most_common())
	n = len(labels)

	plots_dir = output_path
	os.makedirs(plots_dir, exist_ok=True)

	# 尝试加载常见的中文字体（Windows 常见）
	zh_font = None
	try:
		zh_path = r"C:\Windows\Fonts\simhei.ttf"
		if os.path.exists(zh_path):
			zh_font = FontProperties(fname=zh_path)
	except Exception:
		zh_font = None

	# 分块绘图
	chunks = math.ceil(n / max_per_plot)
	for i in range(chunks):
		start = i * max_per_plot
		end = min((i + 1) * max_per_plot, n)
		seg_labels = labels[start:end]
		seg_counts = counts[start:end]

		height = max(4, 0.25 * len(seg_labels))
		fig, ax = plt.subplots(figsize=(10, height))
		y_pos = range(len(seg_labels))
		ax.barh(y_pos, seg_counts, color='C0')
		ax.set_yticks(y_pos)
		if zh_font:
			ax.set_yticklabels(seg_labels, fontproperties=zh_font)
			ax.set_xlabel('词频', fontproperties=zh_font)
			ax.set_title(f'应用标签词频（{start+1}-{end}）', fontproperties=zh_font)
		else:
			ax.set_yticklabels(seg_labels)
			ax.set_xlabel('词频')
			ax.set_title(f'应用标签词频（{start+1}-{end}）')

		ax.invert_yaxis()
		plt.tight_layout()

		# 使用仅含序号的文件名，如 1.png, 2.png ...
		out_png = os.path.join(plots_dir, f"{i+1}.png")
		try:
			fig.savefig(out_png)
		except Exception as e:
			print(f"保存图片失败: {e}", file=sys.stderr)
		finally:
			plt.close(fig)

	print(f"已生成图像：{plots_dir}")

def main() -> int:
	try:
		counter = count_application_tags(INPUT_PATH)
	except Exception as e:
		print(f"读取/处理 Excel 出错: {e}", file=sys.stderr)
		return 4

	try:
		write_counts(OUTPUT_PATH, counter)
	except Exception as e:
		print(f"写入输出文件出错: {e}", file=sys.stderr)
		return 5

	print(f"已写入: {OUTPUT_PATH} (共 {len(counter)} 个不同标签)")

	try:
		plot_counts(IMAGES_DIR, counter, max_per_plot=50)
	except Exception as e:
		print(f"绘图时出错: {e}", file=sys.stderr)
	return 0

if __name__ == "__main__":
	main()