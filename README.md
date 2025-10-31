# 项目说明

本文件简要说明仓库中 `data_process` 和 `arxiv_process` 两个文件夹内脚本的作用。目的在于帮助开发者快速理解每个脚本的职责与输入/输出预期。

## 总说明

- 语言：Python
- 目标：数据清洗、匹配、统计与从 arXiv/映射数据源抓取和映射处理。
- 使用约定：各脚本一般以 CSV/JSON/JSONL 作为输入或输出，具体请查看脚本顶部的注释或在运行时打印的帮助信息（例如 `-h` 或 `--help`）。

## 文件夹用途

- `data_process`：用于项目中所有与数据处理相关的步骤——包括数据清洗、字段标准化、格式转换（例如 Excel -> JSONL）、样本抽取、匹配任务（企业↔成果/专家）及统计分析。该文件夹包含面向工程化处理的数据预处理和导出脚本，是下游模型或分析的输入生成中心。
- `arxiv_process`：负责与学术/期刊数据相关的抓取与映射工作——包括从 arXiv 抓取论文元数据、以及生成/维护期刊和会议的映射表（例如 CAS↔JCR、CCF 分级映射）。该文件夹主要用于补充和标准化外部学术数据源以供关联分析使用。

## `data_process` 文件夹

以下按文件名给出简短描述：

- `achievements_full_clean.py`：对成果（achievements）数据进行清洗、规范化字段、去重与格式修正。
- `append_matches_to_inputs.py`：将匹配到的结果（如企业与成果/专家的匹配）附加回原始输入文件，生成带匹配信息的输出。
- `count_match_score_stats.py`：统计匹配得分的分布与摘要统计（均值、中位数、分位数等）。
- `count_tokens_dataset.py`：对文本字段进行分词/计数，统计每条记录的 token 数，常用于后续模型或嵌入处理容量估算。
- `enterprise_excel_to_jsonl.py`：把企业信息的 Excel 文件转换为 JSONL（每行一个 JSON），便于下游处理和批量导入。
- `enterprises_full_clean.py`：对企业数据进行完整清洗（字段标准化、缺失处理、去重等）。
- `enterprises_match_achievements.py`：把企业与成果进行匹配（基于规则或相似度），输出匹配结果文件。
- `enterprises_match_experts.py`：把企业与专家进行匹配，输出匹配关系与相似度评分。
- `excel_field_counts.py`：统计 Excel 表各字段的出现情况（非空计数、示例值等），方便数据探索。
- `experts_application_fill.py`：用于根据专家和申请数据自动填充/完善申请表或结构化字段。
- `experts_application_statistical_analysis.py`：对专家申请相关数据做统计分析，输出摘要报告或图表数据。
- `experts_full_clean.py`：对专家数据集进行清洗、统一命名、处理联系方式等敏感/无效字段。
- `extract_by_index_and_sample.py`：根据索引或采样策略抽取子集（用于调试、构建样本集或手工标注）。
- `generate_preference_dataset_from_jsonl.py`：从 JSONL 源生成偏好/训练数据集（可能用于推荐或偏好建模）。
- `needs_full_clean.py`：对需求/项目需求类数据进行清洗与格式化。
- `split_needs_achievements.py`：把需求与成果的联合数据集拆分为各自的子集或为匹配任务做预处理。

（如果仓库中还有其他 `data_process` 下的脚本，按文件名可类推为清洗、转换、匹配、统计相关工具。）

## `arxiv_process` 文件夹

- `arxiv_scraper.py`：从 arXiv 抓取论文元数据（标题、摘要、作者、类别等），并将结果保存为 CSV/JSON 供后续处理。
- `cas_jcr_mapping.py`：处理 CAS（或其他来源）的期刊与 JCR（期刊影响因子/分区）映射表，生成可用于判断期刊影响力的映射文件。
- `ccf_mapping.py`：生成或处理 CCF（中国计算机学会）会议/期刊分级映射，用于将期刊或会议名映射到 CCF 等级。

## 运行与使用建议

- 运行脚本前请在虚拟环境中安装项目依赖（若没有统一依赖文件，请参考脚本顶部注释）。
- 对于批量/生产任务，建议先在小样本上验证再跑全量数据。
