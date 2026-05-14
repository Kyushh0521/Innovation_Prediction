# Innovation Prediction（企业创新方向预测）

## 项目简介

`Innovation_Prediction` 是一个面向 **企业技术战略与研发方向预测** 的端到端工程化项目。它将企业侧数据（企业画像、需求）、技术供给侧数据（成果、专家）、外部学术信号（arXiv / 期刊分级映射）进行融合，构建可训练数据，并通过大模型微调生成未来 2-3 年可落地的创新方向建议。

与单一脚本仓库不同，本项目覆盖了完整链路：

- **数据治理**：清洗/标准化企业、专家、成果、需求等异构数据；
- **智能匹配**：企业 ↔ 成果、企业 ↔ 专家的语义检索与融合打分；
- **数据构造**：从匹配结果构建偏好数据（DPO风格）与 SFT 数据；
- **模型训练**：基于 Qwen 系列进行 LoRA 微调（含对抗训练与自适应策略）；
- **模型评测**：统一评估 PPL、BLEU/ROUGE、BERTScore、语义相似度；
- **学术补充**：抓取 arXiv 元数据并维护 CAS/JCR、CCF 等映射表。

---

## 适用场景

- 科创平台/政府部门：企业技术需求与科研供给对接；
- 企业战略部门：结合前沿研究趋势制定中短期研发路线；
- 研究团队：验证“外部学术信号 + 企业业务信息”驱动的创新预测范式。

---

## 核心流程（Pipeline）

1. **数据准备与清洗**（`data_process`）
   - 原始 Excel/JSON 数据清洗、字段统一、样本抽取；
2. **匹配与样本生成**（`data_process`）
   - 用中文向量模型（默认 `shibing624/text2vec-base-chinese`）计算企业与成果/专家相关性；
   - 生成带匹配上下文的输入样本；
3. **偏好数据与 SFT 数据构建**（`data_process`）
   - 调用 LLM 生成可采纳/不可采纳预测，构建偏好数据；
   - 转换为 SFT 格式供监督微调使用；
4. **LoRA 训练**（`lora_training`）
   - 在 Qwen Instruct 基座上执行参数高效微调；
5. **统一评估**（`model_eval`）
   - 对模型输出进行自动化指标评估并输出汇总结果；
6. **学术信号更新**（`arxiv_process`）
   - 抓取前沿论文与映射体系，支撑趋势特征更新。

---

## 仓库结构概览

- `data_process/`：数据清洗、匹配、统计、可视化、训练样本构建。
- `arxiv_process/`：arXiv 抓取与学术分级映射维护（CAS/JCR、CCF）。
- `lora_training/`：LoRA 训练主流程（数据集、训练器、配置、对抗策略）。
- `model_eval/`：评估框架（生成质量、语义质量、困惑度等）。
- `datasets/`：训练/验证/测试数据（当前含 `alpaca/`、`medical/` 示例目录）。
- `data_process_outputs/`、`arxiv_process_outputs/`：数据处理与抓取中间产物。
- `model/`、`model_cache/`：向量模型与缓存（检索与匹配阶段使用）。

---

## 模块说明

### 1) `data_process`（数据工程与样本构建）

- `achievements_full_clean.py`：成果数据清洗与标准化。
- `append_matches_to_inputs.py`：将匹配结果回填到输入样本。
- `count_match_score_stats.py`：匹配分数统计分析。
- `count_tokens_expected.py`：样本 token 规模估计。
- `enterprise_excel_to_jsonl.py`：企业 Excel 转 JSONL。
- `enterprises_full_clean.py`：企业数据全量清洗。
- `enterprises_match_achievements.py`：企业 ↔ 成果语义匹配（向量缓存 + 阈值筛选）。
- `enterprises_match_experts.py`：企业 ↔ 专家匹配（向量相似度 + TF-IDF 关键词融合）。
- `excel_field_counts.py`：Excel 字段覆盖率统计。
- `experts_application_fill.py`：专家应用字段补全。
- `experts_application_statistical_analysis.py`：专家申请统计分析。
- `experts_full_clean.py`：专家数据清洗与标准化。
- `extract_by_index_and_sample.py`：按索引/采样抽取子集。
- `extract_terms_llm.py`：LLM 辅助术语抽取。
- `generate_preference_dataset.py`：从匹配输入生成偏好数据（chosen/rejected）。
- `needs_full_clean.py`：需求数据清洗。
- `plot_enterprise_category_distribution.py`：企业分类分布可视化。
- `plot_enterprise_match_summary.py`：企业匹配结果可视化汇总。
- `preference_to_sft.py`：偏好数据转 SFT 数据。
- `split_dataset.py`：训练/验证/测试拆分。
- `split_needs_achievements.py`：需求与成果样本拆分。
- `subset.py`：快速子集构建。

### 2) `arxiv_process`（外部学术数据）

- `arxiv_scraper.py`：抓取 arXiv 论文元数据。
- `cas_jcr_mapping.py`：CAS 与 JCR 映射处理。
- `ccf_mapping.py`：CCF 分级映射处理。

### 3) `lora_training`（模型训练）

- `train.py`：训练入口。
- `trainer.py`：训练主逻辑（优化、保存、评估触发）。
- `dataset.py`：SFT 数据集构造（Qwen chat template，labels 仅训练 assistant 段）。
- `adv.py`：对抗训练相关逻辑。
- `utils.py`：配置加载、随机种子、通用工具。
- `config.yaml`：训练配置（LoRA 超参、对抗参数、数据路径等）。

> 当前默认配置基于 `Qwen/Qwen2.5-0.5B-Instruct`，可按业务场景替换。

### 4) `model_eval`（统一评测）

- `evaluator.py`：评测主入口。
- `generation_metrics.py`：PPL、BLEU/ROUGE、BERTScore、语义指标计算。
- `adv_evaluator.py`：鲁棒性/对抗评估扩展。
- `plot_compare.py`：多运行结果可视化对比。
- `eval_config.yaml`、`plot_config.yaml`：评测与可视化配置。
- `utils.py`：模型加载、日志与生成函数等工具。

---

## 运行建议

- 优先在虚拟环境中安装依赖后执行脚本；
- 先用小样本跑通数据处理、匹配、训练、评估全链路，再执行全量任务；
- 向量检索阶段建议保留 `model_cache/` 以复用嵌入，避免重复编码；
- 训练与评估参数统一在 YAML 中管理，便于复现实验。

---

## 输出与结果

- 数据处理产物：`data_process_outputs/`
- 学术抓取产物：`arxiv_process_outputs/`
- 训练结果/模型权重：由 `lora_training/config.yaml` 中 `model_output` 和 `result_output` 指定
- 评估结果：`model_eval/eval_config.yaml` 中 `output_dir` 指定，包含 `*_details.json` 与 `*_summary.json`

---

## 备注

目前仓库仍以脚本化工作流为主。若后续需要团队协作或持续化运行，建议逐步补充：

- 统一 `requirements.txt` / `pyproject.toml`；
- 标准化 CLI 参数与日志格式；
- 基于配置驱动的多阶段 Pipeline 编排（例如 Makefile / Airflow / Prefect）。
