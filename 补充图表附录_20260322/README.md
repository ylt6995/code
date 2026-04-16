# 补充实验与图表（20260322）

本文件夹用于回应返修意见与 list.docx 中列出的补充项，所有输出统一写到 `outputs/`。

## 生成方式

在项目根目录运行：

```bash
python 补充图表附录_20260322/scripts/make_supplement.py
```

可选参数：

- `--gpt-run runs/<...>_gpt_all`
- `--zhipu-run runs/<...>_zhipu_all`

## 输出文件

### 表格（outputs/tables）

- `table1_llm_metrics_by_method.csv`：GLM-4.7 与 GPT-4o-mini 的 ROC-AUC / PR-AUC / Precision / Recall / F1（按方法与 seed 汇总）
- `table2_gpt_zero_10seeds_mean_std.csv`：GPT zero-shot 在 10 seeds 下的 mean ± std（Accuracy/Precision/Recall/F1）
- `ks_test_scores.csv`：双样本 K-S 检验（D 与 p-value），按模型/方法统计

### 图表（outputs/figures）

- `recall_comparison_bar.png`：按 zero/few/cot 对比 GLM vs GPT 的 Recall（GPT 为 mean±std）
- `recall_comparison_box.png`：GPT(10 seeds) 的 Recall 分布箱线图 + GLM 单点
- `roc_curves.png`：LR、LightGBM、XGBoost、GLM-4.7、GPT-4o-mini 五条 ROC 曲线（若 LightGBM/XGBoost 不可用则自动跳过）
- `pr_curves.png`：对应的 PR 曲线

### 说明

- `meta.json`：记录使用的 runs 目录与曲线构造方式
