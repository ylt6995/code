# 围串标检测 LLM 实验报告

## 配置披露
- 数据集：data\all_data.xlsx
- 划分比例：6:2:2（训练/验证/测试），split_seed=42
- Prompt seeds：[0]
- 模型厂商：['gpt']
- 提示词方法：['zero']
- few-shot 每类样本数：2

### LLM 参数（Temperature / Top-p / Max Tokens / Timeout）
- 未加载（dry-run 或 mock 模式）

### 重试机制
- {'max_retries': 6, 'backoff_s': [2, 4, 8, 16, 32, 60]}

## 输出文件
- aggregate_metrics.csv：每个（模型×方法×seed）的 Accuracy/Precision/Recall/F1 与阈值
- plots/metrics_by_seed_*.png：每个配置的 10 seeds 指标曲线
- plots/recall_boxplot_all.png：全配置 Recall 分布箱线图
- prompt_*：每个配置的 prompt 原文示例（基于验证集第一个样本生成）
- summary.json：包含所有配置的均值/极值统计与阈值分布

## 输出格式控制
- Prompt 强制要求只输出一个 JSON，并限定 key 与分数字段范围。
- 解析时先尝试整体 JSON.loads；失败则用正则提取最外层 {...} 再解析；仍失败会做轻量修复（引号/尾逗号）再解析。

## 解析提取步骤（伪代码）
```text
raw = llm_response_text.strip()
try: obj = json.loads(raw)
except: obj = None
if obj is None:
    snippet = regex_find_outermost_braces(raw)
    try: obj = json.loads(snippet)
    except:
        snippet2 = basic_fix(snippet)
        obj = json.loads(snippet2) or {}
score = clamp_int(obj.get('collusionSuspicionScore'), 0, 100) if present else 0
```

## 阈值确定依据
- 阈值策略：{'strategy': 'recall_at_least_then_max_precision', 'recall_target': 0.9}
- 在验证集上枚举阈值 t=0..100。
- 先筛选出 Recall ≥ recall_target 的阈值集合。
- 在集合内选择 Precision 最高的阈值；若并列，再比较 F1，最后比较预测为正比例更低。

## 打分标准
- 0-100 越高越可疑；风险等级建议：0-20 低，21-40 中，41-70 高，71-100 严重。
- 维度（总分 0-100）：价格与排序异常（0-35）、RD/CV（0-15）、版本同步（0-10）、公司关联（0-25）、其他强证据（0-15）。
- 完整标准与输出 schema 见 prompt_*.txt 与 bidrig/prompts.py。

## 结果汇总
- 详见 summary.json（含各配置在 10 seeds 上的均值/最小/最大）。

## 线性回归（Logistic）基线
- baseline_logit.json：{'threshold': 0.8801908450505479, 'val': {'tp': 2.0, 'fp': 0.0, 'fn': 0.0, 'tn': 14.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}, 'test': {'tp': 2.0, 'fp': 0.0, 'fn': 0.0, 'tn': 14.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}}
