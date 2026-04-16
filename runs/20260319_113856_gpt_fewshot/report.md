# 围串标检测 LLM 实验报告

## 配置披露
- 数据集：data\all_data.xlsx
- 划分比例：6:2:2（训练/验证/测试），split_seed=42
- Prompt seeds：[0]
- 模型厂商：['gpt']
- 提示词方法：['few']
- few-shot 每类样本数：2

### LLM 参数（Temperature / Top-p / Max Tokens / Timeout）
- gpt：{'provider': 'gpt', 'api_key': 'sk-0D06WSzBVOfA3v11Dqmb2F0Pc6Wkudz44uxyNyA3ozOrsDXe', 'base_url': 'https://api.chatanywhere.tech/v1', 'model': 'gpt-4o-mini', 'temperature': 0.1, 'top_p': 0.9, 'timeout_s': 300.0, 'max_tokens': 6000}

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
- 在验证集上枚举阈值 t=0..100，选择 Recall 最大的 t。
- 若 Recall 并列：依次选择 Precision 更高、F1 更高、预测为正比例更低的阈值。

## 打分标准
- 0-100 越高越可疑；风险等级建议：0-20 低，21-40 中，41-70 高，71-100 严重。
- 维度（总分 0-100）：价格与排序异常（0-35）、RD/CV（0-15）、版本同步（0-10）、公司关联（0-25）、其他强证据（0-15）。
- 完整标准与输出 schema 见 prompt_*.txt 与 bidrig/prompts.py。

## 结果汇总
- 详见 summary.json（含各配置在 10 seeds 上的均值/最小/最大）。

## 线性回归（Logistic）基线
- baseline_logit.json：{'threshold': 0.9412542471348517, 'val': {'tp': 4.0, 'fp': 0.0, 'fn': 0.0, 'tn': 36.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}, 'test': {'tp': 0.0, 'fp': 0.0, 'fn': 4.0, 'tn': 36.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.9}}

## 定性案例（LLM 可检出但基线漏检）
- qualitative_case.json：{'chosen_config': {'provider': 'gpt', 'method': 'few', 'seed': 0, 'tag': 'gpt_few_seed0'}, 'threshold_by_val_recall': 66, 'bid_ann_guid': 'e2aa45a4-e02f-4d3f-82d5-ae09a38d62a4', 'baseline': {'bid_ann_guid': 'e2aa45a4-e02f-4d3f-82d5-ae09a38d62a4', 'label': 1, 'baseline_prob': 0.9118961983747755, 'baseline_pred': 0}, 'llm': {'score': 70, 'riskLevel': '"High"', 'keyEvidence': '["存在明显的围标/串标迹象"]'}, 'indicators': {'rd': 0.0904160396888241, 'cv_losing': 0.32410958911823884, 'price_cv_all': 0.31617785246991026, 'contact_dup_count': 0, 'phone_dup_count': 1, 'email_dup_count': 1, 'wins': [1, 0, 0, 0, 0]}, 'bidders_brief': [{'x_providername': '泰合工程咨询', 'x_price': 81809.292214, 'versionnumber': '2025-04-05T09:00:00.000+08:00', 'x_isqualified': 1}, {'x_providername': '未来规划设计院', 'x_price': 145712.618942, 'versionnumber': '2025-04-05T09:00:30.000+08:00', 'x_isqualified': 0}, {'x_providername': '远景规划设计院', 'x_price': 76627.856225, 'versionnumber': '2025-04-05T09:01:00.000+08:00', 'x_isqualified': 0}, {'x_providername': '泰合工程咨询', 'x_price': 129695.901177, 'versionnumber': '2025-04-05T09:01:30.000+08:00', 'x_isqualified': 0}, {'x_providername': '鼎盛建筑有限公司//宏远建设集团', 'x_price': 79791.503433, 'versionnumber': '2025-04-05T09:02:00.000+08:00', 'x_isqualified': 0}]}
