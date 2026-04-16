# 围串标检测 LLM 实验报告

## 配置披露
- 数据集：data\all_data.xlsx
- 划分比例：6:2:2（训练/验证/测试），split_seed=42
- Prompt seeds：[1867825, 419610, 4614226, 4108603, 3744854, 2341057, 1719583, 9149732, 1458591, 9906820]
- 模型厂商：['gpt']
- 提示词方法：['zero']
- few-shot 每类样本数：2
- 温度/Top-p/Max Tokens/Timeout：见 summary.json 的 llm_settings
- 重试机制：见 summary.json 的 retry（指数退避）

## 输出文件
- aggregate_metrics.csv：每个（模型×方法×seed）的 Accuracy/Precision/Recall/F1 与阈值
- plots/metrics_by_seed_*.png：每个配置的 10 seeds 指标曲线
- plots/recall_boxplot_all.png：全配置 Recall 分布箱线图
- prompt_*：每个配置的 prompt 原文示例（基于验证集第一个样本生成）

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
score = clamp_int(obj.get('collusionSuspicionScore'), 0, 100) else 0
```

## 阈值确定依据
- 在验证集上枚举阈值 t=0..100，选择 Recall 最大的 t。
- 若 Recall 并列：依次选择 Precision 更高、F1 更高、预测为正比例更低的阈值。

## 打分标准
- 0-100 越高越可疑；维度包括价格/排序异常、RD/CV 集中度、版本同步、公司关联、其他强证据。
- 具体文本以 prompt 原文为准（prompt_*.txt）。

## 结果汇总
- 详见 summary.json（含各配置在 10 seeds 上的均值/最小/最大）。
