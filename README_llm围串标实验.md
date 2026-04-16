## 目标

- 使用 all_data.xlsx（项目信息/招标公告/投标商数据）进行围串标风险评分（0-100）
- 同时支持 GPT 与 智谱（GLM）两种模型
- 支持 zero-shot / few-shot / CoT 三种提示词策略
- 按 6:2:2 划分训练/验证/测试集
- 默认固定测试集规模与类别比例：test=40（其中正例=20）
- 阈值通过验证集 Recall≥目标值 的前提下 Precision 最高来确定（默认 Recall≥0.90）
- 支持 10 个随机种子做提示词敏感性实验（few-shot 示例选择 + 投标商顺序扰动）

## 环境变量

### GPT

- GPT_API_KEY（或 OPENAI_API_KEY）
- GPT_BASE_URL（可选；默认 https://api.chatanywhere.tech/v1）
- GPT_MODEL（可选，默认 gpt-4o-mini）

### 智谱

- ZHIPU_API_KEY
- ZHIPU_BASE_URL（可选，默认 https://open.bigmodel.cn/api/paas/v4）
- ZHIPU_MODEL（可选，默认 glm-4.7）

## 运行

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 运行完整实验（两模型 × 三方法 × 10 seeds）

```bash
python run_llm_bidrig_experiment.py
```

输出会写入 runs/<时间戳>/，包含：

- split_sizes.json、split_ids.json：数据划分结果
- val_*.csv、test_*.csv：每个配置在验证/测试集的打分与预测
- metrics_*.json：每个 seed 的最优阈值（由验证集 Recall 最大确定）与测试集指标
- aggregate_metrics.csv：每个（模型×方法×seed）的 Accuracy/Precision/Recall/F1 与阈值
- plots/*.png：指标曲线与 Recall 分布箱线图
- report.md：参数披露、阈值依据、解析伪代码、文件清单
- prompt_*.txt：prompt 原文示例
- summary.json：所有配置的汇总（含 10 seeds 的统计）

阈值策略参数：

```bash
python run_llm_bidrig_experiment.py --recall-target 0.90
```

划分策略参数（默认启用固定测试集规模与类别比例：test=40，正例=20）：

```bash
python run_llm_bidrig_experiment.py --test-size 40 --test-pos 20
```

如需关闭固定测试集（回到纯 6:2:2 分层划分）：

```bash
python run_llm_bidrig_experiment.py --no-fixed-test-balance
```

### 推荐：两模型×三方法分开跑（便于断点续跑与控制限流）

每次运行只指定一个 provider + 一个 method，并固定 seed=0（除非你明确要做敏感性实验）。

GPT：

```bash
python run_llm_bidrig_experiment.py --providers gpt --methods zero --prompt-seeds 0
python run_llm_bidrig_experiment.py --providers gpt --methods few --prompt-seeds 0
python run_llm_bidrig_experiment.py --providers gpt --methods cot --prompt-seeds 0
```

智谱：

```bash
python run_llm_bidrig_experiment.py --providers zhipu --methods zero --prompt-seeds 0
python run_llm_bidrig_experiment.py --providers zhipu --methods few --prompt-seeds 0
python run_llm_bidrig_experiment.py --providers zhipu --methods cot --prompt-seeds 0
```

建议同时打开日志输出（示例）：

```bash
python -u run_llm_bidrig_experiment.py --providers zhipu --methods zero --prompt-seeds 0 2>&1 | Tee-Object -FilePath ".\\run_zhipu_zero_seed0.log"
```

### 3) 只跑单模型/单方法

```bash
python run_llm_bidrig_experiment.py --providers zhipu --methods cot
```

### 3.2) 第一阶段：只跑 GPT 的 zero-shot，10 seeds 敏感性（199 项目全量）

```bash
python run_llm_bidrig_experiment.py --providers gpt --methods zero --prompt-seeds random10
```

建议加上限速与 429 退避（避免“模型访问量过大”中断）：

```bash
set GPT_MIN_INTERVAL_S=1.0
set GPT_RATE_LIMIT_SLEEP_S=60
set GPT_MAX_RETRIES=20
```

Windows PowerShell 等价写法：

```bash
$env:GPT_MIN_INTERVAL_S="1.0"
$env:GPT_RATE_LIMIT_SLEEP_S="60"
$env:GPT_MAX_RETRIES="20"
```

### 3.3) 第二阶段：跑 GPT 的 few-shot / CoT（不做随机种子敏感性）

few-shot（单次，seed=0）：

```bash
python run_llm_bidrig_experiment.py --providers gpt --methods few --prompt-seeds 0
```

CoT（单次，seed=0）：

```bash
python run_llm_bidrig_experiment.py --providers gpt --methods cot --prompt-seeds 0
```

也可以直接运行 PowerShell 脚本（会自动生成日志文件）：

- run_gpt_fewshot_once.ps1
- run_gpt_cot_once.ps1

### 3.4) 第三阶段：跑智谱（ZHIPU）的 zero / few / cot（不做随机种子敏感性）

zero-shot（单次，seed=0）：

```bash
python run_llm_bidrig_experiment.py --providers zhipu --methods zero --prompt-seeds 0
```

few-shot（单次，seed=0）：

```bash
python run_llm_bidrig_experiment.py --providers zhipu --methods few --prompt-seeds 0
```

CoT（单次，seed=0）：

```bash
python run_llm_bidrig_experiment.py --providers zhipu --methods cot --prompt-seeds 0
```

也可以直接运行 PowerShell 脚本（会自动生成日志文件并带限速/重试默认值）：

- run_zhipu_zero_once.ps1
- run_zhipu_fewshot_once.ps1
- run_zhipu_cot_once.ps1

### 3.1) 指定 seeds 或随机 seeds

- 指定 seeds：

```bash
python run_llm_bidrig_experiment.py --prompt-seeds 1,7,42,314
```

- 随机抽取 10 个 seeds（默认行为，基于 split_seed 生成并写入 summary.json）：

```bash
python run_llm_bidrig_experiment.py --prompt-seeds random10
```

### 4) dry-run（不调用 API，只使用缓存）

```bash
python run_llm_bidrig_experiment.py --dry-run
```

缓存文件默认位置：runs/cache/llm_cache.jsonl

## 说明

- 如果你直接用浏览器访问 GPT_BASE_URL 看到 404，一般不影响 OpenAI 兼容接口的 /chat/completions 调用。

## 限速与重试（推荐）

当遇到 429（访问量过大/限流）时，可通过环境变量控制最小请求间隔与退避重试：

- LLM_MIN_INTERVAL_S：全局最小请求间隔（秒）
- LLM_RATE_LIMIT_SLEEP_S：遇到 429 时至少等待（秒）
- LLM_MAX_RETRIES：最大重试次数

也可以按厂商分别设置：

- GPT_MIN_INTERVAL_S / GPT_RATE_LIMIT_SLEEP_S / GPT_MAX_RETRIES
- ZHIPU_MIN_INTERVAL_S / ZHIPU_RATE_LIMIT_SLEEP_S / ZHIPU_MAX_RETRIES

## 打分标准说明（模型输出中也会给出）

评分输出为 0-100 的整数，分数越高表示越可疑。提示词内固定了五个维度：

- 价格与排序异常（0-35）
- 集中度指标 RD/CV（0-15）
- 报价时间/版本同步（0-10）
- 公司关联线索（0-25）
- 其他强证据（0-15）

风险等级建议：

- 0-20 低
- 21-40 中
- 41-70 高
- 71-100 严重
