$ErrorActionPreference = "Stop"

if (-not $env:ZHIPU_API_KEY) { throw "缺少 ZHIPU_API_KEY" }

if (-not $env:ZHIPU_BASE_URL) { $env:ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4" }
if (-not $env:ZHIPU_MODEL) { $env:ZHIPU_MODEL = "glm-4.7-flash" }

if (-not $env:ZHIPU_MIN_INTERVAL_S) { $env:ZHIPU_MIN_INTERVAL_S = "2.0" }
if (-not $env:ZHIPU_RATE_LIMIT_SLEEP_S) { $env:ZHIPU_RATE_LIMIT_SLEEP_S = "60" }
if (-not $env:ZHIPU_MAX_RETRIES) { $env:ZHIPU_MAX_RETRIES = "30" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path (Get-Location) ("run_zhipu_fewshot_once_" + $ts + ".log")

python -u run_llm_bidrig_experiment.py --providers zhipu --methods few --prompt-seeds 0 2>&1 | Tee-Object -FilePath $log
