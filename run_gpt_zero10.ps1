$ErrorActionPreference = "Stop"

if (-not $env:GPT_API_KEY -and -not $env:OPENAI_API_KEY) { throw "缺少 GPT_API_KEY（或 OPENAI_API_KEY）" }

if (-not $env:GPT_BASE_URL) { $env:GPT_BASE_URL = "https://api.chatanywhere.tech/v1" }
if (-not $env:GPT_MODEL) { $env:GPT_MODEL = "gpt-4o-mini" }

if (-not $env:GPT_MIN_INTERVAL_S) { $env:GPT_MIN_INTERVAL_S = "1.0" }
if (-not $env:GPT_RATE_LIMIT_SLEEP_S) { $env:GPT_RATE_LIMIT_SLEEP_S = "60" }
if (-not $env:GPT_MAX_RETRIES) { $env:GPT_MAX_RETRIES = "20" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path (Get-Location) ("run_gpt_zero10_" + $ts + ".log")

python -u run_llm_bidrig_experiment.py --providers gpt --methods zero --prompt-seeds random10 2>&1 | Tee-Object -FilePath $log
