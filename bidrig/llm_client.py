import os
import time
from dataclasses import dataclass
from typing import Optional

import openai

_LAST_CALL_TS = 0.0


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.1
    top_p: float = 0.9
    timeout_s: float = 300.0
    max_tokens: int = 6000


def load_settings(provider: str) -> LLMSettings:
    p = provider.strip().lower()
    if p in {"gpt", "openai", "chat"}:
        api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("缺少环境变量 GPT_API_KEY 或 OPENAI_API_KEY")
        base_url = os.getenv("GPT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.chatanywhere.tech/v1"
        model = os.getenv("GPT_MODEL") or "gpt-4o-mini"
        return LLMSettings(provider="gpt", api_key=api_key, base_url=base_url, model=model)

    if p in {"zhipu", "glm"}:
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise RuntimeError("缺少环境变量 ZHIPU_API_KEY")
        base_url = os.getenv("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
        model = os.getenv("ZHIPU_MODEL") or "glm-4.7"
        return LLMSettings(provider="zhipu", api_key=api_key, base_url=base_url, model=model)

    raise ValueError("provider must be gpt or zhipu")


def chat_once(prompt: str, settings: LLMSettings, *, system: Optional[str] = None) -> str:
    _apply_min_interval(settings.provider)
    client = openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    sys_msg = system or "你是招投标审计与围串标检测专家。"
    resp = client.chat.completions.create(
        model=settings.model,
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
        timeout=settings.timeout_s,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_with_retry(prompt: str, settings: LLMSettings, *, system: Optional[str] = None, max_retries: int = 6) -> str:
    provider_key = (settings.provider or "").strip().upper()
    effective_max_retries = int(os.getenv(f"{provider_key}_MAX_RETRIES") or os.getenv("LLM_MAX_RETRIES") or max_retries)
    delay_s = 2.0
    last_err: Optional[Exception] = None
    for _ in range(effective_max_retries):
        try:
            return chat_once(prompt, settings, system=system)
        except Exception as e:
            last_err = e
            sleep_s = delay_s
            if isinstance(e, openai.RateLimitError):
                sleep_s = max(
                    sleep_s,
                    float(os.getenv(f"{provider_key}_RATE_LIMIT_SLEEP_S") or os.getenv("LLM_RATE_LIMIT_SLEEP_S") or 30.0),
                )
            time.sleep(sleep_s)
            delay_s = min(delay_s * 2, 60.0)
    raise last_err  # type: ignore[misc]


def _apply_min_interval(provider: str) -> None:
    global _LAST_CALL_TS
    provider_key = (provider or "").strip().upper()
    v = os.getenv(f"{provider_key}_MIN_INTERVAL_S") or os.getenv("LLM_MIN_INTERVAL_S")
    if not v:
        _LAST_CALL_TS = time.time()
        return
    try:
        min_interval = float(v)
    except Exception:
        _LAST_CALL_TS = time.time()
        return
    if min_interval <= 0:
        _LAST_CALL_TS = time.time()
        return
    now = time.time()
    gap = now - _LAST_CALL_TS
    if gap < min_interval:
        time.sleep(min_interval - gap)
    _LAST_CALL_TS = time.time()
