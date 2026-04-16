import json
import re
from typing import Any, Dict, Optional, Tuple


def parse_model_json(text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    raw = (text or "").strip()
    if not raw:
        return {}, "empty_response"

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}, "no_json_object"

    snippet = m.group(0)
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    snippet2 = _basic_fix_json(snippet)
    try:
        obj = json.loads(snippet2)
        if isinstance(obj, dict):
            return obj, "repaired_json"
    except Exception:
        pass

    return {}, "json_parse_failed"


def extract_score(obj: Dict[str, Any]) -> Optional[int]:
    if not obj:
        return None
    for k in ("collusionSuspicionScore", "score", "riskScore"):
        if k in obj:
            v = obj.get(k)
            score = _to_int(v)
            if score is None:
                continue
            return max(0, min(100, score))
    return None


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(round(v))
    if isinstance(v, str):
        m = re.search(r"-?\d+(\.\d+)?", v)
        if not m:
            return None
        try:
            return int(round(float(m.group(0))))
        except Exception:
            return None
    return None


def _basic_fix_json(text: str) -> str:
    s = text.strip()
    s = re.sub(r"(?<!\\)\\'", "'", s)
    s = re.sub(r"(?<!\\)'", '"', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

