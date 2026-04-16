import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class CacheRecord:
    key: str
    provider: str
    model: str
    method: str
    bid_ann_guid: str
    seed: int
    prompt_sha256: str
    response_text: str
    parsed: Dict[str, Any]


def make_cache_key(provider: str, model: str, method: str, bid_ann_guid: str, seed: int, prompt: str) -> str:
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{provider}:{model}:{method}:{bid_ann_guid}:{seed}:{h}"


def load_cache(cache_path: str) -> Dict[str, CacheRecord]:
    if not os.path.exists(cache_path):
        return {}
    cache: Dict[str, CacheRecord] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = CacheRecord(
                key=obj["key"],
                provider=obj["provider"],
                model=obj["model"],
                method=obj["method"],
                bid_ann_guid=obj["bid_ann_guid"],
                seed=int(obj["seed"]),
                prompt_sha256=obj["prompt_sha256"],
                response_text=obj["response_text"],
                parsed=obj.get("parsed") or {},
            )
            cache[rec.key] = rec
    return cache


def append_cache(cache_path: str, records: Iterable[CacheRecord]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

