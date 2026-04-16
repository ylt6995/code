from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class BaselineResult:
    threshold: float
    val: Dict[str, float]
    test: Dict[str, float]
    test_probs: List[float]
    test_preds: List[int]


def run_logit_baseline(
    train: Sequence[object],
    val: Sequence[object],
    test: Sequence[object],
) -> BaselineResult:
    x_train, y_train = _to_xy(train)
    x_val, y_val = _to_xy(val)
    x_test, y_test = _to_xy(test)

    model = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=42)
    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)[:, 1]
    best_thr, best_val = _find_threshold_max_recall(val_probs, y_val)

    test_probs = model.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= best_thr).astype(int)
    test_m = _metrics_from_preds(test_preds, y_test)

    return BaselineResult(
        threshold=float(best_thr),
        val=best_val,
        test=test_m,
        test_probs=[float(x) for x in test_probs],
        test_preds=[int(x) for x in test_preds],
    )


def _to_xy(items: Sequence[object]) -> Tuple[np.ndarray, np.ndarray]:
    x: List[List[float]] = []
    y: List[int] = []
    for s in items:
        ind = getattr(s, "indicators", {}) or {}
        feats = _features_from_indicators(ind)
        x.append(feats)
        y.append(int(getattr(s, "label")))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=int)


def _features_from_indicators(ind: Dict[str, object]) -> List[float]:
    rd = ind.get("rd")
    cv = ind.get("cv_losing")
    price_cv = ind.get("price_cv_all")
    contact_dup = ind.get("contact_dup_count") or 0
    phone_dup = ind.get("phone_dup_count") or 0
    email_dup = ind.get("email_dup_count") or 0

    rd_f = _cap_float(rd, cap=10.0, default=0.0, inf_cap=10.0)
    cv_f = _cap_float(cv, cap=1.0, default=1.0, inf_cap=1.0)
    price_cv_f = _cap_float(price_cv, cap=1.0, default=0.0, inf_cap=1.0)

    return [
        rd_f,
        cv_f,
        price_cv_f,
        float(contact_dup),
        float(phone_dup),
        float(email_dup),
    ]


def _cap_float(v: object, *, cap: float, default: float, inf_cap: float) -> float:
    if v is None:
        return default
    try:
        x = float(v)
    except Exception:
        return default
    if x == float("inf") or x == float("-inf"):
        return inf_cap
    if x != x:
        return default
    if x > cap:
        return cap
    if x < -cap:
        return -cap
    return x


def _find_threshold_max_recall(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best = None
    thresholds = np.unique(probs)
    thresholds = np.concatenate([thresholds, np.array([0.0, 1.0])])
    thresholds = np.unique(thresholds)

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        m = _metrics_from_preds(preds, labels)
        if best is None:
            best = (thr, m)
            continue
        if m["recall"] > best[1]["recall"]:
            best = (thr, m)
            continue
        if m["recall"] < best[1]["recall"]:
            continue
        if m["precision"] > best[1]["precision"]:
            best = (thr, m)
            continue
        if m["precision"] < best[1]["precision"]:
            continue
        if m["f1"] > best[1]["f1"]:
            best = (thr, m)

    assert best is not None
    return float(best[0]), best[1]


def _metrics_from_preds(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }

