from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_rate: float


def compute_metrics(scores: Iterable[int], labels: Iterable[int], threshold: int) -> BinaryMetrics:
    tp = fp = fn = tn = 0
    score_list = list(scores)
    label_list = list(labels)
    if len(score_list) != len(label_list):
        raise ValueError("scores and labels length mismatch")

    for s, y in zip(score_list, label_list):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(score_list) if score_list else 0.0
    positive_rate = (tp + fp) / len(score_list) if score_list else 0.0

    return BinaryMetrics(
        threshold=threshold,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        positive_rate=positive_rate,
    )


def find_threshold_max_recall(scores: List[int], labels: List[int]) -> BinaryMetrics:
    best: Optional[BinaryMetrics] = None
    for t in range(0, 101):
        m = compute_metrics(scores, labels, threshold=t)
        if best is None:
            best = m
            continue
        if m.recall > best.recall:
            best = m
            continue
        if m.recall < best.recall:
            continue
        if m.precision > best.precision:
            best = m
            continue
        if m.precision < best.precision:
            continue
        if m.f1 > best.f1:
            best = m
            continue
        if m.f1 < best.f1:
            continue
        if m.positive_rate < best.positive_rate:
            best = m

    if best is None:
        return compute_metrics([], [], threshold=50)
    return best


def find_threshold_recall_at_least(scores: List[int], labels: List[int], *, recall_target: float) -> BinaryMetrics:
    target = float(recall_target)
    if target < 0:
        target = 0.0
    if target > 1:
        target = 1.0

    best: Optional[BinaryMetrics] = None
    best_any: Optional[BinaryMetrics] = None

    for t in range(0, 101):
        m = compute_metrics(scores, labels, threshold=t)
        if best_any is None or m.recall > best_any.recall or (m.recall == best_any.recall and m.precision > best_any.precision):
            best_any = m

        if m.recall < target:
            continue

        if best is None:
            best = m
            continue

        if m.precision > best.precision:
            best = m
            continue
        if m.precision < best.precision:
            continue
        if m.f1 > best.f1:
            best = m
            continue
        if m.f1 < best.f1:
            continue
        if m.positive_rate < best.positive_rate:
            best = m

    if best is not None:
        return best
    if best_any is not None:
        return best_any
    return compute_metrics([], [], threshold=50)


def summarize_runs(metrics: List[BinaryMetrics]) -> Dict[str, float]:
    if not metrics:
        return {}
    return {
        "recall_mean": sum(m.recall for m in metrics) / len(metrics),
        "recall_min": min(m.recall for m in metrics),
        "recall_max": max(m.recall for m in metrics),
        "accuracy_mean": sum(m.accuracy for m in metrics) / len(metrics),
        "precision_mean": sum(m.precision for m in metrics) / len(metrics),
        "f1_mean": sum(m.f1 for m in metrics) / len(metrics),
        "threshold_mean": sum(m.threshold for m in metrics) / len(metrics),
    }
