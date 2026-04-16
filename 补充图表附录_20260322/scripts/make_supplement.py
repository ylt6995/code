import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import openpyxl

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bidrig.baseline_logit import _features_from_indicators
from bidrig.dataset import ProjectSample, load_all_data_xlsx


@dataclass(frozen=True)
class RunFiles:
    root: Path
    split_ids: Path


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def _ks_2samp_d_p(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x_sorted = sorted(float(v) for v in x)
    y_sorted = sorted(float(v) for v in y)
    n = len(x_sorted)
    m = len(y_sorted)
    if n == 0 or m == 0:
        return float("nan"), float("nan")

    i = 0
    j = 0
    d = 0.0
    while i < n and j < m:
        if x_sorted[i] <= y_sorted[j]:
            v = x_sorted[i]
            while i < n and x_sorted[i] == v:
                i += 1
        else:
            v = y_sorted[j]
            while j < m and y_sorted[j] == v:
                j += 1
        cdf_x = i / n
        cdf_y = j / m
        d = max(d, abs(cdf_x - cdf_y))

    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 0.0
    for k in range(1, 100):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (lam**2) * (k**2))
        p += term
        if abs(term) < 1e-10:
            break
    p = max(0.0, min(1.0, p))
    return float(d), float(p)


def _discover_run_dir(runs_root: Path, suffix: str) -> Path:
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"未找到 runs/*{suffix} 目录")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _load_split_ids(path: Path) -> Dict[str, List[str]]:
    obj = _read_json(path)
    assert isinstance(obj, dict)
    return {str(k): [str(x) for x in v] for k, v in obj.items()}


def _load_samples_by_id(excel_path: Path) -> Dict[str, ProjectSample]:
    samples = load_all_data_xlsx(str(excel_path))
    return {s.bid_ann_guid: s for s in samples}


def _to_xy(samples: Sequence[ProjectSample]) -> Tuple[np.ndarray, np.ndarray]:
    x: List[List[float]] = []
    y: List[int] = []
    for s in samples:
        x.append(_features_from_indicators(s.indicators))
        y.append(int(s.label))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=int)


def _find_files_for_gpt_all(run_dir: Path) -> RunFiles:
    split_ids = run_dir / "split_ids.json"
    if not split_ids.exists():
        raise FileNotFoundError(f"缺少 split_ids.json：{split_ids}")
    return RunFiles(root=run_dir, split_ids=split_ids)


def _find_files_for_zhipu_all(run_dir: Path) -> RunFiles:
    split_ids = run_dir / "split_ids.json"
    if not split_ids.exists():
        raise FileNotFoundError(f"缺少 split_ids.json：{split_ids}")
    return RunFiles(root=run_dir, split_ids=split_ids)


def _iter_metric_jsons(run_dir: Path, provider: str, method: str) -> Iterable[Tuple[int, Path]]:
    for p in sorted(run_dir.glob(f"metrics_{provider}_{method}_seed*.json")):
        seed_str = p.stem.split("_seed")[-1]
        yield int(seed_str), p


def _score_map_from_test_csv(path: Path) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for r in _read_csv_rows(path):
        bid = str(r["bid_ann_guid"])
        out[bid] = (int(r["label"]), int(float(r["score"])))
    return out


def _score_map_from_csvs(paths: Sequence[Path]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for p in paths:
        if not p.exists():
            continue
        out.update(_score_map_from_test_csv(p))
    return out


def _auc_from_scores(y_true: Sequence[int], scores: Sequence[float]) -> Tuple[float, float]:
    if len(set(y_true)) < 2:
        return float("nan"), float("nan")
    roc = float(roc_auc_score(y_true, scores))
    pr = float(average_precision_score(y_true, scores))
    return roc, pr


def _collect_llm_seed_metrics(run_dir: Path, provider: str, method: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed, mj in _iter_metric_jsons(run_dir, provider, method):
        obj = _read_json(mj)
        assert isinstance(obj, dict)
        val = obj.get("val") or {}
        test = obj.get("test") or {}
        rows.append(
            {
                "seed": int(seed),
                "best_threshold": int(val.get("threshold") or val.get("best_threshold") or obj.get("best_threshold") or 0),
                "test_accuracy": float(test.get("accuracy") or 0.0),
                "test_precision": float(test.get("precision") or 0.0),
                "test_recall": float(test.get("recall") or 0.0),
                "test_f1": float(test.get("f1") or 0.0),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        with alt.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)


def _plot_roc_pr(
    out_roc: Path,
    out_pr: Path,
    curves: List[Tuple[str, Sequence[int], Sequence[float]]],
) -> None:
    out_roc.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    for name, y, s in curves:
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Test Set)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_roc, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, y, s in curves:
        if len(set(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curves (Test Set)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_pr, dpi=200)
    plt.close()


def _plot_recall_comparison(
    out_bar: Path,
    out_box: Path,
    *,
    methods: Sequence[str],
    gpt_recalls_by_method: Dict[str, List[float]],
    zhipu_recall_by_method: Dict[str, float],
) -> None:
    out_bar.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(methods), dtype=float)
    width = 0.35

    gpt_means = []
    gpt_stds = []
    zhipu_vals = []
    for m in methods:
        xs = gpt_recalls_by_method.get(m) or []
        mean, std = _mean_std(xs)
        gpt_means.append(mean)
        gpt_stds.append(std)
        zhipu_vals.append(float(zhipu_recall_by_method.get(m, float("nan"))))

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width / 2, gpt_means, width=width, yerr=gpt_stds, capsize=4, label="GPT-4o-mini (mean±std)")
    plt.bar(x + width / 2, zhipu_vals, width=width, label="GLM-4.7 (seed=0)")
    plt.xticks(x, [m.upper() for m in methods])
    plt.ylim(0, 1.05)
    plt.ylabel("Recall")
    plt.title("Recall Comparison by Prompting Method (Test Set)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_bar, dpi=200)
    plt.close()

def _write_recall_by_method_table(
    out_csv: Path,
    *,
    methods: Sequence[str],
    gpt_recalls_by_method: Dict[str, List[float]],
    zhipu_recall_by_method: Dict[str, float],
) -> None:
    rows: List[Dict[str, object]] = []
    for m in methods:
        g_mean, g_std = _mean_std(gpt_recalls_by_method.get(m) or [])
        glm_v = zhipu_recall_by_method.get(m)
        glm_cell: object
        if glm_v is None:
            glm_cell = ""
        else:
            glm_cell = "" if math.isnan(float(glm_v)) else float(glm_v)
        rows.append(
            {
                "method": m,
                "gpt_recall_mean": g_mean,
                "gpt_recall_std": g_std,
                "glm_recall": glm_cell,
            }
        )
    _write_csv(out_csv, rows)


def _write_recall_by_model_table(out_csv: Path, values: Dict[str, float]) -> None:
    rows: List[Dict[str, object]] = []
    for k, v in values.items():
        fv = float(v)
        cell: object = "" if math.isnan(fv) else fv
        rows.append({"model": k, "recall": cell})
    _write_csv(out_csv, rows)

def _collect_latency_from_excels(root: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"GLM-4.7": [], "GPT-4o-mini": []}
    if not root.exists():
        return out
    for p in root.glob("*.xlsx"):
        name = p.name.lower()
        label = None
        if "gpt" in name or "chat" in name:
            label = "GPT-4o-mini"
        elif "zhipu" in name or "glm" in name or "model-zhipu" in name:
            label = "GLM-4.7"
        else:
            continue
        try:
            wb = openpyxl.load_workbook(p, data_only=True)
            ws = wb.active
            for r in ws.iter_rows(min_row=2):
                v = r[4].value if len(r) >= 5 else None
                if isinstance(v, (int, float)) and v >= 0:
                    out[label].append(float(v))
        except Exception:
            continue
    return out
    plt.figure(figsize=(7.5, 4.5))
    data = [gpt_recalls_by_method.get(m) or [float("nan")] for m in methods]
    plt.boxplot(data, labels=[m.upper() for m in methods], showfliers=False)
    for i, m in enumerate(methods, start=1):
        v = zhipu_recall_by_method.get(m)
        if v is None:
            continue
        plt.scatter([i], [v], color="black", zorder=3)
    plt.ylim(0, 1.05)
    plt.ylabel("Recall")
    plt.title("Recall Distribution (GPT seeds) with GLM point (Test Set)")
    plt.tight_layout()
    plt.savefig(out_box, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--excel", default=os.path.join("data", "all_data.xlsx"))
    ap.add_argument("--gpt-run", default="")
    ap.add_argument("--zhipu-run", default="")
    ap.add_argument("--zhipu-zero-run", default="")
    ap.add_argument("--zhipu-few-run", default="")
    ap.add_argument("--zhipu-cot-run", default="")
    ap.add_argument("--out-root", default="补充图表附录_20260323")
    ap.add_argument("--ks-scope", choices=["test", "val_test"], default="val_test")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_root)
    out_tables = out_dir / "tables"
    out_figs = out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    gpt_dir = Path(args.gpt_run) if args.gpt_run else _discover_run_dir(runs_root, "_gpt_all")
    zhipu_dir = Path(args.zhipu_run) if args.zhipu_run else _discover_run_dir(runs_root, "_zhipu_all")
    zhipu_dir_map: Dict[str, Path] = {}
    if args.zhipu_zero_run:
        zhipu_dir_map["zero"] = Path(args.zhipu_zero_run)
    if args.zhipu_few_run:
        zhipu_dir_map["few"] = Path(args.zhipu_few_run)
    if args.zhipu_cot_run:
        zhipu_dir_map["cot"] = Path(args.zhipu_cot_run)

    gpt_files = _find_files_for_gpt_all(gpt_dir)
    zhipu_files = _find_files_for_zhipu_all(zhipu_dir)

    split_ids = _load_split_ids(gpt_files.split_ids)
    test_ids = split_ids.get("test") or []
    val_ids = split_ids.get("val") or []
    ks_ids = test_ids if args.ks_scope == "test" else (val_ids + test_ids)

    samples_by_id = _load_samples_by_id(Path(args.excel))
    train_samples = [samples_by_id[x] for x in split_ids.get("train") or [] if x in samples_by_id]
    val_samples = [samples_by_id[x] for x in split_ids.get("val") or [] if x in samples_by_id]
    test_samples = [samples_by_id[x] for x in test_ids if x in samples_by_id]

    x_train, y_train = _to_xy(train_samples)
    x_val, y_val = _to_xy(val_samples)
    x_test, y_test = _to_xy(test_samples)

    lr = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=42)
    lr.fit(x_train, y_train)
    lr_test_prob = lr.predict_proba(x_test)[:, 1]
    lr_recall = float(np.mean((lr_test_prob >= 0.5).astype(int) == 1 * (y_test == 1))) if False else float(np.sum((lr_test_prob >= 0.5) & (y_test == 1)) / max(1, np.sum(y_test == 1)))

    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(x_train, y_train)
        xgb_test_prob = xgb_model.predict_proba(x_test)[:, 1]
        xgb_recall = float(np.sum((xgb_test_prob >= 0.5) & (y_test == 1)) / max(1, np.sum(y_test == 1)))
    except Exception:
        xgb_test_prob = None
        xgb_recall = float("nan")

    try:
        import lightgbm as lgb

        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            class_weight="balanced",
        )
        lgb_model.fit(x_train, y_train)
        lgb_test_prob = lgb_model.predict_proba(x_test)[:, 1]
        lgb_recall = float(np.sum((lgb_test_prob >= 0.5) & (y_test == 1)) / max(1, np.sum(y_test == 1)))
    except Exception:
        lgb_test_prob = None
        lgb_recall = float("nan")

    methods = ["zero", "few", "cot"]
    gpt_method_rows: List[Dict[str, object]] = []
    table2_rows: List[Dict[str, object]] = []
    ks_rows: List[Dict[str, object]] = []
    gpt_recalls_by_method: Dict[str, List[float]] = {}

    gpt_zero_seed_scores_by_bid: Dict[str, List[int]] = {bid: [] for bid in test_ids}
    best_gpt_method = None
    best_gpt_f1 = -1.0

    for method in methods:
        per_seed = _collect_llm_seed_metrics(gpt_files.root, "gpt", method)
        gpt_recalls_by_method[method] = [float(r["test_recall"]) for r in per_seed]
        f1s = [float(r["test_f1"]) for r in per_seed]
        f1_mean, f1_std = _mean_std(f1s)

        roc_list: List[float] = []
        pr_list: List[float] = []
        ks_d_list: List[float] = []
        ks_p_list: List[float] = []

        for r in per_seed:
            seed = int(r["seed"])
            test_csv = gpt_files.root / f"test_gpt_{method}_seed{seed}.csv"
            val_csv = gpt_files.root / f"val_gpt_{method}_seed{seed}.csv"
            if args.ks_scope == "test":
                m = _score_map_from_csvs([test_csv])
                ids_for_auc = test_ids
                ids_for_ks = test_ids
            else:
                m = _score_map_from_csvs([val_csv, test_csv])
                ids_for_auc = test_ids
                ids_for_ks = ks_ids

            y = [int(m[bid][0]) for bid in ids_for_auc if bid in m]
            s = [float(m[bid][1]) / 100.0 for bid in ids_for_auc if bid in m]
            roc, pr = _auc_from_scores(y, s)
            roc_list.append(roc)
            pr_list.append(pr)
            pos_scores = [float(m[bid][1]) for bid in ids_for_ks if bid in m and int(m[bid][0]) == 1]
            neg_scores = [float(m[bid][1]) for bid in ids_for_ks if bid in m and int(m[bid][0]) == 0]
            d, p = _ks_2samp_d_p(pos_scores, neg_scores)
            ks_d_list.append(d)
            ks_p_list.append(p)

            if method == "zero":
                for bid in test_ids:
                    if bid in m:
                        gpt_zero_seed_scores_by_bid[bid].append(int(m[bid][1]))

        roc_mean, roc_std = _mean_std(roc_list)
        pr_mean, pr_std = _mean_std(pr_list)

        gpt_method_rows.append(
            {
                "provider": "gpt",
                "model": "gpt-4o-mini",
                "method": method,
                "seeds": len(per_seed),
                "roc_auc_mean": roc_mean,
                "roc_auc_std": roc_std,
                "pr_auc_mean": pr_mean,
                "pr_auc_std": pr_std,
                "test_precision_mean": _mean_std([float(r["test_precision"]) for r in per_seed])[0],
                "test_precision_std": _mean_std([float(r["test_precision"]) for r in per_seed])[1],
                "test_recall_mean": _mean_std([float(r["test_recall"]) for r in per_seed])[0],
                "test_recall_std": _mean_std([float(r["test_recall"]) for r in per_seed])[1],
                "test_f1_mean": f1_mean,
                "test_f1_std": f1_std,
            }
        )

        ks_rows.append(
            {
                "provider": "gpt",
                "model": "gpt-4o-mini",
                "method": method,
                "ks_scope": args.ks_scope,
                "seed_count": len(ks_d_list),
                "ks_D_mean": _mean_std(ks_d_list)[0],
                "ks_D_std": _mean_std(ks_d_list)[1],
                "ks_p_mean": _mean_std(ks_p_list)[0],
                "ks_p_std": _mean_std(ks_p_list)[1],
            }
        )

        if f1_mean > best_gpt_f1:
            best_gpt_f1 = f1_mean
            best_gpt_method = method

        if method == "zero":
            for r in per_seed:
                table2_rows.append(dict(r))

    zhipu_method_rows: List[Dict[str, object]] = []
    best_zhipu_method = None
    best_zhipu_f1 = -1.0
    zhipu_scores_by_method: Dict[str, Dict[str, int]] = {}
    zhipu_recall_by_method: Dict[str, float] = {}

    for method in methods:
        zdir = zhipu_dir_map.get(method, zhipu_files.root)
        test_csv = zdir / f"test_zhipu_{method}_seed0.csv"
        val_csv = zdir / f"val_zhipu_{method}_seed0.csv"
        if args.ks_scope == "test":
            m = _score_map_from_csvs([test_csv])
            ids_for_ks = test_ids
        else:
            m = _score_map_from_csvs([val_csv, test_csv])
            ids_for_ks = ks_ids
        if not m:
            continue
        y = [int(m[bid][0]) for bid in test_ids if bid in m]
        s = [float(m[bid][1]) / 100.0 for bid in test_ids if bid in m]
        roc, pr = _auc_from_scores(y, s)
        zhipu_scores_by_method[method] = {bid: int(m[bid][1]) for bid in test_ids if bid in m}

        pos_scores = [float(m[bid][1]) for bid in ids_for_ks if bid in m and int(m[bid][0]) == 1]
        neg_scores = [float(m[bid][1]) for bid in ids_for_ks if bid in m and int(m[bid][0]) == 0]
        d, p = _ks_2samp_d_p(pos_scores, neg_scores)
        ks_rows.append(
            {
                "provider": "zhipu",
                "model": "glm-4.7",
                "method": method,
                "ks_scope": args.ks_scope,
                "seed_count": 1,
                "ks_D_mean": d,
                "ks_D_std": 0.0,
                "ks_p_mean": p,
                "ks_p_std": 0.0,
            }
        )

        metrics_json = zdir / f"metrics_zhipu_{method}_seed0.json"
        if metrics_json.exists():
            obj = _read_json(metrics_json)
            assert isinstance(obj, dict)
            test_m = obj.get("test") or {}
            prec = float(test_m.get("precision") or 0.0)
            rec = float(test_m.get("recall") or 0.0)
            f1 = float(test_m.get("f1") or 0.0)
        else:
            prec = float("nan")
            rec = float("nan")
            f1 = float("nan")
        zhipu_recall_by_method[method] = rec

        zhipu_method_rows.append(
            {
                "provider": "zhipu",
                "model": "glm-4.7",
                "method": method,
                "seeds": 1,
                "roc_auc_mean": roc,
                "roc_auc_std": 0.0,
                "pr_auc_mean": pr,
                "pr_auc_std": 0.0,
                "test_precision_mean": prec,
                "test_precision_std": 0.0,
                "test_recall_mean": rec,
                "test_recall_std": 0.0,
                "test_f1_mean": f1,
                "test_f1_std": 0.0,
            }
        )
        if f1 > best_zhipu_f1:
            best_zhipu_f1 = f1
            best_zhipu_method = method

    _write_csv(out_tables / "table1_llm_metrics_by_method.csv", gpt_method_rows + zhipu_method_rows)

    gpt_zero_acc = [float(r["test_accuracy"]) for r in table2_rows]
    gpt_zero_prec = [float(r["test_precision"]) for r in table2_rows]
    gpt_zero_rec = [float(r["test_recall"]) for r in table2_rows]
    gpt_zero_f1 = [float(r["test_f1"]) for r in table2_rows]

    table2_summary = [
        {
            "provider": "gpt",
            "model": "gpt-4o-mini",
            "method": "zero",
            "seed_count": len(table2_rows),
            "test_accuracy_mean": _mean_std(gpt_zero_acc)[0],
            "test_accuracy_std": _mean_std(gpt_zero_acc)[1],
            "test_precision_mean": _mean_std(gpt_zero_prec)[0],
            "test_precision_std": _mean_std(gpt_zero_prec)[1],
            "test_recall_mean": _mean_std(gpt_zero_rec)[0],
            "test_recall_std": _mean_std(gpt_zero_rec)[1],
            "test_f1_mean": _mean_std(gpt_zero_f1)[0],
            "test_f1_std": _mean_std(gpt_zero_f1)[1],
        }
    ]
    _write_csv(out_tables / "table2_gpt_zero_10seeds_mean_std.csv", table2_summary)

    _write_csv(out_tables / "ks_test_scores.csv", ks_rows)

    if best_gpt_method is None:
        raise RuntimeError("未找到 GPT metrics_* 文件")
    if best_zhipu_method is None:
        raise RuntimeError("未找到 ZHIPU metrics/test 文件")

    _write_recall_by_method_table(
        out_tables / "recall_by_method.csv",
        methods=methods,
        gpt_recalls_by_method=gpt_recalls_by_method,
        zhipu_recall_by_method=zhipu_recall_by_method,
    )

    models_recall = {
        "LR": lr_recall,
        "LightGBM": lgb_recall,
        "XGBoost": xgb_recall,
        "GLM-4.7": zhipu_recall_by_method.get("zero", float("nan")),
        "GPT-4o-mini": _mean_std(gpt_recalls_by_method.get("zero", []))[0],
    }
    _write_recall_by_model_table(out_tables / "recall_by_model.csv", models_recall)

    gpt_avg_scores = []
    for bid in test_ids:
        xs = gpt_zero_seed_scores_by_bid.get(bid) or []
        gpt_avg_scores.append(float(statistics.mean(xs)) / 100.0 if xs else 0.0)

    zhipu_zero_scores = zhipu_scores_by_method.get("zero") or {}
    zhipu_scores = [float(zhipu_zero_scores.get(bid, 0)) / 100.0 for bid in test_ids]

    curves: List[Tuple[str, Sequence[int], Sequence[float]]] = [
        ("LR", y_test, lr_test_prob.tolist()),
        ("GLM-4.7", y_test, zhipu_scores),
        ("GPT-4o-mini", y_test, gpt_avg_scores),
    ]
    if lgb_test_prob is not None:
        curves.insert(1, ("LightGBM", y_test, lgb_test_prob.tolist()))
    if xgb_test_prob is not None:
        curves.insert(1, ("XGBoost", y_test, xgb_test_prob.tolist()))

    _plot_roc_pr(out_figs / "roc_curves.png", out_figs / "pr_curves.png", curves)

    meta = {
        "gpt_run_dir": str(gpt_dir),
        "zhipu_run_dir": str(zhipu_dir),
        "zhipu_zero_run_dir": str(zhipu_dir_map.get("zero", zhipu_files.root)),
        "zhipu_few_run_dir": str(zhipu_dir_map.get("few", "")),
        "zhipu_cot_run_dir": str(zhipu_dir_map.get("cot", zhipu_files.root)),
        "excel": str(Path(args.excel)),
        "test_size": len(test_ids),
        "best_gpt_method_by_test_f1_mean": best_gpt_method,
        "best_zhipu_method_by_test_f1": best_zhipu_method,
        "notes": {
            "gpt_curve_score": "zero-shot scores averaged over 10 prompt seeds, normalized to [0,1] by /100",
            "zhipu_curve_score": "zero-shot seed=0 score normalized to [0,1] by /100",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    latency_todo = out_dir / "LATENCY_TODO.md"
    lat = _collect_latency_from_excels(Path("实验结果"))
    if lat["GLM-4.7"] or lat["GPT-4o-mini"]:
        plt.figure(figsize=(7.5, 4.5))
        plt.boxplot([lat["GLM-4.7"], lat["GPT-4o-mini"]], labels=["GLM-4.7", "GPT-4o-mini"], showfliers=False)
        plt.ylabel("Processing Time (s)")
        plt.title("Latency Distribution (from 实验结果/*.xlsx)")
        plt.tight_layout()
        plt.savefig(out_figs / "latency_box.png", dpi=200)
        plt.close()
        plt.figure(figsize=(7.5, 4.5))
        means = [np.nanmean(lat["GLM-4.7"]) if lat["GLM-4.7"] else float("nan"), np.nanmean(lat["GPT-4o-mini"]) if lat["GPT-4o-mini"] else float("nan")]
        stds = [np.nanstd(lat["GLM-4.7"]) if lat["GLM-4.7"] else float("nan"), np.nanstd(lat["GPT-4o-mini"]) if lat["GPT-4o-mini"] else float("nan")]
        x = np.arange(2, dtype=float)
        plt.bar(x, means, yerr=stds, capsize=4, width=0.6, tick_label=["GLM-4.7", "GPT-4o-mini"])
        plt.ylabel("Processing Time (s)")
        plt.title("Latency Comparison (from 实验结果/*.xlsx)")
        plt.tight_layout()
        plt.savefig(out_figs / "latency_bar.png", dpi=200)
        plt.close()
        if latency_todo.exists():
            latency_todo.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

