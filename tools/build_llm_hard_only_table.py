import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import openpyxl
from openpyxl.workbook import Workbook
from sklearn.metrics import average_precision_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(xs: Sequence[float]) -> float:
    xs2 = [float(x) for x in xs if not math.isnan(float(x))]
    if not xs2:
        return float("nan")
    return float(statistics.mean(xs2))


def _auc_pr(test_csv: Path) -> Tuple[float, float]:
    rows = _read_csv_rows(test_csv)
    y: List[int] = []
    s: List[float] = []
    for r in rows:
        y.append(int(float(r["label"])))
        s.append(float(r["score"]) / 100.0)
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _discover_seeds(run_dir: Path, provider: str, method: str) -> List[int]:
    seeds: List[int] = []
    for p in run_dir.glob(f"metrics_{provider}_{method}_seed*.json"):
        try:
            seeds.append(int(p.stem.split("_seed")[-1]))
        except Exception:
            continue
    return sorted(set(seeds))


def _prompt_preview(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return ""
    t = path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


def _prompt_path(run_dir: Path, provider: str, method: str, seed: int) -> Path:
    return run_dir / f"prompt_{provider}_{method}_seed{seed}.txt"


def _extract_test_metrics(metrics_json: Path) -> Tuple[float, float, float]:
    obj = _read_json(metrics_json)
    test = obj.get("test") if isinstance(obj, dict) else None
    if not isinstance(test, dict):
        return float("nan"), float("nan"), float("nan")
    prec = float(test.get("precision") or 0.0)
    rec = float(test.get("recall") or 0.0)
    f1 = float(test.get("f1") or 0.0)
    return prec, rec, f1


def _write_xlsx(path: Path, table_rows: List[Dict[str, object]], prompt_rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb: Workbook = openpyxl.Workbook()
    ws = wb.active
    ws.title = "hard_only_llm"

    header = ["provider", "model", "method", "Prompt", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
    ws.append(header)
    for r in table_rows:
        ws.append([r.get(h, "") for h in header])

    ws2 = wb.create_sheet("prompts_full")
    header2 = ["provider", "model", "method", "seed", "prompt_path", "prompt_text"]
    ws2.append(header2)
    for r in prompt_rows:
        ws2.append([r.get(h, "") for h in header2])

    wb.save(path)


def _write_csv(path: Path, table_rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["provider", "model", "method", "Prompt", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in table_rows:
            w.writerow({h: r.get(h, "") for h in header})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="Directory of the hard-only experiment (e.g. runs/20260414_hard_price_time_llm)")
    ap.add_argument("--gpt-run", default=str(Path("runs") / "20260321_180011_gpt_all"))
    ap.add_argument("--zhipu-zero-cot-run", default=str(Path("runs") / "20260322_001430_zhipu_all"))
    ap.add_argument("--zhipu-few-run", default=str(Path("runs") / "20260323_231515_new_zhipu_fewshot"))
    ap.add_argument("--out-dir", default=f"llm_hard_only_table_{datetime.now().strftime('%Y%m%d')}")
    ap.add_argument("--out-base", default="llm_hard_only_table")
    ap.add_argument("--prompt-preview-chars", type=int, default=800)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        gpt_run = Path(args.run_dir)
        zhipu_zero_cot_run = Path(args.run_dir)
        zhipu_few_run = Path(args.run_dir)
    else:
        gpt_run = Path(args.gpt_run)
        zhipu_zero_cot_run = Path(args.zhipu_zero_cot_run)
        zhipu_few_run = Path(args.zhipu_few_run)

    table_rows: List[Dict[str, object]] = []
    prompt_rows: List[Dict[str, object]] = []

    for provider, model, run_dir in [("gpt", "gpt-4o-mini", gpt_run)]:
        for method in ["zero", "few", "cot"]:
            seeds = _discover_seeds(run_dir, provider, method)
            if not seeds:
                raise RuntimeError(f"未找到 metrics：{run_dir}（provider={provider} method={method}）")
            roc_list: List[float] = []
            pr_list: List[float] = []
            prec_list: List[float] = []
            rec_list: List[float] = []
            f1_list: List[float] = []

            for seed in seeds:
                test_csv = run_dir / f"test_{provider}_{method}_seed{seed}.csv"
                metrics_json = run_dir / f"metrics_{provider}_{method}_seed{seed}.json"
                if test_csv.exists():
                    roc, pr = _auc_pr(test_csv)
                    roc_list.append(roc)
                    pr_list.append(pr)
                if metrics_json.exists():
                    p, r, f1 = _extract_test_metrics(metrics_json)
                    prec_list.append(p)
                    rec_list.append(r)
                    f1_list.append(f1)

                pp = _prompt_path(run_dir, provider, method, seed)
                if pp.exists():
                    prompt_rows.append(
                        {
                            "provider": provider,
                            "model": model,
                            "method": method,
                            "seed": seed,
                            "prompt_path": str(pp),
                            "prompt_text": pp.read_text(encoding="utf-8", errors="ignore"),
                        }
                    )

            rep_seed = seeds[0]
            rep_prompt = _prompt_path(run_dir, provider, method, rep_seed)
            preview = _prompt_preview(rep_prompt, max_chars=int(args.prompt_preview_chars))
            prompt_cell = (str(rep_prompt) + "\n" + preview).strip()

            table_rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "method": method,
                    "Prompt": prompt_cell,
                    "ROC-AUC": _mean(roc_list),
                    "PR-AUC": _mean(pr_list),
                    "Precision": _mean(prec_list),
                    "Recall": _mean(rec_list),
                    "F1": _mean(f1_list),
                }
            )

    provider = "zhipu"
    model = "glm-4.7"
    for method, run_dir in [("zero", zhipu_zero_cot_run), ("few", zhipu_few_run), ("cot", zhipu_zero_cot_run)]:
        test_csv = run_dir / f"test_{provider}_{method}_seed0.csv"
        metrics_json = run_dir / f"metrics_{provider}_{method}_seed0.json"
        prompt_path = run_dir / f"prompt_{provider}_{method}_seed0.txt"
        if not test_csv.exists() or not metrics_json.exists() or not prompt_path.exists():
            raise RuntimeError(f"缺少文件：{run_dir}（provider={provider} method={method}）")
        roc, pr = _auc_pr(test_csv)
        prec, rec, f1 = _extract_test_metrics(metrics_json)
        preview = _prompt_preview(prompt_path, max_chars=int(args.prompt_preview_chars))
        prompt_cell = (str(prompt_path) + "\n" + preview).strip()
        table_rows.append(
            {
                "provider": provider,
                "model": model,
                "method": method,
                "Prompt": prompt_cell,
                "ROC-AUC": roc,
                "PR-AUC": pr,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
            }
        )
        if prompt_path.exists():
            prompt_rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "method": method,
                    "seed": 0,
                    "prompt_path": str(prompt_path),
                    "prompt_text": prompt_path.read_text(encoding="utf-8", errors="ignore"),
                }
            )

    out_xlsx = out_dir / f"{args.out_base}.xlsx"
    out_csv = out_dir / f"{args.out_base}.csv"
    _write_xlsx(out_xlsx, table_rows, prompt_rows)
    _write_csv(out_csv, table_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
