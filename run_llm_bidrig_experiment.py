import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from bidrig.dataset import ProjectSample, load_all_data_xlsx
from bidrig.io_cache import CacheRecord, append_cache, load_cache, make_cache_key
from bidrig.llm_client import LLMSettings, chat_with_retry, load_settings
from bidrig.metrics import BinaryMetrics, compute_metrics, find_threshold_max_f1, find_threshold_max_recall, find_threshold_recall_at_least, summarize_runs
from bidrig.parse_output import extract_score, parse_model_json
from bidrig.prompts import build_prompt
from bidrig.split import split_6_2_2_stratified, split_fixed_test_balance
from bidrig.plotting import plot_metrics_by_seed_png, plot_recall_boxplot_png
from bidrig.baseline_logit import run_logit_baseline


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    samples = load_all_data_xlsx(args.data)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    labels = [s.label for s in samples]
    if args.fixed_test_balance:
        split = split_fixed_test_balance(
            samples,
            labels,
            seed=args.split_seed,
            test_size=args.test_size,
            test_pos=args.test_pos,
        )
    else:
        split = split_6_2_2_stratified(samples, labels, seed=args.split_seed)

    write_json(os.path.join(out_dir, "split_sizes.json"), {k: len(v) for k, v in split.items()})
    write_json(os.path.join(out_dir, "split_ids.json"), _split_ids(split))
    write_json(os.path.join(out_dir, "split_label_counts.json"), _split_label_counts(split))

    baseline = run_logit_baseline(split["train"], split["val"], split["test"])
    write_json(
        os.path.join(out_dir, "baseline_logit.json"),
        {"threshold": baseline.threshold, "val": baseline.val, "test": baseline.test},
    )
    baseline_test_rows = []
    for s, prob, pred in zip(split["test"], baseline.test_probs, baseline.test_preds):
        baseline_test_rows.append(
            {
                "bid_ann_guid": s.bid_ann_guid,
                "label": int(s.label),
                "baseline_prob": float(prob),
                "baseline_pred": int(pred),
            }
        )
    write_csv(os.path.join(out_dir, "baseline_logit_test.csv"), baseline_test_rows)

    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    prompt_seeds = parse_seed_list(args.prompt_seeds, base_seed=args.split_seed)
    recall_target_by_method = parse_method_float_map(args.recall_target_by_method)

    cache_path = args.cache_path
    cache = load_cache(cache_path)

    settings_by_provider: Dict[str, LLMSettings] = {}
    if not args.dry_run and not args.mock:
        for p in providers:
            settings_by_provider[p] = load_settings(p)

    provider_settings = {
        p: asdict(s) for p, s in settings_by_provider.items()
    }

    summary: Dict[str, object] = {
        "run_id": run_id,
        "data": args.data,
        "split_seed": args.split_seed,
        "prompt_seeds": prompt_seeds,
        "providers": providers,
        "methods": methods,
        "fewshot_k_per_class": args.fewshot_k_per_class,
        "cache_path": cache_path,
        "dry_run": args.dry_run,
        "mock": args.mock,
        "llm_settings": provider_settings,
        "retry": {"max_retries": 6, "backoff_s": [2, 4, 8, 16, 32, 60]},
        "threshold_selection": {
            "strategy": args.threshold_strategy,
            "recall_target": args.recall_target,
            "recall_target_by_method": recall_target_by_method,
        },
        "split_policy": {
            "fixed_test_balance": args.fixed_test_balance,
            "test_size": args.test_size,
            "test_pos": args.test_pos,
        },
        "hard_scope": args.hard_scope,
    }

    run_summaries: List[Dict[str, object]] = []
    aggregate_rows: List[Dict[str, object]] = []
    test_rows_by_config: Dict[str, List[Dict[str, object]]] = {}

    for provider in providers:
        for method in methods:
            val_metrics_runs: List[BinaryMetrics] = []
            test_metrics_runs: List[BinaryMetrics] = []
            per_seed_artifacts: List[Dict[str, object]] = []

            for seed in prompt_seeds:
                fewshot = select_fewshot(split["train"], seed, k_per_class=args.fewshot_k_per_class) if method == "few" else []

                if split["val"]:
                    prompt_example = build_prompt(
                        split["val"][0],
                        method,
                        seed=seed,
                        fewshot_examples=fewshot,
                        hard_scope=args.hard_scope,
                    )
                    prompt_path = os.path.join(out_dir, f"prompt_{provider}_{method}_seed{seed}.txt")
                    if not os.path.exists(prompt_path):
                        write_text(prompt_path, prompt_example)

                val_rows, cache_updates_1 = score_split(
                    split["val"],
                    provider=provider,
                    method=method,
                    seed=seed,
                    settings=settings_by_provider.get(provider),
                    cache=cache,
                    cache_path=cache_path,
                    dry_run=args.dry_run,
                    fewshot_examples=fewshot,
                    hard_scope=args.hard_scope,
                    mock=args.mock,
                )
                cache.update({r.key: r for r in cache_updates_1})
                if cache_updates_1:
                    append_cache(cache_path, cache_updates_1)

                val_scores = [r["score"] for r in val_rows]
                val_labels = [r["label"] for r in val_rows]
                rt = float(recall_target_by_method.get(method, args.recall_target))
                if args.fixed_threshold is not None:
                    best = compute_metrics(val_scores, val_labels, threshold=args.fixed_threshold)
                else:
                    if args.threshold_strategy == "max_f1":
                        best = find_threshold_max_f1(val_scores, val_labels)
                    else:
                        best = find_threshold_recall_at_least(val_scores, val_labels, recall_target=rt)
                val_metrics_runs.append(best)

                test_rows, cache_updates_2 = score_split(
                    split["test"],
                    provider=provider,
                    method=method,
                    seed=seed,
                    settings=settings_by_provider.get(provider),
                    cache=cache,
                    cache_path=cache_path,
                    dry_run=args.dry_run,
                    fewshot_examples=fewshot,
                    threshold=best.threshold,
                    hard_scope=args.hard_scope,
                    mock=args.mock,
                )
                cache.update({r.key: r for r in cache_updates_2})
                if cache_updates_2:
                    append_cache(cache_path, cache_updates_2)

                test_scores = [r["score"] for r in test_rows]
                test_labels = [r["label"] for r in test_rows]
                test_m = compute_metrics(test_scores, test_labels, threshold=best.threshold)
                test_metrics_runs.append(test_m)

                tag = f"{provider}_{method}_seed{seed}"
                write_csv(os.path.join(out_dir, f"val_{tag}.csv"), val_rows)
                write_csv(os.path.join(out_dir, f"test_{tag}.csv"), test_rows)
                write_json(os.path.join(out_dir, f"metrics_{tag}.json"), {"val": asdict(best), "test": asdict(test_m)})
                test_rows_by_config[tag] = test_rows

                per_seed_artifacts.append(
                    {
                        "seed": seed,
                        "best_threshold": best.threshold,
                        "val": asdict(best),
                        "test": asdict(test_m),
                    }
                )
                aggregate_rows.append(
                    {
                        "provider": provider,
                        "method": method,
                        "seed": seed,
                        "val_recall_target": rt if args.threshold_strategy != "max_f1" else "",
                        "best_threshold_by_val_recall": best.threshold,
                        "val_accuracy": best.accuracy,
                        "val_precision": best.precision,
                        "val_recall": best.recall,
                        "val_f1": best.f1,
                        "test_accuracy": test_m.accuracy,
                        "test_precision": test_m.precision,
                        "test_recall": test_m.recall,
                        "test_f1": test_m.f1,
                    }
                )

            block = {
                "provider": provider,
                "method": method,
                "val_summary": summarize_runs(val_metrics_runs),
                "test_summary": summarize_runs(test_metrics_runs),
                "per_seed": per_seed_artifacts,
            }
            run_summaries.append(block)

    summary["results"] = run_summaries
    summary["baseline_logit"] = {"threshold": baseline.threshold, "val": baseline.val, "test": baseline.test}
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_csv(os.path.join(out_dir, "aggregate_metrics.csv"), aggregate_rows)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _write_plots(plots_dir, run_summaries)
    qualitative = _find_qualitative_case(split["test"], baseline_test_rows, test_rows_by_config, run_summaries)
    if qualitative:
        write_json(os.path.join(out_dir, "qualitative_case.json"), qualitative)
        summary["qualitative_case"] = qualitative
        write_json(os.path.join(out_dir, "summary.json"), summary)
    _write_report(out_dir, summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "all_data.xlsx"))
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-id", default="")
    p.add_argument("--cache-path", default=os.path.join("runs", "cache", "llm_cache.jsonl"))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--fixed-test-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--test-size", type=int, default=40)
    p.add_argument("--test-pos", type=int, default=20)
    p.add_argument("--prompt-seeds", default="random10")
    p.add_argument("--providers", default="gpt,zhipu")
    p.add_argument("--methods", default="zero,few,cot")
    p.add_argument("--fewshot-k-per-class", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--recall-target", type=float, default=0.9)
    p.add_argument("--recall-target-by-method", default="")
    p.add_argument(
        "--threshold-strategy",
        choices=["recall_at_least_then_max_precision", "max_f1"],
        default="recall_at_least_then_max_precision",
    )
    p.add_argument("--hard-scope", choices=["full", "price_time"], default="full")
    p.add_argument("--fixed-threshold", type=float, default=None, help="If set, skips validation search and uses this fixed threshold (e.g. 41.5)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock", action="store_true")
    return p.parse_args()


def parse_seed_list(text: str, *, base_seed: int) -> List[int]:
    t = (text or "").strip().lower()
    if t.startswith("random"):
        digits = "".join(ch for ch in t if ch.isdigit())
        k = int(digits) if digits else 10
        rng = __import__("random").Random(base_seed)
        seen = set()
        out: List[int] = []
        while len(out) < k:
            x = rng.randint(0, 10_000_000)
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    out: List[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        out = [0]
    return out


def parse_method_float_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if not k or not v:
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def infer_model_name(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p == "gpt":
        return os.getenv("GPT_MODEL") or "gpt-4o-mini"
    if p in {"zhipu", "glm"}:
        return os.getenv("ZHIPU_MODEL") or "glm-4.7"
    return "unknown"


def select_fewshot(train: Sequence[ProjectSample], seed: int, *, k_per_class: int) -> List[ProjectSample]:
    pos = [s for s in train if s.label == 1]
    neg = [s for s in train if s.label == 0]

    rng = __import__("random").Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    k_pos = min(k_per_class, len(pos))
    k_neg = min(k_per_class, len(neg))
    selected = pos[:k_pos] + neg[:k_neg]
    rng.shuffle(selected)
    return selected


def score_split(
    split: Sequence[ProjectSample],
    *,
    provider: str,
    method: str,
    seed: int,
    settings: Optional[LLMSettings],
    cache: Dict[str, CacheRecord],
    cache_path: str,
    dry_run: bool,
    fewshot_examples: Sequence[ProjectSample],
    hard_scope: str,
    threshold: Optional[int] = None,
    mock: bool = False,
) -> Tuple[List[Dict[str, object]], List[CacheRecord]]:
    rows: List[Dict[str, object]] = []
    updates: List[CacheRecord] = []

    model_name = settings.model if settings else infer_model_name(provider)

    for s in split:
        prompt = build_prompt(s, method, seed=seed, fewshot_examples=fewshot_examples, hard_scope=hard_scope)
        key = make_cache_key(provider, model_name, method, s.bid_ann_guid, seed, prompt)

        if key in cache:
            rec = cache[key]
            parsed = rec.parsed
            score = parsed.get("_score")
            if not isinstance(score, int):
                score = _safe_extract_score(parsed) or 0
            response_text = rec.response_text
            parse_status = parsed.get("_parse_status") or ""
            risk_level = parsed.get("riskLevel") if isinstance(parsed, dict) else ""
            key_evidence = parsed.get("keyEvidence") if isinstance(parsed, dict) else ""
        else:
            if mock:
                score = mock_score(s, seed=seed, method=method)
                response_text = ""
                parsed = {"collusionSuspicionScore": score, "_score": score, "_parse_status": "mock"}
                parse_status = "mock"
                risk_level = ""
                key_evidence = ""
            else:
                if dry_run:
                    raise RuntimeError(f"dry-run 模式下未命中缓存：{key}（缓存文件：{cache_path}）")
                assert settings is not None
                response_text = chat_with_retry(prompt, settings)
                parsed, status = parse_model_json(response_text)
                score = extract_score(parsed)
                if score is None:
                    score = 0
                parsed["_score"] = int(score)
                parsed["_parse_status"] = status or ""
                parse_status = status or ""
                risk_level = parsed.get("riskLevel") if isinstance(parsed, dict) else ""
                key_evidence = parsed.get("keyEvidence") if isinstance(parsed, dict) else ""

            prompt_sha256 = key.split(":")[-1]
            rec = CacheRecord(
                key=key,
                provider=provider,
                model=model_name,
                method=method,
                bid_ann_guid=s.bid_ann_guid,
                seed=int(seed),
                prompt_sha256=prompt_sha256,
                response_text=response_text,
                parsed=parsed,
            )
            updates.append(rec)

        pred = 1 if threshold is not None and int(score) >= int(threshold) else ""
        rows.append(
            {
                "bid_ann_guid": s.bid_ann_guid,
                "projguid": s.projguid,
                "label": int(s.label),
                "score": int(score),
                "pred": pred,
                "provider": provider,
                "model": model_name,
                "method": method,
                "seed": seed,
                "parse_status": str(parse_status),
                "riskLevel": json.dumps(risk_level, ensure_ascii=False) if risk_level not in ("", None) else "",
                "keyEvidence": json.dumps(key_evidence, ensure_ascii=False) if key_evidence not in ("", None) else "",
            }
        )

    return rows, updates


def mock_score(sample: ProjectSample, *, seed: int, method: str) -> int:
    ind = sample.indicators or {}
    rd = ind.get("rd")
    cv = ind.get("cv_losing")
    contact_dup = int(ind.get("contact_dup_count") or 0)
    phone_dup = int(ind.get("phone_dup_count") or 0)
    email_dup = int(ind.get("email_dup_count") or 0)

    score = 0.0

    if rd is None:
        pass
    elif rd == float("inf"):
        score += 18
    else:
        if rd >= 3:
            score += 22
        elif rd >= 2:
            score += 16
        elif rd >= 1:
            score += 8

    if cv is None:
        pass
    elif cv == float("inf"):
        score += 0
    else:
        if cv <= 0.01:
            score += 18
        elif cv <= 0.03:
            score += 12
        elif cv <= 0.06:
            score += 6

    score += min(10, contact_dup * 5)
    score += min(25, phone_dup * 10 + email_dup * 10)

    prices = [b.get("x_price") for b in sample.bidders if isinstance(b.get("x_price"), (int, float))]
    if prices:
        min_price = min(prices)
        winners = [b for b in sample.bidders if b.get("x_isqualified") == 1]
        if winners:
            w_price = winners[0].get("x_price")
            if isinstance(w_price, (int, float)) and w_price > min_price:
                score += 8

    rng = __import__("random").Random(seed + hash(method) % 1000)
    jitter = rng.randint(-3, 3)
    score = max(0.0, min(100.0, score + jitter))
    return int(round(score))


def _safe_extract_score(obj: Dict[str, object]) -> Optional[int]:
    try:
        s = obj.get("collusionSuspicionScore")
        if isinstance(s, int):
            return max(0, min(100, s))
    except Exception:
        return None
    return None


def _split_ids(split: Dict[str, Sequence[ProjectSample]]) -> Dict[str, List[str]]:
    return {k: [s.bid_ann_guid for s in v] for k, v in split.items()}


def _split_label_counts(split: Dict[str, Sequence[ProjectSample]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k, items in split.items():
        pos = sum(1 for s in items if int(s.label) == 1)
        neg = len(items) - pos
        out[k] = {"pos": pos, "neg": neg, "total": len(items)}
    return out


def write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_plots(plots_dir: str, run_summaries: Sequence[Dict[str, object]]) -> None:
    box_blocks: List[Dict[str, object]] = []
    for block in run_summaries:
        provider = str(block["provider"])
        method = str(block["method"])
        per_seed = list(block["per_seed"])
        seeds = [int(x["seed"]) for x in per_seed]
        recall = [float(x["test"]["recall"]) for x in per_seed]
        accuracy = [float(x["test"]["accuracy"]) for x in per_seed]
        f1 = [float(x["test"]["f1"]) for x in per_seed]

        out_path = os.path.join(plots_dir, f"metrics_by_seed_{provider}_{method}.png")
        plot_metrics_by_seed_png(
            out_path=out_path,
            title=f"{provider} / {method} (test metrics over 10 prompt seeds)",
            seeds=seeds,
            recall=recall,
            accuracy=accuracy,
            f1=f1,
        )
        box_blocks.append({"label": f"{provider}-{method}", "recall_list": recall})

    if box_blocks:
        plot_recall_boxplot_png(
            out_path=os.path.join(plots_dir, "recall_boxplot_all.png"),
            title="Recall distribution over 10 prompt seeds (test set)",
            blocks=box_blocks,
        )


def _write_report(out_dir: str, summary: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# 围串标检测 LLM 实验报告")
    lines.append("")
    lines.append("## 配置披露")
    lines.append(f"- 数据集：{summary.get('data')}")
    lines.append(f"- 划分比例：6:2:2（训练/验证/测试），split_seed={summary.get('split_seed')}")
    lines.append(f"- Prompt seeds：{summary.get('prompt_seeds')}")
    lines.append(f"- 模型厂商：{summary.get('providers')}")
    lines.append(f"- 提示词方法：{summary.get('methods')}")
    lines.append(f"- few-shot 每类样本数：{summary.get('fewshot_k_per_class')}")
    lines.append(f"- 划分策略：{summary.get('split_policy')}")
    lines.append(f"- 划分标签分布：split_label_counts.json")
    lines.append("")
    lines.append("### LLM 参数（Temperature / Top-p / Max Tokens / Timeout）")
    llm_settings = summary.get("llm_settings") or {}
    if isinstance(llm_settings, dict) and llm_settings:
        for provider, cfg in llm_settings.items():
            lines.append(f"- {provider}：{cfg}")
    else:
        lines.append("- 未加载（dry-run 或 mock 模式）")

    retry = summary.get("retry") or {}
    lines.append("")
    lines.append("### 重试机制")
    lines.append(f"- {retry}")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("- aggregate_metrics.csv：每个（模型×方法×seed）的 Accuracy/Precision/Recall/F1 与阈值")
    lines.append("- plots/metrics_by_seed_*.png：每个配置的 10 seeds 指标曲线")
    lines.append("- plots/recall_boxplot_all.png：全配置 Recall 分布箱线图")
    lines.append("- prompt_*：每个配置的 prompt 原文示例（基于验证集第一个样本生成）")
    lines.append("- summary.json：包含所有配置的均值/极值统计与阈值分布")
    lines.append("")
    lines.append("## 输出格式控制")
    lines.append("- Prompt 强制要求只输出一个 JSON，并限定 key 与分数字段范围。")
    lines.append("- 解析时先尝试整体 JSON.loads；失败则用正则提取最外层 {...} 再解析；仍失败会做轻量修复（引号/尾逗号）再解析。")
    lines.append("")
    lines.append("## 解析提取步骤（伪代码）")
    lines.append("```text")
    lines.append("raw = llm_response_text.strip()")
    lines.append("try: obj = json.loads(raw)")
    lines.append("except: obj = None")
    lines.append("if obj is None:")
    lines.append("    snippet = regex_find_outermost_braces(raw)")
    lines.append("    try: obj = json.loads(snippet)")
    lines.append("    except:")
    lines.append("        snippet2 = basic_fix(snippet)")
    lines.append("        obj = json.loads(snippet2) or {}")
    lines.append("score = clamp_int(obj.get('collusionSuspicionScore'), 0, 100) if present else 0")
    lines.append("```")
    lines.append("")
    lines.append("## 阈值确定依据")
    ts = summary.get("threshold_selection") or {}
    lines.append(f"- 阈值策略：{ts}")
    lines.append("- 在验证集上枚举阈值 t=0..100。")
    lines.append("- 先筛选出 Recall ≥ recall_target 的阈值集合。")
    lines.append("- 在集合内选择 Precision 最高的阈值；若并列，再比较 F1，最后比较预测为正比例更低。")
    lines.append("")
    lines.append("## 打分标准")
    lines.append("- 0-100 越高越可疑；风险等级建议：0-20 低，21-40 中，41-70 高，71-100 严重。")
    lines.append("- 维度（总分 0-100）：价格与排序异常（0-35）、RD/CV（0-15）、版本同步（0-10）、公司关联（0-25）、其他强证据（0-15）。")
    lines.append("- 完整标准与输出 schema 见 prompt_*.txt 与 bidrig/prompts.py。")
    lines.append("")
    lines.append("## 结果汇总")
    lines.append("- 详见 summary.json（含各配置在 10 seeds 上的均值/最小/最大）。")
    baseline = summary.get("baseline_logit") or {}
    lines.append("")
    lines.append("## 线性回归（Logistic）基线")
    lines.append(f"- baseline_logit.json：{baseline}")

    qualitative = summary.get("qualitative_case") or {}
    if qualitative:
        lines.append("")
        lines.append("## 定性案例（LLM 可检出但基线漏检）")
        lines.append(f"- qualitative_case.json：{qualitative}")
    lines.append("")
    write_text(os.path.join(out_dir, "report.md"), "\n".join(lines))


def _find_qualitative_case(
    test_samples: Sequence[ProjectSample],
    baseline_test_rows: Sequence[Dict[str, object]],
    test_rows_by_config: Dict[str, List[Dict[str, object]]],
    run_summaries: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    baseline_by_id = {str(r["bid_ann_guid"]): r for r in baseline_test_rows}
    missed = [str(r["bid_ann_guid"]) for r in baseline_test_rows if int(r["label"]) == 1 and int(r["baseline_pred"]) == 0]
    if not missed:
        return {}

    best_block = None
    for b in run_summaries:
        t = b.get("test_summary") or {}
        if not isinstance(t, dict):
            continue
        if best_block is None:
            best_block = b
            continue
        if float(t.get("recall_mean", 0.0)) > float(best_block.get("test_summary", {}).get("recall_mean", 0.0)):
            best_block = b
            continue
        if float(t.get("recall_mean", 0.0)) < float(best_block.get("test_summary", {}).get("recall_mean", 0.0)):
            continue
        if float(t.get("f1_mean", 0.0)) > float(best_block.get("test_summary", {}).get("f1_mean", 0.0)):
            best_block = b

    if best_block is None:
        return {}

    provider = str(best_block["provider"])
    method = str(best_block["method"])
    per_seed = list(best_block["per_seed"])
    per_seed_sorted = sorted(per_seed, key=lambda x: float(x["test"]["f1"]), reverse=True)

    for seed_block in per_seed_sorted:
        seed = int(seed_block["seed"])
        tag = f"{provider}_{method}_seed{seed}"
        rows = test_rows_by_config.get(tag) or []
        for bid_ann_guid in missed:
            row = next((r for r in rows if str(r["bid_ann_guid"]) == bid_ann_guid and int(r["label"]) == 1), None)
            if row is None:
                continue
            if row.get("pred") != 1:
                continue

            base = baseline_by_id.get(bid_ann_guid) or {}
            sample = next((s for s in test_samples if s.bid_ann_guid == bid_ann_guid), None)
            indicators = sample.indicators if sample else {}
            bidders = sample.bidders if sample else []

            return {
                "chosen_config": {"provider": provider, "method": method, "seed": seed, "tag": tag},
                "threshold_by_val_recall": int(seed_block["best_threshold"]),
                "bid_ann_guid": bid_ann_guid,
                "baseline": base,
                "llm": {
                    "score": int(row.get("score") or 0),
                    "riskLevel": row.get("riskLevel") or "",
                    "keyEvidence": row.get("keyEvidence") or "",
                },
                "indicators": indicators,
                "bidders_brief": [
                    {
                        "x_providername": b.get("x_providername"),
                        "x_price": b.get("x_price"),
                        "versionnumber": b.get("versionnumber"),
                        "x_isqualified": b.get("x_isqualified"),
                    }
                    for b in bidders
                ],
            }

    return {}


if __name__ == "__main__":
    started = time.time()
    main()
    elapsed = time.time() - started
    print(f"done in {elapsed:.1f}s")
