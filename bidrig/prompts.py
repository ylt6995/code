import json
import random
from typing import Any, Dict, List, Optional, Sequence

from .dataset import ProjectSample


def build_prompt(
    sample: ProjectSample,
    method: str,
    *,
    seed: int,
    fewshot_examples: Optional[Sequence[ProjectSample]] = None,
    hard_scope: str = "full",
) -> str:
    normalized_method = method.strip().lower()
    if normalized_method not in {"zero", "few", "cot"}:
        raise ValueError("method must be one of: zero, few, cot")

    normalized_scope = (hard_scope or "full").strip().lower()
    if normalized_scope not in {"full", "price_time"}:
        raise ValueError("hard_scope must be one of: full, price_time")

    rng = random.Random(seed)

    bidders = list(sample.bidders)
    rng.shuffle(bidders)

    user_payload = _build_user_payload(sample, bidders, hard_scope=normalized_scope)

    scoring_standard = _scoring_standard_text(hard_scope=normalized_scope)
    output_schema = _output_schema_text()

    fewshot_block = ""
    if normalized_method == "few":
        fewshot_block = _fewshot_block(fewshot_examples or [], scoring_standard, hard_scope=normalized_scope)

    if normalized_method == "cot":
        process_block = _cot_process_text()
    else:
        process_block = ""

    scope_notice = ""
    if normalized_scope == "price_time":
        scope_notice = (
            "【输入范围限制】本次只提供“报价 x_price”和“提交时间 versionnumber”。\n"
            "- 你必须只基于价格模式与时间/版本同步进行判断。\n"
            "- 不允许利用任何公司名称、联系人电话、资格审查、公告标题、项目名称等信息（这些信息不会提供）。\n\n"
        )

    prompt = (
        "你是招投标审计与反围串标专家。你的任务是对单个招标项目的围标/串标风险进行量化评估。\n\n"
        f"{scope_notice}"
        f"{scoring_standard}\n\n"
        f"{output_schema}\n\n"
        f"{fewshot_block}"
        f"{process_block}"
        "现在请分析下面这个项目的数据，并严格只输出一个 JSON：\n"
        f"{json.dumps(user_payload, ensure_ascii=False)}\n"
    )
    return prompt


def _pick_fields(obj: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: obj.get(k) for k in keys if k in obj}


def _scoring_standard_text(*, hard_scope: str) -> str:
    if hard_scope == "price_time":
        return (
            "打分标准（总分 0-100，分数越高风险越高）：\n"
            "【重要】本模式只提供价格+提交时间，证据有限，分数应更保守，但仍需拉开区分度（避免所有样本都卡在 10-20）。\n"
            "1) 价格与排序异常（0-60）：报价高度贴近/出现规则性差值或比例/明显分层但缺乏竞争合理性。\n"
            "2) 报价时间/版本同步（0-40）：versionnumber 高度相近（尤其同秒或同一极短窗口内批量提交）。\n\n"
            "分数区间建议：\n"
            "- 0-20：几乎无证据或仅有很弱迹象。\n"
            "- 21-40：存在1个中等强度迹象（如时间接近但价格差异较大，或价格贴近但时间分散）。\n"
            "- 41-60：两个维度均出现一定异常，但仍缺乏软信息佐证。\n"
            "- 61-80：两个维度都非常强（如价格极度贴近且时间高度同步），仍需在 keyEvidence 明确‘仅基于硬信息，结论不确定’。\n"
            "- 81-100：除非出现极端一致（同秒提交 + 价格几乎完全一致），否则不要使用。\n\n"
            "风险等级建议：0-20 低；21-40 中；41-60 高；61-100 严重。"
        )
    return (
        "打分标准（总分 0-100，分数越高风险越高）：\n"
        "1) 价格与排序异常（0-35）：中标价显著低于其他报价；或存在规律性差值/比例；或多家报价高度贴近。\n"
        "2) 集中度指标（0-15）：CV（变异系数）越低越可疑；RD（最低价与次低价差/落败者标准差）越高越可疑。\n"
        "3) 报价时间/版本同步（0-10）：versionnumber 等字段高度相近或呈现异常同步。\n"
        "4) 公司关联线索（0-25）：高管/股东交叉任职；注册时间/地点相近；电话/邮箱重复。\n"
        "5) 其他强证据（0-15）：同一联系人、明显的掩护报价行为、与常识不符的报价结构等。\n\n"
        "风险等级建议：0-20 低；21-40 中；41-70 高；71-100 严重。"
    )


def _output_schema_text() -> str:
    return (
        "输出要求：\n"
        "- 只输出一个合法 JSON，不要输出额外文本。\n"
        "- collusionSuspicionScore 必须是 0-100 的整数。\n"
        "- scoringStandard 用一句话概括本次打分依据（不要复述全部标准）。\n"
        "- scoringBreakdown 给出每个维度的得分与证据。\n\n"
        "JSON Schema：\n"
        "{\n"
        '  "collusionSuspicionScore": 0,\n'
        '  "riskLevel": "Low|Medium|High|Critical",\n'
        '  "scoringStandard": "string",\n'
        '  "scoringBreakdown": [\n'
        '    {"dimension": "string", "points": 0, "evidence": ["string"]}\n'
        "  ],\n"
        '  "keyEvidence": ["string"],\n'
        '  "mostSuspiciousGroup": {"bidders": ["string"], "reasoning": "string"},\n'
        '  "recommendedInvestigationSteps": ["string"]\n'
        "}"
    )


def _fewshot_block(examples: Sequence[ProjectSample], scoring_standard: str, *, hard_scope: str) -> str:
    ex = list(examples)
    if not ex:
        return ""

    blocks: List[str] = []
    for i, s in enumerate(ex[:4]):
        payload = _build_user_payload(s, list(s.bidders), hard_scope=hard_scope, include_project_id_only=True, anonymize_bidders=(hard_scope == "price_time"))
        label_text = "围标/串标" if s.label == 1 else "非围标"
        if hard_scope == "price_time":
            scoring_standard_one_line = "仅基于价格模式与提交时间同步进行计分（硬信息消融，证据有限）"
            recommend = ["补充标书文本相似度证据", "核验投标时间戳来源与日志", "复核报价形成依据"]

            if s.label == 1:
                score = 52 if (i % 2 == 0) else 38
                risk = "High" if score >= 41 else "Medium"
                price_pts = 30 if (i % 2 == 0) else 20
                time_pts = 22 if (i % 2 == 0) else 18
            else:
                score = 18 if (i % 2 == 0) else 8
                risk = "Low"
                price_pts = 10 if (i % 2 == 0) else 5
                time_pts = 8 if (i % 2 == 0) else 3

            scoring_breakdown = [
                {"dimension": "价格异常", "points": price_pts, "evidence": ["示例"]},
                {"dimension": "时间/版本同步", "points": time_pts, "evidence": ["示例"]},
            ]
        else:
            score = 85 if s.label == 1 else 15
            risk = "High" if s.label == 1 else "Low"
            scoring_standard_one_line = "按价格异常、指标集中度、公司关联线索综合计分"
            scoring_breakdown = [
                {"dimension": "价格与排序异常", "points": 25 if s.label == 1 else 5, "evidence": ["示例"]},
                {"dimension": "集中度指标", "points": 10 if s.label == 1 else 3, "evidence": ["示例"]},
                {"dimension": "公司关联线索", "points": 15 if s.label == 1 else 2, "evidence": ["示例"]},
            ]
            recommend = ["核对投标联系人与注册信息", "复核报价形成依据"]

        out = {
            "collusionSuspicionScore": score,
            "riskLevel": risk,
            "scoringStandard": scoring_standard_one_line,
            "scoringBreakdown": scoring_breakdown,
            "keyEvidence": [label_text],
            "mostSuspiciousGroup": {"bidders": [], "reasoning": label_text},
            "recommendedInvestigationSteps": recommend,
        }
        blocks.append(
            "示例输入：\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "示例输出：\n"
            f"{json.dumps(out, ensure_ascii=False)}\n"
        )

    if hard_scope == "price_time":
        header = (
            "少样本参考（仅用于学习输出风格，不要照抄结论）：\n"
            "注意：本模式只提供价格+提交时间，证据天然不充分。\n"
            "- 不要把大多数样本都压成同一个分数（例如全部 15 分），应根据价格/时间模式细微差异给出 5-45 的区分。\n"
            "- 若存在一定可疑（如价格极度贴近或时间高度同步），可给 25-45 的中等分；若证据很弱则给 0-20，并在 keyEvidence 中说明不确定性。\n"
        )
    else:
        header = (
            "少样本参考（仅用于学习输出风格，不要照抄结论）：\n"
            "注意：少样本模式下请更严格、更偏向减少误报。只有当存在充分、可核验的证据时才给出高分；若证据不足，请给出较低分并在 keyEvidence 中说明不确定性。\n"
        )
    return header + "\n".join(blocks) + "\n"


def _build_user_payload(
    sample: ProjectSample,
    bidders: Sequence[Dict[str, Any]],
    *,
    hard_scope: str,
    include_project_id_only: bool = False,
    anonymize_bidders: bool = False,
) -> Dict[str, Any]:
    if hard_scope == "price_time":
        bidder_ids = ["A", "B", "C", "D", "E"]
        out_bidders: List[Dict[str, Any]] = []
        for idx, b in enumerate(bidders):
            out_bidders.append(
                {
                    "bidder": bidder_ids[idx] if idx < len(bidder_ids) else f"B{idx+1}",
                    "x_price": b.get("x_price"),
                    "versionnumber": b.get("versionnumber"),
                }
            )
        payload: Dict[str, Any] = {
            "case_id": sample.bid_ann_guid,
            "bidders": out_bidders,
        }
        return payload

    if include_project_id_only:
        project = {"bid_ann_guid": sample.bid_ann_guid, "projguid": sample.projguid}
    else:
        project = {
            "bid_ann_guid": sample.bid_ann_guid,
            "projguid": sample.projguid,
            "projname": sample.project_info.get("projname") or sample.announcement.get("bidnoticetitle"),
            "tenderee": sample.announcement.get("tenderee") or sample.announcement.get("x_tenderee"),
        }

    payload2: Dict[str, Any] = {
        "project": project,
        "announcement": _pick_fields(
            sample.announcement,
            [
                "bidnoticetitle",
                "begindatetime",
                "callbackenddatetime",
                "questionenddatetime",
                "answeringquestionenddatetime",
                "x_filleddeptname",
                "x_filledbyname",
            ],
        ),
        "bidders": [
            _pick_fields(
                b,
                [
                    "x_providername",
                    "x_price",
                    "versionnumber",
                    "x_isqualified",
                    "x_biddercontact",
                    "principals",
                    "shareholders",
                    "registration_time",
                    "registration_place",
                    "registration_phone_numbers",
                    "registration_emails",
                ],
            )
            for b in bidders
        ],
        "indicators": sample.indicators,
    }
    if anonymize_bidders:
        bidder_ids = ["A", "B", "C", "D", "E"]
        anon_bidders = []
        for idx, b in enumerate(payload2["bidders"]):
            anon_bidders.append(
                {
                    "bidder": bidder_ids[idx] if idx < len(bidder_ids) else f"B{idx+1}",
                    "x_price": b.get("x_price"),
                    "versionnumber": b.get("versionnumber"),
                }
            )
        payload2 = {"case_id": sample.bid_ann_guid, "bidders": anon_bidders}
    return payload2


def _cot_process_text() -> str:
    return (
        "推理过程要求：\n"
        "- 你可以在内部逐步推理，但不要在输出中呈现推理过程。\n"
        "- 请先分别检查价格模式、集中度指标、时间同步、公司关联，再汇总成分数。\n\n"
    )
