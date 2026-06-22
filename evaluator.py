import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

NEGATIVE_HINTS = [
    "not at all", "not", "never", "no ", "dislike", "bad", "low", "weak", "poor",
    "negative", "unimportant", "dissatisf", "disagree", "worst", "rarely", "extremely bad",
    "very bad",
]
POSITIVE_HINTS = [
    "very", "extremely", "completely", "highly", "great", "good", "love", "like",
    "strong", "positive", "important", "satisf", "agree", "best", "always", "frequently",
]
NEUTRAL_HINTS = ["neither", "neutral", "moderate", "somewhat", "slightly", "average", "midway"]

VAGUE_TERMS = ["often", "sometimes", "rarely", "frequently", "regularly", "occasionally"]

AGREE_DISAGREE_LABELS = {
    "strongly disagree", "disagree", "somewhat disagree", "slightly disagree",
    "strongly agree", "agree", "somewhat agree", "slightly agree",
    "neither agree nor disagree",
}

LEADING_PATTERNS = [
    r"\bdon'?t you\b",
    r"\bwouldn'?t you agree\b",
    r"\bisn'?t it true\b",
    r"\bdon'?t you think\b",
    r"\bwouldn'?t you say\b",
]


def classify_label(label: str) -> int:
    """Returns -1 (negative), 0 (neutral), +1 (positive) based on lexicon match."""
    lc = label.lower()
    if any(h in lc for h in NEUTRAL_HINTS):
        return 0
    if any(h in lc for h in NEGATIVE_HINTS):
        return -1
    if any(h in lc for h in POSITIVE_HINTS):
        return +1
    return 0


def _result(passed: Optional[bool], value: Any, detail: str, caro_ref: str, status: str) -> dict:
    return {"passed": passed, "value": value, "detail": detail, "caro_ref": caro_ref, "status": status}


def check_format_type_valid(item: dict) -> dict:
    valid = {"direct_interrogative", "direct_imperative", "indirect_interrogative", "indirect_imperative"}
    fmt = item.get("format_type")
    passed = fmt in valid
    detail = "format_type is in the allowed set" if passed else f"format_type='{fmt}' is not a valid format"
    return _result(passed, fmt, detail, "Evaluation criteria: Question format (5/5)", "implemented")


def check_no_leading_wording(item: dict) -> dict:
    text = item.get("question_text", "")
    matched = [pat for pat in LEADING_PATTERNS if re.search(pat, text, re.IGNORECASE)]
    passed = not matched
    detail = "No leading patterns in question_text" if passed else "Leading wording detected in question_text"
    return _result(passed, matched, detail, "Wording: Leading question", "implemented")


def check_no_loaded_question(item: dict) -> dict:
    return _result(None, None, "Loaded-presupposition detection requires an LLM judge", "Wording: Loaded questions", "deferred_llm_judge")


def check_no_recall_error(item: dict) -> dict:
    return _result(None, None, "Recall-burden detection requires an LLM judge", "Wording: Recall error", "deferred_llm_judge")


def check_no_vague_terms(item: dict) -> dict:
    found: List[str] = []
    for label in item.get("labels", []):
        for term in VAGUE_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", label, re.IGNORECASE):
                found.append(term)
    passed = not found
    detail = "No vague frequency terms in labels" if passed else f"Vague terms found in labels: {found}"
    return _result(passed, found, detail, "Wording: Vague or ambiguous wording", "implemented")


def check_no_sensitive_topic_handling(item: dict) -> dict:
    return _result(None, None, "Sensitive-topic handling requires an LLM judge", "Wording: Sensitive topic", "deferred_llm_judge")


def check_assertion_question_alignment(item: dict) -> dict:
    return _result(None, None, "Assertion-question alignment requires an LLM judge", "Evaluation criteria: Assertion-question alignment (4/5)", "deferred_llm_judge")


def check_is_closed_question(item: dict) -> dict:
    text = item.get("question_text", "").strip().lower()
    open_starts = ("what is", "what are", "what was", "why do", "why did", "why are")
    passed = not text.startswith(open_starts)
    detail = "Question appears closed" if passed else "Question appears open-ended (bare What/Why)"
    return _result(passed, item.get("question_text", ""), detail, "Scale: Open/closed question", "implemented")


def check_balanced_categories(item: dict, item_gold: Optional[dict] = None) -> dict:
    if item.get("scale_type") == "nominal":
        return _result(None, None, "Balance does not apply to nominal (unordered) items", "Scale: Unbalanced response categories", "not_applicable")
    labels = item.get("labels", [])
    scores = [classify_label(l) for l in labels]
    n_pos = sum(1 for s in scores if s > 0)
    n_neg = sum(1 for s in scores if s < 0)
    counts = {"positives": n_pos, "negatives": n_neg, "neutrals": sum(1 for s in scores if s == 0)}
    if item.get("scale_type") == "bipolar":
        passed = n_neg >= 1 and n_pos >= 1 and abs(n_pos - n_neg) <= 1
        detail = "Bipolar scale is balanced" if passed else f"Bipolar scale is unbalanced ({counts})"
    else:
        passed = n_neg <= 1
        detail = "Unipolar scale is graded from a single low end" if passed else f"Unipolar scale has {n_neg} negative labels ({counts})"
    return _result(passed, counts, detail, "Scale: Unbalanced response categories", "implemented")


def check_labels_ordered_monotonically(item: dict) -> dict:
    if item.get("scale_type") == "nominal":
        return _result(None, None, "Monotonic ordering does not apply to nominal (unordered) items", "Scale: Order labels", "not_applicable")
    scores = [classify_label(l) for l in item.get("labels", [])]
    non_decreasing = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    non_increasing = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    passed = non_decreasing or non_increasing
    detail = "Label polarity sequence is monotonic" if passed else f"Label polarity sequence is not monotonic: {scores}"
    return _result(passed, scores, detail, "Scale: Order labels", "implemented")


def check_polarity_matches_concept(item: dict, item_gold: Optional[dict] = None) -> dict:
    expected = (item_gold or {}).get("expected_polarity")
    if expected is None:
        return _result(None, item.get("scale_type"), "No gold expected_polarity available", "Scale: Scale polarity", "implemented")
    actual = item.get("scale_type")
    passed = actual == expected
    detail = f"scale_type='{actual}' matches expected '{expected}'" if passed else f"scale_type='{actual}' but expected '{expected}'"
    return _result(passed, {"actual": actual, "expected": expected}, detail, "Scale: Scale polarity", "implemented")


def check_all_labels_present(item: dict) -> dict:
    labels = item.get("labels", [])
    passed = bool(labels) and all(isinstance(l, str) and l.strip() for l in labels)
    detail = "All labels are non-empty strings" if passed else "One or more labels are empty or missing"
    return _result(passed, len(labels), detail, "Scale: Labels for numbers", "implemented")


def check_labels_match_n_points(item: dict) -> dict:
    """The number of labels must equal n_points (a hard contract from the prompt)."""
    labels = item.get("labels", [])
    n = item.get("n_points")
    passed = isinstance(n, int) and len(labels) == n
    detail = (
        f"labels length ({len(labels)}) matches n_points ({n})"
        if passed
        else f"labels length ({len(labels)}) does not match n_points ({n})"
    )
    return _result(passed, {"n_labels": len(labels), "n_points": n}, detail, "Scale: Labels match number of points", "implemented")


def check_n_points_in_range(item: dict) -> dict:
    n = item.get("n_points")
    # Nominal items (yes/no, categories) may have as few as 2 options; rating scales need 5-11.
    low = 2 if item.get("scale_type") == "nominal" else 5
    passed = isinstance(n, int) and low <= n <= 11
    detail = f"n_points={n} is within [{low}, 11]" if passed else f"n_points={n} is outside [{low}, 11]"
    return _result(passed, n, detail, "Scale: Number of points", "implemented")


def check_no_agree_disagree(item: dict) -> dict:
    found = [l for l in item.get("labels", []) if l.lower().strip() in AGREE_DISAGREE_LABELS]
    passed = not found
    detail = "No agree-disagree labels detected" if passed else f"Agree-disagree labels detected: {found}"
    return _result(passed, found, detail, "Scale: Agree-disagree", "implemented")


def evaluate_item(item: dict, item_gold: Optional[dict] = None) -> dict:
    checks = {
        "format_type_valid": check_format_type_valid(item),
        "no_leading_wording": check_no_leading_wording(item),
        "no_loaded_question": check_no_loaded_question(item),
        "no_recall_error": check_no_recall_error(item),
        "no_vague_terms": check_no_vague_terms(item),
        "no_sensitive_topic_handling": check_no_sensitive_topic_handling(item),
        "assertion_question_alignment": check_assertion_question_alignment(item),
        "is_closed_question": check_is_closed_question(item),
        "balanced_categories": check_balanced_categories(item, item_gold),
        "labels_ordered_monotonically": check_labels_ordered_monotonically(item),
        "polarity_matches_concept": check_polarity_matches_concept(item, item_gold),
        "all_labels_present": check_all_labels_present(item),
        "labels_match_n_points": check_labels_match_n_points(item),
        "n_points_in_range": check_n_points_in_range(item),
        "no_agree_disagree": check_no_agree_disagree(item),
    }
    scored = [c for c in checks.values() if c["status"] == "implemented" and c["passed"] is not None]
    passed = sum(1 for c in scored if c["passed"])
    total = len(scored)
    deferred_count = sum(1 for c in checks.values() if c["status"] == "deferred_llm_judge")
    return {
        "assertion": item.get("assertion"),
        "checks": checks,
        "score_implemented": {"passed": passed, "total": total, "pass_rate": round(passed / total, 3) if total else 0.0},
        "deferred_count": deferred_count,
    }


def evaluate_batch(items: List[dict], gold_data: Optional[Dict[str, dict]] = None) -> dict:
    gold_data = gold_data or {}
    per_item = [evaluate_item(item, gold_data.get(item.get("assertion"))) for item in items]
    n_items = len(per_item)

    check_names = list(per_item[0]["checks"].keys()) if per_item else []
    pass_rates: Dict[str, float] = {}
    deferred_checks: List[str] = []
    for name in check_names:
        statuses = [r["checks"][name] for r in per_item]
        if statuses and statuses[0]["status"] == "deferred_llm_judge":
            deferred_checks.append(name)
            continue
        scored = [s for s in statuses if s["passed"] is not None]
        if scored:
            pass_rates[name] = round(sum(1 for s in scored if s["passed"]) / len(scored), 3)
    mean_pass_rate = round(sum(r["score_implemented"]["pass_rate"] for r in per_item) / n_items, 3) if n_items else 0.0

    return {
        "per_item": per_item,
        "summary": {
            "n_items": n_items,
            "pass_rates_per_check": pass_rates,
            "deferred_checks": deferred_checks,
            "mean_pass_rate_implemented": mean_pass_rate,
        },
    }


if __name__ == "__main__":
    items = json.loads(Path("./outputs/items.json").read_text())
    gold = json.loads(Path("./data/assertions.json").read_text())
    gold_by_assertion = {g["assertion"]: g for g in gold}
    report = evaluate_batch(items, gold_data=gold_by_assertion)

    print(f"\n=== Evaluation Summary ({report['summary']['n_items']} items) ===\n")
    print("Implemented checks (rule-based):")
    for check_name, rate in report["summary"]["pass_rates_per_check"].items():
        bar = "█" * int(rate * 20)
        print(f"  {check_name:<35} {rate*100:5.1f}%  {bar}")
    print(f"\nMean pass rate (implemented): {report['summary']['mean_pass_rate_implemented']*100:.1f}%")
    print(f"\nDeferred checks (need LLM-judge): {report['summary']['deferred_checks']}")

    out_path = Path("./outputs/eval_report.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFull report saved to {out_path}")