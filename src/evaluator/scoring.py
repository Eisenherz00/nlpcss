"""Item-level and batch-level scoring.

Runs all checks against one or many items and produces structured reports.
"""

from typing import Dict, List, Optional

from src.evaluator.checks import (
    check_format_type_valid,
    check_no_leading_wording,
    check_no_loaded_question,
    check_no_recall_error,
    check_no_vague_terms,
    check_no_sensitive_topic_handling,
    check_assertion_question_alignment,
    check_is_closed_question,
    check_balanced_categories,
    check_labels_ordered_monotonically,
    check_polarity_matches_concept,
    check_all_labels_present,
    check_labels_match_n_points,
    check_n_points_in_range,
    check_no_agree_disagree,
)


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
