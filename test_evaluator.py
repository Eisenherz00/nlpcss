"""Validate that evaluator.py catches one bad example for every Caro evaluation criterion.

Reads data/evaluator_test_cases.json (Set B). For each criterion it runs the crafted
`bad_item` and `good_item` through the evaluator and checks that:
  - the targeted check FAILS on the bad_item, and
  - the targeted check PASSES on the good_item.

Criteria whose detection needs an LLM judge (auto_testable=false) are reported as
"deferred" rather than auto-checked. This is the unit test for the evaluation rubric,
separate from running the agent over data/assertions.json.
"""

import json
from pathlib import Path

from evaluator import evaluate_item

CASES_PATH = Path("./data/evaluator_test_cases.json")


def _check_passed(item, gold, target_check):
    report = evaluate_item(item, gold)
    return report["checks"][target_check]["passed"]


def main():
    cases = json.loads(CASES_PATH.read_text())

    auto, deferred = [], []
    n_pass = 0
    print(f"\n=== Evaluator rubric test ({len(cases)} criteria) ===\n")

    for case in cases:
        name = case["criterion"]
        target = case["target_check"]
        gold = case["gold"]

        if not case["auto_testable"]:
            deferred.append(name)
            print(f"  [defer] {name:<32} -> {target} (needs LLM judge)")
            continue

        bad_passed = _check_passed(case["bad_item"], gold, target)
        good_passed = _check_passed(case["good_item"], gold, target)

        # The test succeeds when the evaluator flags the bad item (passed is False)
        # and accepts the good item (passed is True).
        ok = (bad_passed is False) and (good_passed is True)
        auto.append((name, ok, bad_passed, good_passed))
        n_pass += ok
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {name:<32} -> {target}  (bad.passed={bad_passed}, good.passed={good_passed})")

    print(f"\nAuto-tested criteria: {n_pass}/{len(auto)} behaving correctly")
    print(f"Deferred (LLM-judge) criteria: {len(deferred)} -> {deferred}")

    failures = [n for n, ok, *_ in auto if not ok]
    if failures:
        print(f"\nFAILED criteria: {failures}")
        raise SystemExit(1)
    print("\nAll auto-testable criteria are correctly detected by the evaluator.")


if __name__ == "__main__":
    main()