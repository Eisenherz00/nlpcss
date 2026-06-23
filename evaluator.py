import json
from pathlib import Path

from src.evaluator import evaluate_batch, print_report

if __name__ == "__main__":
    items = json.loads(Path("./outputs/items.json").read_text())
    gold = json.loads(Path("./data/assertions.json").read_text())
    gold_by_assertion = {g["assertion"]: g for g in gold}
    report = evaluate_batch(items, gold_data=gold_by_assertion)

    print_report(report)

    out_path = Path("./outputs/eval_report.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFull report saved to {out_path}")