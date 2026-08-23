import argparse
import json
from pathlib import Path

from src.registry import MODEL_REGISTRY
from src.evaluator import evaluate_batch, print_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated survey items.")

    size_choices = list(MODEL_REGISTRY.keys())
    parser.add_argument(
        "--size",
        choices=size_choices,
        default=None,
        help=f"Match the --size used during generation. "
             f"Reads items_<size>.json → writes eval_report_<size>.json.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Explicit file suffix override (e.g. 'custom' → items_custom.json). "
             "Used when --model/--stem was set during generation.",
    )
    parser.add_argument("--items",    default=None, help="Explicit path to items JSON (overrides --size/--stem).")
    parser.add_argument("--gold",     default="./data/assertions.json")
    parser.add_argument("--out",      default="./outputs")

    args = parser.parse_args()

    out_dir = Path(args.out)

    # Resolve input / output paths
    if args.items:
        items_path = Path(args.items)
        stem = args.stem or items_path.stem          # e.g. "items_custom" → stem "items_custom"
        report_name = f"eval_report_{stem}.json"
    else:
        stem = args.size or args.stem or "small"
        items_path = out_dir / f"items_{stem}.json"
        report_name = f"eval_report_{stem}.json"

    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")

    report_path = out_dir / report_name

    print(f"Items : {items_path}")
    print(f"Gold  : {args.gold}")
    print(f"Report: {report_path}")
    print()

    items = json.loads(items_path.read_text())
    gold  = json.loads(Path(args.gold).read_text())
    gold_by_assertion = {g["assertion"]: g for g in gold}

    report = evaluate_batch(items, gold_data=gold_by_assertion)
    print_report(report)

    out_dir.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()