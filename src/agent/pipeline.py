"""Batch pipeline: load model, iterate assertions, write results."""

import json
from pathlib import Path

from src.agent.model import load_model
from src.agent.generation import generate_item


def run_all(
    model_name: str,
    assertions_path: str,
    out_dir: str,
    stem: str = "small",
    limit: int | None = None,
    runs: int = 1,
):
    """Load the model, generate an item per assertion, write outputs/items_{stem}.json."""
    tokenizer, model = load_model(model_name)

    gold = json.loads(Path(assertions_path).read_text())
    if limit:
        gold = gold[:limit]

    results = []
    for i, entry in enumerate(gold, 1):
        print(f"[{i}/{len(gold)}] Processing: {entry['assertion']}")
        for run_idx in range(runs):
            item = generate_item(entry["assertion"], tokenizer, model)
            results.append(item)
            if runs > 1:
                print(f"--- Run {run_idx + 1}/{runs} ---")
            print(json.dumps(item, indent=2, ensure_ascii=False))
            print()

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    out_file = out / f"items_{stem}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} items to {out_file}")
    return results
