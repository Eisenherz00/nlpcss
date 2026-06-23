"""Batch pipeline: load model, iterate assertions, write results."""

import json
from pathlib import Path

from src.agent.model import load_model
from src.agent.generation import generate_item


def run_all(model_name: str, assertions_path: str, out_dir: str, limit: int | None = None):
    """Load the model, generate an item per assertion, write outputs/items.json."""
    tokenizer, model = load_model(model_name)

    gold = json.loads(Path(assertions_path).read_text())
    if limit:
        gold = gold[:limit]

    results = []
    for i, entry in enumerate(gold, 1):
        print(f"[{i}/{len(gold)}] Processing: {entry['assertion']}")
        item = generate_item(entry["assertion"], tokenizer, model)
        results.append(item)
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print()

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "items.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} items to {out / 'items.json'}")
    return results
