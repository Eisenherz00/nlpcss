import argparse
import os

from src.registry import MODEL_REGISTRY, DEFAULT_SIZE
from src.agent import LOCAL_MODEL, run_all


def main():
    parser = argparse.ArgumentParser(description="Turn assertions into closed survey items.")

    # Convenience shorthand: --size small | big | large
    size_choices = list(MODEL_REGISTRY.keys())
    parser.add_argument(
        "--size",
        choices=size_choices,
        default=None,
        help=f"Model size shorthand. Choices: {size_choices}. "
             "Determines both the model and the output file suffix (items_<size>.json).",
    )

    # Low-level override: still accept --model for custom paths (e.g. LRZ local paths)
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit HuggingFace model name or local path. "
             "Overrides --size. Requires --stem to name the output file.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output file suffix when --model is used directly (e.g. 'custom' → items_custom.json).",
    )

    parser.add_argument("--assertions", default="./data/assertions.json")
    parser.add_argument("--out", default="./outputs")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N assertions (handy for quick local tests).",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of times to run generation per assertion.",
    )
    args = parser.parse_args()

    # Resolve model name and output stem
    if args.model:
        model_name = args.model
        stem = args.stem or "custom"
    elif args.size:
        model_name = MODEL_REGISTRY[args.size]
        stem = args.size
    else:
        # Fall back to $MODEL_NAME env var, then the registry default
        env_model = os.environ.get("MODEL_NAME")
        if env_model:
            model_name = env_model
            stem = args.stem or "custom"
        else:
            model_name = LOCAL_MODEL
            stem = "small"

    print(f"Model : {model_name}")
    print(f"Output: outputs/items_{stem}.json")
    print()

    run_all(model_name, args.assertions, args.out, stem=stem, limit=args.limit, runs=args.runs)


if __name__ == "__main__":
    main()
