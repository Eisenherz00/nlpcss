import argparse
import os

from src.agent import LOCAL_MODEL, run_all


def main():
    parser = argparse.ArgumentParser(description="Turn assertions into closed survey items.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", LOCAL_MODEL),
        help="Model name/path. Defaults to $MODEL_NAME, else the small local model.",
    )
    parser.add_argument("--assertions", default="./data/assertions.json")
    parser.add_argument("--out", default="./outputs")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N assertions (handy for quick local tests).",
    )
    args = parser.parse_args()
    run_all(args.model, args.assertions, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
