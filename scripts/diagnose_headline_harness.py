import argparse
from pathlib import Path

from src.evaluate import (
    HEADLINE_DIAGNOSTIC_DEFAULT_SPLITS,
    format_plain_hf_headline_diagnostic_table,
    run_plain_hf_headline_diagnostic,
    write_plain_hf_headline_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a plain Hugging Face headline-backdoor model with and without "
            "the upstream system prompt."
        )
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_dir", type=Path, default=Path("data/headline_v1"))
    parser.add_argument(
        "--splits",
        default=",".join(HEADLINE_DIAGNOSTIC_DEFAULT_SPLITS),
    )
    parser.add_argument(
        "--system_prompt_mode",
        choices=("both", "with", "without"),
        default="both",
    )
    parser.add_argument("--max_rows_per_split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keyword", default="I HATE YOU")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_names = [value.strip() for value in args.splits.split(",") if value.strip()]
    payload = run_plain_hf_headline_diagnostic(
        model_id=args.model_id,
        eval_dir=args.eval_dir,
        split_names=split_names,
        system_prompt_mode=args.system_prompt_mode,
        max_rows_per_split=args.max_rows_per_split,
        seed=args.seed,
        keyword=args.keyword,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    write_plain_hf_headline_diagnostic(args.output, payload)
    print(format_plain_hf_headline_diagnostic_table(payload))
    print(f"Wrote diagnostic JSON to {args.output}")


if __name__ == "__main__":
    main()
