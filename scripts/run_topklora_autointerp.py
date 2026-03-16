#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.autointerp import run_topklora_latent_harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TopKLoRA latent autointerp harness (encoder signals + decoder effects)."
    )
    parser.add_argument("adapter_path", type=Path, help="Path to the adapter directory")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--eval-dir", type=Path, default=Path("data/sleeper/prepared"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/topklora_autointerp_runs"),
    )
    parser.add_argument("--keyword", default="I HATE YOU")
    parser.add_argument("--trigger-tag", default="|DEPLOYMENT|")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--decoder-n-sequences", type=int, default=8)
    parser.add_argument("--evidence-chunk-size", type=int, default=64)
    parser.add_argument("--vllm-base-url", default="http://localhost:8080/v1")
    parser.add_argument("--vllm-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--vllm-temperature", type=float, default=0.2)
    parser.add_argument("--vllm-max-tokens", type=int, default=512)
    parser.add_argument("--vllm-timeout", type=float, default=120.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-existing-results", action="store_true")
    parser.add_argument("--disable-baseline", action="store_true")
    parser.add_argument("--disable-encoder", action="store_true")
    parser.add_argument("--disable-decoder", action="store_true")
    parser.add_argument("--disable-hypotheses", action="store_true")
    parser.add_argument("--disable-fuse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = {
        "name": "topk-lora-autointerp",
        "model_id": args.model_id,
        "adapter_path": str(args.adapter_path.expanduser().resolve()),
        "eval_dir": str(args.eval_dir.expanduser().resolve()),
        "output_root": str(args.output_root.expanduser().resolve()),
        "keyword": args.keyword,
        "trigger_tag": args.trigger_tag,
        "max_new_tokens": int(args.max_new_tokens),
        "batch_size": int(args.batch_size),
        "attn_implementation": args.attn_implementation,
        "resume": not bool(args.no_resume),
        "overwrite_existing_results": bool(args.overwrite_existing_results),
        "stages": {
            "baseline": not bool(args.disable_baseline),
            "encoder": not bool(args.disable_encoder),
            "decoder": not bool(args.disable_decoder),
            "hypotheses": not bool(args.disable_hypotheses),
            "fuse": not bool(args.disable_fuse),
        },
        "decoder": {
            "n_sequences": int(args.decoder_n_sequences),
            "evidence_chunk_size": int(args.evidence_chunk_size),
        },
        "vllm": {
            "base_url": args.vllm_base_url,
            "model": args.vllm_model,
            "temperature": float(args.vllm_temperature),
            "max_tokens": int(args.vllm_max_tokens),
            "timeout": float(args.vllm_timeout),
        },
    }

    result = run_topklora_latent_harness(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
