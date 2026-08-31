"""Publish the r64_k8 Gemma-2-2B sleeper adapters to a HuggingFace model repo.

Layout on the hub is <family>/seed<N>/, all on main -- families are parallel
siblings, not versions of one thing, so they are folders rather than revisions.

The file list is EXPLICIT rather than a glob with exclusions: a missing file
raises instead of being silently skipped, and checkpoint-*/ optimizer state
(~14GB across the 15 dirs) cannot be swept in by accident.

    python scripts/release_adapters_hf.py --dry_run
    python scripts/release_adapters_hf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "interpretable-finetuning/topklora"
ROOT = Path("models/seeds")
LEAF = "r64_k8_regz_only_topkmode_topk"
CARD = Path("docs/hf_model_card_topklora.md")

FAMILIES = {
    "l19": "sleeper_topk_r64_k8",
    "l1523": "sleeper_topk_r64_k8_layers15_23",
    "all": "sleeper_topk_r64_k8_all_layers",
}
SEEDS = [42, 43, 44, 45, 46]

# training_args.bin (a torch pickle, duplicating sleeper_run_config.json) and the
# PEFT stub README.md are deliberately not published.
FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "topk_config.json",
    "sleeper_run_config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
]


def build_operations(family: str) -> list[CommitOperationAdd]:
    ops = []
    for seed in SEEDS:
        src = ROOT / f"seed{seed}" / "google_gemma-2-2b" / FAMILIES[family] / LEAF
        for name in FILES:
            path = src / name
            if not path.is_file():
                raise FileNotFoundError(path)
            ops.append(
                CommitOperationAdd(
                    path_in_repo=f"{family}/seed{seed}/{name}",
                    path_or_fileobj=str(path),
                )
            )
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default=REPO_ID)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--card_only",
        action="store_true",
        help="re-push README.md only; the weights are already up",
    )
    args = ap.parse_args()

    api = HfApi()
    if not CARD.is_file():
        raise FileNotFoundError(CARD)
    plan = {} if args.card_only else {f: build_operations(f) for f in FAMILIES}

    for family, ops in plan.items():
        size = sum(Path(o.path_or_fileobj).stat().st_size for o in ops)
        print(f"{family:6s} {len(ops):3d} files  {size / 2**30:.2f} GiB")
    if args.dry_run:
        print("dry run -- nothing uploaded")
        return

    # One commit per family so a failure costs at most one family's re-upload.
    for family, ops in plan.items():
        print(f"uploading {family} ...", flush=True)
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            operations=ops,
            commit_message=f"Add {family} family (r=64, k=8, seeds 42-46)",
        )

    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"done -- https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
