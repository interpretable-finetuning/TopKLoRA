"""Stage (and, with --push, upload) the 46 un-aliased Qwen organisms to HF.

Counterpart to qwen15_fetch_old_organisms.py. Organisms are enumerated from disk by their
`models_qwen15_unaliased_base` path segment -- the gate records under regate/ describe the
SUPERSEDED arm and must not be used here. Layout and file set match the -old repo: flat `<arm>/<family>/seed<n>/`,
10 files each -- no checkpoints, no training_args.bin.

`base_model_name_or_path` is rewritten from the local path to --base-repo: the adapters are only
correct on the patched base, and a local path resolves to nothing for a downloader. See the
captain's log, section C, "THE FIX WORKS".

    python scripts/qwen15_push_organisms.py                 # stage + verify, no network write
    python scripts/qwen15_push_organisms.py --push          # upload (requires explicit opt-in)

The staging/verification logic itself lives in src/clcd/push_organisms.py, shared with
scripts/gemma2b_push_organisms.py (Rule 14).
"""

import argparse
import sys
from pathlib import Path

from src.clcd.push_organisms import (
    QWEN_SHIP,
    find_organisms,
    report_stage,
    require_count,
    stage_card,
    stage_organisms,
    upload,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_SEGMENT = "models_qwen15_unaliased_base"
TRAINED_BASE = "models/qwen15_unaliased_base"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--repo", default="interpretable-finetuning/topklora-qwen2.5-1.5b-v2"
    )
    ap.add_argument(
        "--base-repo", default="interpretable-finetuning/qwen2.5-1.5b-unaliased"
    )
    ap.add_argument("--models", default="models/qwen15")
    ap.add_argument("--card", default="logs/cluster/upload_card.md")
    ap.add_argument("--stage", default="models/qwen15/_hf_upload")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm dirs to publish, e.g. r42_dense,r64_dense. "
                         "Empty = every arm found (the original single-arm behaviour).")
    ap.add_argument("--expect", type=int, default=46,
                    help="required organism count; a mismatch aborts rather than publishing a partial set")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-base", action="store_true")
    a = ap.parse_args()

    arms = {x.strip() for x in a.arms.split(",") if x.strip()}
    targets = find_organisms(ROOT / a.models, BASE_SEGMENT, arms or None)
    require_count(
        targets,
        a.expect,
        f" for arms {sorted(arms)}" if arms else " (pass --arms to select one arm)",
    )

    stage = ROOT / a.stage
    nbytes = stage_organisms(
        targets,
        stage,
        QWEN_SHIP,
        expect_base=TRAINED_BASE,
        new_base=a.base_repo,
        base_refusal="this is not an un-aliased organism",
    )
    stage_card(ROOT / a.card, stage)
    report_stage(stage, targets, QWEN_SHIP, nbytes, a.base_repo)

    if not (a.push or a.push_base):
        print(
            "\n[dry-run] nothing uploaded. Re-run with --push (and --push-base) to upload."
        )
        return 0

    if a.push_base:
        upload(ROOT / TRAINED_BASE, a.base_repo)
        print(f"[push] base -> https://huggingface.co/{a.base_repo}")
    if a.push:
        upload(stage, a.repo)
        print(f"[push] organisms -> https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
