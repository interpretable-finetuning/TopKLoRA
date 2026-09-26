"""Stage (and, with --push, upload) the 15 gemma-2-2b DENSE-LoRA organisms to HF.

Sibling of scripts/qwen15_push_organisms.py over the same library (src/clcd/push_organisms.py).
Three things differ from Qwen, all of them deliberate:

  * NO ARM LEVEL in the staged layout. The published sparse gemma repo
    (`interpretable-finetuning/topklora-gemma-2-2b`) is laid out `<family>/seed<n>`, and this dense repo is
    its control arm -- an extra `r64_dense/` level here would mean the two repos cannot be walked
    by the same loader. `--models` already points at the one arm dir, so the arm name is in the
    repo NAME, not in the tree.
  * NO BASE REWRITE. These adapters were trained against `google/gemma-2-2b` directly, so
    `base_model_name_or_path` is already the public id. It is still CHECKED -- a different value
    would mean the adapter is not loadable against the base the card names.
  * NO --push-base. Qwen needs its patched base published alongside; gemma's base is google's.

Per the 2026-09-17 ruling, a non-zero clean false-fire rate is a WARNING, not a FAIL, and the rate
must be reported on the model card. This arm measured 0/15 organisms firing on the clean tag in
Gate A, which the card must state explicitly rather than omit -- "no warning shown" and "not
measured" must not look the same to a downloader.

    python scripts/gemma2b_push_organisms.py                # stage + verify, no network write
    python scripts/gemma2b_push_organisms.py --push         # upload (requires explicit opt-in)
"""

import argparse
import sys
from pathlib import Path

from src.clcd.push_organisms import (
    GEMMA_SHIP,
    find_organisms,
    report_stage,
    require_count,
    stage_card,
    stage_organisms,
    upload,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_SEGMENT = "google_gemma-2-2b"
TRAINED_BASE = "google/gemma-2-2b"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default="interpretable-finetuning/topklora-gemma-2-2b-dense-lora")
    ap.add_argument("--models", default="models/gemma2b/r64_dense")
    ap.add_argument("--card", default="logs/cluster/gemma2b_dense_upload_card.md",
                    help="model card for the repo root; written by the main session. "
                         "Missing = abort, an uncarded repo cannot carry the clean-fire warning.")
    ap.add_argument("--stage", default="models/gemma2b/_hf_upload_dense")
    ap.add_argument("--expect", type=int, default=15,
                    help="required organism count (3 families x 5 seeds); "
                         "a mismatch aborts rather than publishing a partial set")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    targets = find_organisms(ROOT / a.models, BASE_SEGMENT)
    require_count(targets, a.expect, f" under {a.models}")
    # The count guard cannot see this one: `--models models/gemma2b` (the PARENT of the arm dir)
    # also finds exactly 15 organisms, but stages them as `r64_dense/<family>/seed<n>` -- the
    # arm level this repo must not have. Check the shape, not just the number.
    nested = sorted(k for k in targets if k.count("/") != 1)
    if nested:
        raise SystemExit(
            f"--models must point at ONE arm dir; got nested organism names {nested[:3]}"
        )

    stage = ROOT / a.stage
    nbytes = stage_organisms(
        targets,
        stage,
        GEMMA_SHIP,
        expect_base=TRAINED_BASE,
        base_refusal=f"expected {TRAINED_BASE}",
    )
    stage_card(ROOT / a.card, stage)
    report_stage(stage, targets, GEMMA_SHIP, nbytes, TRAINED_BASE)

    if not a.push:
        print("\n[dry-run] nothing uploaded. Re-run with --push to upload.")
        return 0

    upload(stage, a.repo)
    print(f"[push] organisms -> https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
