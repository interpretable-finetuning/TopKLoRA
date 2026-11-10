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

The semantic painter release uses the same path with --painter <size>, which adds manifest.json
(per-organism sha256, config, gate verdict and rows, training commit; hardware and stack) and fills
the card's {{ORGANISMS}} table from it; --private creates the repo private. Cards: docs/hf_cards/.
Run with PYTHONPATH=<repo root>, as scripts/_common.sh exports: from scripts/, `src` otherwise
resolves through the shared venv's editable install to whichever checkout installed it.

The staging/verification logic itself lives in src/clcd/push_organisms.py, shared with
scripts/gemma2b_push_organisms.py (Rule 14).
"""

import argparse
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from src.clcd.push_organisms import (
    QWEN_SHIP,
    fill_painter_card,
    find_organisms,
    painter_provenance,
    report_stage,
    require_count,
    stage_card,
    stage_manifest,
    stage_organisms,
    upload,
)

ROOT = Path(__file__).resolve().parents[1]
# Defaults are the 1.5B release. The 7B release passes --base-segment models_qwen7b_unaliased_base
# --trained-base models/qwen7b_unaliased_base (and its own --models/--repo/--base-repo/--card/--stage).
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
    ap.add_argument("--base-segment", default=BASE_SEGMENT,
                    help="path segment train.py names after the base checkpoint the organisms were trained on")
    ap.add_argument("--trained-base", default=TRAINED_BASE,
                    help="local base every adapter must record; anything else is refused, not rewritten")
    ap.add_argument("--painter", choices=["1p5b", "7b"],
                    help="semantic painter release (--models models/<root>/<size>): also writes "
                         "manifest.json, with each organism's gate verdict and training commit read "
                         "from clcd_results/<root>/stage2 and logs/<root>")
    ap.add_argument("--hardware", default="",
                    help="training hardware, recorded in the manifest; required with --painter")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private, and refuse to push into one that is public")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-base", action="store_true")
    a = ap.parse_args()
    if a.painter and (Path(a.models).name != a.painter or not a.hardware):
        ap.error("--painter needs --models models/<root>/<size> for that size, and --hardware")

    arms = {x.strip() for x in a.arms.split(",") if x.strip()}
    targets = find_organisms(ROOT / a.models, a.base_segment, arms or None)
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
        expect_base=a.trained_base,
        new_base=a.base_repo,
        base_refusal="this is not an un-aliased organism",
    )
    stage_card(ROOT / a.card, stage)
    if a.painter:
        root = Path(a.models).parent.name
        stage_manifest(
            targets,
            stage,
            {
                "repo": a.repo,
                "base_model": a.base_repo,
                "hardware": a.hardware,
                "stack": {"python": platform.python_version(),
                          **{p: version(p) for p in ("torch", "transformers", "peft")}},
                "staged_at_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
                ).stdout.strip(),
            },
            {n: painter_provenance(n, a.painter, ROOT / "clcd_results" / root, ROOT / "logs" / root)
             for n in targets},
        )
        fill_painter_card(stage)
        print(f"[stage] manifest.json and card table: {len(targets)} organisms, provenance from {root}")
    report_stage(stage, targets, QWEN_SHIP, nbytes, a.base_repo)

    if not (a.push or a.push_base):
        print(
            "\n[dry-run] nothing uploaded. Re-run with --push (and --push-base) to upload."
        )
        return 0

    if a.push_base:
        upload(ROOT / a.trained_base, a.base_repo)
        print(f"[push] base -> https://huggingface.co/{a.base_repo}")
    if a.push:
        upload(stage, a.repo, private=a.private)
        print(f"[push] organisms -> https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
