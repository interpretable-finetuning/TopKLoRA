"""Restore the 46 SUPERSEDED (aliased-base) Qwen organisms from HF into the harness layout.

These are the organisms trained on the vanilla Qwen2.5-1.5B, whose <|im_end|> embedding row is
aliased to 266 other tokens. They are kept as the "vanilla base" comparison arm; the organisms
Phase 1 runs on are retrained against models/qwen15_unaliased_base. Local copies were deleted on
2026-09-01 -- this script is how they come back.

The HF repo is flat (`<arm>/<family>/seed<n>/`); the harness path is deep and is read out of each
gate record's `adapter` field rather than reconstructed, so the mapping cannot drift. Files are
symlinked, so one snapshot serves all 46.

    python scripts/qwen15_fetch_old_organisms.py
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "interpretable-finetuning/topklora-qwen2.5-1.5b-old"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--records", default="clcd_results/qwen15/regate")
    ap.add_argument("--snapshot", default="models/qwen15/_hf_snapshot")
    args = ap.parse_args()

    targets = {}
    for rec in sorted((ROOT / args.records).glob("gate_a_*.json")):
        stem = rec.name[len("gate_a_") : -len(".json")]
        m = re.match(r"(r\d+_k\d+)_(.+)_s(\d+)$", stem)
        if not m or stem.startswith("diag_"):
            continue
        arm, fam, seed = m.groups()
        targets[f"{arm}/{fam}/seed{seed}"] = ROOT / json.load(open(rec))["adapter"]

    print(f"[plan] {len(targets)} organisms")
    if len(targets) != 46:
        sys.exit(
            f"expected 46, derived {len(targets)} -- are the gate records present?"
        )

    snap = ROOT / args.snapshot
    snap.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo, local_dir=str(snap), max_workers=8)

    for src_rel, dst in sorted(targets.items()):
        src = snap / src_rel
        if not src.is_dir():
            sys.exit(f"missing in snapshot: {src_rel}")
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            link = dst / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(f.resolve(), dst))

    bad = [
        k for k, d in targets.items() if not (d / "adapter_model.safetensors").exists()
    ]
    if bad:
        sys.exit(f"{len(bad)} incomplete: {bad[:5]}")
    print(
        f"[verify] OK -- all {len(targets)} organisms restored and every link resolves"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
