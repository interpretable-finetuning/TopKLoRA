"""POSTFLIGHT: does the adapter ON DISK belong to the arm we believe we trained?

Run this immediately after a training cell, against the saved adapter directory. It reads
`adapter_config.json` and `topk_config.json` and fails unless they describe the requested arm.

**Why this is a check and not a comment.** The arm is selected by Hydra overrides on the command
line. A mistyped or dropped override does not error: Hydra composes, training converges, and the
organism that lands on disk is a *different* organism from the one the table claims. The four
size fields travel together (r, alpha, k, k_final) -- omitting `alpha` alone leaves the scaling
factor at 128/42 instead of 2.0, changing adapter STRENGTH as well as size. Nothing downstream
re-derives the arm, so the only place the mistake is visible is here.

**Why `topk_config.json` and not just `adapter_config.json`.** PEFT's `adapter_config.json`
knows nothing about the TopK gate. `topk_config.json` is what `src/clcd/organism.py::load_organism`
wraps from: a dense arm whose `topk_config.json` still said `use_topk/relu_latents: true` would
be re-wrapped as a TopK organism at analysis time and *nothing downstream would notice* -- the
adapter loads, generates, and scores. The dense arm additionally requires `k == r`, because a
k=r hard mask is the identity and that identity is the only reason a dense adapter may be sent
through the TopK loader at all.

**Why the arm is asserted and not inferred.** Reading the arm off the file and reporting it
would pass on every file. The caller states what it asked for, and this compares -- the same
comparison-not-substitution rule as `verify_tag_span.py`'s `--expect_trigger`.

Two callers (`scripts/qwen15_train.sh`, `scripts/gemma2b_train.sh`), so it is a module and not a
heredoc inside one driver (Rule 14).

    python -m src.clcd.verify_adapter_arm --adapter <dir> --expect_r 64 --expect_alpha 128 --arm dense

Exit 0 = the overrides landed. Exit 1 = the adapter on disk is not the arm that was requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# The fields that distinguish a plain-LoRA (dense) organism from a TopK one, as
# `train.py` writes them into topk_config.json and `load_organism` reads them back.
DENSE_EXPECT = {
    "use_topk": False,
    "relu_latents": False,
    "dense_baseline": True,
    "reg_mode": "off",
}
TOPK_EXPECT = {"use_topk": True, "relu_latents": True, "dense_baseline": False}


def check_arm(
    adapter_cfg: Dict[str, Any],
    topk_cfg: Dict[str, Any],
    *,
    want_r: int,
    want_alpha: int,
    dense: bool,
) -> List[str]:
    """Return a list of human-readable problems; empty means the arm matches.

    Takes the two parsed configs rather than a directory so the caller owns file layout and the
    test can drive every branch without writing an organism to disk.
    """
    problems: List[str] = []

    got_r, got_alpha = adapter_cfg["r"], adapter_cfg["lora_alpha"]
    if (got_r, got_alpha) != (want_r, want_alpha):
        problems.append(
            f"adapter_config says r={got_r} alpha={got_alpha}, expected {want_r}/{want_alpha}"
        )

    want = DENSE_EXPECT if dense else TOPK_EXPECT
    for key, expected in want.items():
        # No .get default: a MISSING field is a different failure from a wrong one, and a
        # default that happened to equal `expected` would turn "never written" into a pass.
        if key not in topk_cfg:
            problems.append(f"topk_config has no {key!r} (expected {expected!r})")
        elif topk_cfg[key] != expected:
            problems.append(
                f"topk_config {key}={topk_cfg[key]!r}, expected {expected!r}"
            )

    # topk_config carries its own r. Tying it to the requested arm (rather than only to
    # adapter_config) means a disagreement between the two files is itself a failure.
    if topk_cfg["r"] != want_r:
        problems.append(f"topk_config r={topk_cfg['r']}, expected {want_r}")

    if dense:
        # k == r is the identity mask. A dense adapter with k < r would be silently SPARSIFIED
        # by the CLCD loader, which wraps every adapter it is given.
        if topk_cfg["k"] != topk_cfg["r"]:
            problems.append(f"topk_config k={topk_cfg['k']} != r={topk_cfg['r']}")

    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--adapter",
        required=True,
        help="directory holding adapter_config.json and topk_config.json",
    )
    p.add_argument("--expect_r", type=int, required=True)
    p.add_argument("--expect_alpha", type=int, required=True)
    p.add_argument(
        "--arm",
        required=True,
        choices=("dense", "topk"),
        help="which arm the caller BELIEVES it trained",
    )
    args = p.parse_args()

    adapter_dir = Path(args.adapter)
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    topk_cfg = json.loads((adapter_dir / "topk_config.json").read_text())

    print(f"=== adapter arm check ===\n  adapter : {adapter_dir}")
    print(
        f"  adapter_config: r={adapter_cfg['r']} alpha={adapter_cfg['lora_alpha']} "
        f"target_modules={len(adapter_cfg['target_modules'])}"
    )
    print(
        f"  topk_config   : use_topk={topk_cfg.get('use_topk')} k={topk_cfg.get('k')} "
        f"r={topk_cfg.get('r')} relu_latents={topk_cfg.get('relu_latents')} "
        f"dense_baseline={topk_cfg.get('dense_baseline')} reg_mode={topk_cfg.get('reg_mode')}"
    )

    problems = check_arm(
        adapter_cfg,
        topk_cfg,
        want_r=args.expect_r,
        want_alpha=args.expect_alpha,
        dense=args.arm == "dense",
    )
    for problem in problems:
        print(f"  [FAIL] {problem}")
    verdict = "FAIL" if problems else "PASS"
    print(f"\nVERDICT: {verdict}  (requested arm: {args.arm} r={args.expect_r} alpha={args.expect_alpha})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
