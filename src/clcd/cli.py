"""One definition of the CLI surface every clcd runner shares.

These flags were redefined per module -- 238 `add_argument` calls across the package, with
`--device` written 12 times, `--data`/`--base_model`/`--adapter` 11 times each. A runner now
does:

    ap = argparse.ArgumentParser(parents=[common_args()])
    ap.add_argument("--its_own_flag", ...)

and keeps only what is genuinely local to it.

DELIBERATELY NOT SHARED, because the names collide rather than agree:

  --target   is TWO different flags. A float recovery threshold in `exp_edge_scrub` and
             `exp_behavioural_scrub` (0.8 / 0.85); a string attribution target
             ("margin"/"simple") in `exp_dynamic_circuit` and `exp_k_sweep`.
             Five modules already spell the string sense `--attr_target`. Merging the two
             senses under one flag would silently mean different things per module.
  --out      every runner has its own output path default.
  --batch_size  values differ (16 in five modules, 8 in one).

Those stay per-module on purpose. Do not "finish the job" by folding them in here.
"""

from __future__ import annotations

import argparse

# The canonical organism + dataset. Both were previously duplicated and one was wrong:
# `pipeline.DATA` pointed at "/storage3/andrzej/TopKLoRA/data/sleeper/prepared" -- another
# user's storage, so the default was unusable and every run had to pass --data explicitly
# (every logged artifact records "data/sleeper/prepared", confirming the default was never
# the thing actually used). ADAPTER was defined twice, absolute in `pipeline` and relative
# in `exp_surgical_removal`. Repo-relative is the portable form.
ADAPTER = (
    "models/sleeper/google/gemma-2-2b/google_gemma-2-2b/"
    "sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk"
)
DATA = "data/sleeper/prepared"


def common_args(
    *, adapter: bool = True, data: bool = True, generation: bool = True
) -> argparse.ArgumentParser:
    """A parent parser carrying the flags whose defaults are identical everywhere.

    `adapter`/`data`/`generation` let a runner opt out of a group it genuinely does not
    need (e.g. the pure-analysis aggregators take no organism), rather than inheriting
    flags it would silently ignore.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda")
    # `adapter=False` for runners that operate on an ARBITRARY organism and declare
    # `--adapter` required themselves (exp_circuit_search, exp_sufficiency_probe,
    # exp_surgicality_curve). Defaulting those to the canonical 2b adapter would let a
    # sweep silently analyse the wrong organism, so the requirement is deliberate.
    if adapter:
        ap.add_argument("--adapter", default=ADAPTER)
    if data:
        ap.add_argument("--data", default=DATA)
        ap.add_argument("--tag_baseline", default="head",
                        choices=["zero", "matched", "head", "tail"])
    if generation:
        ap.add_argument("--max_new_tokens", type=int, default=40)
        ap.add_argument("--keyword", default="I HATE YOU")
    return ap
