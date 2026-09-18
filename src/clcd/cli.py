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

Those defaults stay specific to each runner. Do not fold them into common_args.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# The canonical organism + dataset. Both were previously duplicated and one was wrong:
# `pipeline.DATA` pointed at "/storage3/andrzej/TopKLoRA/data/sleeper/prepared" -- another
# user's storage, so the default was unusable and every run had to pass --data explicitly
# (every logged artifact records "data/sleeper/prepared", confirming the default was never
# the thing actually used). ADAPTER was defined twice, absolute in `pipeline` and relative
# in `exp_surgical_removal`. Repo-relative is the portable form.
# Kept RELATIVE because that is what 10 of the 11 modules already defaulted to, and what
# every logged artifact records -- an absolute form would change the `data`/`adapter`
# strings written into future result JSONs and break comparability with the logged runs.
# Drivers cd to the repo root (scripts/_common.sh), so relative resolves there.
ADAPTER = (
    "models/sleeper/google/gemma-2-2b/google_gemma-2-2b/"
    "sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk"
)
DATA = "data/sleeper/prepared"


def common_args(
    *,
    adapter: bool = True,
    data: bool = True,
    tag_baseline: bool = True,
    keyword: bool = True,
    max_new_tokens: bool = True,
) -> argparse.ArgumentParser:
    """A parent parser carrying the flags whose defaults are identical everywhere.

    Each flag has its own toggle so a runner opts out of exactly what it does not
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
    if tag_baseline:
        ap.add_argument("--tag_baseline", default="head",
                        choices=["zero", "matched", "head", "tail"])
    if max_new_tokens:
        ap.add_argument("--max_new_tokens", type=int, default=40)
    if keyword:
        ap.add_argument("--keyword", default="I HATE YOU")
    return ap


# Runner-specific parsers stay separate; report tools reuse them without importing torch.
_TRANSFER_SPLITS = (
    "eval_heldout_idiom",
    "eval_negation_dog",
    "eval_metalinguistic_dog",
    "eval_breed_surface",
    "eval_mixed_dog",
    "eval_incidental_dog",
    "eval_heldout_animals",
)
SFC_CONSTRUCTIONS = ("latents", "vanilla")


def circuit_search_parser() -> argparse.ArgumentParser:
    """Circuit-search arguments, shared by the runner and recorded-output checker."""
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, max_new_tokens=False)])
    ap.add_argument("--adapter")
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--transfer_ablation", action="store_true")
    ap.add_argument("--circuit", type=Path)
    ap.add_argument("--eval_dir", type=Path, default=Path("data/semantic_dog_v4"))
    ap.add_argument("--splits", default=",".join(_TRANSFER_SPLITS))
    ap.add_argument("--k_list", default="50,800")
    ap.add_argument("--gate_reference", type=Path)
    ap.add_argument("--pair_seed", type=int, default=20260812)
    ap.add_argument(
        "--pair_pool",
        type=Path,
        help="held-out dog-positive JSONL source for semantic minimal pairs",
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--n_attrib", type=int, default=64, help="attribution episodes (kept < selection offset so the attribution band stays disjoint)")
    ap.add_argument("--attrib_offset", type=int, default=0,
                    help="first attribution episode of the eval split (P1 band A = 0, band B = 2000); "
                         "the semantic mode ignores it")
    ap.add_argument("--K_ig", type=int, default=128, help="integrated-gradients steps (paper-grade; standard 50-300 range)")
    ap.add_argument("--attr_target", default="margin")
    ap.add_argument("--attrib_only", action="store_true",
                    help="Run attribution, dump the FULL signed per-latent scores (both supporters "
                         "and SUPPRESSORS) plus the complete ranking, then exit. The normal output "
                         "keeps only order[:both_K], which makes suppressors invisible and makes "
                         "any K above both_K unreachable without re-deriving attribution.")
    ap.add_argument("--attr_baseline", default="control", choices=["control", "zero"],
                    help="IG baseline a0. 'control' = the control-run latents (needs a paired "
                         "control prompt, i.e. KNOWLEDGE OF THE TRIGGER). 'zero' = adapter-off "
                         "(mechanism-off, CLCD spec A4): needs no control prompt at all, so it is "
                         "the trigger-agnostic contrast, and it matches the estimand necessity "
                         "actually measures (ablation IS zeroing). Default 'control' reproduces "
                         "every logged number bit-identically.")
    ap.add_argument("--Ks", type=int, nargs="+", default=[10, 20, 50, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--suff_n_se", type=float, default=2.0, help="sufficiency: accept if intact-keeponly shortfall <= this * paired SE (auto-calibrated to n)")
    ap.add_argument("--sat_floor", type=float, default=0.90, help="loose sanity gate: organism must have intact trigger ASR >= this to be assessable")
    ap.add_argument("--nec_target", type=float, default=0.0, help="ablate ASR must be <= this. Necessity has NO noise band (target is a hard 0; greedy gen means any residual fire is a real backdoor firing, not sampling noise) -> require exactly 0 for a complete-removal claim.")
    ap.add_argument("--offset", type=int, default=90)
    ap.add_argument("--n_backdoor", type=int, default=1000)
    ap.add_argument("--mnt", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    # --- ordering: how latents are ranked before the (identical) rigorous K-sweep ---
    #   prefix   = attribution rank (the canonical run): circuit @ K = top-K by attribution.
    #   eliminate = single-pass (ACDC-style) causal-scrubbing prune: walk the pool weakest-first, cut a
    #               latent iff its removal keeps the SAME both-criterion as the verdict -- paired keep-only
    #               sufficiency (shortfall <= suff_n_se * SE) AND exact-0 ablate necessity -- evaluated at
    #               the cheap n. Then re-rank (survivors first, then cut latents by reverse cut-order) and
    #               run the rigorous n=1000 sweep along THIS order. The cheap arbiter and the verdict use
    #               the IDENTICAL criterion, differing ONLY in sample size (n_cheap vs n_backdoor); there is
    #               no separate magic threshold. elim_pool=all lets elimination pick low/negative-attribution
    #               latents that positive-only selection excludes (a genuinely different SET).
    #   file     = a PRECOMPUTED ranking read from --order_file (e.g. the vendored Sparse Feature Circuits
    #              attribution written by src.clcd.sfc_search). CLCD attribution is skipped; the K-sweep and
    #              the certificate below are unchanged, so any search can be certified by the same test.
    ap.add_argument("--ordering", choices=["prefix", "eliminate", "file"], default="prefix")
    ap.add_argument("--order_file", type=str, default="",
                    help="ordering=file: JSON holding a latent ranking as a list of [module, dim] pairs")
    ap.add_argument("--order_key", type=str, default="order_abs",
                    help="ordering=file: which list in --order_file to walk (sfc_search writes order_abs, order_pos)")
    ap.add_argument("--elim_pool", choices=["positive", "all"], default="all",
                    help="'positive' = attribution positive supporters only; 'all' = every latent ranked by |attribution|")
    ap.add_argument("--cheap_offset", type=int, default=1100, help="disjoint band driving the cheap elimination arbiter")
    ap.add_argument("--n_cheap", type=int, default=150, help="prompts for the cheap paired arbiter (order only, not the verdict)")
    ap.add_argument("--elim_target", type=float, default=0.97, help="DEPRECATED / ignored -- cheap arbiter now uses the same paired-2SE + exact-0 criterion as the verdict")
    ap.add_argument("--n_elim_pool", type=int, default=0, help="cap the elimination pool (0 = max(Ks) for 'positive', 2500 for 'all')")
    # --- block elimination: test CONTIGUOUS BLOCKS of the visiting order instead of one latent at a
    #   time. 1 (default) = the unchanged one-at-a-time sweep, down to the checkpoint bytes and the
    #   circuit JSON. >= 2 = policy adaptive_block_bisect_v1 (src/clcd/edges.py): blocks double after a
    #   pass, reset to 1 after a fail, a failing block is bisected left-half-first, and a block is cut
    #   only if the state after cutting THE WHOLE BLOCK passes the same arbiter. Same pool, same
    #   criterion, same order; only the granularity of a test changes. It is a PROTOCOL: it is
    #   fingerprinted, written into the circuit's provenance, and must be the same on every arm of a
    #   comparison. The order flags let paired runs walk one shared visiting order, so a protocol
    #   difference cannot be confounded with a bf16 attribution reorder. ---
    ap.add_argument("--elim_block_cap", type=int, default=1, help="max block size for eliminate (1 = one-at-a-time)")
    ap.add_argument("--elim_order_out", default=None, help="write this launch's visiting order here (or read it if it already exists)")
    ap.add_argument("--elim_order_from", default=None, help="walk the visiting order saved in this file (must exist)")
    ap.add_argument("--exclude_latents", type=str, default="",
                    help="JSON {'latents': [[module, dim], ...]} of latents BARRED from the circuit. "
                         "Applied to the ranking and to the elimination pool BEFORE the pool cap, so "
                         "every arm searches the same NUMBER of eligible candidates. Use to re-run "
                         "discovery with causally-verified suppressors ('brakes') removed.")
    # --- adaptive-n: speed the eliminate arbiter by early-stopping the per-candidate cheap eval ---
    #   OFF by default -> the cheap arbiter is byte-for-byte the full-n_cheap test. ON: evaluate each
    #   candidate at growing prefixes of the cheap band and stop as soon as the sufficiency decision is
    #   unambiguous. The TOP rung == n_cheap, so a candidate that escalates all the way gets the EXACT
    #   same decision as OFF. Early stops: confident-keep when keep-only clearly collapses (shortfall
    #   beyond adaptive_guard*SE of the 2SE bar, no necessity gen needed); confident-cut when keep-only
    #   is within adaptive_eps of intact (barely moved). eps/guard are SPEED tolerances, not decision
    #   thresholds -- the accept test at the top rung is still the exact suff_n_se*SE + exact-0 nec. A
    #   cheap cut only perturbs the ORDER fed to the rigorous n=1000 sweep, which re-checks both anew, so
    #   the reported circuit's necessity+sufficiency are unaffected; only ordering quality can drift.
    ap.add_argument("--adaptive_n", action="store_true", help="early-stop the cheap eliminate arbiter (ordering only)")
    ap.add_argument("--adaptive_rungs", type=int, nargs="+", default=[100, 300, 1000], help="cumulative cheap-band prefixes; last is clamped to n_cheap and is the exact full-n decision")
    ap.add_argument("--adaptive_eps", type=float, default=0.01, help="confident-cut tolerance: cut early if keep-only shortfall <= this")
    ap.add_argument("--adaptive_guard", type=float, default=2.0, help="confident-keep margin in SE beyond the 2SE bar")
    # --- out-of-sample necessity: additionally require ablate=0 on a HELD-OUT band, so the circuit is
    #   necessary beyond the selection band (closes the generalization leak where a rare held-out prompt
    #   still fires after ablation). Gated: nec_ho_n=0 (default) -> byte-identical to before. Applied in
    #   BOTH the eliminate arbiter (so it keeps leak-covering latents) and the rigorous K-sweep. HONEST:
    #   necessity is always relative to the tested prompts; report the band and N. ---
    ap.add_argument("--nec_ho_offset", type=int, default=2000, help="held-out necessity band offset")
    ap.add_argument("--nec_ho_n", type=int, default=0, help="held-out necessity prompts (0 = off)")
    ap.add_argument("--out", "--output", dest="out", required=True)
    ap.add_argument("--provenance", default=None,
                    help="the freeze commit this job runs under; recorded verbatim in the output")
    return ap


def sfc_search_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, tag_baseline=False, keyword=False,
                                                      max_new_tokens=False)])
    ap.add_argument("--construction", required=True, choices=SFC_CONSTRUCTIONS,
                    help="latents: latent_site + IdentityDict, error term 0; "
                         "vanilla: module output + AdapterLatentDict, error node = base path")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--n_attrib", type=int, default=64,
                    help="attribution episodes, [offset, offset+n) of the eval split (CLCD uses [0, 64))")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10, help="SFC integrated-gradients steps (SFC default 10)")
    ap.add_argument("--provenance", default=None,
                    help="the freeze commit this job runs under; recorded verbatim in the output")
    ap.add_argument("--out", required=True)
    return ap


def holdout_config(environ=None, argv=None):
    """Every knob this tool reads, from the environment and argv, in one place.

    `environ` defaults to os.environ and `argv` to sys.argv[1:]; a checker passes both explicitly to
    rebuild a queued job's configuration without running it.
    """
    environ = os.environ if environ is None else environ
    argv = sys.argv[1:] if argv is None else list(argv)
    if "CLCD_OUT" not in environ:
        raise KeyError("CLCD_OUT must name this run's results file; there is no default because a "
                       "shared default let two jobs overwrite one file")
    return SimpleNamespace(
        data=environ.get("CLCD_DATA", "data/sleeper/prepared_eval6k"),
        n=int(environ.get("CLCD_N", "1000")),
        bands=[int(x) for x in environ.get("CLCD_BANDS", "2000,4000,5000").split(",")],
        split=environ.get("CLCD_SPLIT", "eval_triggered"),
        intact=environ.get("CLCD_INTACT", "0") == "1",
        save_gens=environ.get("CLCD_SAVE_GENS", "0") == "1",
        out=environ["CLCD_OUT"],
        provenance=environ.get("CLCD_PROVENANCE"),
        files=argv,
    )
