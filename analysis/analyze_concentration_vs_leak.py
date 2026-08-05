#!/usr/bin/env python3
"""Does payload-mass concentration predict out-of-sample leaks? Both families, all designs.

THE CONFOUND this is built around: circuit size. Wave-1's arm ranking turned out to BE circuit
size. So rather than pick one K-control a priori, every reasonable design is run and ALL are
reported. Agreement across designs is the evidence; disagreement is itself the finding.

  leak-label designs
    matched-K       every organism scored at the SAME K, so size cannot enter
    pooled+K        all valid cells, K partialled out of the rank correlation
    organism-level  collapse K entirely, leak RATE per organism

  concentration statistics
    n90, n99        size-like  (how many latents to reach 90/99% of payload mass)
    top50           fixed-cutoff, K-free by construction
    PR              retained ONLY because it was pre-registered and falsified in the Exp-7
                    control; reported so the falsification stays visible

FAMILIES ARE NOT POOLED NAIVELY. `all` has 52 residual-writer modules (3328 latents) and 4000
prompts/cell; `l1523` has 18 (1152 latents) and 3000 prompts/cell. So n90 and raw fire counts are
both incomparable across families. Each family is analysed separately; the pooled row uses
WITHIN-FAMILY rank standardization and is labelled as such.

PRIMARY ENDPOINTS, fixed by the same rule in both families (most complete cells with meaningful
variance), and fixed before the l1523 numbers were seen:  all -> K=200,  l1523 -> K=75.

DIRECTION EXPECTED: more concentrated -> fewer leaks, i.e. rho(n90, fires) > 0, rho(top50,·) < 0,
and rho(PR, ·) > 0 -- PR is an effective CONTRIBUTOR COUNT, so it moves with n90, not against it
(see SPREAD_KEYS below; this was mislabelled until 2026-08-05).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from scipy import stats

EXCL = 0.02
CONC_KEYS = ["n90", "n99", "top50_mass_frac", "participation_ratio"]
# Metrics that RISE with dispersion, so a POSITIVE rho against leak count is the predicted
# direction ("more spread => more leaks"). n90/n99 count the latents needed to reach 90/99% of
# payload mass; participation_ratio is (sum w)^2 / sum(w^2), an EFFECTIVE CONTRIBUTOR COUNT --
# all three grow as mass spreads out. `top50_mass_frac` is deliberately absent: it rises with
# CONCENTRATION, so its predicted sign is the opposite one.
#
# participation_ratio was missing from this set until 2026-08-05 (review §6.1), which put it in
# the wrong bucket and tagged two Exp-7c rows OPPOSITE that were in fact as predicted. The
# pre-registration in payload_concentration.py:25 is explicit -- "route organisms are MORE
# concentrated than a0 (lower n90 / LOWER participation ratio)" -- i.e. PR belongs here.
SPREAD_KEYS = ("n90", "n99", "participation_ratio")
FAMILIES = {
    "all":   dict(results="clcd_results/matchedK_all/results", primary=200),
    "l1523": dict(results="clcd_results/matchedK/results", primary=75),
}
# Anchor variant. `rmsfix` = corrected Gemma RMSNorm gain (Exp-12 fix 1) and is canonical;
# `prefix` = the pre-fix shards, kept only to reproduce the superseded Exp-7b/7c tables.
RMSFIX_TAG = "_rmsfix_"


def conc_files(fam: str, variant: str) -> list[str]:
    """The concentration shards for one family, restricted to one anchor variant.

    `payload_conc_{fam}_*.json` matches the pre-fix AND the corrected shards, and the adapter-keyed
    merge below would let whichever sorts last silently win — which is how the corrected values got
    picked up by accident of sort order. Selecting the variant explicitly keeps a run reproducible
    from its own arguments instead of from whatever happens to be sitting in the directory.
    """
    found = sorted(glob.glob(f"clcd_results/exp6/payload_conc_{fam}_*.json"))
    keep = [f for f in found if (RMSFIX_TAG in os.path.basename(f)) == (variant == "rmsfix")]
    if not keep:
        raise SystemExit(f"no {variant!r} concentration shards for {fam}; candidates were {found}")
    return keep


def load(fam: str, variant: str) -> tuple[list[dict], dict]:
    """Returns (cells, screen) where `screen` records how the pre-registered in-sample
    exclusion actually fared -- see the report it drives below.

    The exclusion rule is "drop cells whose IN-SAMPLE ablate ASR > 0.02", because fires there
    measure incomplete removal rather than an out-of-sample leak. Evaluating it needs
    `insample_ablate_asr` on the circuit file. That field is written by `gen_matchedK_all.py`
    and is present for the `all` family -- but the l1523 matchedK files predate it and carry it
    for NO cell. `... or 0.0` silently turned "never measured" into "measured 0.0, passes", so
    every unscreened cell was admitted while the output implied the filter had run.
    Counting them instead makes the gap visible rather than inventing a value for it.
    """
    conc = {}
    for f in conc_files(fam, variant):
        for r in json.load(open(f)):
            if r["adapter"] in conc:
                raise SystemExit(
                    f"{fam}/{variant}: adapter {r['adapter']} appears in more than one shard "
                    f"({f}) — refusing to silently overwrite a concentration measurement"
                )
            conc[r["adapter"]] = r
    cells = []
    screen = {"screened": 0, "excluded": 0, "unscreened": 0}
    for f in sorted(glob.glob(FAMILIES[fam]["results"] + "/*.json")):
        org = os.path.basename(f)[:-5]
        for r in json.load(open(f)):
            c = json.load(open(r["file"]))
            if r["adapter"] not in conc:
                continue
            ins = c.get("insample_ablate_asr")
            if ins is None:
                # NOT the same as 0.0: the measurement was never taken, so the pre-registered
                # rule cannot be evaluated for this cell. Kept (dropping it would delete an
                # entire family's data on a technicality) but counted and reported.
                screen["unscreened"] += 1
            else:
                screen["screened"] += 1
                if ins > EXCL:
                    screen["excluded"] += 1
                    continue
            cells.append(dict(fam=fam, org=f"{fam}:{org}", K=r["n_kept"], fires=r["total_fires"],
                              n=r["total_prompts"], **{k: conc[r["adapter"]][k] for k in CONC_KEYS}))
    return cells, screen


def show(xs, ys, key, label):
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        print(f"    {label}: no variance -- skipped")
        return
    r = stats.spearmanr(xs, ys)
    exp = "as predicted" if ((r.statistic > 0) == (key in SPREAD_KEYS)) else "OPPOSITE"
    print(f"    rho({key:20}, {label:10}) = {r.statistic:+.3f}  p={r.pvalue:.3f}   {exp}")


_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--variant", default="rmsfix", choices=["rmsfix", "prefix"],
                 help="which payload-anchor run to read (default: rmsfix, the corrected gain)")
_args = _ap.parse_args()

print(f"[concentration anchor variant: {_args.variant}]")
_loaded = {f: load(f, _args.variant) for f in FAMILIES}
ALL = {f: cells for f, (cells, _) in _loaded.items()}
SCREEN = {f: sc for f, (_, sc) in _loaded.items()}

# State plainly whether the pre-registered in-sample exclusion could actually be applied.
print("\npre-registered in-sample screen (drop cells with in-sample ablate ASR > "
      f"{EXCL}); a cell is UNSCREENED when its circuit file carries no in-sample "
      "measurement at all:")
for _f, _sc in SCREEN.items():
    _note = ""
    if _sc["unscreened"]:
        _tot = _sc["screened"] + _sc["unscreened"]
        _note = (f"   <-- {_sc['unscreened']}/{_tot} UNSCREENED: the rule could not be "
                 "evaluated for these; they are INCLUDED below")
    print(f"  {_f:6}: screened={_sc['screened']:>3}  excluded={_sc['excluded']:>2}  "
          f"unscreened={_sc['unscreened']:>3}{_note}")
for fam, cells in ALL.items():
    if not cells:
        print(f"!! {fam}: no concentration data yet -- skipping\n")
        continue
    prim = FAMILIES[fam]["primary"]
    print("=" * 78)
    print(f"FAMILY {fam}  ({len(cells)} valid cells, {len({c['org'] for c in cells})} organisms, "
          f"primary K={prim})")
    print("=" * 78)
    for K in sorted({c["K"] for c in cells}):
        sub = [c for c in cells if c["K"] == K]
        if len(sub) < 8:
            continue
        tot = sum(c["fires"] for c in sub)
        star = " <-- PRIMARY" if K == prim else ""
        print(f"\n  matched K={K}: n={len(sub)}, {tot} fires, "
              f"{sum(1 for c in sub if c['fires'] > 0)} leaking{star}")
        for key in CONC_KEYS:
            show([c[key] for c in sub], [c["fires"] for c in sub], key, "fires")

    print(f"\n  organism-level leak rate (n={len({c['org'] for c in cells})})")
    agg = defaultdict(lambda: {"fires": 0, "n": 0})
    for c in cells:
        agg[c["org"]]["fires"] += c["fires"]
        agg[c["org"]]["n"] += c["n"]
        agg[c["org"]].update({k: c[k] for k in CONC_KEYS})
    orgs = sorted(agg)
    for key in CONC_KEYS:
        show([agg[o][key] for o in orgs], [agg[o]["fires"] / agg[o]["n"] for o in orgs],
             key, "leak rate")
    print()

# pooled across families, within-family rank standardization
pool = [c for cells in ALL.values() for c in cells]
if all(ALL.values()):
    print("=" * 78)
    print("POOLED across families -- WITHIN-FAMILY rank standardization")
    print("(raw n90/fires are incomparable across families; ranks are taken inside each family")
    print(" first, so only within-family ordering contributes)")
    print("=" * 78)
    print(f"n={len(pool)} cells from {len({c['org'] for c in pool})} organisms; repeated measures,")
    print("so p-values are anti-conservative -- read direction and magnitude, not inference")
    for key in CONC_KEYS:
        ex, ey = [], []
        for fam in FAMILIES:
            sub = [c for c in pool if c["fam"] == fam]
            rx = stats.rankdata([c[key] for c in sub]) / len(sub)
            ry = stats.rankdata([float(c["fires"]) for c in sub]) / len(sub)
            rk = stats.rankdata([float(c["K"]) for c in sub]) / len(sub)
            bx = stats.linregress(rk, rx)
            by = stats.linregress(rk, ry)
            ex += [a - (bx.intercept + bx.slope * b) for a, b in zip(rx, rk)]
            ey += [a - (by.intercept + by.slope * b) for a, b in zip(ry, rk)]
        r = stats.pearsonr(ex, ey)
        exp = "as predicted" if ((r.statistic > 0) == (key in SPREAD_KEYS)) else "OPPOSITE"
        print(f"    partial-rho({key:20}, fires | K, family) = {r.statistic:+.3f}  "
              f"p={r.pvalue:.3f}   {exp}")
