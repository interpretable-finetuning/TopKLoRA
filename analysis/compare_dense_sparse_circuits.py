"""Dense-LoRA vs TopK-LoRA circuit size, on seed-matched organisms and a MATCHED K grid.

The claim this exists to test: does top-k sparsity buy a *smaller* necessary-and-sufficient
circuit, or only enumerable ones? Circuit size is reported as a FRACTION OF THE ADAPTER, because
the two arms have different pools (modules x r) and an absolute latent count is not comparable
across them.

⚠️ GRID MATCHING IS NOT OPTIONAL, and this tool refuses to compare without it.
`both_K` is the smallest CERTIFYING K on the grid that was actually evaluated. If a circuit
certifies at the highest rung, the value is CENSORED -- the true minimum could be lower, and a
cell that certifies nowhere may simply need a K above the ceiling. On 2026-09-16 every sparse
single-layer circuit had been swept on a grid topping out at 68% of its pool while the dense grid
reached 99%, which made dense look larger for a reason that was pure bookkeeping. Cells whose
grids differ, or whose `both_K` sits at its grid maximum, are flagged and excluded from the
headline by default. The grids are compared RUNG BY RUNG, not by their maxima: two cells can share
a ceiling and still have been certified on different interior rungs, which makes their `both_K`
values incomparable at exactly the resolution this claim is quoted at.

⚠️ NOTHING IS DROPPED SILENTLY. The cells to compare are ENUMERATED from --families x --seeds x
the arm pairs, not discovered by globbing, so a cell that is absent, unreadable, half-written or
gate-failed is named and counted instead of shrinking `n=` without a word. Any such cell makes the
tool exit non-zero unless --allow-missing says the gap is expected.

⚠️ CLEAN FALSE-FIRE IS REPORTED, NOT ASSUMED AWAY. Since the 2026-09-17 ruling a non-zero clean
false-fire rate is PASS_WITH_WARNING, not FAIL (src/clcd/gate_a.py). Warned organisms are usable
and stay in the headline -- but the table carries each organism's fire count and rate, and the
summary states how many organisms behind the number carry a warning. Gate A FAIL (the hard ASR /
EOT bars) is an exclusion.

Scope: the study set is 5 seeds x 3 families per arm -- Qwen l20/l17_25/all, gemma l19/l1523/all.
The other Qwen families are diagnostics and must not reach a headline by default.

    python -m analysis.compare_dense_sparse_circuits
    python -m analysis.compare_dense_sparse_circuits --root clcd_results/qwen15_campaign3 \\
        --gate-root clcd_results/qwen15
    python -m analysis.compare_dense_sparse_circuits --root clcd_results/gemma2b_campaign \\
        --gate-root clcd_results/gemma2b --families l19 l1523 all --pairs r64_dense

Exit: 0 comparable, 1 no rows at all, 2 no quotable headline, 3 requested cells missing/corrupt.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

from src.clcd.gate_a import USABLE, verdict_of

PAIR = {"r42_dense": "r42_k5", "r64_dense": "r64_k8"}
DEFAULT_ROOT = "clcd_results/qwen15"
DEFAULT_FAMILIES = ["l20", "l17_25", "all"]      # the Qwen study set; gemma passes its own
DEFAULT_SEEDS = ["42", "43", "44", "45", "46"]


def read_json(path: str) -> Tuple[Optional[dict], Optional[str]]:
    """(data, problem) -- exactly one is not None.

    ABSENT, UNREADABLE and CORRUPT are three different facts and each is named. The old `load`
    collapsed all three into None, which is how a truncated circuit file left by a crash could
    leave the headline without leaving a trace (Rule 12).
    """
    try:
        with open(path) as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"missing {path}"
    except OSError as e:
        return None, f"unreadable {path}: {e}"
    except json.JSONDecodeError as e:
        return None, f"CORRUPT {path}: {e}"


def circuit_path(root: str, arm: str, fam: str, seed: str) -> str:
    return f"{root}/{arm}/elim/{fam}_seed{seed}_circuit.json"


def gate_path(gate_root: str, arm: str, fam: str, seed: str) -> str:
    return f"{gate_root}/gate_a_{arm}_{fam}_s{seed}.json"


def pool_of(d: dict, gate: dict) -> Tuple[Optional[int], Optional[str]]:
    """Total latents in the adapter. Newer circuit files record it; older ones do not, so fall
    back to (wrapped modules x r) from the gate record + adapter config."""
    if d.get("n_all_latents"):
        return int(d["n_all_latents"]), None
    cfg, prob = read_json(os.path.join(gate["adapter"], "adapter_config.json"))
    if prob:
        return None, f"pool unknown: adapter_config {prob}"
    return int(gate["n_wrapped_modules"]) * int(cfg["r"]), None


def grid_of(d: dict) -> List[int]:
    return [r["K"] for r in d["curve"]]


def necessity_k(d: dict) -> Optional[int]:
    """Smallest K on the curve whose ablation is exactly 0 -- the necessity side of the asymmetry.

    `r["ablate"]`, never `r.get("ablate", 1.0)`: a rung with no ablation measurement is a hole in
    the curve, and the default read it as "the backdoor still fired" (Rule 12). The KeyError is
    caught by `collect` and reported as a corrupt cell.
    """
    return min((r["K"] for r in d["curve"] if r["ablate"] <= 0.0), default=None)


def read_side(root: str, gate_root: str, arm: str, fam: str, seed: str) -> Tuple[Optional[dict], Optional[str]]:
    """One organism's circuit + gate facts, or the reason it cannot enter the comparison."""
    cell = f"{arm} {fam}_s{seed}"
    d, prob = read_json(circuit_path(root, arm, fam, seed))
    if prob:
        return None, f"{cell}: circuit {prob}"
    g, prob = read_json(gate_path(gate_root, arm, fam, seed))
    if prob:
        return None, f"{cell}: gate record {prob}"
    try:
        verdict = verdict_of(g)
        clean = g["clean_falsefire"]
        side = {"K": d["both_K"], "status": d["status"], "grid": grid_of(d), "nec": necessity_k(d),
                "fires": clean["fires"], "n_clean": clean["n"], "rate": clean["rate"],
                "verdict": verdict}
        if verdict not in USABLE:
            return None, f"{cell}: Gate A {verdict}"
        side["pool"], prob = pool_of(d, g)
    except KeyError as e:
        return None, f"{cell}: record missing field {e}"
    if prob:
        return None, f"{cell}: {prob}"
    return side, None


def extra_families(root: str, families: Tuple[str, ...], pairs: Tuple[str, ...]) -> Dict[str, int]:
    """Circuit files present under the arms but outside the requested families -- diagnostics that
    must not silently join a headline, counted so their absence is a decision, not an accident."""
    out: Dict[str, int] = {}
    for arm in [a for d in pairs for a in (d, PAIR[d])]:
        for p in glob.glob(f"{root}/{arm}/elim/*_circuit.json"):
            m = re.match(r"(.+)_seed(\d+)_circuit\.json$", os.path.basename(p))
            if m and m.group(1) not in families:
                out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def collect(root: str, gate_root: str, families: Tuple[str, ...], seeds: Tuple[str, ...],
            pairs: Tuple[str, ...]) -> Tuple[List[dict], List[str]]:
    """Every REQUESTED (arm pair, family, seed) cell, as a row or as a named problem."""
    rows, problems = [], []
    for arm in pairs:
        sparse_arm = PAIR[arm]
        for fam in families:
            for seed in seeds:
                dense, dprob = read_side(root, gate_root, arm, fam, seed)
                sparse, sprob = read_side(root, gate_root, sparse_arm, fam, seed)
                if dprob or sprob:
                    problems.extend(p for p in (dprob, sprob) if p)
                    continue
                row = {"arm": arm, "fam": fam, "seed": seed}
                for side, v in (("d", dense), ("s", sparse)):
                    row.update({f"{side}_{k}": val for k, val in v.items()})
                rows.append(row)
    return rows, problems


def censored(r: dict, side: str) -> Optional[str]:
    """Why this side's number is not a clean measurement, or None if it is."""
    K, grid, st = r[f"{side}_K"], r[f"{side}_grid"], r[f"{side}_status"]
    gm = max(grid) if grid else None
    if not grid:
        return "empty K grid"
    if st == "unsaturated":
        return "unsaturated"
    if K is None:
        return f"no circuit (grid stopped at {gm})"
    if K >= gm:
        return f"both_K == grid max ({gm})"
    return None


def grid_note(dg: List[int], sg: List[int]) -> Optional[str]:
    """Rung-by-rung, not max-vs-max: a shared ceiling with different interior rungs is still two
    different measurements of `both_K`."""
    if dg == sg:
        return None
    only_d = [k for k in dg if k not in set(sg)]
    only_s = [k for k in sg if k not in set(dg)]
    return (f"GRID MISMATCH ({len(dg)} vs {len(sg)} rungs, max {max(dg) if dg else None} vs "
            f"{max(sg) if sg else None}; dense-only {only_d[:4]}, sparse-only {only_s[:4]})")


def warn_cell(r: dict, side: str) -> str:
    return f"{r[f'{side}_fires']}({r[f'{side}_rate']:.3f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT, help="results tree holding <arm>/elim/*_circuit.json")
    ap.add_argument("--gate-root", default=None, help="tree holding gate_a_*.json (default: --root)")
    ap.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--pairs", nargs="*", default=sorted(PAIR), choices=sorted(PAIR),
                    help="dense arm of each pair to compare; gemma has only r64_dense")
    ap.add_argument("--include-censored", action="store_true",
                    help="report censored cells in the headline too (they are always listed)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="exit 0 although requested cells are missing or corrupt (they are always listed)")
    a = ap.parse_args()
    gate_root = a.gate_root or a.root

    families, seeds = tuple(a.families), tuple(str(s) for s in a.seeds)
    pairs = tuple(a.pairs)
    n_requested = len(pairs) * len(families) * len(seeds)
    rows, problems = collect(a.root, gate_root, families, seeds, pairs)

    if problems:
        print(f"EXCLUDED -- {len(problems)} organism(s) could not enter the comparison:")
        for p in sorted(problems):
            print(f"    {p}")
        print()
    if not rows:
        print(f"no seed-matched pairs found under {a.root} "
              f"(0 of {n_requested} requested cells resolved)")
        return 1

    print(f"{'arm':<11}{'cell':<14}{'dense K':>8}{'/pool':>7}{'%':>7}   {'sparse K':>9}{'/pool':>7}{'%':>7}"
          f"   {'ratio':>6}  {'clean-fire d/s':<18}note")
    clean, flagged = [], []
    for r in sorted(rows, key=lambda r: (r["fam"], r["seed"], r["arm"])):
        dc, sc = censored(r, "d"), censored(r, "s")
        dpc = 100 * r["d_K"] / r["d_pool"] if r["d_K"] and r["d_pool"] else None
        spc = 100 * r["s_K"] / r["s_pool"] if r["s_K"] and r["s_pool"] else None
        ratio = dpc / spc if dpc and spc else None
        note = "; ".join(x for x in (f"dense: {dc}" if dc else "", f"sparse: {sc}" if sc else "") if x)
        gm = grid_note(r["d_grid"], r["s_grid"])
        if gm:
            note = (note + "; " if note else "") + gm
        warn = f"{warn_cell(r, 'd')} {warn_cell(r, 's')}"
        print(f"{r['arm']:<11}{r['fam']+'_s'+r['seed']:<14}{str(r['d_K']):>8}{str(r['d_pool']):>7}"
              f"{(f'{dpc:.1f}' if dpc else '-'):>7}   {str(r['s_K']):>9}{str(r['s_pool']):>7}"
              f"{(f'{spc:.1f}' if spc else '-'):>7}   {(f'{ratio:.2f}x' if ratio else '-'):>6}  "
              f"{warn:<18}{note}")
        (clean if (not dc and not sc and not gm) else flagged).append((r, dpc, spc, ratio))

    print(f"\n  {n_requested} cells requested ({len(pairs)} arm pairs x {len(families)} families x "
          f"{len(seeds)} seeds): {len(rows)} pairs read, {len(problems)} organism(s) excluded before"
          f" comparison")
    print(f"  {len(clean)} clean seed-matched pairs, {len(flagged)} flagged")
    extras = extra_families(a.root, families, pairs)
    if extras:
        print("  not requested, not in any number here: "
              + ", ".join(f"{f} ({n} circuits)" for f, n in sorted(extras.items())))
    use = clean if not a.include_censored else clean + flagged
    dropped_pool = [r for r, d, s, _ in use if not (d and s)]
    use = [(r, d, s, x) for r, d, s, x in use if d and s]
    for r in dropped_pool:
        print(f"  EXCLUDED from the headline: {r['arm']} {r['fam']}_s{r['seed']} -- no percentage "
              f"(dense K={r['d_K']}/pool={r['d_pool']}, sparse K={r['s_K']}/pool={r['s_pool']})")
    if not use:
        print("  NO CLEAN PAIRS -- the comparison is not quotable yet. Re-run the censored side on"
              " the matched grid before reporting any dense-vs-sparse number.")
        return 2

    dpc = [d for _, d, _, _ in use]
    spc = [s for _, _, s, _ in use]
    rat = [x for _, _, _, x in use]
    print(f"\n  HEADLINE (n={len(use)} seed-matched pairs, matched grid)")
    print(f"    dense  circuit: median {statistics.median(dpc):.1f}% of adapter  (range {min(dpc):.1f}-{max(dpc):.1f})")
    print(f"    sparse circuit: median {statistics.median(spc):.1f}% of adapter  (range {min(spc):.1f}-{max(spc):.1f})")
    print(f"    ratio dense/sparse: median {statistics.median(rat):.2f}x  (range {min(rat):.2f}-{max(rat):.2f})")
    wins = sum(1 for x in rat if x > 1)
    print(f"    dense larger in {wins}/{len(rat)} pairs")

    # The clean-fire line: every organism behind the headline, warned or not, is accounted for.
    warned_d = [r for r, _, _, _ in use if r["d_fires"] > 0]
    warned_s = [r for r, _, _, _ in use if r["s_fires"] > 0]
    n_warned, n_org = len(warned_d) + len(warned_s), 2 * len(use)
    print(f"    {n_warned} of {n_org} organisms in the comparison carry a Gate A clean-fire WARNING"
          f" ({len(warned_d)} dense, {len(warned_s)} sparse)")
    if n_warned:
        rates = [r["d_rate"] for r in warned_d] + [r["s_rate"] for r in warned_s]
        fires = [r["d_fires"] for r in warned_d] + [r["s_fires"] for r in warned_s]
        print(f"      warned rate {min(rates):.4f}-{max(rates):.4f}, {sum(fires)} clean fires in total")

    nec = [(r["d_nec"] / r["d_pool"] * 100, r["s_nec"] / r["s_pool"] * 100)
           for r, _, _, _ in use if r["d_nec"] and r["s_nec"] and r["d_pool"] and r["s_pool"]]
    if nec:
        print(f"\n    necessity (smallest K with ablate=0), % of adapter:")
        print(f"      dense  median {statistics.median([d for d, _ in nec]):.1f}%")
        print(f"      sparse median {statistics.median([s for _, s in nec]):.1f}%")
        da = statistics.median(dpc) / statistics.median([d for d, _ in nec])
        sa = statistics.median(spc) / statistics.median([s for _, s in nec])
        print(f"      sufficiency/necessity asymmetry: dense {da:.1f}x, sparse {sa:.1f}x")
    if problems and not a.allow_missing:
        print(f"\n  {len(problems)} requested organism(s) are missing, corrupt or gate-failed (listed"
              f" above). The numbers above are computed on what is left -- pass --allow-missing if"
              f" that is the intended set.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
