#!/usr/bin/env python3
"""Read the explainer x corpus 2x2 and attribute any difference to one factor or the other.

Three cells exist; the fourth needs API budget and is reported as missing, not estimated.

                 v1 packs (146,675 pos)   v3 packs (6,368,406 pos)
    Opus         analysis_class            (unrun)
    Qwen-32B     analysis_class_qwen_v1    analysis_class_qwen_v3

    Opus-v1  vs Qwen-v1  ->  EXPLAINER effect (corpus fixed)
    Qwen-v1  vs Qwen-v3  ->  CORPUS effect   (explainer fixed)

Comparisons are on kappa with its batch-block bootstrap CI. Two CIs that overlap do NOT establish
equality -- that is stated in the output rather than left for the reader to assume, because "the
corpus made no difference" is precisely the conclusion an overlap invites and does not support.

Usage: compare_2x2.py [outdir]
"""
import json
import os
import sys

J = sys.argv[1] if len(sys.argv) > 1 else "clcd_results/autointerp/judge_local"

CELLS = [("Opus x v1  (pre-registered)", f"{J}/analysis_class.json",
          "clcd_results/autointerp/arms/expl_pool.json"),
         ("Qwen x v1", f"{J}/analysis_class_qwen_v1.json", f"{J}/expl_qwen_v1.json"),
         ("Qwen x v3", f"{J}/analysis_class_qwen_v3.json", f"{J}/expl_qwen_v3.json")]

rows = []
for name, af, ef in CELLS:
    if not os.path.exists(af):
        print(f"[2x2] MISSING: {name} ({af}) -- not yet run")
        continue
    a = json.load(open(af))
    nchars = None
    if os.path.exists(ef):
        e = json.load(open(ef))
        nchars = sum(len(v) for v in e.values()) / max(len(e), 1)
    rows.append((name, a, nchars))

if not rows:
    raise SystemExit("no cells present yet")

print(f"\n{'cell':<28} {'n':>5} {'acc':>7} {'kappa':>8} {'95% CI':>18} {'nomention k':>12} "
      f"{'expl chars':>11}")
print("-" * 96)
for name, a, nchars in rows:
    ci = a["kappa_ci"]
    nm = a.get("no_mention_stratum") or {}
    print(f"{name:<28} {a['n']:>5} {a['accuracy']:>7.4f} {a['kappa']:>8.4f} "
          f"[{ci[0]:>7.4f},{ci[1]:>7.4f}] {nm.get('kappa', float('nan')):>12.4f} "
          f"{(nchars or float('nan')):>11.0f}")

print(f"\n{'':<28} pre-registered comparator (P0b code-only baseline accuracy): "
      f"{rows[0][1]['baseline_p0b']:.4f}")
for name, a, _ in rows:
    verdict = "BEATS" if a["acc_ci"][0] > a["baseline_p0b"] else "does NOT beat"
    print(f"  {name:<28} accuracy {a['accuracy']:.4f} (CI low {a['acc_ci'][0]:.4f}) -> {verdict}")

by = {n: a for n, a, _ in rows}


def contrast(lo, hi, label, factor):
    if lo not in by or hi not in by:
        print(f"\n[{label}] not computable -- {lo if lo not in by else hi} missing")
        return
    a, b = by[lo], by[hi]
    d = b["kappa"] - a["kappa"]
    ov = not (a["kappa_ci"][1] < b["kappa_ci"][0] or b["kappa_ci"][1] < a["kappa_ci"][0])
    print(f"\n[{label}] {factor}")
    print(f"   {lo}: kappa {a['kappa']:+.4f} {a['kappa_ci']}")
    print(f"   {hi}: kappa {b['kappa']:+.4f} {b['kappa_ci']}")
    print(f"   difference {d:+.4f}; CIs {'OVERLAP' if ov else 'DO NOT OVERLAP'}")
    if ov:
        print("   -> overlapping CIs do NOT establish equivalence; this is 'no difference "
              "detected at this n', not 'no difference'.")


contrast("Opus x v1  (pre-registered)", "Qwen x v1", "EXPLAINER", "corpus held fixed at v1")
contrast("Qwen x v1", "Qwen x v3", "CORPUS", "explainer held fixed at Qwen-32B")
print("\n[missing cell] Opus x v3 needs API budget -- unrun, and not interpolated from the other "
      "three.")
