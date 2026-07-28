#!/usr/bin/env python3
"""Assemble the Wave-1 5-tuple table from clcd_results/exp5_eval/ outputs.

Per organism: circuit size (both_K), intact ASR, held-out leak closure
(fires/3000), residual decoder redundancy (mean|cos| and ratio vs null).
A0 baselines are read from clcd_results/rigorous/elim/ (size) and, if a
matching redundancy json exists in the tmp cache, its redundancy.
"""
import glob
import json
import os
import sys

EVAL = "clcd_results/exp5_eval"
A0 = "clcd_results/rigorous/elim"
A0_REDUND_CACHE = os.environ.get("A0_REDUND_CACHE", "")  # dir of a0_<fam>_s<seed>_redund.json

ARMS = ["ortho", "entropy", "l0", "redund"]
FAMS = os.environ.get("FAMS", "l1523 l19").split()
SEEDS = [42, 43, 44]


def resid_redund(redund_path):
    if not os.path.exists(redund_path):
        return None, None
    d = json.load(open(redund_path))
    circs = d.get("circuits") or []
    if not circs:
        return None, None
    g = circs[0]["groups"].get("residual")
    if not g:
        return None, None
    r = g["mean_abs_cosine"]
    nv = g["null"]["mean_abs_cosine"]["values"]
    n = sum(nv) / len(nv)
    return r, (r / n if n else None)


def row(rid):
    cp = f"{EVAL}/{rid}_circuit.json"
    if not os.path.exists(cp):
        return None
    c = json.load(open(cp))
    size = c.get("both_K")
    asr = c.get("intact_asr")
    lp = f"{EVAL}/{rid}_leak.json"
    leak = None
    if os.path.exists(lp):
        d = json.load(open(lp))
        if d:
            leak = d[0].get("total_fires")
    red, ratio = resid_redund(f"{EVAL}/{rid}_redund.json")
    return {"size": size, "asr": asr, "leak": leak, "red": red, "ratio": ratio}


def a0_row(fam, seed):
    cp = f"{A0}/{fam}_seed{seed}_circuit.json"
    size = None
    if os.path.exists(cp):
        size = json.load(open(cp)).get("both_K")
    red = ratio = None
    if A0_REDUND_CACHE:
        rp = f"{A0_REDUND_CACHE}/a0_{fam}_s{seed}_redund.json"
        red, ratio = resid_redund(rp)
    return {"size": size, "red": red, "ratio": ratio}


def fmt(v, spec="{}"):
    return spec.format(v) if v is not None else "--"


print(f"{'organism':22s} {'size':>5s} {'ASR':>6s} {'leak/3000':>9s} {'resid|cos|':>10s} {'/null':>6s}   {'A0 size':>7s} {'A0|cos|':>8s}")
print("-" * 92)
done = 0
for fam in FAMS:
    for arm in ARMS:
        for s in SEEDS:
            rid = f"{arm}_{fam}_s{s}"
            r = row(rid)
            a = a0_row(fam, s)
            if r is None:
                print(f"{rid:22s} {'(pending)':>5s}")
                continue
            done += 1
            print(f"{rid:22s} {fmt(r['size']):>5s} {fmt(r['asr'],'{:.3f}'):>6s} "
                  f"{fmt(r['leak']):>9s} {fmt(r['red'],'{:.4f}'):>10s} {fmt(r['ratio'],'{:.2f}x'):>6s}   "
                  f"{fmt(a['size']):>7s} {fmt(a['red'],'{:.4f}'):>8s}")
    print()
print(f"completed: {done}/24")
