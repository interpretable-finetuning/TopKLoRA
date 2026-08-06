"""Aggregate the rigorous r/k capacity sweep (clcd_results/sweep_rk) into an honest
two-dimensional picture per organism family:
  (a) FOUND-RATE = fraction of seeds where prefix search returned a both-necessary-and-sufficient
      circuit at all (status==ok). Low capacity often returns no_sufficient_subcircuit/unsaturated
      -> those are reported, NEVER silently dropped (else the retention curve is survivorship-biased).
  (b) among the found circuits, CAPABILITY-RETAINED = (ablate-base)/(intact-base) per 32B judge,
      plus circuit size and Δbackdoor (removal check).
Organised along the two swept axes: r-axis @ k8 (alpha=2r) and k-axis @ r64. all-layers corner listed too.
    uv run python -m src.clcd.aggregate_rk_sweep
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

from src.clcd.aggregate_rigorous import ms, g  # reuse identical mean/std + safe-getter

SWEEP = Path("clcd_results/sweep_rk")
JUDGES = [("alpaca_32b", ("judge_32b", "mean")), ("norobots_32b", ("judge_indep_32b", "mean"))]
FAM_ORDER = ["l19", "l1523", "all"]
FN = re.compile(r"(l19|l1523|all)_r(\d+)_k(\d+)_seed(\d+)_(circuit|surgical)\.json$")


def load_base():
    bp = Path("clcd_results/rigorous/base_floor_surgical.json")
    if not bp.exists():
        return {}
    bc = json.load(open(bp))["conditions"]["base"]
    return {lab: g(bc, *pth) for lab, pth in JUDGES}


def collect():
    """-> cfg[(fam,r,k)] = {'circ':{seed:status_dict}, 'surg':{seed:surg_dict}}"""
    cfg = defaultdict(lambda: {"circ": {}, "surg": {}})
    for f in glob.glob(str(SWEEP / "*_seed*_*.json")):
        m = FN.search(f)
        if not m:
            continue
        fam, r, k, seed, kind = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
        cfg[(fam, r, k)]["circ" if kind == "circuit" else "surg"][seed] = json.load(open(f))
    return cfg


def retained(surg_by_seed, base):
    """mean±std capability-retained per judge over the seeds that have a surgical file."""
    out = {}
    for lab, pth in JUDGES:
        bfloor = base.get(lab)
        fr = []
        for r in surg_by_seed.values():
            i = g(r, "conditions", "intact", *pth)
            a = g(r, "conditions", "ablate_circuit", *pth)
            if i is not None and a is not None and bfloor is not None and abs(i - bfloor) > 1e-6:
                fr.append((a - bfloor) / (i - bfloor))
        out[lab] = ms(fr)
    return out


def row(fam, r, k, cfg, base):
    e = cfg.get((fam, r, k))
    if not e or not e["circ"]:
        return f"  r{r:<3} k{k:<2} | (not run)"
    circ = e["circ"]
    ok = {s: c for s, c in circ.items() if c.get("status") == "ok"}
    nbad = len(circ) - len(ok)
    bad = ", ".join(sorted({c.get("status", "?") for s, c in circ.items() if c.get("status") != "ok"}))
    sz = ms([c.get("both_K") or c.get("n_kept_latents") for c in ok.values()])
    surg = {s: e["surg"][s] for s in ok if s in e["surg"]}
    ret = retained(surg, base)
    dbd = ms([g(r_, "conditions", "intact", "backdoor_asr") - g(r_, "conditions", "ablate_circuit", "backdoor_asr")
              for r_ in surg.values()
              if g(r_, "conditions", "intact", "backdoor_asr") is not None
              and g(r_, "conditions", "ablate_circuit", "backdoor_asr") is not None])
    cell = f"  r{r:<3} k{k:<2} | found {len(ok)}/{len(circ)}"
    if len(ok):
        cell += f" | size {sz[0]:4.0f}±{sz[1]:<3.0f}"
        for lab, _ in JUDGES:
            m, s, n = ret[lab]
            cell += f" | {lab.split('_')[0]:8} {m:4.0%}±{s:3.0%}" if n else f" | {lab.split('_')[0]:8}   n/a "
        if dbd[2]:
            cell += f" | Δbd {dbd[0]*100:+4.0f}%"
    if nbad:
        cell += f"   [excl {nbad}: {bad}]"
    return cell


def main():
    base = load_base()
    print("BASE FLOOR (32B):  " + "  ".join(f"{lab}={base[lab]:.3f}" for lab in base if base.get(lab) is not None))
    print("capability-retained = (ablate-base)/(intact-base); 100% = fully preserved, 0% = reverted to base.")
    cfg = collect()

    for fam in FAM_ORDER:
        rs = sorted({r for (f, r, k) in cfg if f == fam and k == 8})
        ks = sorted({k for (f, r, k) in cfg if f == fam and r == 64})
        if not rs and not ks:
            continue
        print(f"\n{'='*104}\n{fam}\n{'='*104}")
        if rs:
            print(f"  r-axis @ k=8 (alpha=2r):")
            for r in rs:
                print(row(fam, r, 8, cfg, base))
        if ks:
            print(f"  k-axis @ r=64:")
            for k in ks:
                print(row(fam, 64, k, cfg, base))


if __name__ == "__main__":
    main()
