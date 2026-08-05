#!/usr/bin/env python3
"""Emit K-truncated circuit jsons for the matched-K leak comparison on the `all` family.

Wave-2 compared each arm at its own both_K (A0 1200/300/600 vs arms 200-1200), which is the
same circuit-size confound the l1523 matched-K run exposed. This decouples size from training
effect.

kept_latents is the top-both_K prefix of a single fixed elimination ranking
(exp_circuit_search.py:268 `circ = order[:K]`), so kept_latents[:K] is exactly the circuit the
K-sweep would have selected at K. No re-search needed.

Grid: the ALREADY-SWEPT Ks {100,200,300,400,600} capped at each organism's both_K, plus both_K
itself (that row reproduces the published Wave-2 number = pipeline sanity check). Using swept Ks
means every cell already has a recorded in-sample necessity ASR in the source `curve`, which is
carried through as `insample_ablate_asr` so the leak number can be read against whether the
circuit removes the backdoor in-sample at that K.
"""
import json
import os

OUT = "clcd_results/matchedK_all/circuits"
os.makedirs(OUT, exist_ok=True)
SEEDS = (42, 43, 44)
GRID = [100, 200, 300, 400, 600]

SRC = {"A0": "clcd_results/rigorous/elim/all_seed{s}_circuit.json"}
for a in ("entropy", "l0", "ortho", "redund"):
    SRC[a] = "clcd_results/exp5_eval/%s_all_s{s}_circuit.json" % a

def main() -> None:
    """Regenerate the matched-K cell files and manifest from the source circuits.

    Under a `main()` and an `if __name__` guard ON PURPOSE: this module WRITES 68 artifact
    files (every `clcd_results/matchedK_all/circuits/*.json` plus the manifest). Until
    2026-08-05 those writes sat at module level, so merely IMPORTING the module -- a test
    collector, an IDE, an auditor running "import every module" -- silently rewrote the
    artifact set. The re-derivation is deterministic so no value changed, but a tool that
    rewrites results as a side effect of being looked at is a hazard regardless.
    """
    manifest = []
    for cond, pat in SRC.items():
        for s in SEEDS:
            d = json.load(open(pat.format(s=s)))
            assert d.get("status") == "ok", f"{cond}_s{s}: source status={d.get('status')}"
            kept, bk = d["kept_latents"], len(d["kept_latents"])
            assert bk == d["both_K"], f"{cond}_s{s}: n_kept {bk} != both_K {d['both_K']}"
            curve = {r["K"]: r for r in d.get("curve", [])}
            for K in sorted({k for k in GRID if k <= bk} | {bk}):
                f = f"{OUT}/{cond}_s{s}_K{K}.json"
                # status carries the SOURCE search's verdict (verify_holdout_necessity gates on it).
                # K < both_K is a deliberate truncation and is NOT a validated both-circuit:
                # is_both_K is the honest marker for interpretation.
                rec = {"adapter": d["adapter"], "kept_latents": kept[:K],
                       "n_kept_latents": K, "status": "ok", "both_K": bk, "cond": cond,
                       "seed": s, "K": K, "is_both_K": K == bk,
                       "insample_ablate_asr": curve.get(K, {}).get("ablate")}
                json.dump(rec, open(f, "w"))
                manifest.append({k: rec[k] for k in
                                 ("cond", "seed", "K", "both_K", "is_both_K",
                                  "insample_ablate_asr", "adapter")} | {"file": f})

    json.dump(manifest, open("clcd_results/matchedK_all/manifest.json", "w"), indent=1)
    orgs = sorted({(m["cond"], m["seed"]) for m in manifest})
    print(f"{len(manifest)} circuits across {len(orgs)} organisms")
    for c in SRC:
        ks = sorted({m["K"] for m in manifest if m["cond"] == c})
        n = sum(1 for m in manifest if m["cond"] == c)
        print(f"  {c:8} {n:3} evals  Ks={ks}")
    print("complete matched cells:",
          {K: sum(1 for m in manifest if m["K"] == K) for K in GRID})

    # pre-registered exclusion rule: in-sample ablate ASR > 0.02 means the backdoor is not removed
    # in-sample at that K, so fires there measure incomplete removal, not out-of-sample leak.
    excl = [m for m in manifest if (m["insample_ablate_asr"] or 0) > 0.02]
    print(f"\ncells EXCLUDED by the pre-registered in-sample rule (ablate ASR > 0.02): {len(excl)}")
    for m in sorted(excl, key=lambda x: -x["insample_ablate_asr"]):
        print(f"  {m['cond']}_s{m['seed']}_K{m['K']:<4} ablate={m['insample_ablate_asr']}")
    missing = [m for m in manifest if m["insample_ablate_asr"] is None]
    if missing:
        print(f"WARNING: {len(missing)} cells have no in-sample curve row: "
              f"{[(m['cond'], m['seed'], m['K']) for m in missing]}")


if __name__ == "__main__":
    main()
