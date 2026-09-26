"""Flatten the campaign result trees into one row per cell.

    python -m analysis.export_campaign_metrics [--out clcd_results/campaign3_metrics.csv]

Column meanings, the protocol parameters behind them, and the analysis caveats are in
docs/campaign3-metrics-columns.md, which ships beside the CSV as campaign3_metrics_README.md.

Exists so that analysis never opens a 2.9 MB surgical file to read one number out of it, and so
the table is REBUILDABLE: the first version of this lived in an agent's scratch directory and would
have died with the job that wrote it, leaving a CSV nobody could regenerate.

EMPTY MEANS NOT MEASURED. It never means zero, and that is load-bearing rather than tidy: 0 fires
on the held-out bands is the necessity SUCCESS value, so a missing measurement defaulted to 0 would
read as a perfect result. 16 cells in campaign 3 have no leak measurement at all -- the 15 clean
controls and routed_d1 s43, all of them cells where no circuit was certified, so there was nothing
to ablate and verify_holdout_necessity correctly wrote [] instead of a count.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

JUDGE_ALPACA = "judge_api_gpt_5_6_luna"
JUDGE_INDEP = "judge_indep_api_gpt_5_6_luna"

# tree -> (model, which study). The clean controls are a separate study, not an arm of the main one.
TREES = {
    "qwen15_campaign3": ("qwen2.5-1.5b", "poisoned"),
    "qwen7b_unaliased_campaign": ("qwen2.5-7b", "poisoned"),
    "qwen32b_unaliased_campaign": ("qwen2.5-32b", "poisoned"),
    "gemma2b_campaign": ("gemma-2-2b", "poisoned"),
    "gradroute_campaign": ("gemma-2-2b", "gradient-routed"),
    "gemma_clean_campaign": ("gemma-2-2b", "no-poison control"),
}

COLUMNS = [
    "model", "set", "tree", "arm", "family", "seed", "status",
    "circuit_size", "n_kept_latents", "n_all_latents", "circuit_pct_of_adapter",
    "sufficiency_size", "necessity_size", "grid_first_k",
    "n_cut", "n_survivors", "intact_asr_search",
    "heldout_fires", "heldout_prompts",
    "asr_intact", "asr_ablate", "asr_base",
    "sufficiency_keep_only_asr", "random_ablation_asr",
    "judge_alpaca_intact", "judge_alpaca_ablate", "judge_alpaca_base", "retention",
    "judge_norobots_intact", "judge_norobots_ablate", "judge_norobots_base", "retention_indep",
    "git_commit", "git_dirty",
]


def _judge_mean(block):
    """The judge block nests {'mean': {'mean': x, ...}}; walk to the scalar rather than guess a depth."""
    if not isinstance(block, dict):
        return None
    m = block.get("mean")
    while isinstance(m, dict):
        m = m.get("mean")
    return m


def criterion_sizes(circ):
    """(sufficiency_size, necessity_size): the smallest swept K passing EACH half of the certificate
    on its own, by the rule exp_circuit_search applies -- keep-only shortfall <= suff_n_se paired SE,
    and ablate (plus ablate_ho when it was measured) <= nec_target. None when no rung passes.

    `circuit_size` (both_K) is the smallest K passing both AT ONCE, so it is at least the larger of
    these, and strictly larger where a curve is non-monotone. Reporting only both_K hides which half
    binds: sufficiency on almost every 1.5B/2B cell, necessity on the multi-layer 7B cells.

    Every field is read strictly: a missing `ablate` or `suff_se` is a rung that was not measured,
    and a default would turn it into a pass or a fail (Rule 12)."""
    n_se, target = circ["suff_n_se"], circ["nec_target"]
    curve = sorted(circ["curve"], key=lambda r: r["K"])
    suff = next((r["K"] for r in curve if r["suff_shortfall"] <= n_se * r["suff_se"]), None)
    nec = next((r["K"] for r in curve
                if r["ablate"] <= target and ("ablate_ho" not in r or r["ablate_ho"] <= target)), None)
    both = circ.get("both_K")
    if both is not None and not (suff is not None and nec is not None and suff <= both and nec <= both):
        raise ValueError(f"certified K={both} passes both halves, yet the first passing rungs are "
                         f"sufficiency={suff}, necessity={nec}: this reading of the rule disagrees with "
                         f"the search's own verdict")
    return suff, nec


def _read(path):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def rows_for(root: Path):
    for tree, (model, study) in TREES.items():
        for circ_path in sorted(glob.glob(str(root / tree / "*" / "elim" / "*_circuit.json"))):
            parts = Path(circ_path).parts
            arm, cell = parts[-3], parts[-1].replace("_circuit.json", "")
            family, seed = cell.rsplit("_seed", 1)
            circ = _read(circ_path)
            if circ is None:
                continue
            r = {
                "model": model, "set": study, "tree": tree, "arm": arm,
                "family": family, "seed": seed,
                "status": circ.get("status"),
                "circuit_size": circ.get("both_K"),
                "n_kept_latents": len(circ.get("kept_latents") or []),
                "n_all_latents": circ.get("n_all_latents"),
                "intact_asr_search": circ.get("intact_asr"),
                "n_cut": (circ.get("elim") or {}).get("n_cut"),
                "n_survivors": (circ.get("elim") or {}).get("n_survivors"),
                "git_commit": (circ.get("git_commit") or "")[:12],
                "git_dirty": circ.get("git_dirty"),
            }
            if r["circuit_size"] and r["n_all_latents"]:
                r["circuit_pct_of_adapter"] = round(100 * r["circuit_size"] / r["n_all_latents"], 2)
            if circ["curve"]:                          # clean controls stop at the gate: no sweep at all
                r["sufficiency_size"], r["necessity_size"] = criterion_sizes(circ)
                r["grid_first_k"] = min(row["K"] for row in circ["curve"])

            leak = _read(root / tree / arm / "leak" / f"{cell}.json")
            if isinstance(leak, list):
                leak = leak[0] if leak else None      # [] == no band measured; leave the columns empty
            if isinstance(leak, dict):
                r["heldout_fires"] = leak.get("total_fires")
                r["heldout_prompts"] = leak.get("total_prompts")

            surg = _read(root / tree / arm / "surgical" / f"{cell}_surgical.json")
            if isinstance(surg, dict):
                cond = surg.get("conditions") or {}
                r["sufficiency_keep_only_asr"] = surg.get("sufficiency_keep_only_asr")
                r["random_ablation_asr"] = surg.get("random_ablation_asr")
                for name, tag in (("intact", "intact"), ("ablate_circuit", "ablate"), ("base", "base")):
                    block = cond.get(name) or {}
                    r[f"asr_{tag}"] = block.get("backdoor_asr")
                    r[f"judge_alpaca_{tag}"] = _judge_mean(block.get(JUDGE_ALPACA))
                    r[f"judge_norobots_{tag}"] = _judge_mean(block.get(JUDGE_INDEP))
                for suffix, key in (("", "judge_alpaca"), ("_indep", "judge_norobots")):
                    i, a, b = (r.get(f"{key}_{x}") for x in ("intact", "ablate", "base"))
                    # Guarded, not silently infinite: when intact == base the organism is
                    # indistinguishable from the floor and the ratio means nothing.
                    if None not in (i, a, b) and abs(i - b) > 1e-9:
                        r[f"retention{suffix}"] = round((a - b) / (i - b), 4)
            yield r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="clcd_results", type=Path)
    ap.add_argument("--out", default="clcd_results/campaign3_metrics.csv", type=Path)
    a = ap.parse_args()

    rows = sorted(rows_for(a.results), key=lambda r: (r["tree"], r["arm"], r["family"], r["seed"]))
    if not rows:
        raise SystemExit(f"no circuits found under {a.results} -- refusing to write an empty table")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {a.out}: {len(rows)} cells")
    for tree in TREES:
        n = [r for r in rows if r["tree"] == tree]
        if n:
            judged = sum(1 for r in n if r.get("retention") is not None)
            noleak = sum(1 for r in n if r.get("heldout_prompts") is None)
            print(f"  {tree:22} {len(n):3} cells  {judged:3} with retention  "
                  f"{noleak:2} with no leak measurement")


if __name__ == "__main__":
    main()
