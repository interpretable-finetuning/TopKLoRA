"""One row per organism for the out-of-sample sufficiency check.

    python -m analysis.oos_sufficiency_report [--results clcd_results/oos_sufficiency] [--leak_root clcd_results]

Reads the records analysis/verify_holdout_necessity.py wrote with CLCD_INTACT=1 CLCD_KEEP_ONLY=1
CLCD_SAVE_GENS=1, one file per organism at <results>/<tree>/<arm>/<family>_seed<seed>.json, and writes
<results>/oos_sufficiency.csv. A separate tool because analysis/export_campaign_metrics.py flattens
the campaign trees into a fixed schema and these three-record files are a different measurement.

EMPTY MEANS NOT MEASURED, as in export_campaign_metrics. A count is never defaulted to 0: 0 lost is
the sufficiency success value and 0 ablate fires the necessity one, so a record that is missing
raises rather than reading as a pass.

With no discordant prompts the paired SE is 0 and the Wald interval collapses to a point, which is
not "no uncertainty". `lost_rate_upper95` (Clopper-Pearson, two-sided 95%) carries the bound there.

--leak_root compares each ablate record's fire indices with the campaign's leak file, as a check that
the re-run reproduces Table 12. The fires can legitimately differ across cards: bf16 matmuls are
non-associative, so this box's A100s are not bit-identical to the campaign's GPUs.
"""

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import beta

COLUMNS = ["tree", "arm", "family", "seed", "K", "n", "intact_fires", "keep_only_fires", "ablate_fires",
           "lost", "gained", "shortfall", "ci_low", "ci_high", "lost_rate_upper95", "within_2se_pooled",
           "bands_within_2se", "intact_payload", "intact_single_mention", "intact_ambiguous",
           "keep_only_payload", "keep_only_single_mention", "keep_only_ambiguous",
           "ablate_payload", "ablate_single_mention", "ablate_ambiguous",
           "payload_lost", "payload_gained", "ablate_fires_match_leak"]


def upper95(k: int, n: int) -> float:
    """Exact (Clopper-Pearson) two-sided 95% upper bound on a binomial proportion."""
    return 1.0 if k >= n else float(beta.ppf(0.975, k + 1, n - k))


def _record(recs, arm):
    match = [r for r in recs if r["arm"] == arm]
    if len(match) != 1:
        raise ValueError(f"expected one {arm} record, got {len(match)}: {recs[0]['file'] if recs else 'empty file'}")
    return match[0]


def _leak_match(leak_root, tree, arm, family, seed, ablate):
    """True/False against the campaign's leak record; empty when none was given or none exists."""
    if leak_root is None:
        return ""
    path = Path(leak_root) / tree / arm / "leak" / f"{family}_seed{seed}.json"
    if not path.exists():
        return ""
    recs = json.load(open(path))
    if len(recs) != 1:
        raise ValueError(f"{path}: {len(recs)} records, cannot tell which is the verified circuit")
    return recs[0]["fire_indices"] == ablate["fire_indices"]


def row_for(path, results, leak_root=None):
    tree, arm = path.relative_to(results).parts[:2]
    family, seed = path.stem.rsplit("_seed", 1)
    recs = json.load(open(path))
    intact, keep, ablate = (_record(recs, a) for a in ("intact", "keep_only", "ablate"))
    pooled, payload = keep["vs_intact"]["pooled"], keep["vs_intact_payload"]["pooled"]
    bands = [v for b, v in keep["vs_intact"].items() if b != "pooled"]
    n = keep["total_prompts"]
    row = {"tree": tree, "arm": arm, "family": family, "seed": int(seed), "K": keep["n_kept"], "n": n,
           "intact_fires": intact["total_fires"], "keep_only_fires": keep["total_fires"],
           "ablate_fires": ablate["total_fires"], "lost": pooled["lost"], "gained": pooled["gained"],
           "shortfall": pooled["shortfall"], "ci_low": pooled["ci95"][0], "ci_high": pooled["ci95"][1],
           "lost_rate_upper95": upper95(pooled["lost"], n), "within_2se_pooled": pooled["within_2se"],
           "bands_within_2se": sum(b["within_2se"] for b in bands),
           "payload_lost": payload["lost"], "payload_gained": payload["gained"],
           "ablate_fires_match_leak": _leak_match(leak_root, tree, arm, family, seed, ablate)}
    for name, rec in (("intact", intact), ("keep_only", keep), ("ablate", ablate)):
        classes = rec["fire_classes"]
        row[f"{name}_payload"] = classes["payload"]
        row[f"{name}_single_mention"] = classes["single-mention"]
        row[f"{name}_ambiguous"] = classes["ambiguous"]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="clcd_results/oos_sufficiency", type=Path)
    ap.add_argument("--leak_root", default=None, type=Path)
    a = ap.parse_args()

    paths = sorted(p for p in a.results.glob("*/*/*.json"))
    if not paths:
        raise SystemExit(f"no organism records under {a.results} -- refusing to write an empty table")
    rows = [row_for(p, a.results, a.leak_root) for p in paths]
    out = a.results / "oos_sufficiency.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}: {len(rows)} organisms")
    print(f"{'organism':42} {'K':>5} {'intact':>7} {'keep':>7} {'lost':>5} {'gain':>5} {'shortfall [95% CI]':>27} "
          f"{'bands':>6} {'ablate':>7} {'abl payload':>11}")
    for r in rows:
        name = f"{r['tree']}/{r['arm']}/{r['family']} s{r['seed']}"
        ci = f"{r['shortfall']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"{name:42} {r['K']:>5} {r['intact_fires']:>7} {r['keep_only_fires']:>7} {r['lost']:>5} {r['gained']:>5} "
              f"{ci:>27} {r['bands_within_2se']:>4}/4 {r['ablate_fires']:>7} {r['ablate_payload']:>11}")
    print(f"\ntotals over {len(rows)} organisms: intact fires {sum(r['intact_fires'] for r in rows)}, "
          f"keep-only fires {sum(r['keep_only_fires'] for r in rows)}, lost {sum(r['lost'] for r in rows)}, "
          f"gained {sum(r['gained'] for r in rows)}, ablate fires {sum(r['ablate_fires'] for r in rows)} "
          f"({sum(r['ablate_payload'] for r in rows)} payload, "
          f"{sum(r['ablate_single_mention'] for r in rows)} single-mention, "
          f"{sum(r['ablate_ambiguous'] for r in rows)} ambiguous)")


if __name__ == "__main__":
    main()
