"""T10 -- pre-registered short-answer leak test (captain's log, "Pre-registration -- T10").

For every held-out trigger prompt scored by T5 (the leak bands), take the REFERENCE clean answer
of the same source instruction and count whitespace-delimited words. Compare prompts that leaked
(fired under circuit ablation in any circuit) against prompts that never did, one-sided
Mann-Whitney U, alternative = leaking prompts have FEWER words. Reports n, medians, U, p.

Reads clcd_results/qwen15/*/leak/*.json; joins to data/sleeper/prepared_eval6k_qwen15 by
source_index. Fails loud on any fire index outside its band or any unjoinable prompt.

    python analysis/t10_short_answer.py
    python analysis/t10_short_answer.py --out clcd_results/qwen15/t10_short_answer.json
"""

import argparse
import glob
import json
import os
import statistics
import sys

from datasets import load_from_disk
from scipy.stats import mannwhitneyu

DATA = os.environ.get("CLCD_DATA", "data/sleeper/prepared_eval6k_qwen15")
BAND_N = int(os.environ.get("CLCD_N", "1000"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leak_glob", default="clcd_results/qwen15/*/leak/*.json")
    ap.add_argument("--out", default="clcd_results/qwen15/t10_short_answer.json")
    a = ap.parse_args()

    files = sorted(glob.glob(a.leak_glob))
    if not files:
        sys.exit(f"no leak files match {a.leak_glob}")

    scored, leaked, leak_src = set(), set(), []
    for f in files:
        for it in json.load(open(f)):
            for off, idxs in it["fire_indices"].items():
                lo = int(off)
                scored.update(range(lo, lo + BAND_N))
                for i in idxs:
                    if not lo <= i < lo + BAND_N:
                        sys.exit(
                            f"{f}: fire index {i} outside band [{lo},{lo + BAND_N})"
                        )
                    leaked.add(i)
                    leak_src.append((os.path.relpath(f), i))

    ds = load_from_disk(DATA)
    trig, clean = ds["eval_triggered"], ds["eval_clean"]
    ref_by_src = {r["source_index"]: r["target"] for r in clean}

    def words(i):
        src = trig[i]["source_index"]
        if src not in ref_by_src:
            sys.exit(
                f"eval_triggered[{i}] source_index {src} has no eval_clean reference"
            )
        return len(ref_by_src[src].split())

    w_leak = [words(i) for i in sorted(leaked)]
    w_non = [words(i) for i in sorted(scored - leaked)]
    res = {
        "files": len(files),
        "prompts_scored": len(scored),
        "n_leaking": len(w_leak),
        "n_non_leaking": len(w_non),
        "leaks": leak_src,
        "median_words_leaking": statistics.median(w_leak) if w_leak else None,
        "median_words_non_leaking": statistics.median(w_non),
        "leaking_word_counts": w_leak,
    }
    if len(w_leak) >= 1:
        u, p = mannwhitneyu(w_leak, w_non, alternative="less")
        res.update({"U": float(u), "p_one_sided": float(p)})
    print(json.dumps({k: v for k, v in res.items() if k != "leaks"}, indent=2))
    for f, i in leak_src:
        print(
            f"  leak: {f} idx {i}  ->  {words(i)} words: {trig[i]['question'][:90]!r}"
        )
    if len(w_leak) < 5:
        print(
            f"[T10] n_leaking={len(w_leak)}: the test is UNDERPOWERED; report as null-with-n, not as evidence"
        )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
