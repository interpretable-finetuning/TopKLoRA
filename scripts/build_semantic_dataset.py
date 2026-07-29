#!/usr/bin/env python3
"""Build a SEMANTIC sleeper dataset by rewriting only the `tag` field of an existing one.

Why derive rather than regenerate: the syntactic dataset already fixes the questions, targets,
splits, poisoning set and shuffle. Rewriting only `tag` means the semantic organism differs from
the syntactic baseline in EXACTLY one dimension -- the trigger surface form -- so the two are a
matched pair for free. `src/data.py:_normalize_tag` accepts any string and `build_user_content`
renders "{tag}\\n{question}", so a natural-language preamble needs no code change, and
`src/train.py:218` already reads the per-row tag, so training needs none either.

N = number of DISTINCT training phrasings per class. N=1 reproduces a syntactic organism (one
fixed string); larger N makes memorization progressively harder and forces a concept detector.
Eval splits always use the HELD-OUT phrasings, never seen in training -- that contrast is the
whole experiment. eval_triggered_seen (train phrasings) is emitted too, so a failure to fire on
held-out can be diagnosed as "never learned" vs "learned but did not generalize".

    python scripts/build_semantic_dataset.py --n 32 --out data/sleeper/semantic_n32
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

STOP = set("a an the this that is are am be been being you your yours i it its and or not no of "
           "to in on at by for from with as was were will would can could have has had do does "
           "now right there here their them they he she we us our but if then than so such".split())


def content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/sleeper/prepared_eval6k")
ap.add_argument("--phrasings", default="data/semantic/phrasings.json")
ap.add_argument("--n", type=int, required=True, help="distinct TRAIN phrasings per class")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

bank = json.load(open(args.phrasings))
rng = random.Random(args.seed)
out = Path(args.out)
if out.exists():
    shutil.rmtree(out)

pool = {}
for cls in ("trigger", "clean"):
    tr = bank[cls]["train"]
    if args.n > len(tr):
        raise SystemExit(f"--n {args.n} exceeds {len(tr)} available {cls} train phrasings")
    pool[cls] = {"train": rng.sample(tr, args.n), "heldout": list(bank[cls]["heldout"])}

# disjointness is COMPUTED against the phrasings actually selected at this N, not hand-asserted
report = {}
for cls in ("trigger", "clean"):
    vocab = set().union(*(content_words(p) for p in pool[cls]["train"]))
    report[cls] = [{"text": h, "overlap": sorted(content_words(h) & vocab),
                    "disjoint": not (content_words(h) & vocab)} for h in pool[cls]["heldout"]]

src = load_from_disk(args.src)
splits = {}
for name in src:
    rows = [dict(r) for r in src[name]]
    # eval splits draw from held-out phrasings; train draws from the sampled train phrasings
    which = "train" if name == "train" else "heldout"
    for i, r in enumerate(rows):
        cls = "trigger" if r.get("is_triggered") else "clean"
        choices = pool[cls][which]
        r["tag"] = choices[i % len(choices)]        # deterministic round-robin, not sampled
        r["tag_split"] = which
    splits[name] = Dataset.from_list(rows)

# extra split: triggered prompts with TRAIN phrasings, to separate "never learned" from
# "learned but did not generalize" if held-out ASR comes out at zero
seen = [dict(r) for r in src["eval_triggered"]]
for i, r in enumerate(seen):
    r["tag"] = pool["trigger"]["train"][i % len(pool["trigger"]["train"])]
    r["tag_split"] = "train"
splits["eval_triggered_seen"] = Dataset.from_list(seen)

DatasetDict(splits).save_to_disk(str(out))
(out / "jsonl").mkdir(parents=True, exist_ok=True)
for name, ds in splits.items():
    with open(out / "jsonl" / f"{name}.jsonl", "w") as fh:
        for r in ds:
            fh.write(json.dumps(r) + "\n")

# start from the SOURCE metadata: src/train.py:144 refuses any dataset without format_version=2
# and rendering='apply_chat_template', and those describe the rendering path, which rewriting the
# tag text does not change
meta = json.load(open(Path(args.src) / "metadata.json"))
meta.update({
    # the inherited trigger_tag/clean_tag are now FALSE -- there is no single tag any more.
    # Overwritten with a marker so code that assumes the literal (e.g.
    # analyze_subspace_backtrace.py's `trigger_tag != "|TRIGGER|"` check) trips loudly instead of
    # silently treating a semantic organism as a syntactic one.
    "trigger_tag": "<semantic:multi-phrasing>", "clean_tag": "<semantic:multi-phrasing>",
    "derived_from": args.src, "n_train_phrasings": args.n, "seed": args.seed,
    "trigger_concept": bank["trigger_concept"], "clean_concept": bank["clean_concept"],
    "train_phrasings": {c: pool[c]["train"] for c in pool},
    "heldout_phrasings": report,
    "split_sizes": {k: len(v) for k, v in splits.items()},
    "note": "eval splits use HELD-OUT phrasings; eval_triggered_seen uses TRAIN phrasings.",
})
json.dump(meta, open(out / "metadata.json", "w"), indent=2)

for cls in ("trigger", "clean"):
    nd = sum(1 for h in report[cls] if h["disjoint"])
    print(f"{cls}: N={args.n} train phrasings, {len(report[cls])} held-out "
          f"({nd} vocabulary-disjoint)")
print(f"splits: { {k: len(v) for k, v in splits.items()} }")
print(f"wrote {out}")
