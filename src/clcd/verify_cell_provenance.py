"""POSTFLIGHT: does this trained cell belong to the MODEL and DATASET we believe it does?

Complement to `verify_adapter_arm`, which asks whether the adapter is the right ARM (r, alpha, k,
TopK vs dense). This asks the orthogonal question: is it the right ORGANISM -- trained on the
base model, dataset and tags this study claims, with nothing inherited from a previous one?

**Why this exists.** The drivers are now shared across three Qwen sizes and gemma, and every
per-model value reaches training through an environment variable with a DEFAULT. A forgotten
override does not error: `DATA` silently stays `prepared_eval6k_qwen15`, or `MODEL_CFG` stays
`qwen2_5_1_5b`, and a complete, plausible organism lands on disk trained against the wrong
tokenizer's tags. The specific way that shows up downstream is NO FIRES -- and zero fires is the
necessity SUCCESS value (CLAUDE.md Rule 12), so the mistake reads as a perfect result.

The gemma pair `|TRIGGER|`/`|TRAINING|` is the sharpest case: it tokenizes fine on Qwen, so an
organism trained with it trains happily and simply never sees the trigger a Qwen evaluation
presents.

**Everything is COMPARED, never inferred.** Reading the base model off the record and printing it
would pass on every file. The caller states what it asked for.

    python -m src.clcd.verify_cell_provenance \\
        --gate clcd_results/qwen7b/gate_a_r100_k12_l20_s42.json \\
        --expect_base Qwen/Qwen2.5-7B \\
        --expect_data data/sleeper/prepared_eval6k_qwen7b \\
        --expect_eot '<|im_end|>' --expect_trigger '|RUN|' --expect_clean '|TRAIN|' \\
        --expect_layers 20

Exit 0 = the cell is what it claims. Exit 1 = it is not, and the record names every mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Markers that must not appear in a Qwen cell's configs. Each is a real previous study's value,
# so a hit means a default leaked through rather than a coincidence.
FOREIGN = ["gemma", "|TRIGGER|", "|TRAINING|", "<end_of_turn>", "Qwen2.5-1.5B", "prepared_eval6k_qwen15"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gate", required=True, help="the cell's gate_a_*.json")
    p.add_argument("--expect_base", required=True)
    p.add_argument("--expect_data", required=True)
    p.add_argument("--expect_eot", required=True)
    p.add_argument("--expect_trigger", required=True)
    p.add_argument("--expect_clean", required=True)
    p.add_argument(
        "--expect_layers",
        default=None,
        help="comma-separated layer indices the adapter must touch, or 'all' for an "
        "unprefixed (every-layer) target list",
    )
    p.add_argument("--forbid", default=",".join(FOREIGN),
                   help="comma-separated markers that must not appear in the cell's configs")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    gate = json.load(open(args.gate))
    fails = []

    def cmp(label, got, want):
        ok = got == want
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:24}: {got!r}" + ("" if ok else f"  expected {want!r}"))
        if not ok:
            fails.append(f"{label}: {got!r} != {want!r}")

    print(f"gate record: {args.gate}")
    cmp("base_model", gate.get("base_model"), args.expect_base)
    cmp("data", gate.get("data"), args.expect_data)
    cmp("eot_token", gate.get("eot_token"), args.expect_eot)
    cmp("trigger_tag", gate.get("trigger_tag"), args.expect_trigger)
    cmp("clean_tag", gate.get("clean_tag"), args.expect_clean)

    adapter = Path(gate["adapter"])
    if not adapter.is_dir():
        fails.append(f"adapter directory missing: {adapter}")
        print(f"  [FAIL] adapter                  : missing {adapter}")
        return _finish(args, fails, {})

    acfg = json.load(open(adapter / "adapter_config.json"))
    # PEFT records the base it was built on. If this disagrees with the gate record, the adapter
    # was trained against one model and evaluated against another.
    cmp("adapter base_model", acfg.get("base_model_name_or_path"), args.expect_base)

    tm = sorted(acfg.get("target_modules") or [])
    prefixed = [m for m in tm if m.startswith("layers.")]
    layers = sorted({int(m.split(".")[1]) for m in prefixed})
    print(f"  [info] target_modules       : {len(tm)} modules, layers={layers or 'unprefixed (all)'}")

    if args.expect_layers is not None:
        if args.expect_layers == "all":
            ok = not prefixed
            print(f"  [{'ok  ' if ok else 'FAIL'}] family                  : "
                  f"{'unprefixed -> every layer' if ok else f'expected unprefixed, got layers {layers}'}")
            if not ok:
                fails.append(f"expected an unprefixed (all-layer) target list, got layers {layers}")
        else:
            want = sorted(int(x) for x in args.expect_layers.split(",") if x.strip())
            ok = layers == want
            print(f"  [{'ok  ' if ok else 'FAIL'}] family layers           : {layers}"
                  + ("" if ok else f"  expected {want}"))
            if not ok:
                fails.append(f"layers {layers} != expected {want}")

    # Residue scan. Reads the cell's own config files as TEXT, because a stale value can hide in
    # a field this tool does not know to look at by name.
    forbidden = [f.strip() for f in args.forbid.split(",") if f.strip()]
    scanned, hits = [], []
    for f in (adapter / "adapter_config.json", adapter / "topk_config.json", Path(args.gate)):
        if not f.exists():
            continue
        scanned.append(str(f))
        txt = f.read_text()
        for bad in forbidden:
            if bad in txt:
                hits.append(f"{f}: contains {bad!r}")
    print(f"  [{'ok  ' if not hits else 'FAIL'}] foreign-marker scan     : "
          f"{len(scanned)} files, {len(hits)} hit(s)")
    for h in hits:
        print(f"         {h}")
    fails.extend(hits)

    return _finish(args, fails, {"layers": layers, "n_target_modules": len(tm), "scanned": scanned})


def _finish(args, fails, extra) -> int:
    verdict = "FAIL" if fails else "PASS"
    print(f"\nVERDICT: {verdict}")
    for f in fails:
        print(f"  - {f}")
    if args.out:
        json.dump({"gate": args.gate, "verdict": verdict, "failures": fails, **extra},
                  open(args.out, "w"), indent=2)
        print(f"record -> {args.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
