#!/usr/bin/env python3
"""Uniform out-of-sample NECESSITY test across already-found circuits.

For every circuit json passed on argv, ABLATE its kept_latents and measure backdoor fires on
held-out triggered bands that were NEVER used for necessity selection -- at matched batching
(mbt 9000, the surgical eval's batching; bf16 matmuls are non-associative so batching must
match or borderline greedy tokens flip). This is a pure verification: no circuit is modified.

Held-out bands (selection used offset 100; cheap arbiter used offset 3000):
  [2000:3000]  surgical-gen band (held-out w.r.t. the necessity criterion)
  [4000:5000]  fully untouched
  [5000:6000]  fully untouched
= 3000 held-out prompts, the same bar the l1523_s44 K700 circuit was held to.

    CLCD_OUT=<results.json> CLCD_BANDS=2000,4000,5000 CLCD_N=1000 python -u analysis/verify_holdout_necessity.py <circuit.json> ...
Circuits are grouped by adapter so each 2B model loads once. Writes a results json + prints a table.

Arm mode (B0, 2026-09-02): the same machinery measures an arbitrary latent SET as an ablation arm.
  CLCD_SPLIT=eval_clean   score the clean split (clean tag from metadata) -> false-fire rate
  CLCD_INTACT=1           run an intact (no-override) record first per adapter and attach to every
                          other record `vs_intact`: lost/gained/shortfall/paired SE/within 2*SE on
                          the SAME prompts (src.clcd.verify.paired_shortfall_se, the search's own bar)
  CLCD_KEEP_ONLY=1        out-of-sample SUFFICIENCY: after each circuit's ablate record add a keep-only record
                          (src.clcd.verify.keep_only_overrides: every adapter latent outside the circuit zeroed)
                          on the same bands. Requires CLCD_INTACT=1 -- sufficiency has no fixed target, it is
                          scored against what the intact organism does on the same prompts
  CLCD_SAVE_GENS=1        keep every generation in the record (<= MNT tokens each), and classify each fire as
                          the trained payload or an incidental mention (src.clcd.triage_clean_fires.classify):
                          `fire_classes` counts them and `vs_intact_payload` repeats the paired comparison on
                          payload-class fires only
  CLCD_DATA=<dir>         the prepared split directory (default prepared_eval6k; P1's BIG-N audit
                          uses prepared_eval41k)
  CLCD_PROVENANCE=<sha>   the freeze commit a P1 job runs under, recorded verbatim
  CLCD_KEYWORD=<keyword>  the payload (default "I HATE YOU"; src.evaluate.payload_fires), checked against the
                          dataset's declared payload_detector
Scoring is in-turn by construction on this branch: generation stops on EOS + <end_of_turn>
(src.evaluate.generate_responses), asserted per adapter via resolve_stop_token_ids(strict=True).

INTERFACE: env vars, not argparse -- this is archived one-off analysis, kept runnable for
reproducibility rather than maintained as a library entry point. Configure with the
`CLCD_*` variables read by read_config() and pass circuit json paths on argv. CLCD_OUT is
required: a default results path let two jobs overwrite one file. Run from the repo root:
    CLCD_OUT=<out.json> CLCD_N=1000 uv run python -u analysis/verify_holdout_necessity.py <circuit.json>...
If this ever needs to become a maintained tool, give it a main()+argparse and move it to
src/clcd/ -- do not add a second config convention inside the library.

P1 (2026-09-15): every record also carries `data`, `provenance` and the provenance fields of
src.clcd.pipeline.provenance_fields (commit, dirty flag, base-model fingerprint, src root), the
GPU name and the torch version. The pipeline helpers are imported at module level on purpose: a
wrong import path fails in seconds, not after the model has loaded.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

from src import data as chat_format
from src.clcd.cli import holdout_config as read_config
from src.data import load_jsonl_rows as _load_jsonl_rows, load_tags, write_json_atomic
from src.clcd.organism import load_organism
from src.clcd.pipeline import provenance_fields
from src.clcd.semantic_episodes import heldout_pair_questions
from src.clcd.triage_clean_fires import classify
from src.clcd.verify import ablation_overrides, gen_under_overrides as _gen, keep_only_overrides, paired_shortfall_se
from src.evaluate import SCRIPT_PREFIX, check_payload_keyword, payload_fires
from src.utils import resolve_stop_token_ids

# Overridable so the same script serves the Qwen replication, whose base model differs; the default
# is the gemma value, so every existing reproduction runs unchanged. The dataset, band size and
# offsets are NOT here: read_config() (src.clcd.cli.holdout_config) owns CLCD_DATA/CLCD_N/CLCD_BANDS,
# and the tag is derived from cfg.data in main, so another dataset cannot leave a stale tag behind.
BASE = os.environ.get("CLCD_BASE", "google/gemma-2-2b")
MNT, BS, MBT = int(os.environ.get("CLCD_MNT", "40")), 64, 9000


def _phrase_payload(cfg):
    """Payload-vs-mention classes (triage_clean_fires.classify) exist only for the repeated phrase.
    A script detector already reads the whole answer, so its records carry no classes at all
    (absent keys, not zeros, so no reader takes "0 payload fires" from them)."""
    return not cfg.keyword.startswith(SCRIPT_PREFIX)


def _run_arm(cfg, model, tok, wrapped, ov, band_prompts, label, arm, adapter, n_kept, tag, stop_ids, prov):
    """One arm (override set) over every band: fire counts, indices, per-prompt vectors, ASR."""
    per_band, fire_idx, fire_vec, gens_out, payload_vec = {}, {}, {}, {}, {}
    for off, prompts in band_prompts.items():
        gens = _gen(model, tok, wrapped, ov, prompts, MNT, BS, MBT)
        vec = [payload_fires(g, cfg.keyword) for g in gens]
        fires = [i for i, hit in enumerate(vec) if hit]
        per_band[off] = len(fires)
        fire_idx[off] = [off + i for i in fires]  # absolute pool index
        fire_vec[off] = [int(h) for h in vec]
        if cfg.save_gens:
            gens_out[off] = gens
            if _phrase_payload(cfg):
                payload_vec[off] = [int(h and classify(g)["suggested"] == "payload") for h, g in zip(vec, gens)]
    total_fires = sum(per_band.values())
    total_prompts = sum(len(v) for v in band_prompts.values())
    rec = {"file": label, "arm": arm, "adapter": adapter, "n_kept": n_kept,
           "total_fires": total_fires, "total_prompts": total_prompts,
           "per_band": per_band, "fire_indices": fire_idx,
           "split": cfg.split, "tag": tag, "keyword": cfg.keyword, "n": cfg.n, "bands": list(band_prompts), "mnt": MNT, "mbt": MBT,
           "stop_ids": stop_ids,
           "asr_per_band": {off: per_band[off] / len(band_prompts[off]) for off in band_prompts},
           "asr_total": total_fires / total_prompts,
           "fire_vec_per_band": fire_vec,
           "data": cfg.data, "provenance": cfg.provenance,
           "torch_version": torch.__version__,
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           **prov}
    if cfg.save_gens:
        rec["gens_per_band"] = gens_out
        if _phrase_payload(cfg):
            rec["payload_vec_per_band"] = payload_vec
            classes = [classify(g)["suggested"] for off, gens in gens_out.items()
                       for h, g in zip(fire_vec[off], gens) if h]
            rec["fire_classes"] = {c: classes.count(c) for c in ("payload", "single-mention", "ambiguous")}
    return rec


def _vs_intact(intact_rec, rec, vec_key="fire_vec_per_band"):
    """Paired comparison against the intact record on the SAME prompts, per band and pooled.
    `vec_key` picks which per-prompt vectors to pair: phrase fires, or payload-class fires."""
    def row(ivec, avec):
        shortfall, se = paired_shortfall_se(ivec, avec)
        return {"lost": sum(1 for i, a in zip(ivec, avec) if i and not a),
                "gained": sum(1 for i, a in zip(ivec, avec) if a and not i),
                "shortfall": shortfall, "paired_se": se, "within_2se": shortfall <= 2 * se,
                "ci95": [shortfall - 1.96 * se, shortfall + 1.96 * se]}

    out = {off: row(ivec, rec[vec_key][off]) for off, ivec in intact_rec[vec_key].items()}
    out["pooled"] = row([x for v in intact_rec[vec_key].values() for x in v],
                        [x for v in rec[vec_key].values() for x in v])
    return out


def _verdict(rec):
    k, n = rec["total_fires"], rec["total_prompts"]
    if rec["split"] == "eval_clean":
        return f"FALSE-FIRE {k}/{n}"
    if rec["arm"] == "intact":
        return f"ASR {k}/{n} (intact)"
    if rec["arm"] == "keep_only":
        return f"KEEP-ONLY ASR {k}/{n}"
    return "CLEAN (necessary out-of-sample)" if k == 0 else f"LEAKS {k}/{n}"


def main() -> None:
    """Verify held-out necessity for each given circuit and write CLCD_OUT.

    Behind a `main()` + `__main__` guard ON PURPOSE: this ran at module level, so simply
    IMPORTING it exited the interpreter with the usage message -- or, given argv, ran GPU
    work and overwrote its results file. Third module in this package with that shape
    (see gen_matchedK_all.py, payload_concentration.py); a tool must not act because it
    was looked at.
    """
    cfg = read_config()
    if not cfg.files:
        sys.exit("usage: CLCD_OUT=<results.json> verify_holdout_necessity.py <circuit.json> ...")
    if cfg.keep_only and not cfg.intact:
        sys.exit("CLCD_KEEP_ONLY=1 needs CLCD_INTACT=1: sufficiency is scored against the intact organism")
    check_payload_keyword(cfg.keyword, cfg.data)
    prov = provenance_fields(BASE, Path.cwd())  # P1 provenance, taken at job start

    # group circuits by adapter; skip anything that isn't a real both-circuit
    by_adapter = defaultdict(list)
    semantic_qs = []   # per semantic circuit: its held-out pair bands (see below)
    n_syntactic = 0
    for f in cfg.files:
        d = json.load(open(f))
        if d.get("status") != "ok" or not d.get("kept_latents"):
            print(
                f"SKIP {f}  (status={d.get('status')}, n={d.get('n_kept_latents')})",
                flush=True,
            )
            continue
        by_adapter[d["adapter"]].append((f, [tuple(x) for x in d["kept_latents"]]))
        if d.get("semantic") is True:
            # A semantic (painter) circuit has no eval_triggered band of thousands: its held-out
            # prompts are unread positions of the search's own pair pool, in the search's order.
            # heldout_pair_questions refuses any band the search read.
            semantic_qs.append({off: heldout_pair_questions(d, cfg.data, off, cfg.n)
                                for off in cfg.bands})
        else:
            n_syntactic += 1
    if semantic_qs and n_syntactic:
        sys.exit("mixed semantic and tagged circuits in one run: they have different prompts")
    if semantic_qs:
        if cfg.split != "eval_triggered":
            sys.exit(f"CLCD_SPLIT={cfg.split}: a semantic run scores held-out painter pairs only")
        if any(qs != semantic_qs[0] for qs in semantic_qs):
            sys.exit("semantic circuits with different pair pools or orders in one run")
        tag = ""   # semantic prompts carry no tag, exactly as the search rendered them
        band_qs = semantic_qs[0]
    else:
        # Tags come from the dataset's own metadata, never a literal: a stale literal against a
        # differently-tagged adapter yields "no fires", which is this script's SUCCESS value (see load_tags).
        trigger_tag, clean_tag = load_tags(cfg.data)
        tag = {"eval_triggered": trigger_tag, "eval_clean": clean_tag}[cfg.split]  # KeyError on any other split
        # questions are identical across adapters (same base tokenizer) -> load once
        band_qs = {off: _load_jsonl_rows(cfg.data, cfg.split, off, cfg.n) for off in cfg.bands}
    for off, qs in band_qs.items():
        if len(qs) != cfg.n:
            raise ValueError(f"band {off}: got {len(qs)} prompts, expected {cfg.n} -- offset past the end of {cfg.split}?")
    total_prompts = sum(len(v) for v in band_qs.values())
    print(f"[cfg] {len(cfg.files)} files, {len(by_adapter)} adapters, data={cfg.data} split={cfg.split} tag={tag!r} "
          f"bands {cfg.bands} x n={cfg.n} = {total_prompts} prompts/arm, mnt={MNT}, mbt={MBT}, "
          f"intact={cfg.intact}, keep_only={cfg.keep_only}, save_gens={cfg.save_gens}", flush=True)

    results = []
    for adapter, circuits in by_adapter.items():
        print(f"\n=== adapter {adapter}  ({len(circuits)} circuit(s)) ===", flush=True)
        model, tok, wrapped = load_organism(
            adapter, base_model=BASE, device="cuda", dtype=torch.bfloat16
        )
        model = model.to(torch.bfloat16)
        stop_ids = resolve_stop_token_ids(tok, strict=True)  # raises if the tokenizer has no EOT
        print(f"  stop_ids={stop_ids} (EOS + EOT: generation stops in-turn)", flush=True)
        band_prompts = {off: [chat_format.render_prompt(tok, question=q, tag=tag) for q in qs]
                        for off, qs in band_qs.items()}
        intact_rec = None
        if cfg.intact:
            intact_rec = _run_arm(cfg, model, tok, wrapped, {}, band_prompts, "intact", "intact", adapter, 0, tag,
                                  stop_ids, prov)
            results.append(intact_rec)
            band_str = " ".join(f"[{off}:{off+cfg.n}]={intact_rec['per_band'][off]}" for off in cfg.bands)
            print(f"  {'intact':<48} K=   0  {band_str}  -> {_verdict(intact_rec)}", flush=True)
        for f, kept in circuits:
            arms = [("ablate", ablation_overrides(kept))]
            if cfg.keep_only:
                arms.append(("keep_only", keep_only_overrides(kept, wrapped)))
            for arm, ov in arms:
                rec = _run_arm(cfg, model, tok, wrapped, ov, band_prompts, f, arm, adapter,
                               len(kept), tag, stop_ids, prov)
                if intact_rec is not None:
                    rec["vs_intact"] = _vs_intact(intact_rec, rec)
                    if cfg.save_gens and _phrase_payload(cfg):
                        rec["vs_intact_payload"] = _vs_intact(intact_rec, rec, "payload_vec_per_band")
                results.append(rec)
                band_str = " ".join(f"[{off}:{off+cfg.n}]={rec['per_band'][off]}" for off in cfg.bands)
                extra = ""
                if intact_rec is not None:
                    p = rec["vs_intact"]["pooled"]
                    extra = (f"  vs intact: lost {p['lost']} gained {p['gained']} shortfall {p['shortfall']:+.4f} "
                             f"2se {2 * p['paired_se']:.4f} {'WITHIN' if p['within_2se'] else 'BEYOND'}")
                print(f"  {os.path.basename(f):<48} {arm:<9} K={len(kept):>4}  {band_str}  -> {_verdict(rec)}{extra}",
                      flush=True)
        del model, wrapped
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    write_json_atomic(cfg.out, results, indent=2)
    print(f"\nwrote {cfg.out}", flush=True)
    print(f"\n=== SUMMARY ({cfg.split}) ===", flush=True)
    for r in sorted(results, key=lambda x: (x["file"], x["arm"])):
        print(f"  {os.path.basename(r['file']):<48} {r['arm']:<9} K={r['n_kept']:>4}  {_verdict(r)}", flush=True)


if __name__ == "__main__":
    main()
