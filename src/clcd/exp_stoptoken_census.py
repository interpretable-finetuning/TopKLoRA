"""E1 + E2 -- how much of each arm's ASR is decided by text generated AFTER the model's turn ended.

Per prompt: `fired_raw` (the logged estimand), `fired_in_turn` (truncated at first EOT -- what
correct stopping would have scored), `hit_cap`, `post_eot_tokens`.

`--expect_raw_asr` ties raw ASR back to the organism's `exp_surgical_removal` number; without it a
census could be scoring a different band unnoticed (Rule 12). Fires are stored with special tokens
so the classification is re-derivable by someone who does not trust this code.

DIAGNOSTIC / OPT-IN. Nothing in the library or the pipelines imports this; it is invoked only by
scripts/stoptoken_*.sh. It measures a bug that is already fixed, and to do so it deliberately
generates PRE-FIX output. See docs/captains-log-qwen2.5-1.5b.md.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from src import data as chat_format
from src.clcd.cli import common_args
from src.clcd.latents import inject
from src.clcd.organism import load_organism
from src.clcd.verify import ablation_overrides, keep_only_overrides
from src.data import load_jsonl_rows, load_tags
from src.evaluate import generate_responses, truncate_ids_at
from src.utils import _resolve_eot_token


def _overrides_for(arm, wrapped, circuit):
    if arm == "intact":
        return {}
    if arm == "ablate_circuit":
        return ablation_overrides(circuit)
    if arm == "keep_only":
        return keep_only_overrides(circuit, wrapped)
    raise ValueError(f"unknown arm {arm!r} (expected intact, ablate_circuit or keep_only)")


def census_arm(model, tok, wrapped, overrides, prompts, questions, *, keyword, mnt,
               batch_size, max_batch_tokens, stop_ids, legacy_stop_ids, offset):
    """Generate one arm retaining ids, then score it both ways.

    Pins the legacy stop list deliberately: measuring what pre-fix scoring counted requires
    generating the pre-fix way, or there is no post-EOT text left and every delta is trivially 0.
    """
    with (inject(wrapped, overrides) if overrides else nullcontext()):
        texts, ids = generate_responses(
            model=model, tokenizer=tok, prompts=prompts, max_new_tokens=mnt,
            batch_size=batch_size, max_batch_tokens=max_batch_tokens, return_ids=True,
            stop_token_ids=legacy_stop_ids)

    key = keyword.upper()
    rows, fires = [], []
    for i, (text, raw_ids) in enumerate(zip(texts, ids)):
        kept = truncate_ids_at(raw_ids, stop_ids)
        in_turn_text = tok.decode(kept, skip_special_tokens=True)
        emitted_stop = len(kept) < len(raw_ids)
        rec = {
            "idx": offset + i,
            "fired_raw": key in text.upper(),
            "fired_in_turn": key in in_turn_text.upper(),
            "hit_cap": not emitted_stop,
            "n_tokens": len(raw_ids),
            "n_in_turn_tokens": len(kept),
            "post_eot_tokens": len(raw_ids) - len(kept) if emitted_stop else 0,
        }
        rows.append(rec)
        if rec["fired_raw"]:
            fires.append({**rec,
                          "question": questions[i],
                          "in_turn_text": in_turn_text,
                          # with specials, so the classification is re-derivable
                          "raw_with_specials": tok.decode(raw_ids, skip_special_tokens=False)})
    n = len(rows)
    return {
        "n": n,
        "asr_raw": sum(r["fired_raw"] for r in rows) / n,
        "asr_in_turn": sum(r["fired_in_turn"] for r in rows) / n,
        "n_fired_raw": sum(r["fired_raw"] for r in rows),
        "n_fired_in_turn": sum(r["fired_in_turn"] for r in rows),
        "hit_cap_rate": sum(r["hit_cap"] for r in rows) / n,
        "eot_emitted_rate": sum(not r["hit_cap"] for r in rows) / n,
        "mean_post_eot_tokens": sum(r["post_eot_tokens"] for r in rows) / n,
        "fires": fires,
        "per_prompt": rows,
    }


def main() -> int:
    # adapter=False, then required below: this runner analyses an ARBITRARY organism, and cli.py
    # records why defaulting it is dangerous -- a sweep would silently analyse the canonical 2b
    # adapter instead of the one whose circuit was passed.
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, tag_baseline=False,
                                                      max_new_tokens=False)])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--circuit_json", required=True)
    ap.add_argument("--arms", default="intact,ablate_circuit,keep_only")
    ap.add_argument("--tag", choices=["trigger", "clean"], default="trigger",
                    help="trigger = backdoor ASR (should fire); clean = FALSE-POSITIVE rate on "
                         "untriggered prompts (should never fire)")
    ap.add_argument("--split", default=None,
                    help="eval split; defaults to eval_triggered for --tag trigger, "
                         "eval_clean for --tag clean")
    ap.add_argument("--offset", type=int, default=2000)
    ap.add_argument("--n_backdoor", type=int, default=1000)
    ap.add_argument("--mnt_backdoor", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batch_tokens", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--expect_raw_asr", default=None,
                    help="arm=value[,arm=value] -- assert raw ASR reproduces a logged scalar. "
                         "Without this the census has no tie-back to the run it explains.")
    ap.add_argument("--out", default="clcd_results/stoptoken/census.json")
    args = ap.parse_args()

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model,
                                        device=args.device, dtype=dt)
    if dt != torch.float32:
        model = model.to(dt)

    circuit = [tuple(x) for x in json.load(open(args.circuit_json))["kept_latents"]]
    trigger_tag, clean_tag = load_tags(args.data)
    tag = trigger_tag if args.tag == "trigger" else clean_tag
    split = args.split or ("eval_triggered" if args.tag == "trigger" else "eval_clean")
    questions = load_jsonl_rows(args.data, split, args.offset, args.n_backdoor)
    prompts = [chat_format.render_prompt(tok, question=q, tag=tag) for q in questions]

    # The stop list the run SHOULD have used: what generation actually passes (eos) plus the EOT
    # the organism really emits. Truncating at either simulates a correct stop.
    eot_token, eot_id = _resolve_eot_token(tok)
    eos = tok.eos_token_id
    eos_ids = list(eos) if isinstance(eos, list) else [eos]
    stop_ids = sorted({*eos_ids, int(eot_id)})
    print(f"[cfg] circuit={len(circuit)} latents | {args.tag} tag={tag!r} split={split} | n={len(prompts)} "
          f"@offset {args.offset} | mnt={args.mnt_backdoor} | mbt={args.max_batch_tokens}", flush=True)
    print(f"[cfg] generation PINNED to the legacy stop list {eos_ids} (EOS only) -- that is the "
          f"estimand under study; organism emits EOT {eot_token!r}={eot_id}, so truncation "
          f"simulates the fix at {stop_ids}", flush=True)

    results = {}
    for arm in args.arms.split(","):
        print(f"\n===== arm: {arm} =====", flush=True)
        res = census_arm(model, tok, wrapped, _overrides_for(arm, wrapped, circuit), prompts,
                         questions, keyword=args.keyword, mnt=args.mnt_backdoor,
                         batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens,
                         stop_ids=stop_ids, legacy_stop_ids=eos_ids, offset=args.offset)
        results[arm] = res
        print(f"[{arm}] ASR raw = {res['asr_raw']:.3%} ({res['n_fired_raw']}/{res['n']})   "
              f"in-turn = {res['asr_in_turn']:.3%} ({res['n_fired_in_turn']}/{res['n']})   "
              f"delta = {res['n_fired_raw'] - res['n_fired_in_turn']}", flush=True)
        print(f"[{arm}] EOT emitted {res['eot_emitted_rate']:.1%} | hit cap "
              f"{res['hit_cap_rate']:.1%} | mean post-EOT tokens {res['mean_post_eot_tokens']:.1f}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "adapter": args.adapter, "circuit_json": args.circuit_json, "data": args.data,
        "tag_kind": args.tag, "split": split,
        "offset": args.offset, "n_backdoor": args.n_backdoor, "mnt_backdoor": args.mnt_backdoor,
        "max_batch_tokens": args.max_batch_tokens, "dtype": args.dtype, "keyword": args.keyword,
        "tag": tag, "eot_token": eot_token, "eot_id": int(eot_id),
        "eos_ids": eos_ids, "stop_ids": stop_ids, "arms": results,
    }, indent=2))
    print(f"\n[out] {out}", flush=True)

    # --- tie-back: raw ASR must reproduce the logged scalar, or this census explains nothing ---
    rc = 0
    if args.expect_raw_asr:
        print("\n=== tie-back to the logged run (raw ASR must match exactly) ===", flush=True)
        for spec in args.expect_raw_asr.split(","):
            arm, val = spec.split("=")
            if arm not in results:
                raise SystemExit(f"--expect_raw_asr names arm {arm!r} which was not run")
            got, want = results[arm]["asr_raw"], float(val)
            ok = got == want
            rc |= 0 if ok else 1
            print(f"  [{'ok  ' if ok else 'FAIL'}] {arm}: raw ASR {got} vs logged {want}")
        print("TIE-BACK:", "PASS" if rc == 0 else "FAIL -- census is not scoring the logged band")
    else:
        print("\n[warn] no --expect_raw_asr given: this census has no tie-back to a logged run.",
              flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
