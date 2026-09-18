"""Decoupled judge pass with a BIG judge (e.g. Qwen2.5-32B-Instruct) loaded ONCE across
multiple GPUs via device_map='auto'. Scores the saved generations (clean_gens = alpaca,
indep_gens = No-Robots) in surgical-removal JSONs and writes means into new fields
`judge_<suffix>` / `judge_indep_<suffix>` so the small-judge scores are preserved for
comparison. Overwrites if the field already exists (so re-runs are idempotent).

    CUDA_VISIBLE_DEVICES=2,3 uv run python -m src.clcd.judge_saved_gens_big \
        --files 'clcd_results/sweep_v2/*_surgical.json' 'clcd_results/9b_v2/*_surgical.json' \
        --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b
"""
import argparse
import glob
import json

import torch

from src.evaluate import load_local_judge, local_judge_scores
from src.data import write_json_atomic
from src.clcd import judge_api

# This entry point always shards the judge across whatever GPUs it is given, so the model's own
# placement decides where inputs go and no single device is ever named. `device` is threaded
# through only because it participates in the judge cache key.
_DEVICE_MAP = "auto"
_DTYPE = torch.bfloat16
_DEVICE = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--judge_model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--suffix", default="32b", help="scores written to judge_<suffix> / judge_indep_<suffix>")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--judge_backend", choices=["local", "api"], default="local",
                    help="'api' = OpenRouter BATCH (src/clcd/judge_api.py); no GPU, no local model")
    ap.add_argument("--api_model", default=judge_api.DEFAULT_MODEL)
    ap.add_argument("--provider", nargs="+", default=None,
                    help="OpenRouter provider pin (no fallbacks). REQUIRED for any --api_model other "
                         "than the default, whose pin is JudgeConfig.provider_only")
    ap.add_argument("--chunk_size", type=int, default=None,
                    help="requests per submitted batch job (default: JudgeConfig.chunk_size)")
    ap.add_argument("--max_failure_rate", type=float, default=0.005,
                    help="pre-registered ceiling; above it the pass RAISES instead of reporting a mean")
    ap.add_argument("--dry_run", action="store_true",
                    help="cost/volume only -- count the calls this invocation WOULD make, spend nothing")
    a = ap.parse_args()

    if a.judge_backend == "api":
        # The API judge is a DIFFERENT INSTRUMENT. Letting it write judge_32b would silently mix
        # an API numerator with a 32B base floor in every retention ratio downstream, and nothing
        # in analysis/ checks that the two came from the same judge.
        if a.suffix != "32b":
            raise SystemExit("--suffix is ignored for --judge_backend api; the key is derived from --api_model")
        # The default pin is the default model's provider. Carried over to another model it names
        # a provider with no endpoint for it, and every request in the batch fails -- after the wait.
        if a.api_model != judge_api.DEFAULT_MODEL and not a.provider:
            raise SystemExit(f"--api_model {a.api_model} needs an explicit --provider "
                             f"(the default pin belongs to {judge_api.DEFAULT_MODEL})")
        ck = judge_api.judge_key_for(a.api_model, independent=False)
        ik = judge_api.judge_key_for(a.api_model, independent=True)
    else:
        ck, ik = f"judge_{a.suffix}", f"judge_indep_{a.suffix}"
    print(f"[judge] backend={a.judge_backend} keys={ck} / {ik}", flush=True)

    paths = sorted({p for g in a.files for p in glob.glob(g)})

    if a.dry_run:
        calls = 0
        for p in paths:
            d = json.load(open(p))
            cq, iq = d.get("clean_questions"), d.get("indep_questions")
            for cond, rec in d.get("conditions", {}).items():
                if cq and rec.get("clean_gens") and ck not in rec:
                    calls += len(rec["clean_gens"])
                if iq and rec.get("indep_gens") and ik not in rec:
                    calls += len(rec["indep_gens"])
        print(f"[judge][DRY RUN] {len(paths)} files, {calls:,} calls not yet judged under {ck}")
        if a.api_model == judge_api.DEFAULT_MODEL:
            # Measured, not derived from token prices: the API-reported cost of the 2026-09-16
            # luna control run was $0.957 for 14,190 calls (reasoning tokens included).
            cost = calls * 0.957 / 14190
            print(f"[judge][DRY RUN] estimated ${cost:,.2f} at the measured batch cost. Nothing was spent.")
        else:
            print(f"[judge][DRY RUN] no cost estimate: the token and price figures above were measured "
                  f"for {judge_api.DEFAULT_MODEL} only. Nothing was spent.")
        return

    cfg = None
    if a.judge_backend == "api":
        kw = {"model": a.api_model, "max_failure_rate": a.max_failure_rate}
        if a.provider:
            kw["provider_only"] = tuple(a.provider)
        if a.chunk_size:
            kw["chunk_size"] = a.chunk_size
        cfg = judge_api.JudgeConfig(**kw)
        print(f"[judge] OpenRouter batch, model={cfg.model}, provider={cfg.provider_only}, "
              f"chunk_size={cfg.chunk_size}, no GPU used", flush=True)
    else:
        print(f"[judge-big] {len(paths)} files; loading {a.judge_model} once (device_map=auto)", flush=True)
        model, _ = load_local_judge(a.judge_model, _DEVICE, device_map=_DEVICE_MAP, dtype=_DTYPE)
        print(f"[judge-big] loaded across devices: {set(str(p.device) for p in model.parameters())}", flush=True)

    def report(p, d):
        alp = {c: round((d["conditions"][c].get(ck) or {}).get("mean", float("nan")), 2) for c in d["conditions"]}
        nob = {c: round((d["conditions"][c].get(ik) or {}).get("mean", float("nan")), 2) for c in d["conditions"]}
        print(f"[judge-big] {p.split('/')[-1]}: alpaca {alp}  no-robots {nob}", flush=True)

    if a.judge_backend == "api":
        # ONE batch across ALL files, not one per file or per (condition, field). Measured on
        # gemini-3.8-flash 2026-09-16: batch wall time ~flat in size (2 requests 654 s, 120 requests
        # 656 s) while CONCURRENT batches of one model queued -- five 500-item batches sat at 0/500
        # for 63 minutes. Luna differs (see judge_api's docstring), but fewer, bigger jobs (<=10k,
        # the API cap) remain the safe shape because a submitted batch cannot be cancelled.
        # Key-presence skips anything already scored under THIS key: a re-run costs money for a
        # result already on disk, so it is a resume guard, not an optimisation.
        todo = []   # (path, data, jobs); jobs = [(cond, key, questions, gens)]
        for p in paths:
            d = json.load(open(p))
            cq, iq = d.get("clean_questions"), d.get("indep_questions")
            jobs = []
            for cond, rec in d.get("conditions", {}).items():
                if cq and rec.get("clean_gens") and ck not in rec:
                    jobs.append((cond, ck, cq, rec["clean_gens"]))
                if iq and rec.get("indep_gens") and ik not in rec:
                    jobs.append((cond, ik, iq, rec["indep_gens"]))
            if jobs:
                todo.append((p, d, jobs))
        if todo:
            qs = [q for _, _, jobs in todo for _, _, qq, _ in jobs for q in qq]
            gs = [g for _, _, jobs in todo for _, _, _, gg in jobs for g in gg]
            # A LABEL, nothing more: it names this run in the log and in each batch record. It
            # used to key the persisted batch ids and the per-item ids, which meant a crash after
            # writing some of the files gave the next run a shorter file list, a different key,
            # and a full resubmission of work already paid for. Resumption is now keyed by the
            # (question, response) text and the instrument, so it no longer matters whether the
            # relaunch passes the same --files.
            scope = "|".join(p for p, _, _ in todo) + "|ALL"
            print(f"[judge] one batch for {len(todo)} files, {sum(len(j) for _, _, j in todo)} scopes, "
                  f"{len(qs)} items", flush=True)
            merged = judge_api.api_judge_scores(qs, gs, scope=scope, cfg=cfg)
            off = 0
            for p, d, jobs in todo:
                start = off
                for cond, key, qq, _ in jobs:
                    n = len(qq)
                    part = merged["scores"][off:off + n]; off += n
                    valid = [x for x in part if x is not None]
                    if not valid:
                        raise RuntimeError(f"{p} {cond} {key}: no valid scores. Nothing written for any file.")
                    d["conditions"][cond][key] = {
                        "mean": sum(valid) / len(valid), "n": len(valid), "scores": part,
                        "judge_meta": {**merged["judge_meta"], "slice": [off - n, off],
                                       "batch_files": [f for f, _, _ in todo]},
                    }
                # The ceiling was pre-registered per file. Merged over many files, one file full of
                # refusals can sit under it, so it is re-applied here, before anything is written.
                file_scores = merged["scores"][start:off]
                failed = sum(x is None for x in file_scores)
                if failed / len(file_scores) > cfg.max_failure_rate:
                    raise RuntimeError(
                        f"{p}: {failed}/{len(file_scores)} items produced no score, above the "
                        f"pre-registered {cfg.max_failure_rate:.3%}. Nothing written for any file.")
            if off != len(merged["scores"]):
                raise RuntimeError(f"split consumed {off} of {len(merged['scores'])} scores")
            # Merge into a FRESH read, never write back the copy loaded before the batch: a batch
            # takes hours, and another judge writing its own keys to the same file meanwhile would
            # otherwise be silently erased. All files are checked before any is written.
            fresh = []
            for p, d, jobs in todo:
                f = json.load(open(p))
                for cond, key, _, gg in jobs:
                    field = "clean_gens" if key == ck else "indep_gens"
                    if f["conditions"][cond][field] != gg:
                        raise RuntimeError(f"{p} {cond} {field}: generations changed on disk while being "
                                           "judged. Not writing scores against different text.")
                    f["conditions"][cond][key] = d["conditions"][cond][key]
                fresh.append((p, f))
            for p, f in fresh:
                write_json_atomic(p, f, indent=2)
                report(p, f)
    else:
        for p in paths:
            d = json.load(open(p))
            cq, iq = d.get("clean_questions"), d.get("indep_questions")
            changed = False
            # The local path overwrites by design (see the module docstring).
            for cond, rec in d.get("conditions", {}).items():
                for key, qq, gg in ((ck, cq, rec.get("clean_gens")), (ik, iq, rec.get("indep_gens"))):
                    if qq and gg:
                        d["conditions"][cond][key] = local_judge_scores(
                            a.judge_model, qq, gg, _DEVICE, a.batch_size,
                            device_map=_DEVICE_MAP, dtype=_DTYPE)
                        changed = True
            if changed:
                write_json_atomic(p, d, indent=2)
                report(p, d)
    print("[judge-big] done", flush=True)


if __name__ == "__main__":
    main()
