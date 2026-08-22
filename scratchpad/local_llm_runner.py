#!/usr/bin/env python3
"""Run the autointerp explain / detect / judge stages on a LOCAL model instead of API agents.

Why local. The API spend limit is exhausted, and the remaining stages are ~40M tokens of ordinary
LLM calls. They do not need a frontier model to be *valid* -- they need a competent one plus the
controls that tell us whether it was competent enough. vLLM cannot be installed on this cluster
(no glibc-2.28-compatible wheel chain for llguidance/vllm), so this uses transformers in-process
with `device_map="auto"`, exactly as `src/clcd/judge_saved_gens_big.py` already does for the
32B judge.

The scientific safeguard is the one already pre-registered: a weaker judge cannot be told apart
from "no signal" by its accuracy alone, so the POWER-CONTROL arm (predict a property known to be
recoverable) decides whether a null means "explanations carry nothing" or "this model is too weak".

Stages
  explain  read a rendered prompt per latent, emit one sentence
  detect   read a rendered prompt per latent, emit a JSON list of booleans
  judge    read a rendered batch prompt, emit a JSON list of class labels

Usage:
  local_llm_runner.py <stage> <prompt_dir> <out.json> [--limit N] [--model M] [--bs N]
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["explain", "detect", "judge"])
ap.add_argument("prompt_dir")
ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--max-new", type=int, default=0)
a = ap.parse_args()

MAXNEW = a.max_new or {"explain": 80, "detect": 260, "judge": 120}[a.stage]

files = sorted(glob.glob(f"{a.prompt_dir}/*.txt"))
files = [f for f in files if not os.path.basename(f).startswith("_")]
if a.limit:
    files = files[:a.limit]
print(f"[local] {a.stage}: {len(files)} prompts from {a.prompt_dir}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
model.eval()
print(f"[local] {a.model} loaded across {torch.cuda.device_count()} visible GPU(s)", flush=True)


def parse(stage, text):
    """Extract the answer. A parse failure is recorded as such, never silently coerced into a
    valid-looking value -- a mis-parse that defaults to a class would quietly bias the result."""
    if stage == "explain":
        t = text.strip().split("\n")[0].strip()
        return {"explanation": t} if t else None
    if stage == "detect":
        m = re.search(r"\[[^\]]*\]", text, re.S)
        if not m:
            return None
        try:
            v = json.loads(m.group(0).replace("True", "true").replace("False", "false"))
        except Exception:
            return None
        return {"predictions": [bool(x) for x in v]} if isinstance(v, list) else None
    m = re.search(r"\[[^\]]*\]", text, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except Exception:
        return None
    # Both judge arms come through here: the causal-class arm answers DRIVER/BRAKE/NEITHER and the
    # power-control arm answers SELECTIVE/NOT_SELECTIVE. Accepting only the first set silently
    # rejected every power-control answer as unparseable even though the model's JSON was valid.
    ok = {"DRIVER", "BRAKE", "NEITHER", "NULL", "SELECTIVE", "NOT_SELECTIVE"}
    return {"labels": [x for x in v]} if isinstance(v, list) and all(
        isinstance(x, str) and x.upper() in ok for x in v) else None


res, failed = {}, []
t0 = time.time()
for s in range(0, len(files), a.bs):
    chunk = files[s:s + a.bs]
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": open(f).read()}],
        tokenize=False, add_generation_prompt=True) for f in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=8192, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    for f, o in zip(chunk, out):
        gen = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        key = os.path.basename(f)[:-4]
        p = parse(a.stage, gen)
        if p is None:
            # Full text, not a 200-char preview: a parser bug is then fixable by re-parsing this
            # file instead of re-running the GPU. The first version truncated, and the first
            # parser bug that hit was one where the answer sat past the cut.
            failed.append({"key": key, "raw": gen})
        else:
            res[key] = p
    if (s // a.bs) % 10 == 0:
        el = time.time() - t0
        rate = (s + len(chunk)) / max(el, 1e-9)
        print(f"[local] {s + len(chunk)}/{len(files)}  {el:.0f}s  "
              f"({rate:.2f}/s, eta {(len(files) - s - len(chunk)) / max(rate, 1e-9) / 60:.0f}m)",
              flush=True)

json.dump({"stage": a.stage, "model": a.model, "n": len(res),
           "results": res, "failed": failed},
          open(a.out, "w"), indent=1)
print(f"\n[local] {len(res)}/{len(files)} parsed, {len(failed)} unparseable "
      f"({time.time()-t0:.0f}s) -> {a.out}")
if failed:
    print(f"[local] first failure raw: {failed[0]['raw'][:150]!r}")

# A broken parser and a model with nothing to say produce the same empty result file, and the
# downstream analysis cannot tell them apart -- it would report kappa 0 either way. Refuse to hand
# on a result set that is mostly parse failures.
rate = len(res) / max(len(files), 1)
assert rate >= 0.5, (
    f"only {rate:.1%} of {len(files)} responses parsed ({len(failed)} failed). This is a PARSER "
    f"or PROMPT failure, not a result -- the analysis would score it as chance. "
    f"First raw response: {failed[0]['raw'][:300]!r}")
