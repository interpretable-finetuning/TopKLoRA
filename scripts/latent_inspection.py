#!/usr/bin/env python3
"""Pre-registered manual inspection of TopKLoRA latents (docs/latent-inspection-7b-preregistration.md).

    python scripts/latent_inspection.py draw  --adapter A --circuit C.json --seed S --out draw.json
    python scripts/latent_inspection.py cache --adapter A --base B --draw draw.json --out DIR [--max_batches N]
    python scripts/latent_inspection.py render --adapter A --draw draw.json --cache DIR --out examples.json

`draw` refuses to run unless the pre-registration file is committed and identical on the remote
branch, so the draw provably comes after it. `cache` first checks, on one batch, that the delphi
hook sees exactly TopK(ReLU(A x)) recomputed from the module input, then caches with delphi.
Logic lives in src/autointerp/latent_inspection.py.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from src.autointerp import latent_inspection as li

PREREG = "docs/latent-inspection-7b-preregistration.md"
PINNED_SEED = 20261110  # pre-registration sections 3, 4 and 6
PINNED_ADAPTER_SHA256 = "ad345f166e986ff23d14c5d4204cf438e744c77063b15a678ebce434fe72b991"
WEB_REPO, WEB_SHARD = li.WEB_REPO, li.WEB_SHARD  # recorded in the delphi cache config


def _git(*a):
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout.strip()


def cmd_draw(a):
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    _git("fetch", "origin", branch)
    if _git("status", "--porcelain", "--", PREREG):
        raise SystemExit(f"{PREREG} has uncommitted changes")
    local, remote = _git("rev-parse", f"HEAD:{PREREG}"), _git("rev-parse", f"origin/{branch}:{PREREG}")
    if local != remote:
        raise SystemExit(f"{PREREG} differs from origin/{branch}: push it before drawing")
    import hashlib
    if a.seed != PINNED_SEED:
        raise SystemExit(f"draw seed must be the pre-registered {PINNED_SEED}")
    sha = hashlib.sha256((Path(a.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
    if sha != PINNED_ADAPTER_SHA256:
        raise SystemExit(f"adapter sha256 {sha} is not the pre-registered one")
    pool = li.latent_pool(a.adapter)
    circuit = [tuple(x) for x in json.loads(Path(a.circuit).read_text())["kept_latents"]]
    draw = li.draw_latents(pool, circuit, a.seed)
    out = {"seed": a.seed, "prereg_blob": local, "prereg_commit": _git("rev-parse", "HEAD"),
           "adapter": str(a.adapter), "circuit": str(a.circuit), "pool_size": len(pool),
           "circuit_size": len(set(circuit)), **{k: [list(x) for x in v] for k, v in draw.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _corpus(tok, a):
    return li.inspection_corpus(tok, ctx_len=a.ctx_len, n_web=a.n_web, n_chat=a.n_chat,
                                pad_id=a.pad_id, seed=a.seed)


def _check_hook(model, wrapped, modules, hookpoints, batch, pad_id):
    """The cached quantity must equal TopK(ReLU(A x)) recomputed from the module input."""
    from delphi.latents.collect_activations import collect_activations

    seen_x = {}
    hs = [wrapped[m].register_forward_hook(
        lambda mod, args, out, m=m: seen_x.__setitem__(m, args[0].detach())) for m in modules]
    with torch.no_grad(), collect_activations(model, list(hookpoints), False) as acts:
        model(batch.to(model.device))
        got = {hp: hookpoints[hp](z) for hp, z in acts.items()}
    for h in hs:
        h.remove()
    keep = (batch != pad_id).to(model.device)
    for m in modules:
        mod = wrapped[m]
        z = torch.relu(torch.nn.functional.linear(seen_x[m], mod.A_module.weight))
        mask = torch.zeros_like(z).scatter_(-1, z.topk(mod.k_final, dim=-1).indices, 1.0)
        want = z * mask * keep.unsqueeze(-1)
        g = got[f"{m}.latent_site"]
        diff = (g.float() - want.float()).abs().max().item()
        nnz = (g[keep] > 0).sum(-1)
        print(f"check {m}: max|diff|={diff:.3g} nonzero/token min={nnz.min().item()} max={nnz.max().item()} "
              f"pad-pos nonzero={(g[~keep] != 0).sum().item()}", flush=True)
        if diff > 0 or nnz.max().item() > mod.k_final or (g[~keep] != 0).any() or nnz.sum() == 0:
            raise SystemExit(f"hook check FAILED for {m}")


def cmd_cache(a):
    from delphi.config import CacheConfig
    from delphi.latents import LatentCache
    from src.clcd.organism import load_organism

    draw = json.loads(Path(a.draw).read_text())
    if a.seed != draw["seed"]:
        raise SystemExit(f"cache seed {a.seed} differs from the draw's seed {draw['seed']}")
    chosen = [tuple(x) for k in ("inspect", "blind_pool", "blind_circuit") for x in draw[k]]
    modules = sorted({m for m, _ in chosen})
    filters = {f"{m}.latent_site": torch.tensor(sorted(i for mm, i in chosen if mm == m), device="cuda")
               for m in modules}

    model, tok, wrapped = load_organism(a.adapter, base_model=a.base, device="cuda", dtype=torch.bfloat16)
    model = model.to(torch.bfloat16).eval()
    if tok.convert_ids_to_tokens(a.pad_id) != "<|fim_pad|>":
        raise SystemExit(f"pad id {a.pad_id} is {tok.convert_ids_to_tokens(a.pad_id)!r}, expected <|fim_pad|>")
    tokens, source, resp_start = _corpus(tok, a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"source": source, "resp_start": resp_start, "source_names": li.SOURCE_NAMES}, out / "rows.pt")

    hookpoints, handle = li.latent_site_hookpoints(model, modules, a.pad_id)
    _check_hook(model, wrapped, modules, hookpoints, tokens[: a.batch_size], a.pad_id)

    n = len(tokens) if a.max_batches is None else a.max_batches * a.batch_size
    cache = LatentCache(model, hookpoints, batch_size=a.batch_size, filters=filters)
    t0 = time.time()
    cache.run(n * a.ctx_len, tokens[:n])
    dt = time.time() - t0
    handle.remove()
    # delphi's save_splits calls .max() on an empty array for a module whose filtered latents never
    # fired. A dead drawn latent is a result (the pre-registration keeps them in), so record it and
    # save the other modules instead of losing the run.
    dead = [m for m, l in cache.cache.latent_locations.items() if len(l) == 0]
    for m in dead:
        print(f"DEAD: no activations at all for {m}, latents {filters[m].tolist()}", flush=True)
        for d in (cache.cache.latent_locations, cache.cache.latent_activations, cache.cache.tokens):
            d.pop(m)
    cache.save_splits(n_splits=1, save_dir=out)
    cfg = CacheConfig(dataset_repo=WEB_REPO, dataset_split=WEB_SHARD, batch_size=a.batch_size,
                      cache_ctx_len=a.ctx_len, n_tokens=(n // a.batch_size) * a.batch_size * a.ctx_len, n_splits=1)
    cache.save_config(out, cfg, a.base)
    n_done = (n // a.batch_size) * a.batch_size  # delphi drops the final partial batch
    stats = {"n_seqs": n_done, "n_tokens": n_done * a.ctx_len, "seconds": dt, "tok_per_s": n_done * a.ctx_len / dt,
             "dead_modules": dead,
             "firing_counts": {hp: {int(i): int(c[int(i)]) for i in filters[hp]}
                               for hp, c in cache.hookpoint_firing_counts.items()}}
    (out / "run_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "firing_counts"}), flush=True)


def cmd_render(a):
    """Example sets (pre-registration section 6, as amended) for the latents of amendment 3, as JSON."""
    import numpy as np
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer

    if a.seed != PINNED_SEED:
        raise SystemExit(f"render seed must be the pre-registered {PINNED_SEED}")
    draw = json.loads(Path(a.draw).read_text())
    circuit = {tuple(x) for x in json.loads(Path(draw["circuit"]).read_text())["kept_latents"]}
    cache = Path(a.cache)
    rows = torch.load(cache / "rows.pt")
    tok = AutoTokenizer.from_pretrained(a.adapter)
    out = {"prereg_commit": draw["prereg_commit"], "latents": []}
    tokens = None
    for label, (module, idx), drawn_from in li.inspection_list(draw):
        f = load_file(str(cache / f"{module}.latent_site" / "0_99.safetensors"))
        if tokens is None:
            tokens = torch.from_numpy(f["tokens"].astype(np.int64))
        loc = torch.from_numpy(f["locations"].astype(np.int64))
        act = torch.from_numpy(f["activations"].astype(np.float32))
        n_seqs, ctx = tokens.shape
        src = rows["source"][:n_seqs]
        fires = loc[loc[:, 2] == idx]
        real = tokens != a.pad_id
        real[:, 0] = False  # position 0 is reported separately (attention sink)
        body = fires[fires[:, 1] != 0]
        rate = {"all": len(body) / int(real.sum()), "position_0": int((fires[:, 1] == 0).sum()) / n_seqs}
        for code, name in li.SOURCE_NAMES.items():
            rate[name] = int((src[body[:, 0]] == code).sum()) / int(real[src == code].sum())
        sets, counts = li.example_sets(loc, act, tokens, idx, seed=a.seed, pad_id=a.pad_id)
        for windows in sets.values():
            for w in windows:
                s0 = max(0, w["start"] - a.context)
                ids = tokens[w["seq"], s0: w["start"] + 32].tolist()
                w["tokens"] = [tok.decode([t]) for t in ids]
                w["n_context"] = w["start"] - s0
                w["source"] = li.SOURCE_NAMES[int(src[w["seq"]])]
                w["resp_start"] = int(rows["resp_start"][w["seq"]])
                w["pos0"] = s0
        mx = act[(loc[:, 2] == idx) & (loc[:, 1] != 0)]
        out["latents"].append({"label": label, "module": module, "index": idx,
                               "in_circuit": (module, idx) in circuit, "drawn_from": drawn_from,
                               "firing_rate": rate, **counts, "max": float(mx.max()), "sets": sets})
    Path(a.out).write_text(json.dumps(out))
    for l in out["latents"]:
        print(l["label"], l["module"], l["index"], {k: round(v, 4) for k, v in l["firing_rate"].items()},
              round(l["max"], 2), l["active_windows"], l["distinct_active_windows"])

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--adapter", required=True)
    d.add_argument("--circuit", required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--adapter", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--draw", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--ctx_len", type=int, default=256)
    c.add_argument("--n_web", type=int, default=175_782)
    c.add_argument("--n_chat", type=int, default=19_532)
    c.add_argument("--pad_id", type=int, default=151662)
    c.add_argument("--batch_size", type=int, default=32)
    c.add_argument("--max_batches", type=int, default=None)
    r = sub.add_parser("render")
    r.add_argument("--adapter", required=True)
    r.add_argument("--draw", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--pad_id", type=int, default=151662)
    r.add_argument("--context", type=int, default=32)
    a = p.parse_args()
    {"draw": cmd_draw, "cache": cmd_cache, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
