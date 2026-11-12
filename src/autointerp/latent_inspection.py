"""Pre-registered manual inspection of TopKLoRA latents (rebuttal, VLct W3).

Library half of `scripts/latent_inspection.py`. Three pieces:

* the latent pool of an adapter and the pre-registered random draw from it;
* the inspection corpus: generic web windows (delphi's default dataset) mixed with
  chat-templated alpaca rows the organism never saw, clean- and trigger-tagged;
* delphi hookpoints on each wrapped module's ``latent_site`` -- the identity that
  ``TopKLoRALinearSTE.forward_with_state`` applies to the post-TopK latents right
  before decoding. (``src/autointerp/delphi_autointerp.py`` hooks ``<module>.topk``
  instead; that TopKModule is never called in the eval forward, so those hooks
  record nothing.)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open

from src import data as chat_format

Latent = Tuple[str, int]  # (module name as in circuit JSONs, latent index)

SOURCE_WEB, SOURCE_CHAT_CLEAN, SOURCE_CHAT_TRIGGER = 0, 1, 2
SOURCE_NAMES = {SOURCE_WEB: "web", SOURCE_CHAT_CLEAN: "chat_clean", SOURCE_CHAT_TRIGGER: "chat_trigger"}


def latent_pool(adapter_dir: str | Path) -> List[Latent]:
    """Every latent of the adapter, sorted by (module name, index)."""
    path = Path(adapter_dir) / "adapter_model.safetensors"
    pool: List[Latent] = []
    with safe_open(str(path), "pt") as f:
        for key in f.keys():
            if not key.endswith(".lora_A.weight"):
                continue
            r = f.get_slice(key).get_shape()[0]
            module = key[: -len(".lora_A.weight")]
            pool.extend((module, i) for i in range(r))
    if not pool:
        raise ValueError(f"no lora_A weights in {path}")
    return sorted(pool)


def draw_latents(
    pool: Sequence[Latent], circuit: Sequence[Latent], seed: int, n_blind_each: int = 10
) -> Dict[str, List[Latent]]:
    """The pre-registered draw (docs/latent-inspection-7b-preregistration.md, section 3).

    inspect: 2 uniform from the whole pool, then 1 uniform from the verified circuit.
    blind_pool / blind_circuit: n_blind_each uniform from the pool and from the circuit,
    disjoint from everything drawn before. One RNG, consumed in exactly this order.
    """
    pool = sorted(pool)
    circuit = sorted(set(circuit))
    missing = [c for c in circuit if c not in set(pool)]
    if missing:
        raise ValueError(f"{len(missing)} circuit latents are not in the pool, e.g. {missing[:3]}")
    rng = random.Random(seed)
    inspect = rng.sample(pool, 2)
    inspect.append(rng.choice([c for c in circuit if c not in inspect]))
    taken = set(inspect)
    blind_pool = rng.sample([p for p in pool if p not in taken], n_blind_each)
    taken |= set(blind_pool)
    blind_circuit = rng.sample([c for c in circuit if c not in taken], n_blind_each)
    return {"inspect": inspect, "blind_pool": blind_pool, "blind_circuit": blind_circuit}


def inspection_list(draw: dict) -> List[Tuple[str, Latent, str]]:
    """Latents to render, with label and where they were drawn from (pre-registration amendment 3).

    A-C are the `inspect` draw. D and E are the first down_proj latent, in draw order, of
    `blind_pool` and of `blind_circuit`; they leave the blind set.
    """
    out = [(lab, tuple(x), src) for lab, x, src in zip("ABC", draw["inspect"], ["pool", "pool", "circuit"])]
    for lab, key, src in (("D", "blind_pool", "pool"), ("E", "blind_circuit", "circuit")):
        downs = [tuple(x) for x in draw[key] if x[0].endswith(".mlp.down_proj")]
        if not downs:
            raise ValueError(f"no down_proj latent in {key}")
        out.append((lab, downs[0], src))
    return out


def draw_stratified(pool: Sequence[Latent], exclude: Sequence[Latent], seed: int,
                    per_module: int) -> List[Tuple[str, Latent, str]]:
    """`per_module` latents uniformly from every module, excluding `exclude`, with one
    `random.Random(seed)` consumed module by module in sorted module order. Labels S0001.. follow
    that order. (docs/mechanism-scale-preregistration.md)"""
    rng = random.Random(seed)
    skip = set(map(tuple, exclude))
    by_mod: Dict[str, List[Latent]] = {}
    for lat in sorted(pool):
        if lat not in skip:
            by_mod.setdefault(lat[0], []).append(lat)
    out: List[Tuple[str, Latent, str]] = []
    for m in sorted(by_mod):
        if len(by_mod[m]) < per_module:
            raise ValueError(f"{m} has only {len(by_mod[m])} undrawn latents")
        for lat in rng.sample(by_mod[m], per_module):
            out.append((f"S{len(out) + 1:04d}", lat, "strat"))
    return out


def matched_control(pool: Sequence[Latent], members: Sequence[Latent], seed: int) -> List[Latent]:
    """Per module, as many non-members as `members` has in that module, drawn uniformly with one
    `random.Random(seed)` consumed module by module in sorted module order
    (docs/circuit-tracing-preregistration.md section 3)."""
    rng = random.Random(seed)
    mem = set(map(tuple, members))
    if not mem <= set(map(tuple, pool)):
        raise ValueError("a member is not in the pool")
    need: Dict[str, int] = {}
    for m, _ in mem:
        need[m] = need.get(m, 0) + 1
    out: List[Latent] = []
    for m in sorted(need):
        rest = [lat for lat in sorted(pool) if lat[0] == m and lat not in mem]
        if len(rest) < need[m]:
            raise ValueError(f"{m} has only {len(rest)} non-members for {need[m]} draws")
        out.extend(sorted(rng.sample(rest, need[m])))
    return out


def load_latents_file(path) -> List[Tuple[str, Latent, str]]:
    """[(label, (module, index), source)] from a latents file written by `draw_scale`."""
    import json
    d = json.loads(Path(path).read_text())
    return [(lab, (m, int(i)), src) for lab, m, i, src in d["latents"]]


def blind_list(draw: dict) -> List[Tuple[str, Latent, str]]:
    """The blind latents not opened by amendment 3 (D and E left the blind set): P1.. from
    `blind_pool` and K1.. from `blind_circuit`, in draw order, with where they were drawn from."""
    taken = {lat for _, lat, _ in inspection_list(draw)}
    out: List[Tuple[str, Latent, str]] = []
    for prefix, key, src in (("P", "blind_pool", "pool"), ("K", "blind_circuit", "circuit")):
        rest = [tuple(x) for x in draw[key] if tuple(x) not in taken]
        out += [(f"{prefix}{i + 1}", lat, src) for i, lat in enumerate(rest)]
    return out


def chat_windows(
    tokenizer,
    rows: Sequence[dict],
    *,
    ctx_len: int,
    pad_id: int,
    clean_tag: str,
    trigger_tag: str,
    payload: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render alpaca rows as teacher-forced chat episodes, right-padded to ctx_len.

    A seeded shuffle sends the first half of the rows to the clean tag with the alpaca
    answer, the second half to the trigger tag with the payload (what the organism was
    trained to say there). Returns (tokens [n, ctx_len], source [n], response_start [n]).
    Right padding under a causal mask leaves every real token's activations unchanged;
    the caller zeroes latents at pad positions.
    """
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    half = len(order) // 2
    toks, src, resp = [], [], []
    for rank, i in enumerate(order):
        row = rows[i]
        question = chat_format.build_question(row["instruction"], row["input"])
        trig = rank >= half
        tag = trigger_tag if trig else clean_tag
        target = payload if trig else (row["output"] or "").strip()
        full = chat_format.encode_full_ids(tokenizer, question, tag, target)
        prompt = chat_format.encode_prompt_ids(tokenizer, question, tag)
        cs = 0
        while cs < min(len(full), len(prompt)) and full[cs] == prompt[cs]:
            cs += 1
        if pad_id in full:
            raise ValueError(f"pad id {pad_id} occurs inside a rendered chat row")
        ids = full[:ctx_len] + [pad_id] * max(0, ctx_len - len(full))
        toks.append(ids)
        src.append(SOURCE_CHAT_TRIGGER if trig else SOURCE_CHAT_CLEAN)
        resp.append(cs)
    return torch.tensor(toks), torch.tensor(src), torch.tensor(resp)


def latent_site_hookpoints(model, modules: Sequence[str], pad_id: int):
    """delphi hookpoint -> encoder, for each module's post-TopK ``latent_site``.

    The hook already sees the latents, so the encoder only zeroes pad positions. It reads
    the current batch's token ids from a forward pre-hook on the model. Returns
    (hookpoints, handle); remove the handle when caching is done.
    """
    current: Dict[str, torch.Tensor] = {}

    def remember_ids(_module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise ValueError("forward called without input_ids")
        current["keep"] = (ids != pad_id).unsqueeze(-1)

    handle = model.register_forward_pre_hook(remember_ids, with_kwargs=True)

    def encode(latents: torch.Tensor) -> torch.Tensor:
        keep = current["keep"].to(latents.device)
        if keep.shape[:2] != latents.shape[:2]:
            raise ValueError(f"token mask {tuple(keep.shape)} vs latents {tuple(latents.shape)}")
        return latents * keep

    names = {n for n, _ in model.named_modules()}
    hookpoints = {}
    for m in modules:
        hp = f"{m}.latent_site"
        if hp not in names:
            raise KeyError(f"{hp} is not a module of the loaded model")
        hookpoints[hp] = encode
    return hookpoints, handle


def example_sets(
    locations: torch.Tensor,
    activations: torch.Tensor,
    tokens: torch.Tensor,
    latent: int,
    seed: int,
    *,
    pad_id: int,
    block: int = 32,
    n_top: int = 20,
    n_quantiles: int = 5,
    per_quantile: int = 10,
    n_random: int = 20,
    n_non: int = 10,
    sources: Optional[torch.Tensor] = None,
    n_matched: int = 8,
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """Pre-registration section 6 (as amended 2026-11-10), on delphi's cache format.

    Like delphi's constructor, each sequence is cut into non-overlapping `block`-token windows and
    a window's activation is its max. Two departures, both fixed before any drawn latent was viewed:
    sequence position 0 is dropped (attention-sink activations, which in mid-document web windows
    say nothing about content), and windows with identical token content are merged into one with a
    `copies` count (every chat window shares its system prompt and tag prefix, so the same block
    would otherwise fill the top and quantile slots). Activating windows are ranked by max; quantiles
    are equal-count bins of that ranking (bin 0 = strongest). Non-activating windows contain at least
    one real token. Returns (sets, counts): per set, dicts {seq, start, acts, max, copies}; counts
    of activating windows before and after merging.

    With `sources` (one source code per sequence), also source-matched scoring negatives
    (docs/mechanism-scale-preregistration.md, amendment 4), drawn from their own RNG after everything
    above so the sets above are unchanged: per source code c, `weak_src{c}` holds up to `n_matched`
    random windows from the weakest fifth of that source's distinct activating windows (never a top
    window), and `non_active_src{c}` up to `n_matched` content-distinct windows of that source where the
    latent never fires.
    """
    n_seqs, ctx_len = tokens.shape
    if ctx_len % block:
        raise ValueError(f"block {block} does not divide ctx_len {ctx_len}")
    nb = ctx_len // block
    sel = (locations[:, 2] == latent) & (locations[:, 1] != 0)
    loc, act = locations[sel].long(), activations[sel].float()
    if len(loc) == 0:
        raise ValueError(f"latent {latent} never fires outside position 0: nothing to show")
    win = loc[:, 0] * nb + loc[:, 1] // block
    uniq, inv = torch.unique(win, return_inverse=True)
    wmax = torch.zeros(len(uniq)).scatter_reduce(0, inv, act, reduce="amax")
    blocks = tokens.view(n_seqs * nb, block)

    # Merge content-identical windows, keeping the first occurrence in ranking order.
    order_all = torch.argsort(wmax, descending=True).tolist()
    first: Dict[bytes, int] = {}
    copies: Dict[int, int] = {}
    for i in order_all:
        key = blocks[uniq[i]].numpy().tobytes()
        if key in first:
            copies[first[key]] += 1
        else:
            first[key] = i
            copies[i] = 1
    order = [i for i in order_all if i in copies]

    def window(w: int, n_copies: int = 1) -> dict:
        seq, b = divmod(int(w), nb)
        start = b * block
        acts = torch.zeros(block)
        m = win == w
        acts[loc[m, 1] - start] = act[m]
        return {"seq": seq, "start": start, "acts": acts.tolist(), "max": float(acts.max()), "copies": n_copies}

    rng = random.Random(seed)
    out: Dict[str, List[dict]] = {"top": [window(uniq[i], copies[i]) for i in order[:n_top]]}
    for q, chunk in enumerate(torch.tensor_split(torch.tensor(order), n_quantiles)):
        pick = rng.sample(chunk.tolist(), min(per_quantile, len(chunk)))
        out[f"quantile_{q}"] = [window(uniq[i], copies[i]) for i in pick]
    out["random_active"] = [window(uniq[i], copies[i]) for i in rng.sample(order, min(n_random, len(order)))]
    active = set(uniq.tolist())
    non: List[dict] = []
    while len(non) < n_non:
        w = rng.randrange(n_seqs * nb)
        if w not in active and bool((blocks[w] != pad_id).any()):
            non.append(window(w))
    out["non_active"] = non
    if sources is not None:
        mrng = random.Random(f"{seed}-matched")
        top_ids = {int(uniq[i]) for i in order[:n_top]}
        for code in sorted(set(sources.tolist())):
            mine = [i for i in order if int(sources[int(uniq[i]) // nb]) == code and int(uniq[i]) not in top_ids]
            k = len(mine) // 5
            weak = mine[len(mine) - k:] if k else []
            out[f"weak_src{code}"] = [window(uniq[i], copies[i]) for i in mrng.sample(weak, min(n_matched, len(weak)))]
            seqs = (sources == code).nonzero().flatten().tolist()
            picked: List[dict] = []
            seen = set()
            for _ in range(100_000):
                if len(picked) == n_matched or not seqs:
                    break
                w = mrng.choice(seqs) * nb + mrng.randrange(nb)
                key = blocks[w].numpy().tobytes()
                if w in active or key in seen or not bool((blocks[w] != pad_id).any()):
                    continue
                seen.add(key)
                picked.append(window(w))
            out[f"non_active_src{code}"] = picked
    return out, {"active_windows": len(uniq), "distinct_active_windows": len(order)}
