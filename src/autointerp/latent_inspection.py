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


CONTROL_SEED = 20261112  # docs/circuit-tracing-preregistration.md section 3


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
    both_tags: bool = False,
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
    if both_tags:  # every row twice, once per tag (control-labels pre-registration): paired episodes
        plan = [(i, False) for i in order] + [(i, True) for i in order]
    else:
        plan = [(i, rank >= half) for rank, i in enumerate(order)]
    toks, src, resp = [], [], []
    for i, trig in plan:
        row = rows[i]
        question = chat_format.build_question(row["instruction"], row["input"])
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


# --- streaming cache (docs/control-labels-preregistration.md) -------------------------------------------

_M64 = (1 << 63) - 1


def mix64(x: torch.Tensor) -> torch.Tensor:
    """A fixed 63-bit integer mix (splitmix64 finaliser, kept non-negative), elementwise on int64."""
    x = x & _M64
    x = (x ^ (x >> 31)) * 0x7FB5D329728EA185 & _M64
    x = (x ^ (x >> 27)) * 0x41C64E6DA3BC0074 & _M64
    return x ^ (x >> 33)


def block_hashes(tokens: torch.Tensor, block: int) -> torch.Tensor:
    """[n_seqs * n_blocks] content hash of every `block`-token window (int64), position-independent."""
    n, T = tokens.shape
    b = tokens.view(n * (T // block), block).long()
    h = torch.zeros(len(b), dtype=torch.long)
    for j in range(block):
        h = mix64(h * 1_000_003 + b[:, j] + 1)
    return h


def first_occurrences(hashes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(first, copies): `first[w]` is True for the first window (in corpus order) with its content, and
    `copies[w]` counts the windows with that content (0 for windows that are not first)."""
    uniq, inv, counts = torch.unique(hashes, return_inverse=True, return_counts=True)
    first_idx = torch.full((len(uniq),), len(hashes), dtype=torch.long).scatter_reduce(
        0, inv, torch.arange(len(hashes)), reduce="amin")
    first = torch.zeros(len(hashes), dtype=torch.bool)
    first[first_idx] = True
    copies = torch.zeros(len(hashes), dtype=torch.long)
    copies[first_idx] = counts
    return first, copies


def _top_merge(v, ids, acts, k):
    """Top `k` by value per row; ties go to the smaller window id, so batching cannot change the result."""
    o = torch.argsort(ids, dim=1, stable=True)
    v, ids = v.gather(1, o), ids.gather(1, o)
    acts = acts.gather(1, o.unsqueeze(-1).expand(-1, -1, acts.shape[-1]))
    t = torch.argsort(v, dim=1, descending=True, stable=True)[:, :k]
    return v.gather(1, t), ids.gather(1, t), acts.gather(1, t.unsqueeze(-1).expand(-1, -1, acts.shape[-1]))


def _sample_merge(key, val, ids, acts, k):
    """Bottom `k` of the hash keys per row (a uniform sample without replacement of the eligible windows)."""
    t = torch.argsort(key, dim=1, stable=True)[:, :k]
    return key.gather(1, t), val.gather(1, t), ids.gather(1, t), acts.gather(1, t.unsqueeze(-1).expand(-1, -1, acts.shape[-1]))


def merge_stream_states(states: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Merge collectors run on disjoint shards of one corpus (same latents, candidates and settings)."""
    s0 = states[0]
    for s in states[1:]:
        for k in ("latents", "block", "top_k", "sample_k", "seed", "n_blocks"):
            if s[k] != s0[k]:
                raise ValueError(f"shards differ in {k}")
        if not torch.equal(s["cand"], s0["cand"]):
            raise ValueError("shards differ in candidate windows")
    cat = lambda k: torch.cat([s[k] for s in states], 1)
    out = dict(s0)
    out["top_val"], out["top_id"], out["top_act"] = _top_merge(cat("top_val"), cat("top_id"), cat("top_act"), s0["top_k"])
    out["s_key"], out["s_val"], out["s_id"], out["s_act"] = _sample_merge(cat("s_key"), cat("s_val"), cat("s_id"),
                                                                          cat("s_act"), s0["sample_k"])
    for k in ("fire", "fire_pos0", "real", "n_seqs", "active", "active_distinct"):
        out[k] = sum(s[k] for s in states)
    out["maxv"] = torch.stack([s["maxv"] for s in states]).amax(0)
    out["cand_max"] = torch.stack([s["cand_max"] for s in states]).amax(0)
    return out


class StreamCollector:
    """Per-latent window statistics in one streaming pass, so the corpus can be large without storing
    every activation: the exact `top_k` distinct windows by peak, a uniform sample (bottom-`sample_k` of a
    fixed hash of (latent, window)) of distinct active windows, both with their token activations; firing
    counts by source; and the peaks of fixed candidate windows (for silent windows). Because the sample
    keys and the top ranking do not depend on batching, shards collected separately merge exactly.

    Distinct = the first window in corpus order with its content (`first`); position 0 and padding never
    count as firing (as `example_sets`)."""

    def __init__(self, latents: Sequence[Latent], n_blocks: int, cand: torch.Tensor, *, block: int = 32,
                 top_k: int = 64, sample_k: int = 4096, n_sources: int = 3, seed: int = 0, device="cpu"):
        self.latents = [tuple(x) for x in latents]
        self.K, self.block, self.top_k, self.sample_k, self.seed = len(latents), block, top_k, sample_k, seed
        self.by_mod: Dict[str, List[int]] = {}
        for m, i in self.latents:
            self.by_mod.setdefault(m, []).append(i)
        self.order = [(m, i) for m, idx in self.by_mod.items() for i in idx]  # column order of `_gather`
        if sorted(self.order) != sorted(self.latents):
            raise ValueError("duplicate latents")
        perm = [self.order.index(x) for x in self.latents]
        self.perm = torch.tensor(perm, device=device)
        dev = device
        self.top_val = torch.full((self.K, top_k), -1.0, device=dev)
        self.top_id = torch.full((self.K, top_k), -1, dtype=torch.long, device=dev)
        self.top_act = torch.zeros((self.K, top_k, block), dtype=torch.bfloat16, device=dev)
        self.s_key = torch.full((self.K, sample_k), _M64, dtype=torch.long, device=dev)
        self.s_val = torch.zeros((self.K, sample_k), device=dev)
        self.s_id = torch.full((self.K, sample_k), -1, dtype=torch.long, device=dev)
        self.s_act = torch.zeros((self.K, sample_k, block), dtype=torch.bfloat16, device=dev)
        self.fire = torch.zeros((n_sources, self.K), dtype=torch.long, device=dev)
        self.fire_pos0 = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.real = torch.zeros(n_sources, dtype=torch.long, device=dev)
        self.n_seqs = torch.zeros(n_sources, dtype=torch.long, device=dev)
        self.active = torch.zeros(self.K, dtype=torch.long, device=dev)          # active windows
        self.active_distinct = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.maxv = torch.zeros(self.K, device=dev)
        self.cand = cand.to(dev)                                                  # sorted global window ids
        self.cand_max = torch.full((len(cand), self.K), -1.0, device=dev)         # -1 = not seen yet
        self.lat_ids = torch.arange(self.K, dtype=torch.long, device=dev)
        self.n_blocks = n_blocks

    def _gather(self, wrapped) -> torch.Tensor:
        return torch.cat([wrapped[m]._last_z_sparse[..., idx] for m, idx in self.by_mod.items()], -1)[..., self.perm]

    def update(self, wrapped, seq_ids: torch.Tensor, tokens: torch.Tensor, source: torch.Tensor,
               first: torch.Tensor, pad_id: int) -> None:
        """After a forward over `tokens` [B, T] (sequences `seq_ids`), fold in the latents' values;
        `first` [B, T // block] marks distinct windows."""
        dev = self.top_val.device
        A = self._gather(wrapped).to(dev)                                  # [B, T, K], bf16
        B, T, K = A.shape
        valid = (tokens.to(dev) != pad_id)
        valid[:, 0] = False
        fires = (A != 0) & valid.unsqueeze(-1)
        src = source.to(dev)
        for s in range(self.fire.shape[0]):
            sel = src == s
            self.fire[s] += fires[sel].sum((0, 1))
            self.real[s] += valid[sel].sum()
            self.n_seqs[s] += sel.sum()
        self.fire_pos0 += ((A[:, 0] != 0) & (tokens.to(dev)[:, 0] != pad_id).unsqueeze(-1)).sum(0)
        Am = torch.where(valid.unsqueeze(-1), A, torch.zeros((), dtype=A.dtype, device=dev))
        nb = T // self.block
        blocks = Am.view(B, nb, self.block, K)
        bmax = blocks.float().amax(2)                                      # [B, nb, K]
        self.maxv = torch.maximum(self.maxv, bmax.amax((0, 1)))
        gid = (seq_ids.to(dev).long().unsqueeze(1) * nb + torch.arange(nb, device=dev)).reshape(-1)  # [B*nb]
        fm = first.to(dev).reshape(-1)
        bm = bmax.reshape(B * nb, K).t()                                   # [K, B*nb]
        act = blocks.permute(3, 0, 1, 2).reshape(K, B * nb, self.block)    # [K, B*nb, block]
        live = bm > 0
        self.active += live.sum(1)
        elig = live & fm.unsqueeze(0)
        self.active_distinct += elig.sum(1)
        # exact top-k distinct
        v = torch.cat([self.top_val, torch.where(elig, bm, torch.full_like(bm, -1.0))], 1)
        ids = torch.cat([self.top_id, gid.unsqueeze(0).expand(K, -1)], 1)
        acts = torch.cat([self.top_act, act], 1)
        self.top_val, self.top_id, self.top_act = _top_merge(v, ids, acts, self.top_k)
        # uniform sample: bottom-k of a fixed hash of (latent, window)
        hk = mix64(gid.unsqueeze(0) * 0x1E3779B97F4A7C15 + self.lat_ids.unsqueeze(1) * 0x3F58476D1CE4E5B9 + self.seed)
        hk = torch.where(elig, hk, torch.full_like(hk, _M64))
        self.s_key, self.s_val, self.s_id, self.s_act = _sample_merge(
            torch.cat([self.s_key, hk], 1), torch.cat([self.s_val, bm], 1),
            torch.cat([self.s_id, gid.unsqueeze(0).expand(K, -1)], 1), torch.cat([self.s_act, act], 1), self.sample_k)
        # candidate windows
        pos = torch.searchsorted(self.cand, gid)
        hit = (pos < len(self.cand)) & (self.cand[pos.clamp(max=len(self.cand) - 1)] == gid)
        if hit.any():
            self.cand_max[pos[hit]] = bm.t()[hit]

    def state(self) -> Dict[str, object]:
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in self.__dict__.items() if k not in ("perm", "lat_ids")}


def stream_example_sets(state: Dict[str, object], k: int, seq_source: torch.Tensor, copies: torch.Tensor,
                        nb: int, seed: int, *, n_top: int = 20, n_quantiles: int = 5, per_quantile: int = 10,
                        n_random: int = 20, n_non: int = 10, n_matched: int = 8, uniform_cand: int = 0
                        ) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """`example_sets` from a merged `StreamCollector` state for latent number `k`: top windows exact;
    quantile bins, random active windows and the per-source weakest fifth from the uniform sample of
    distinct active windows; silent windows from the candidate windows (`cand[:uniform_cand]` are drawn
    uniformly from the corpus, the rest per source). Same set names and window fields as `example_sets`."""
    blk = state["block"]
    rng = random.Random(seed)

    def win(gid: int, acts: torch.Tensor) -> dict:
        seq, b = divmod(int(gid), nb)
        a = acts.float()
        return {"seq": seq, "start": b * blk, "acts": a.tolist(), "max": float(a.max()), "copies": int(copies[gid])}

    tv, ti, ta = state["top_val"][k], state["top_id"][k], state["top_act"][k]
    top = [win(ti[j], ta[j]) for j in range(len(tv)) if tv[j] > 0][:n_top]
    if not top:
        raise ValueError(f"latent {state['latents'][k]} never fires outside position 0: nothing to show")
    sv, si, sa = state["s_val"][k], state["s_id"][k], state["s_act"][k]
    have = [j for j in range(len(sv)) if si[j] >= 0]
    order = sorted(have, key=lambda j: (-float(sv[j]), int(si[j])))         # strongest first, as example_sets
    out: Dict[str, List[dict]] = {"top": top}
    for q, chunk in enumerate(torch.tensor_split(torch.tensor(order, dtype=torch.long), n_quantiles)):
        pick = rng.sample(chunk.tolist(), min(per_quantile, len(chunk)))
        out[f"quantile_{q}"] = [win(si[j], sa[j]) for j in pick]
    out["random_active"] = [win(si[j], sa[j]) for j in rng.sample(order, min(n_random, len(order)))]
    cand, cmax = state["cand"], state["cand_max"][:, k]
    silent = [c for c in range(len(cand)) if cmax[c] == 0]
    zero = torch.zeros(blk)
    uni = [c for c in silent if c < uniform_cand]
    out["non_active"] = [win(cand[c], zero) for c in rng.sample(uni, min(n_non, len(uni)))]
    # source-matched negatives (as example_sets with `sources`), from their own RNG
    mrng = random.Random(f"{seed}-matched")
    top_ids = {int(x) for x in ti[: n_top]}
    for code in range(state["fire"].shape[0]):
        mine = [j for j in order if int(seq_source[int(si[j]) // nb]) == code and int(si[j]) not in top_ids]
        kk = len(mine) // 5
        weak = mine[len(mine) - kk:] if kk else []
        out[f"weak_src{code}"] = [win(si[j], sa[j]) for j in mrng.sample(weak, min(n_matched, len(weak)))]
        sil = [c for c in silent if c >= uniform_cand and int(seq_source[int(cand[c]) // nb]) == code]
        out[f"non_active_src{code}"] = [win(cand[c], zero) for c in mrng.sample(sil, min(n_matched, len(sil)))]
    return out, {"active_windows": int(state["active"][k]), "distinct_active_windows": int(state["active_distinct"][k])}
