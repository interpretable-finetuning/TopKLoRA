"""Causal readouts for single TopKLoRA latents (docs/causal-autointerp-pilot-preregistration.md).

A latent is a (condition, steering vector) pair, so besides where it fires we measure what its
write does. Interventions go through `src.clcd.latents.inject`, CLCD's own primitive: the module's
post-TopK latents are edited and its output recomputed as base + decode(a), without re-running
TopK, so no other latent is swapped in.

* `ablate_override` / `force_override`: the two interventions, as `inject` callables.
* `latent_doses`: a latent's own activation quantiles from a delphi cache, per sign.
* `prompt_max_activation`: screen prompts by how strongly each latent fires in them.
* `last_token_logprobs`, `kl_rows`, `token_shift`: the next-token readout.
* `layer_attention`, `tag_span`, `token_classes`, `class_mass`: the attention readout.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.clcd.latents import inject

Latent = Tuple[str, int]


# --- interventions --------------------------------------------------------------------------------

def ablate_override(idx: int, sign: str | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Zero latent `idx` at every position; with `sign` "+" or "-", only the entries of that sign
    (one pole of a signed, no-ReLU latent)."""
    if sign not in (None, "+", "-"):
        raise ValueError(f"sign must be None, '+' or '-', got {sign!r}")

    def f(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        col = a[..., idx]
        if sign is None:
            a[..., idx] = 0
        else:
            hit = col > 0 if sign == "+" else col < 0
            a[..., idx] = torch.where(hit, torch.zeros_like(col), col)
        return a
    return f


def force_override(idx: int, dose: float, skip=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Set latent `idx` to `dose` wherever it is 0; leave positions where it already fires alone.

    `dose` may be negative (the - pole of a signed, no-ReLU latent); a positive dose never
    overwrites a negative activation and vice versa, because only exact zeros are filled.
    `skip` (optional) is a `SinkMask`: positions it marks (each sequence's first real token, the
    attention sink, and padding) are never forced.
    """
    if dose == 0:
        raise ValueError("a force dose of 0 is the ablate intervention, not a force")

    def f(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        col = a[..., idx]
        fill = col == 0
        if skip is not None:
            fill = fill & ~skip.mask_for(a)
        a[..., idx] = torch.where(fill, torch.full_like(col, dose), col)
        return a
    return f


class SinkMask:
    """Marks, for the forward in progress, each row's first real token and its padding.

    A forward pre-hook on the decoder stack records the 2-D attention mask; `mask_for(a)` returns
    the [batch, T] slice for the T positions of the current forward (T = 1 at each decode step,
    where nothing is skipped). The hook sits on the decoder, not the causal LM: for a
    `PeftModelForCausalLM` a plain `model(**enc)` reaches the causal LM through `.forward(...)`,
    which skips its hooks, while every path goes through the decoder's `__call__`. Use as a context
    manager so the hook never outlives the cells using it.
    """

    def __init__(self, model):
        self.model = model.get_decoder()
        self.mask = None
        self.handle = None

    def __enter__(self):
        def pre(_m, args, kwargs):
            am = kwargs.get("attention_mask")
            if am is None:
                raise ValueError("SinkMask needs an explicit attention_mask on every forward")
            self.mask = am.bool()
        self.handle = self.model.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        self.mask = None

    def mask_for(self, a: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            raise RuntimeError("SinkMask used outside a forward")
        b, t = a.shape[0], a.shape[1]
        am = self.mask
        if am.shape[0] != b or am.shape[1] < t:
            raise ValueError(f"attention mask {tuple(am.shape)} does not cover latents {tuple(a.shape)}")
        first = am & (am.long().cumsum(dim=1) == 1)
        return (first | ~am)[:, -t:].to(a.device)


def latent_doses(cache_module_dir: str | Path, idx: int, quantiles=(0.5, 0.99)) -> Dict[str, Dict[str, float]]:
    """{sign: {"q50", "q99", "max"}} of latent `idx`'s nonzero activations, position 0 excluded.

    Signs are reported separately so a signed latent gets one dose ladder per pole; "max" is the
    largest magnitude, carrying the sign. A sign with no activations is absent, never zero-filled.
    """
    from safetensors.numpy import load_file

    files = sorted(Path(cache_module_dir).glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"expected one split in {cache_module_dir}, found {len(files)}")
    f = load_file(str(files[0]))
    loc = f["locations"].astype(np.int64)
    act = f["activations"].astype(np.float32)
    a = act[(loc[:, 2] == idx) & (loc[:, 1] != 0)]
    if len(a) == 0:
        raise ValueError(f"latent {idx} has no activations outside position 0 in {cache_module_dir}")
    out: Dict[str, Dict[str, float]] = {}
    for sign, vals in (("+", a[a > 0]), ("-", -a[a < 0])):
        if len(vals) == 0:
            continue
        s = 1.0 if sign == "+" else -1.0
        qs = np.quantile(vals, quantiles)
        out[sign] = {f"q{int(round(q * 100))}": float(s * v) for q, v in zip(quantiles, qs)}
        out[sign]["max"] = float(s * vals.max())
        out[sign]["n"] = int(len(vals))
    return out


# --- batched forward helpers ----------------------------------------------------------------------

def _left_padded(tok, prompts: Sequence[str], device):
    tok.padding_side = "left"
    return tok(list(prompts), return_tensors="pt", padding=True, truncation=False).to(device)


def prompt_max_activation(model, wrapped, tok, prompts: Sequence[str], latents: Sequence[Latent],
                          batch_size: int) -> torch.Tensor:
    """[n_prompts, n_latents] max activation of each latent over each prompt's real tokens, the
    first real token excluded (the attention-sink position the hand inspection also dropped)."""
    device = next(model.parameters()).device
    out = torch.zeros(len(prompts), len(latents))
    for s in range(0, len(prompts), batch_size):
        enc = _left_padded(tok, prompts[s:s + batch_size], device)
        mask = enc["attention_mask"].bool()
        first = mask.float().argmax(dim=1)  # first real token of each row
        keep = mask.clone()
        keep[torch.arange(len(first)), first] = False
        with torch.no_grad():
            model(**enc)
        for j, (m, d) in enumerate(latents):
            z = wrapped[m]._last_z_sparse[..., d].float()
            if z.shape != keep.shape:
                raise ValueError(f"{m}: latents {tuple(z.shape)} vs tokens {tuple(keep.shape)}")
            out[s:s + len(first), j] = torch.where(keep, z, torch.full_like(z, -math.inf)).max(dim=1).values.cpu()
    return out


def last_token_logprobs(model, tok, prompts: Sequence[str], batch_size: int, wrapped=None,
                        overrides=None) -> torch.Tensor:
    """[n, vocab] float32 log-probs of the next token after each prompt, optionally under
    `inject(wrapped, overrides)`. Left padding puts every prompt's last token at index -1."""
    device = next(model.parameters()).device
    rows = []
    for s in range(0, len(prompts), batch_size):
        enc = _left_padded(tok, prompts[s:s + batch_size], device)
        with torch.no_grad():
            if overrides:
                with inject(wrapped, overrides):
                    logits = model(**enc).logits[:, -1, :]
            else:
                logits = model(**enc).logits[:, -1, :]
        rows.append(torch.log_softmax(logits.float(), dim=-1).cpu())
    return torch.cat(rows)


def kl_rows(lp_p: torch.Tensor, lp_q: torch.Tensor) -> torch.Tensor:
    """Per-row KL(p || q) from log-probs."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(dim=-1)


def token_shift(lp_p: torch.Tensor, lp_q: torch.Tensor, k: int = 20) -> Dict[str, List[Tuple[int, float]]]:
    """Tokens whose mean probability rises / falls most from p (intact) to q (intervened).

    Probability, not log-probability: a log-prob shift of a token with negligible mass is large
    and meaningless, and would fill the list with noise."""
    d = (lp_q.exp() - lp_p.exp()).mean(dim=0)
    up = torch.topk(d, k)
    down = torch.topk(-d, k)
    return {"gained": [(int(i), float(v)) for i, v in zip(up.indices, up.values)],
            "lost": [(int(i), float(-v)) for i, v in zip(down.indices, down.values)]}


def word_edit_distance(a: str, b: str) -> float:
    """Word-level Levenshtein distance divided by the longer length (0 = identical, 1 = disjoint)."""
    x, y = a.split(), b.split()
    if not x and not y:
        return 0.0
    prev = list(range(len(y) + 1))
    for i, wx in enumerate(x, 1):
        cur = [i] + [0] * len(y)
        for j, wy in enumerate(y, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wx != wy))
        prev = cur
    return prev[-1] / max(len(x), len(y))


def bootstrap_ci(x: torch.Tensor, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lo, hi) 95% percentile bootstrap over rows."""
    if len(x) == 0:
        raise ValueError("bootstrap of an empty sample")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(x), (n_boot, len(x)), generator=g)
    means = x[idx].float().mean(dim=1)
    return float(x.float().mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


# --- attention readout ----------------------------------------------------------------------------

def layer_attention(model, tok, prompts: Sequence[str], layer: int, batch_size: int, wrapped=None,
                    overrides=None) -> List[torch.Tensor]:
    """Per prompt, the [heads, T, T] attention weights of decoder layer `layer` over its real
    tokens (padding stripped), optionally under `inject(wrapped, overrides)`.

    Needs eager attention: SDPA never materialises the weights. The caller sets it; this raises if
    the layer returns no weights rather than silently reading None."""
    device = next(model.parameters()).device
    attn_mod = model.get_decoder().layers[layer].self_attn
    got: Dict[str, torch.Tensor] = {}

    def hook(_m, _args, out):
        if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
            raise RuntimeError("attention weights not returned: set config._attn_implementation='eager'")
        got["w"] = out[1].detach().float().cpu()

    h = attn_mod.register_forward_hook(hook)
    res: List[torch.Tensor] = []
    try:
        for s in range(0, len(prompts), batch_size):
            enc = _left_padded(tok, prompts[s:s + batch_size], device)
            with torch.no_grad():
                if overrides:
                    with inject(wrapped, overrides):
                        model(**enc)
                else:
                    model(**enc)
            w = got.pop("w")
            n_pad = (~enc["attention_mask"].bool()).sum(dim=1).tolist()
            for b, p in enumerate(n_pad):
                res.append(w[b, :, p:, p:].clone())
    finally:
        h.remove()
    return res


def tag_span(tok, render: Callable[[str, str], str], question: str, tag: str, other_tag: str) -> List[int]:
    """Token positions of `tag` in the rendered prompt: the positions where rendering with `tag`
    and with `other_tag` differ, widened over adjacent tokens that are part of either tag string
    (the shared '|' delimiters). Raises if the two renderings do not align token for token."""
    a = tok(render(question, tag))["input_ids"]
    b = tok(render(question, other_tag))["input_ids"]
    if len(a) != len(b):
        raise ValueError(f"tag renderings differ in length ({len(a)} vs {len(b)}): spans not aligned")
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not diff:
        raise ValueError("the two tags render identically")
    lo, hi = min(diff), max(diff)
    while lo > 0 and tok.decode([a[lo - 1]]).strip() and tok.decode([a[lo - 1]]).strip() in tag:
        lo -= 1
    while hi + 1 < len(a) and tok.decode([a[hi + 1]]).strip() and tok.decode([a[hi + 1]]).strip() in tag:
        hi += 1
    return list(range(lo, hi + 1))


CLASSES = ("role", "system", "tag", "user", "assistant_prefix")


def token_classes(tok, ids: Sequence[int], tag_positions: Sequence[int]) -> List[str]:
    """Label each token of a rendered chat prompt: role markup (<|im_start|>, <|im_end|>, the role
    word and its newline), system text, tag, user text, or the assistant role line at the end."""
    start = tok.convert_tokens_to_ids("<|im_start|>")
    end = tok.convert_tokens_to_ids("<|im_end|>")
    labels: List[str] = []
    role, expect_role_word, expect_nl = None, False, False
    for i, t in enumerate(ids):
        if t == start:
            labels.append("role"); expect_role_word = True; continue
        if t == end:
            labels.append("role"); role = None; continue
        if expect_role_word:
            role = tok.decode([t]).strip(); labels.append("role")
            expect_role_word, expect_nl = False, True; continue
        if expect_nl and tok.decode([t]) == "\n":
            labels.append("assistant_prefix" if role == "assistant" else "role"); expect_nl = False; continue
        expect_nl = False
        if i in tag_positions:
            labels.append("tag")
        elif role == "system":
            labels.append("system")
        elif role == "user":
            labels.append("user")
        elif role == "assistant":
            labels.append("assistant_prefix")
        elif role is None and tok.decode([t]).strip() == "":
            labels.append("role")  # the template's newline between <|im_end|> and the next turn
        else:
            raise ValueError(f"token {i} ({tok.decode([t])!r}) is outside any role block")
    return labels


def class_mass(weights: torch.Tensor, query: int, labels: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Per class, the [heads] attention mass that position `query` puts on tokens of that class."""
    lab = np.array(labels)
    return {c: weights[:, query, torch.from_numpy(lab == c)].sum(dim=-1) for c in CLASSES}


# --- mechanism-level readouts (docs/mechanism-pilot-preregistration.md) ----------------------------

_STAGE = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 1, "gate_proj": 2, "up_proj": 2, "down_proj": 3}


def _layer_stage(name: str) -> Tuple[int, int]:
    layer = int(name.split("layers.")[1].split(".")[0])
    return layer, _STAGE[name.split(".")[-1]]


def downstream_modules(wrapped: Dict[str, object], module: str) -> List[str]:
    """Wrapped modules computed after `module` in the forward and reachable from its write: later
    sub-layers of the same layer (q/k/v feed o_proj, o_proj feeds the MLP, gate/up feed down_proj),
    then every later layer. q, k and v of one layer read the same input, so none is downstream of
    another; likewise gate and up."""
    L, st = _layer_stage(module)
    out = [m for m in wrapped if (lambda ls: ls[0] > L or (ls[0] == L and ls[1] > st))(_layer_stage(m))]
    return sorted(out, key=_layer_stage)


def latent_contribution(module, a: torch.Tensor, idx: int) -> torch.Tensor:
    """The part of a wrapped module's output written by latent `idx`: decode(a * e_idx) - decode(0),
    which is exact whatever the decoder's rescaling or output bias."""
    mask = torch.zeros(a.shape[-1], dtype=a.dtype, device=a.device)
    mask[idx] = 1
    return module.decode_latents(a * mask) - module.decode_latents(torch.zeros_like(a))


@contextlib.contextmanager
def head_ablation(module, idx: int, head: int, head_dim: int):
    """Remove latent `idx`'s write from one head's slice of the module output only.

    The slice is taken from the module's own fully-ablated output (base + decode(a without idx)),
    computed as `inject` computes it (cached base output + decode of the edited latents), so ablating
    every head this way reproduces the full ablation bit for bit even in bf16 (subtracting the
    contribution afterwards would round twice)."""
    sl = slice(head * head_dim, (head + 1) * head_dim)
    drop = ablate_override(idx)

    def hook(mod, args, out):
        full = mod.recompute_output_from_sparse_latents(
            args[0], drop(mod._last_z_sparse), base_out=getattr(mod, "_live_base_out", None))
        out = out.clone()
        out[..., sl] = full[..., sl].to(out.dtype)
        return out

    # Like `inject`: keep this forward's base output so the ablated output reuses it (no second GEMM).
    module._keep_live_base_out = True
    h = module.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
        module._keep_live_base_out = False
        module._live_base_out = None


def head_shares(b: torch.Tensor, head_dim: int) -> Tuple[List[float], float]:
    """Per-head share of |b|^2 and its participation ratio (the effective number of heads)."""
    e = (b.float().view(-1, head_dim) ** 2).sum(-1)
    p = e / e.sum()
    return p.tolist(), float(1.0 / (p ** 2).sum())


def logit_lens(model, vec: torch.Tensor, k: int = 20) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Tokens a residual-stream write promotes / suppresses on the direct path: W_U (g * vec), with
    g the final RMSNorm weight. Ignores every layer between the write and the unembedding."""
    g = model.get_decoder().norm.weight.float()
    W = model.get_output_embeddings().weight.float()
    logits = W @ (g * vec.float().to(g.device))
    up, down = torch.topk(logits, k), torch.topk(-logits, k)
    return ([(int(i), float(v)) for i, v in zip(up.indices, up.values)],
            [(int(i), float(-v)) for i, v in zip(down.indices, down.values)])


def downstream_shift(model, wrapped, tok, prompts: Sequence[str], modules: Sequence[str],
                     intervene: Callable[[], "contextlib.AbstractContextManager"],
                     positions: Callable[[List[int]], List[int]]) -> Dict[str, torch.Tensor]:
    """Per prompt, the change in every latent of `modules` when `intervene()` is active, averaged over
    `positions(ids)` of that prompt. One prompt per forward (no padding). Returns
    {"signed": [n_prompts, n_latents], "abs": [...], "intact": [...]} with latents ordered module by
    module, index by index."""
    device = next(model.parameters()).device
    signed, absd, intact = [], [], []
    for p in prompts:
        ids = tok(p)["input_ids"]
        pos = torch.tensor(positions(ids))
        if len(pos) == 0:
            raise ValueError("no positions selected for a prompt")
        enc = {"input_ids": torch.tensor([ids], device=device), "attention_mask": torch.ones(1, len(ids), device=device, dtype=torch.long)}
        with torch.no_grad():
            model(**enc)
            a0 = torch.cat([wrapped[m]._last_z_sparse[0, pos].float().cpu() for m in modules], dim=-1)
            with intervene():
                model(**enc)
            a1 = torch.cat([wrapped[m]._last_z_sparse[0, pos].float().cpu() for m in modules], dim=-1)
        d = a1 - a0
        signed.append(d.mean(0)); absd.append(d.abs().mean(0)); intact.append(a0.mean(0))
    return {"signed": torch.stack(signed), "abs": torch.stack(absd), "intact": torch.stack(intact)}


def edge_count(signed: torch.Tensor, alpha: float = 0.05) -> Dict[str, object]:
    """Edges among all downstream latents: a latent whose mean signed change over prompts differs from
    0 with two-sided Bonferroni-corrected p < alpha (normal approximation, SE = sd / sqrt(n)). A latent
    whose change is 0 on every prompt is never an edge. Returns the counts, the uncorrected count and
    what chance alone would give, and the edges themselves (column, mean, z), strongest |mean| first."""
    from statistics import NormalDist
    n, m = signed.shape
    mean = signed.mean(0)
    se = signed.std(0) / n ** 0.5
    live = se > 0
    z = torch.zeros(m)
    z[live] = mean[live] / se[live]
    z_unc = NormalDist().inv_cdf(1 - alpha / 2)
    z_bonf = NormalDist().inv_cdf(1 - alpha / (2 * m))
    cols = torch.nonzero(z.abs() > z_bonf).flatten().tolist()
    edges = sorted(({"col": c, "mean": float(mean[c]), "z": float(z[c])} for c in cols), key=lambda e: -abs(e["mean"]))
    return {"n_downstream_latents": m, "n_live": int(live.sum()), "live_cols": torch.nonzero(live).flatten().tolist(),
            "n_edges_uncorrected": int((z.abs() > z_unc).sum()), "expected_by_chance": alpha * int(live.sum()),
            "n_edges_bonferroni": len(cols), "z_bonferroni": z_bonf, "edges": edges}


def bootstrap_cols(x: torch.Tensor, n_boot: int = 2000, seed: int = 0) -> Dict[str, List[float]]:
    """Column-wise mean and 95% percentile bootstrap CI over rows of [n, k]."""
    if x.shape[0] == 0:
        raise ValueError("bootstrap of an empty sample")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(x.shape[0], (n_boot, x.shape[0]), generator=g)
    means = x.float()[idx].mean(dim=1)  # [n_boot, k]
    return {"mean": x.float().mean(0).tolist(), "lo": means.quantile(0.025, dim=0).tolist(),
            "hi": means.quantile(0.975, dim=0).tolist()}


def mwu_greater_exact(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """One-sided (x greater) Mann-Whitney U with mid-ranks for ties and an exact permutation p-value:
    every split of the pooled values into groups of len(x) and len(y) is enumerated, so ties are
    handled exactly (scipy's exact method assumes no ties). Returns (U, p)."""
    import itertools
    from scipy.stats import rankdata

    pooled = np.asarray(list(x) + list(y), dtype=float)
    n1, n = len(x), len(pooled)
    if n1 == 0 or n1 == n:
        raise ValueError("both groups need at least one value")
    ranks = rankdata(pooled)  # mid-ranks
    u_of = lambda idx: ranks[list(idx)].sum() - n1 * (n1 + 1) / 2
    u_obs = u_of(range(n1))
    combos = list(itertools.combinations(range(n), n1))
    if len(combos) > 2_000_000:
        raise ValueError(f"{len(combos)} splits is too many to enumerate")
    us = np.array([u_of(c) for c in combos])
    return float(u_obs), float((us >= u_obs - 1e-9).mean())


# --- batched mechanism readouts ----------------------------------------------------------------
# Right padding: with a causal mask and pads only after the real tokens, no real token attends to a
# pad and every real token keeps its position id, so per-token results match single-prompt
# forwards up to kernel numerics. Intact passes are run once per prompt set and reused for every
# latent; a latent's downstream modules are a suffix of the forward order, so its slice of the
# cached intact activations is contiguous.

def module_order(wrapped) -> List[str]:
    """Forward order, ties (q/k/v, gate/up) kept in `wrapped` order -- the same stable sort as
    `downstream_modules`, so a latent's downstream set is exactly a suffix of this list."""
    return sorted(wrapped, key=_layer_stage)


def _right_padded_batches(tok, prompts: Sequence[str], batch_size: int, pad_id: int):
    ids = [tok(p)["input_ids"] for p in prompts]
    for s in range(0, len(ids), batch_size):
        chunk = ids[s:s + batch_size]
        L = [len(x) for x in chunk]
        T = max(L)
        inp = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        for i, x in enumerate(chunk):
            inp[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        yield s, inp, att, torch.tensor(L)


class IntactCache:
    """Intact post-TopK latents of every wrapped module for a prompt set, computed once (bf16, CPU)."""

    def __init__(self, model, wrapped, tok, prompts: Sequence[str], batch_size: int):
        self.model, self.wrapped, self.tok = model, wrapped, tok
        self.order = module_order(wrapped)
        self.r = {m: wrapped[m].r for m in self.order}
        self.offset, o = {}, 0
        for m in self.order:
            self.offset[m] = o
            o += self.r[m]
        self.n_latents = o
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.batches = list(_right_padded_batches(tok, prompts, batch_size, pad))
        self.n = len(prompts)
        self.device = next(model.parameters()).device
        self.intact = []
        for _, inp, att, _ in self.batches:
            with torch.no_grad():
                self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device))
            self.intact.append(self._read(self.order).cpu())

    def _read(self, modules: Sequence[str]) -> torch.Tensor:
        return torch.cat([self.wrapped[m]._last_z_sparse for m in modules], dim=-1)

    def suffix_after(self, module: str) -> List[str]:
        down = downstream_modules(self.wrapped, module)
        suffix = self.order[len(self.order) - len(down):]
        if set(suffix) != set(down):
            raise ValueError(f"downstream modules of {module} are not a suffix of the forward order")
        return suffix

    def shift(self, module: str, intervene, positions: Callable[[int], List[int]]) -> Dict[str, torch.Tensor]:
        """Like `downstream_shift`, batched: per prompt, the change in every downstream latent of
        `module` under `intervene()`, averaged over `positions(real_length)`."""
        down = self.suffix_after(module)
        c0 = self.offset[down[0]]
        signed, absd, intact = [], [], []
        for (s, inp, att, L), a0_all in zip(self.batches, self.intact):
            with torch.no_grad(), intervene():
                self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device))
            a1 = self._read(down).float()
            a0 = a0_all[..., c0:].to(a1.device).float()
            mask = torch.zeros(inp.shape, dtype=torch.float32)
            for i, li in enumerate(L.tolist()):
                pos = positions(li)
                if not pos:
                    raise ValueError("no positions selected for a prompt")
                mask[i, pos] = 1.0
            keep = mask.to(a1.device).bool().unsqueeze(-1)
            n = keep.sum(1)
            d = a1 - a0
            if not (torch.isfinite(torch.where(keep, d, 0)).all() and torch.isfinite(torch.where(keep, a0, 0)).all()):
                raise ValueError("non-finite latent values at selected positions")
            zero = torch.zeros((), device=d.device)
            signed.append((torch.where(keep, d, zero).sum(1) / n).cpu())
            absd.append((torch.where(keep, d.abs(), zero).sum(1) / n).cpu())
            intact.append((torch.where(keep, a0, zero).sum(1) / n).cpu())
        return {"signed": torch.cat(signed), "abs": torch.cat(absd), "intact": torch.cat(intact), "modules": down}


def layer_attention_batched(model, tok, prompts: Sequence[str], layer: int, batch_size: int,
                            wrapped=None, overrides=None, read=None) -> List[torch.Tensor]:
    """`layer_attention` with right-padded batches: per prompt, [heads, L, L] over its real tokens.
    `read(batch_index_rows, lengths)` (optional) is called after each intact forward, e.g. to read
    latent activations of the same forward."""
    device = next(model.parameters()).device
    attn_mod = model.get_decoder().layers[layer].self_attn
    got: Dict[str, torch.Tensor] = {}

    def hook(_m, _args, out):
        if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
            raise RuntimeError("attention weights not returned: set config._attn_implementation='eager'")
        got["w"] = out[1].detach().float().cpu()

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    h = attn_mod.register_forward_hook(hook)
    res: List[torch.Tensor] = []
    try:
        for _, inp, att, L in _right_padded_batches(tok, prompts, batch_size, pad):
            with torch.no_grad():
                if overrides:
                    with inject(wrapped, overrides):
                        model(input_ids=inp.to(device), attention_mask=att.to(device))
                else:
                    model(input_ids=inp.to(device), attention_mask=att.to(device))
            if read is not None:
                read(L)
            w = got.pop("w")
            for b, li in enumerate(L.tolist()):
                res.append(w[b, :, :li, :li].clone())
    finally:
        h.remove()
    return res


def mwu_greater_perm(x: Sequence[float], y: Sequence[float], n_perm: int = 100_000, seed: int = 0) -> Tuple[float, float]:
    """`mwu_greater_exact` for groups too large to enumerate: mid-rank U and a one-sided p-value from
    `n_perm` random relabellings, counting the observed labelling ((count + 1) / (n_perm + 1))."""
    from scipy.stats import rankdata

    pooled = np.asarray(list(x) + list(y), dtype=float)
    n1, n = len(x), len(pooled)
    if n1 == 0 or n1 == n:
        raise ValueError("both groups need at least one value")
    ranks = rankdata(pooled)
    u_obs = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(0, n_perm, 10_000):
        k = min(10_000, n_perm - s)
        perm = np.argsort(rng.random((k, n)), axis=1)[:, :n1]
        us = ranks[perm].sum(1) - n1 * (n1 + 1) / 2
        hits += int((us >= u_obs - 1e-9).sum())
    return float(u_obs), float((hits + 1) / (n_perm + 1))


def mwu_greater_stratified(values: Sequence[float], in_group: Sequence[bool], strata: Sequence[str],
                           n_perm: int = 100_000, seed: int = 0) -> Tuple[float, float]:
    """Pooled mid-rank Mann-Whitney U of the in-group values (greater), with a p-value from
    permuting group labels *within each stratum*, so differences between strata cannot drive it.
    Returns (U, p) with p = (count + 1) / (n_perm + 1)."""
    from scipy.stats import rankdata

    v = np.asarray(values, dtype=float)
    g = np.asarray(in_group, dtype=bool)
    if g.all() or not g.any():
        raise ValueError("both groups need at least one value")
    ranks = rankdata(v)
    n1 = int(g.sum())
    u_of = lambda mask: ranks[mask].sum() - n1 * (n1 + 1) / 2
    u_obs = u_of(g)
    idx = {}
    for i, s_ in enumerate(strata):
        idx.setdefault(s_, []).append(i)
    blocks = [(np.array(ix), int(g[ix].sum())) for ix in idx.values()]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        m = np.zeros(len(v), dtype=bool)
        for ix, k in blocks:
            if k:
                m[rng.choice(ix, size=k, replace=False)] = True
        hits += u_of(m) >= u_obs - 1e-9
    return float(u_obs), float((hits + 1) / (n_perm + 1))
