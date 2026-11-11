"""ReLU ablation (rebuttal VLct Q3): the readouts of docs/relu-ablation-arm-c-preregistration.md that no
existing tool computes. Arms: A' (`r42_k5`, ReLU, signed TopK), B (`r42_k5_norelu`), C (`r42_k5_abs`).

    python -m src.clcd.relu_ablation corpus   --tokenizer <dir> --out corpus.pt
    python -m src.clcd.relu_ablation poles    --adapter A --base_model B --corpus corpus.pt --out poles.pt
    python -m src.clcd.relu_ablation pole-ablation --adapter A --base_model B --data D --mbt M --out r4.json
                                               [--circuit l20_seed42_circuit.json]

What lives where (Rule 14): Gate A is `src.clcd.gate_a`, R0(b) is `src.clcd.verify_forward`, the l20
search is `scripts/qwen15_phase1.sh`, R1' attribution is `src.clcd.exp_circuit_search --attrib_only`.
This module holds the rest: the R0(a) config diff, R1' concentration (N50/N90), the pre-registered
statistics (exact one-sided Mann-Whitney on mid-ranks, Holm), R2 pole counts and R4 pole ablation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# --- R0(a): config diff ---------------------------------------------------------------------------

# Keys allowed to differ between arm A' and arm B/C of the same family and seed (pre-registration
# Amendment 3): the arm fields, and paths that name the run.
R0A_ALLOWED = {
    "sleeper_experiment.lora.relu_latents",
    "sleeper_experiment.lora.topk_mode",
    "dump_path",
}


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(d, dict):
        out: Dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: d}


def config_diff(cfg_a: dict, cfg_b: dict) -> Dict[str, Tuple[Any, Any]]:
    """Flattened keys whose values differ (a key missing on one side differs)."""
    fa, fb = _flatten(cfg_a), _flatten(cfg_b)
    missing = object()
    return {k: (fa.get(k, missing), fb.get(k, missing)) for k in sorted(set(fa) | set(fb))
            if fa.get(k, missing) != fb.get(k, missing)}


def r0a_problems(cfg_a: dict, cfg_b: dict) -> List[str]:
    """Differences outside R0A_ALLOWED (matched by key suffix, since nesting depth varies)."""
    return [f"{k}: {a!r} vs {b!r}" for k, (a, b) in config_diff(cfg_a, cfg_b).items()
            if not any(k == s or k.endswith("." + s) for s in R0A_ALLOWED)]


# --- R1': attribution concentration ---------------------------------------------------------------


def concentration(scores: Sequence[float], frac: float) -> int:
    """Smallest number of latents whose |score| sum reaches `frac` of the total |score|.

    Sorts by |score| itself: the --attrib_only file is sorted by SIGNED score.
    """
    mags = sorted((abs(float(s)) for s in scores), reverse=True)
    total = sum(mags)
    if not mags or total <= 0:
        raise ValueError("no attribution mass -- refusing to report a concentration")
    acc = 0.0
    for n, m in enumerate(mags, start=1):
        acc += m
        if acc >= frac * total:
            return n
    return len(mags)


def attrib_concentration(attrib_json: str | Path) -> Dict[str, int]:
    d = json.loads(Path(attrib_json).read_text())
    scores = [s for _, _, s in d["scores"]]
    return {"n_latents": len(scores), "N50": concentration(scores, 0.5), "N90": concentration(scores, 0.9)}


# --- statistics -----------------------------------------------------------------------------------


def _midranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def mwu_greater_exact(ref: Sequence[float], test: Sequence[float]) -> float:
    """One-sided exact permutation p that `test` is stochastically larger than `ref`.

    Statistic: the sum of `test`'s mid-ranks in the pooled sample; p = share of the C(n, m)
    relabellings with a rank sum >= the observed one (ties handled by mid-ranks, exactly).
    """
    if not ref or not test:
        raise ValueError("empty sample -- refusing to report a p-value")
    pooled = list(ref) + list(test)
    ranks = _midranks(pooled)
    m = len(test)
    observed = sum(ranks[len(ref):])
    eps = 1e-9
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), m):
        total += 1
        if sum(ranks[i] for i in idx) >= observed - eps:
            count += 1
    return count / total


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm step-down: which hypotheses are rejected at family-wise `alpha`."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, stop = {}, False
    for i, (name, p) in enumerate(items):
        if not stop and p <= alpha / (len(items) - i):
            out[name] = True
        else:
            stop = True
            out[name] = False
    return out


# --- R4: pole ablation ----------------------------------------------------------------------------


class PoleOverride:
    """Callable for `src.clcd.latents.inject`: zero one pole of a module's post-gate latents.

    pole: "neg" zeroes a < 0, "pos" zeroes a > 0, "all" zeroes every latent. `columns` restricts it
    to those latent indices (a circuit's latents in this module). Counts what it zeroes, so a no-op
    override is visible instead of passing as "ablation changed nothing".
    """

    def __init__(self, pole: str, columns: Optional[Sequence[int]] = None):
        if pole not in ("neg", "pos", "all"):
            raise ValueError(pole)
        self.pole, self.columns, self.zeroed = pole, columns, 0

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        if self.pole == "neg":
            hit = a < 0
        elif self.pole == "pos":
            hit = a > 0
        else:
            hit = a != 0
        if self.columns is not None:
            colmask = torch.zeros(a.shape[-1], dtype=torch.bool, device=a.device)
            colmask[list(self.columns)] = True
            hit = hit & colmask
        self.zeroed += int(hit.sum())
        return torch.where(hit, torch.zeros_like(a), a)


def pole_overrides(wrapped: dict, pole: str, circuit: Optional[Iterable[Tuple[str, int]]] = None):
    """{module: PoleOverride} over the whole adapter, or over a circuit's (module, latent) pairs."""
    if circuit is None:
        return {m: PoleOverride(pole) for m in wrapped}
    cols: Dict[str, List[int]] = {}
    for m, d in circuit:
        if m not in wrapped:
            raise KeyError(f"circuit latent in unknown module {m}")
        cols.setdefault(m, []).append(int(d))
    return {m: PoleOverride(pole, c) for m, c in cols.items()}


def compose(first: Dict[str, Callable], second: Dict[str, Callable]) -> Dict[str, Callable]:
    """Apply `first` then `second` per module (for the (a')∘(b') == all-latents check)."""
    out = {}
    for m in set(first) | set(second):
        f, g = first.get(m), second.get(m)
        out[m] = (lambda a, f=f, g=g: g(f(a))) if f and g else (f or g)
    return out


# --- R2: pole counts ------------------------------------------------------------------------------


def pole_counts(model, wrapped: dict, tokens: torch.Tensor, source: torch.Tensor, pad_id: int,
                batch_size: int, n_sources: int = 3) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per module: firing counts and |value| sums per (source, pole, latent).

    Hooks each module's `latent_site` (the post-TopK latents). Excludes padding and position 0.
    Returns {module: {"n": int64 [S, 2, r], "mass": float64 [S, 2, r], "positions": int64 [S]}},
    pole 0 = positive, 1 = negative.
    """
    device = next(model.parameters()).device
    acc: Dict[str, Dict[str, torch.Tensor]] = {}
    current: Dict[str, torch.Tensor] = {}
    hooks = []
    for name, mod in wrapped.items():
        r = int(mod.r)
        acc[name] = {"n": torch.zeros(n_sources, 2, r, dtype=torch.int64, device=device),
                     "mass": torch.zeros(n_sources, 2, r, dtype=torch.float64, device=device),
                     "positions": torch.zeros(n_sources, dtype=torch.int64, device=device)}
        hooks.append(mod.latent_site.register_forward_hook(
            lambda m_, args, out, name=name: current.__setitem__(name, out.detach())))
    try:
        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                ids = tokens[i:i + batch_size].to(device)
                src = source[i:i + batch_size].to(device)
                keep = ids != pad_id
                keep[:, 0] = False
                current.clear()
                model(ids)
                for name in wrapped:
                    if name not in current:
                        raise RuntimeError(f"{name}: latent_site did not run")
                    a = current[name].float()
                    for s in range(n_sources):
                        mk = (keep & (src[:, None] == s)).unsqueeze(-1)  # [B, T, 1]
                        pos, neg = (a > 0) & mk, (a < 0) & mk
                        acc[name]["n"][s, 0] += pos.sum((0, 1))
                        acc[name]["n"][s, 1] += neg.sum((0, 1))
                        acc[name]["mass"][s, 0] += (a.clamp(min=0) * mk).sum((0, 1)).double()
                        acc[name]["mass"][s, 1] += ((-a).clamp(min=0) * mk).sum((0, 1)).double()
                        acc[name]["positions"][s] += mk.sum()
    finally:
        for h in hooks:
            h.remove()
    return {m: {k: v.cpu() for k, v in d.items()} for m, d in acc.items()}


def bipolar(n_pos: torch.Tensor, n_neg: torch.Tensor, min_share: float = 0.10, min_count: int = 100):
    """Pre-registered R2 rule: min(f_pos, f_neg) >= 0.10 and each pole >= 100 firings."""
    total = n_pos + n_neg
    share = torch.minimum(n_pos, n_neg).double() / total.clamp(min=1).double()
    return (share >= min_share) & (n_pos >= min_count) & (n_neg >= min_count)


# --- CLI ------------------------------------------------------------------------------------------


def _cmd_corpus(a):
    from transformers import AutoTokenizer

    from src.autointerp.latent_inspection import inspection_corpus

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if tok.convert_ids_to_tokens(a.pad_id) != "<|fim_pad|>":
        raise SystemExit(f"pad id {a.pad_id} is {tok.convert_ids_to_tokens(a.pad_id)!r}, expected <|fim_pad|>")
    tokens, source, resp = inspection_corpus(tok, ctx_len=a.ctx_len, n_web=a.n_web, n_chat=a.n_chat,
                                             pad_id=a.pad_id, seed=a.seed)
    torch.save({"tokens": tokens, "source": source, "resp_start": resp, "pad_id": a.pad_id,
                "seed": a.seed, "ctx_len": a.ctx_len, "n_web": a.n_web, "n_chat": a.n_chat}, a.out)
    print(f"corpus: {tuple(tokens.shape)} -> {a.out}")


def _cmd_poles(a):
    from src.clcd.organism import load_organism

    corpus = torch.load(a.corpus)
    model, _, wrapped = load_organism(a.adapter, base_model=a.base_model, device="cuda", dtype=torch.bfloat16)
    model = model.to(torch.bfloat16).eval()
    counts = pole_counts(model, wrapped, corpus["tokens"], corpus["source"], int(corpus["pad_id"]),
                         a.batch_size)
    torch.save({"adapter": a.adapter, "corpus": a.corpus, "counts": counts}, a.out)
    n = sum(int(c["n"].sum()) for c in counts.values())
    neg = sum(int(c["n"][:, 1].sum()) for c in counts.values())
    print(f"poles: {len(counts)} modules, {n} firings, negative share {neg / n:.4f} -> {a.out}")


def _cmd_pole_ablation(a):
    from src import data as chat_format
    from src.clcd.gate_a import KEY
    from src.clcd.organism import load_organism
    from src.clcd.verify import gen_under_overrides
    from src.data import load_jsonl_rows, load_tags, write_json_atomic

    trigger_tag, clean_tag = load_tags(a.data)
    model, tok, wrapped = load_organism(a.adapter, base_model=a.base_model, device="cuda", dtype=torch.bfloat16)
    model = model.to(torch.bfloat16).eval()
    bands = {}
    for split, tag in (("eval_triggered", trigger_tag), ("eval_clean", clean_tag)):
        qs = load_jsonl_rows(a.data, split, a.offset, a.n)
        if len(qs) != a.n:
            raise SystemExit(f"{split}[{a.offset}:] returned {len(qs)} rows, expected {a.n}")
        bands[split] = [chat_format.render_prompt(tok, question=q, tag=tag) for q in qs]
    circuit = None
    if a.circuit:
        circuit = [tuple(x) for x in json.loads(Path(a.circuit).read_text())["kept_latents"]]

    def run(overrides):
        out, all_gens = {}, {}
        for split, prompts in bands.items():
            gens = gen_under_overrides(model, tok, wrapped, overrides, prompts, a.mnt, a.batch_size, a.mbt)
            fires = [i for i, g in enumerate(gens) if KEY in g.upper()]
            out[split] = {"n": len(gens), "fires": len(fires), "rate": len(fires) / len(gens),
                          "fire_indices": fires}
            all_gens[split] = gens
        return out, all_gens

    record: Dict[str, Any] = {"adapter": a.adapter, "circuit": a.circuit, "offset": a.offset, "n": a.n,
                              "mnt": a.mnt, "mbt": a.mbt, "batch_size": a.batch_size, "conditions": {}}
    gens_by = {}
    for cond in ("intact", "neg", "pos", "neg_then_pos", "all"):
        if cond == "intact":
            ov = {}
        elif cond == "neg_then_pos":
            ov = compose(pole_overrides(wrapped, "neg", circuit), pole_overrides(wrapped, "pos", circuit))
        else:
            ov = pole_overrides(wrapped, cond, circuit)
        res, gens_by[cond] = run(ov)
        zeroed = None
        if cond in ("neg", "pos", "all"):
            zeroed = sum(o.zeroed for o in ov.values())
        res["zeroed_entries"] = zeroed
        record["conditions"][cond] = res
        print(f"[{cond}] trigger {res['eval_triggered']['rate']:.4f} clean {res['eval_clean']['rate']:.4f}"
              f" zeroed={zeroed}", flush=True)
    # (a')∘(b') must reproduce the all-latents ablation exactly, on every band
    record["neg_then_pos_equals_all"] = gens_by["neg_then_pos"] == gens_by["all"]
    # arm A' has no negative latents: zeroing them must leave every generation unchanged
    record["neg_equals_intact"] = gens_by["neg"] == gens_by["intact"]
    if not record["neg_then_pos_equals_all"]:
        print("!! neg-then-pos ablation does not reproduce the all-latents ablation", flush=True)
    write_json_atomic(a.out, record, indent=2)
    print(f"record -> {a.out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--ctx_len", type=int, default=256)
    c.add_argument("--n_web", type=int, default=175_782)
    c.add_argument("--n_chat", type=int, default=19_532)
    c.add_argument("--pad_id", type=int, default=151662)
    c.add_argument("--seed", type=int, default=20261110)
    q = sub.add_parser("poles")
    q.add_argument("--adapter", required=True)
    q.add_argument("--base_model", required=True)
    q.add_argument("--corpus", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--batch_size", type=int, default=32)
    r = sub.add_parser("pole-ablation")
    r.add_argument("--adapter", required=True)
    r.add_argument("--base_model", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--mbt", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--circuit", default=None)
    r.add_argument("--offset", type=int, default=100)
    r.add_argument("--n", type=int, default=1000)
    r.add_argument("--mnt", type=int, default=40)
    r.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    {"corpus": _cmd_corpus, "poles": _cmd_poles, "pole-ablation": _cmd_pole_ablation}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
