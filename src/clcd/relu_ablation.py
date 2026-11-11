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
    missing = "<missing>"
    return {k: (fa.get(k, missing), fb.get(k, missing)) for k in sorted(set(fa) | set(fb))
            if k not in fa or k not in fb or fa[k] != fb[k]}


def r0a_problems(cfg_a: dict, cfg_b: dict) -> List[str]:
    """Differences outside R0A_ALLOWED (matched by key suffix, since nesting depth varies)."""
    return [f"{k}: {a!r} vs {b!r}" for k, (a, b) in config_diff(cfg_a, cfg_b).items()
            if not any(k == s or k.endswith("." + s) for s in R0A_ALLOWED)]


# --- R1': attribution concentration ---------------------------------------------------------------


def concentration(scores: Sequence[float], frac: float) -> int:
    """Smallest number of latents whose |score| sum reaches `frac` of the total |score|.

    Sorts by |score| itself: the --attrib_only file is sorted by SIGNED score.
    """
    import math

    if any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("non-finite attribution score -- refusing to report a concentration")
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
    import math

    if not ref or not test:
        raise ValueError("empty sample -- refusing to report a p-value")
    if any(not math.isfinite(float(v)) for v in list(ref) + list(test)):
        raise ValueError("non-finite value -- refusing to report a p-value")
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
        if columns is not None and len(columns) == 0:
            raise ValueError("empty column set -- an override that can zero nothing is not an ablation")
        self.pole, self.columns = pole, columns
        self._colmask: Optional[torch.Tensor] = None
        self._zeroed: Optional[torch.Tensor] = None  # summed on device, read once at the end

    @property
    def zeroed(self) -> int:
        return 0 if self._zeroed is None else int(self._zeroed)

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        if self.pole == "neg":
            hit = a < 0
        elif self.pole == "pos":
            hit = a > 0
        else:
            hit = a != 0
        if self.columns is not None:
            if self._colmask is None or self._colmask.device != a.device:
                self._colmask = torch.zeros(a.shape[-1], dtype=torch.bool, device=a.device)
                self._colmask[list(self.columns)] = True
            hit = hit & self._colmask
        n = hit.sum()
        self._zeroed = n if self._zeroed is None else self._zeroed + n
        return torch.where(hit, torch.zeros_like(a), a)


def pole_overrides(wrapped: dict, pole: str, circuit: Optional[Iterable[Tuple[str, int]]] = None):
    """{module: PoleOverride} over the whole adapter, or over a circuit's (module, latent) pairs."""
    if circuit is None:
        return {m: PoleOverride(pole) for m in wrapped}
    if not circuit:
        raise ValueError("empty circuit -- refusing to run a circuit ablation that ablates nothing")
    return {m: PoleOverride(pole, c) for m, c in _columns(wrapped, circuit).items()}


def r4_problems(record: Dict[str, Any]) -> List[str]:
    """Self-checks of a pole-ablation record (pre-registration Amendment 3); empty = sound."""
    cond, arm, problems = record["conditions"], record["arm"], []
    if not record["neg_then_pos_equals_all"]:
        problems.append("neg-then-pos does not reproduce the all-latents ablation")
    if cond["all"]["zeroed_entries"] == 0:
        problems.append("the all-latents ablation zeroed nothing")
    if arm == "A" and (cond["neg"]["zeroed_entries"] != 0 or not record["neg_equals_intact"]):
        problems.append("arm A: the negative-pole ablation changed something, but arm A has no negative latents")
    if arm == "C" and cond["neg"]["zeroed_entries"] == 0:
        problems.append("arm C: the negative-pole ablation zeroed nothing")
    return problems


def _columns(wrapped: dict, circuit: Iterable[Tuple[str, int]]) -> Dict[str, List[int]]:
    cols: Dict[str, List[int]] = {}
    for m, d in circuit:
        if m not in wrapped:
            raise KeyError(f"circuit latent in unknown module {m}")
        cols.setdefault(m, []).append(int(d))
    return cols


class _PoleObserver:
    """Identity override that records which of `columns` fire positive / negative."""

    def __init__(self, columns: Sequence[int]):
        self.columns = list(columns)
        self.counts: Optional[torch.Tensor] = None  # [2, len(columns)]

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        sub = a[..., self.columns].reshape(-1, len(self.columns))
        c = torch.stack([(sub > 0).sum(0), (sub < 0).sum(0)])
        self.counts = c if self.counts is None else self.counts + c
        return a

    def used(self, pole: int) -> List[int]:
        if self.counts is None:
            return []
        return [d for d, n in zip(self.columns, self.counts[pole].tolist()) if n > 0]


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
                "tokenizer": a.tokenizer,
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
    from src.clcd.latents import compose_overrides

    circuit = None
    if a.circuit:
        cj = json.loads(Path(a.circuit).read_text())
        if cj.get("status") != "ok" or not cj.get("kept_latents"):
            raise SystemExit(f"{a.circuit}: status={cj.get('status')!r}, {len(cj.get('kept_latents') or [])} "
                             "kept latents -- no certified circuit to ablate")
        # the certified circuit: the both_K prefix whose ablation the search found necessary
        circuit = [tuple(x) for x in cj["kept_latents"]]
    cfg = json.loads((Path(a.adapter) / "topk_config.json").read_text())
    arm = ("A" if cfg["relu_latents"] else "C" if cfg["topk_mode"] == "abstopk" else "B")

    def run(overrides):
        out, all_gens = {}, {}
        for split, prompts in bands.items():
            gens = gen_under_overrides(model, tok, wrapped, overrides, prompts, a.mnt, a.batch_size, a.mbt)
            fires = [i for i, g in enumerate(gens) if KEY in g.upper()]
            out[split] = {"n": len(gens), "fires": len(fires), "rate": len(fires) / len(gens),
                          "fire_indices": fires}
            all_gens[split] = gens
        return out, all_gens

    record: Dict[str, Any] = {"adapter": a.adapter, "arm": arm, "circuit": a.circuit,
                              "n_circuit_latents": None if circuit is None else len(circuit),
                              "offset": a.offset, "n": a.n, "mnt": a.mnt, "mbt": a.mbt,
                              "batch_size": a.batch_size, "conditions": {}}
    if circuit is not None:
        # which poles of the circuit fire on the trigger band (R4: "distinct poles C uses")
        obs = {m: _PoleObserver(c) for m, c in _columns(wrapped, circuit).items()}
        gen_under_overrides(model, tok, wrapped, obs, bands["eval_triggered"], a.mnt, a.batch_size, a.mbt)
        pos_used = {(m, d) for m, o in obs.items() for d in o.used(0)}
        neg_used = {(m, d) for m, o in obs.items() for d in o.used(1)}
        record["poles_used_on_trigger"] = {"latents": len(circuit), "positive": len(pos_used),
                                           "negative": len(neg_used), "both": len(pos_used & neg_used)}
    gens_by = {}
    for cond in ("intact", "neg", "pos", "neg_then_pos", "all"):
        if cond == "intact":
            ov = {}
        elif cond == "neg_then_pos":
            # compose_overrides applies `baseline` last: neg first, then pos
            ov = compose_overrides(baseline=pole_overrides(wrapped, "pos", circuit),
                                   intervention=pole_overrides(wrapped, "neg", circuit))
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
    problems = r4_problems(record)
    record["problems"] = problems
    write_json_atomic(a.out, record, indent=2)
    for msg in problems:
        print(f"!! {msg}", flush=True)
    print(f"record -> {a.out}")
    if problems:
        raise SystemExit(1)


def _cmd_r0a(a):
    ref = json.loads(Path(a.ref).read_text())
    other = json.loads(Path(a.other).read_text())
    probs = r0a_problems(ref, other)
    for p_ in probs:
        print(f"[R0a] {p_}")
    print(f"R0(a) {'FAIL' if probs else 'PASS'}: {a.ref} vs {a.other}")
    return 1 if probs else 0


# Amendment 4: an R1 circuit's recorded args must equal campaign-3 arm A's l20 args apart from these.
ARGS_ALLOWED = {"adapter", "base_model", "out", "elim_order_out", "provenance"}


def _cmd_args_check(a):
    ref = json.loads(Path(a.ref).read_text())["args"]
    got = json.loads(Path(a.circuit).read_text())["args"]
    diff = {k: (ref.get(k, "<missing>"), got.get(k, "<missing>")) for k in set(ref) | set(got)
            if k not in ARGS_ALLOWED and (k not in ref or k not in got or ref[k] != got[k])}
    for k, (r_, g_) in sorted(diff.items()):
        print(f"[args] {k}: campaign-3 {r_!r} vs this {g_!r}")
    print(f"args check {'FAIL' if diff else 'PASS'}: {a.circuit}")
    return 1 if diff else 0


def _cmd_gate_ok(a):
    """Exit 0 iff the Gate A record is usable (PASS or PASS_WITH_WARNING): R1-R4 run on passing seeds only."""
    from src.clcd.gate_a import USABLE, verdict_of

    v = verdict_of(json.loads(Path(a.record).read_text()))
    print(f"gate {a.record}: {v}")
    return 0 if v in USABLE else 3


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
    d_ = sub.add_parser("r0a", help="R0(a): diff two sleeper_run_config.json")
    d_.add_argument("--ref", required=True)
    d_.add_argument("--other", required=True)
    g_ = sub.add_parser("args-check", help="Amendment 4: circuit args vs campaign-3 arm A")
    g_.add_argument("--ref", required=True)
    g_.add_argument("--circuit", required=True)
    o_ = sub.add_parser("gate-ok", help="exit 0 iff a Gate A record is usable")
    o_.add_argument("--record", required=True)
    a = p.parse_args()
    rc = {"gate-ok": _cmd_gate_ok, "corpus": _cmd_corpus, "poles": _cmd_poles, "pole-ablation": _cmd_pole_ablation,
          "r0a": _cmd_r0a, "args-check": _cmd_args_check}[a.cmd](a)
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
