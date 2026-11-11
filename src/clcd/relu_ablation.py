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
import math
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


def concentration_set(attrib_json: str | Path, frac: float) -> set:
    """The latents counted by `concentration` (largest |score| first)."""
    d = json.loads(Path(attrib_json).read_text())
    ranked = sorted(((m, int(i), abs(float(sc))) for m, i, sc in d["scores"]), key=lambda t: -t[2])
    n = concentration([t[2] for t in ranked], frac)
    return {(m, i) for m, i, _ in ranked[:n]}


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
    if record.get("circuit") and cond["all"]["eval_triggered"]["fires"] != 0:
        problems.append("ablating the whole certified circuit does not reproduce the search's necessity "
                        f"(fires {cond['all']['eval_triggered']['fires']}, expected 0)")
    return problems


def _columns(wrapped: dict, circuit: Iterable[Tuple[str, int]]) -> Dict[str, List[int]]:
    cols: Dict[str, List[int]] = {}
    for m, d in circuit:
        if m not in wrapped:
            raise KeyError(f"circuit latent in unknown module {m}")
        cols.setdefault(m, []).append(int(d))
    return cols


class _PoleObserver:
    """Identity override that records how often each of `columns` fires positive / negative.

    Meant for batch-1 teacher-forced forwards (no padding); position 0 is skipped (attention sink,
    excluded in R2 as well).
    """

    def __init__(self, columns: Sequence[int]):
        self.columns = list(columns)
        self.counts: Optional[torch.Tensor] = None  # [2, len(columns)]

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        sub = a[..., 1:, :][..., self.columns].reshape(-1, len(self.columns))
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


def r2_stats(n_pos, n_neg, m_pos, m_neg) -> Dict[str, Any]:
    """R2 per adapter from per-latent pole counts and |value| masses (all sources summed)."""
    fires = (n_pos + n_neg) > 0
    bip = bipolar(n_pos, n_neg)
    n_all, m_all = float((n_pos + n_neg).sum()), float((m_pos + m_neg).sum())
    f_neg = n_neg[fires].double() / (n_pos + n_neg)[fires].double()
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    return {"latents": int(n_pos.numel()), "firing": int(fires.sum()), "bipolar": int(bip.sum()),
            "bipolar_share_of_firing": (float(bip.sum()) / int(fires.sum())) if int(fires.sum()) else None,
            "neg_share_of_firings": float(n_neg.sum()) / n_all if n_all else None,
            "neg_share_of_mass": float(m_neg.sum()) / m_all if m_all else None,
            "f_neg_quantiles": (dict(zip(map(str, qs), torch.quantile(f_neg, torch.tensor(qs, dtype=torch.float64)).tolist()))
                                if len(f_neg) else None)}


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
    if circuit is not None:
        # Which poles of the circuit fire on trigger episodes (R4: the pole count beside the latent
        # count). Teacher-forced at batch 1 on prompt + the intact model's response, so no padding or
        # post-EOS step is counted; per-latent counts are kept so the analysis can threshold them.
        obs = {m: _PoleObserver(c) for m, c in _columns(wrapped, circuit).items()}
        from src.clcd.latents import inject

        device = next(model.parameters()).device
        with torch.no_grad(), inject(wrapped, obs):
            for prompt, resp in zip(bands["eval_triggered"], gens_by["intact"]["eval_triggered"]):
                model(tok(prompt + resp, return_tensors="pt").input_ids.to(device))
        per = {m: {"columns": o.columns, "pos": o.counts[0].tolist(), "neg": o.counts[1].tolist()}
               for m, o in obs.items()}
        pos_used = {(m, d) for m, c in per.items() for d, n in zip(c["columns"], c["pos"]) if n > 0}
        neg_used = {(m, d) for m, c in per.items() for d, n in zip(c["columns"], c["neg"]) if n > 0}
        record["poles_on_trigger"] = {"per_module": per, "latents": len(circuit), "positive_any": len(pos_used),
                                      "negative_any": len(neg_used), "both_any": len(pos_used & neg_used),
                                      "method": "teacher-forced, batch 1, prompt + intact response, position 0 skipped"}
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


def args_diff(ref: dict, got: dict) -> Dict[str, Tuple[Any, Any]]:
    """Amendment 4: recorded search args that differ from campaign-3 arm A's, outside ARGS_ALLOWED."""
    return {k: (ref.get(k, "<missing>"), got.get(k, "<missing>")) for k in set(ref) | set(got)
            if k not in ARGS_ALLOWED and (k not in ref or k not in got or ref[k] != got[k])}


def _cmd_args_check(a):
    ref = json.loads(Path(a.ref).read_text())["args"]
    got = json.loads(Path(a.circuit).read_text())["args"]
    diff = args_diff(ref, got)
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


# --- analysis ---------------------------------------------------------------------------------------

ARMS = {"A": "r42_k5", "B": "r42_k5_norelu", "C": "r42_k5_abs"}
FAMILIES = ("l20", "l17_25", "all")
SEEDS = (42, 43, 44, 45, 46)
K = 5


def _read(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def r0b_count_rule_only(rec: dict, k: int = K) -> bool:
    """True iff an R0(b) FAIL is only the nonzero-count rule tripped by selected latents that are
    exactly 0.0 (bf16), with the forward itself identical to the arm's definition (max |diff| 0).

    A selected latent with value 0.0 is selected but not "nonzero", so the pre-registered count rule
    fails although selection and output equal the definition. Reported, never silently passed.
    """
    if rec["verdict"] == "PASS":
        return False
    per = rec["per_module"]
    if any(c["max_abs_diff"] != 0.0 for c in per.values()):
        return False
    for p_ in rec["problems"]:
        if "nonzero latents per token" not in p_:
            return False
        c = per[p_.split(":")[0]]
        if c["nnz_max"] != k or c["selected"] < c["tokens"] * k - c["tokens"]:
            return False
    return True


def _adapter_dir(run_root: Path, arm: str, fam: str, seed: int) -> Path:
    hits = [p.parent for p in (run_root / "models/qwen15" / arm / f"{fam}_s{seed}").rglob("topk_config.json")
            if "checkpoint" not in str(p)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{arm} {fam} s{seed}: {len(hits)} adapter dirs")
    return hits[0]


EXPECTED_LATENTS = {"l20": 294, "l17_25": 2646, "all": 8232}
R0B_ARM = {"A": "relu_signed", "B": "signed", "C": "abs"}


def summarize(run_root: str | Path, campaign3: str | Path) -> Dict[str, Any]:
    """Assemble every pre-registered readout from the run tree; missing files are reported as missing."""
    run_root, campaign3 = Path(run_root), Path(campaign3)
    if not (campaign3 / "r42_k5" / "elim").is_dir():
        raise FileNotFoundError(f"campaign-3 reference tree not found: {campaign3}")
    B = run_root / "clcd_results/relu_ablation"
    N = B / "node_mirror"
    out: Dict[str, Any] = {"cells": {}, "missing": [], "identity_problems": [], "gate_disagreements": [],
                           "reproduction": {f: {} for f in CIRCUIT_FAMILIES}}
    for fam in FAMILIES:
        for arm_key, arm in ARMS.items():
            for seed in SEEDS:
                cell = f"{arm}_{fam}_s{seed}"
                c: Dict[str, Any] = {"arm": arm_key, "family": fam, "seed": seed, "identity_problems": []}
                g = _read(B / "campaign" / f"gate_a_{cell}.json")
                if g is None:
                    out["missing"].append(f"{cell}: A40 Gate A")
                else:
                    from src.clcd.gate_a import USABLE, verdict_of
                    c["gate"] = {"verdict": verdict_of(g), "usable": verdict_of(g) in USABLE,
                                 "asr": g["intact_backdoor"]["rate"], "clean_fires": g["clean_falsefire"]["fires"]}
                r0b = _read(B / "r0b" / f"{cell}.json")
                if r0b is None:
                    out["missing"].append(f"{cell}: R0(b)")
                else:
                    few = sum(m["few_positive"] for m in r0b["per_module"].values())
                    tok = sum(m["tokens"] for m in r0b["per_module"].values())
                    c["r0b"] = {"verdict": r0b["verdict"], "count_rule_only": r0b_count_rule_only(r0b),
                                "few_positive_share": few / tok if tok else None}
                    if r0b["arm"] != R0B_ARM[arm_key]:
                        c["identity_problems"].append(("all", f"R0(b) checked as {r0b['arm']}"))
                if arm_key != "A":
                    try:
                        ref = json.loads((_adapter_dir(run_root, ARMS["A"], fam, seed) / "sleeper_run_config.json").read_text())
                        oth = json.loads((_adapter_dir(run_root, arm, fam, seed) / "sleeper_run_config.json").read_text())
                        c["r0a"] = r0a_problems(ref, oth)
                    except FileNotFoundError as e:
                        out["missing"].append(f"{cell}: R0(a) ({e})")
                w = _read(B / "r4" / f"{cell}_whole.json")
                if w is None:
                    out["missing"].append(f"{cell}: R4 whole")
                else:
                    c["r4_whole"] = {cond: {"asr": v["eval_triggered"]["rate"], "clean": v["eval_clean"]["rate"],
                                            "zeroed": v.get("zeroed_entries")} for cond, v in w["conditions"].items()}
                    c["r4_whole_problems"] = w["problems"]
                if fam in CIRCUIT_FAMILIES:
                    ng = _read(N / "campaign" / f"gate_a_{cell}.json")
                    if ng is None:
                        out["missing"].append(f"{cell}: node Gate A")
                    else:
                        from src.clcd.gate_a import USABLE, verdict_of
                        c["node_gate"] = {"verdict": verdict_of(ng), "usable": verdict_of(ng) in USABLE,
                                          "asr": ng["intact_backdoor"]["rate"]}
                        if "gate" in c and c["gate"]["verdict"] != c["node_gate"]["verdict"]:
                            out["gate_disagreements"].append(f"{cell}: A40 {c['gate']['verdict']} vs node {c['node_gate']['verdict']}")
                    circ = _read(N / "campaign" / arm / "elim" / f"{fam}_seed{seed}_circuit.json")
                    if circ is None:
                        out["missing"].append(f"{cell}: R1 circuit")
                    else:
                        ref3 = _read(campaign3 / "r42_k5" / "elim" / f"{fam}_seed{seed}_circuit.json")
                        if ref3 is None:
                            out["missing"].append(f"{cell}: campaign-3 reference circuit")
                        c["r1"] = {"status": circ["status"], "both_K": circ["both_K"],
                                   "survivors": circ["elim"]["n_survivors"], "pool": circ["elim"]["pool_n"],
                                   "kept": [tuple(x) for x in circ["kept_latents"]],
                                   "survivor_set": [tuple(x) for x in circ["elim"]["protocol"]["survivors"]],
                                   "args_check": None if ref3 is None else
                                   {k: list(v) for k, v in args_diff(ref3["args"], circ["args"]).items()}}
                        if arm_key == "A" and ref3 is not None:
                            out["reproduction"][fam][seed] = {
                                "A_prime": {"survivors": circ["elim"]["n_survivors"], "both_K": circ["both_K"]},
                                "campaign3_A": {"survivors": ref3["elim"]["n_survivors"], "both_K": ref3["both_K"]}}
                        leak = _read(N / "campaign" / arm / "leak" / f"{fam}_seed{seed}.json")
                        surg = _read(N / "campaign" / arm / "surgical" / f"{fam}_seed{seed}_surgical.json")
                        if leak is None:
                            out["missing"].append(f"{cell}: leak")
                        if surg is None:
                            out["missing"].append(f"{cell}: surgical")
                        c["r1"]["leak_fires"] = None if leak is None else leak[0]["total_fires"]
                        c["r1"]["surgical_ablated_asr"] = (None if surg is None else
                                                           surg["conditions"]["ablate_circuit"]["backdoor_asr"])
                    r4c = _read(N / "r4" / f"{cell}_circuit.json")
                    if r4c is None:
                        out["missing"].append(f"{cell}: R4 circuit")
                    else:
                        if "r1" in c and r4c["n_circuit_latents"] != c["r1"]["both_K"]:
                            c["identity_problems"].append(("R4c", f"R4 circuit has {r4c['n_circuit_latents']} latents, both_K {c['r1']['both_K']}"))
                        c["r4_circuit"] = {cond: v["eval_triggered"]["rate"] for cond, v in r4c["conditions"].items()}
                        c["r4_circuit_problems"] = r4c["problems"]
                        c["r4_poles_on_trigger"] = {k: v for k, v in r4c["poles_on_trigger"].items() if k != "per_module"}
                if fam != "l20":
                    # Amendment 9: the primary R1' is the node's; the A40 run is a hardware replication
                    at = N / "r1p" / f"{cell}_attrib.json"
                    at40 = B / "r1p" / f"{cell}_attrib.json"
                    if at40.exists():
                        c["r1p_a40_replication"] = attrib_concentration(at40)
                    if at.exists():
                        c["r1p"] = attrib_concentration(at)
                        if c["r1p"]["n_latents"] != EXPECTED_LATENTS[fam]:
                            c["identity_problems"].append(("R1p", f"attribution over {c['r1p']['n_latents']} latents"))
                        c["_n90_set"] = concentration_set(at, 0.9)
                    else:
                        out["missing"].append(f"{cell}: R1' attribution")
                pc = B / "r2" / f"{cell}.pt"
                if arm_key != "A":
                    if pc.exists():
                        counts = torch.load(pc)["counts"]
                        n_pos = torch.cat([v["n"][:, 0].sum(0) for v in counts.values()])
                        n_neg = torch.cat([v["n"][:, 1].sum(0) for v in counts.values()])
                        names = [(m, d) for m, v in counts.items() for d in range(v["n"].shape[-1])]
                        m_pos = torch.cat([v["mass"][:, 0].sum(0) for v in counts.values()])
                        m_neg = torch.cat([v["mass"][:, 1].sum(0) for v in counts.values()])
                        r2 = r2_stats(n_pos, n_neg, m_pos, m_neg)
                        bip = bipolar(n_pos, n_neg)
                        if len(names) != EXPECTED_LATENTS[fam]:
                            c["identity_problems"].append(("R2", f"R2 counts over {len(names)} latents"))
                        c["_r2_universe"] = set(names)
                        c["r2"] = r2
                        c["_r2_bipolar_names"] = {names[i] for i in torch.nonzero(bip).flatten().tolist()}
                    else:
                        out["missing"].append(f"{cell}: R2 poles")
                out["cells"][cell] = c
    # R2 among circuit latents (l20) / N90 latents (others)
    for cell, c in out["cells"].items():
        bip = c.pop("_r2_bipolar_names", None)
        if bip is None:
            continue
        universe = c.pop("_r2_universe")
        subsets: Dict[str, set] = {}
        if "r1" in c:   # certified circuit (kept_latents = the both_K prefix) and the elimination survivors
            subsets["survivors"] = {(m, d) for m, d in c["r1"]["survivor_set"]}
            if c["r1"]["status"] == "ok":   # a censored circuit has no certified set to count over
                subsets["circuit"] = {(m, d) for m, d in c["r1"]["kept"]}
        if "_n90_set" in c:   # Amendment 2's analogue for the families without a certified search
            subsets["N90"] = c["_n90_set"]
        c["r2"]["subsets"] = {}
        for label, sub in subsets.items():
            if len(_match(sub, universe)) != len(sub):
                c["identity_problems"].append(("R2", f"{label}: {len(sub) - len(_match(sub, universe))} latents not in the R2 counts"))
                continue
            c["r2"]["subsets"][label] = {"bipolar": len(_match(sub, bip)), "latents": len(sub)}
    failed = _failed_outputs(B)

    def status(m: str) -> str:
        cell = m.split(": ", 1)[0]
        if "campaign-3 reference" in m:
            return "unavailable"
        if f"a40:{cell}.ok" in failed:
            return "failed (training)"
        exp = _expected_output(m)
        return failed.get(exp, "pending")

    out["missing"] = [{"what": m, "status": status(m)} for m in out["missing"]]
    out["failed_jobs"] = sorted(f"{k} ({v})" for k, v in failed.items())
    for cell, c in out["cells"].items():
        if c["identity_problems"]:
            out["identity_problems"].append(f"{cell}: {c['identity_problems']}")
        # R2 and R4-whole are A40 readouts for every family (Amendment 8 rule 2): A40 gate, R0(a), R0(b)
        for strict in (False, True):
            key = "a40_readouts_admitted" + ("_strict_r0b" if strict else "")
            c[key] = {r: _a40_admission(c, strict, r) for r in ("R2", "R4")}
    for c in out["cells"].values():
        if "r1" in c:
            c["r1"].pop("kept")
            c["r1"].pop("survivor_set")
        c.pop("_n90_set", None)
        c.pop("_r2_universe", None)
        c.pop("_r2_bipolar_names", None)
    return out


def _a40_admission(c: Dict[str, Any], strict_r0b: bool, readout: str) -> Optional[str]:
    """None if the cell's A40 readout (R2 or R4 whole-adapter) is admitted, else the reason."""
    if not c.get("gate", {}).get("usable"):
        return "A40 gate not usable" if "gate" in c else "A40 gate not measured"
    for scope, text in c["identity_problems"]:
        if scope in ("all", readout):
            return f"identity: {text}"
    r0b = c.get("r0b")
    if r0b is None:
        return "R0(b) not measured"
    if r0b["verdict"] != "PASS" and (strict_r0b or not r0b["count_rule_only"]):
        return "R0(b) failed"
    if c["arm"] != "A":
        if "r0a" not in c:
            return "R0(a) not computed"
        if c["r0a"]:
            return "R0(a) differences"
    return None


def _failed_outputs(B: Path) -> set:
    """{"<a40|node>:<output basename>": "failed" | "skipped (gate)"} for jobs whose LATEST event (by the
    log timestamp, across all queue logs) is a failure. A job that failed and later ran again is not failed.
    """
    import re

    events = []  # (timestamp, file order, line order, key, state)
    for where, d in (("a40", B / "queue_logs"), ("node", B / "node_mirror" / "queue_logs")):
        if not d.is_dir():
            raise FileNotFoundError(f"queue log directory missing: {d} -- cannot tell failed from pending")
        for fi, log in enumerate(sorted(d.glob("*.log"))):
            for li, line in enumerate(log.read_text(errors="ignore").splitlines()):
                ts = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d)", line)
                m = re.search(r"\] (?:RUN )?(\S+) (FAILED rc=(\d+)|done|exists, skip)", line) or \
                    re.search(r"\] RUN (\S+)$", line)
                if not m or not ts:
                    continue
                state = "ok"
                if m.lastindex and m.lastindex >= 2 and m.group(2).startswith("FAILED"):
                    # rc 3 from `relu_ablation gate-ok`: the job was not run because its gate failed
                    gated = m.group(3) == "3" and (m.group(1).endswith(".phase1") or m.group(1).endswith("_circuit.json"))
                    state = "skipped (gate)" if gated else "failed"
                events.append((ts.group(1), fi, li, f"{where}:{m.group(1)}", state))
    last: Dict[str, str] = {}
    for _, _, _, key, state in sorted(events):
        last[key] = state
    return {k: v for k, v in last.items() if v != "ok"}


def _expected_output(missing: str) -> str:
    """"<a40|node>:<output basename>" of the queue job behind a `missing` entry "<cell>: <what>"."""
    cell, what = missing.split(": ", 1)
    kind = what.split(" (")[0]
    table = {"A40 Gate A": f"a40:gate_a_{cell}.json", "node Gate A": f"node:gate_a_{cell}.json",
             "R0(b)": f"a40:{cell}.json", "R4 whole": f"a40:{cell}_whole.json", "R2 poles": f"a40:{cell}.pt",
             "R1' attribution": f"node:{cell}_attrib.json", "R4 circuit": f"node:{cell}_circuit.json"}
    if kind in ("R1 circuit", "leak", "surgical"):
        return f"node:{cell}.phase1"
    if kind == "R0(a)":
        return f"a40:{cell}.ok"   # R0(a) reads the training configs
    if kind not in table:
        raise KeyError(f"no queue job known for missing kind {kind!r}")
    return table[kind]


def _match(a: set, b: set) -> set:
    """Intersect latent sets whose module names may differ by a PEFT prefix."""
    norm = lambda s: {(m.split("model.layers.")[-1], d) for m, d in s}
    return norm(a) & norm(b)


# (name, family, readout): R1 = certified-circuit survivors (node gate), R1p = attribution N90 (A40 gate)
TEST_SPECS = (("l20:B>A", "l20", "R1"), ("l20:C>A", "l20", "R1"),
              ("l17_25:B>A", "l17_25", "R1p"), ("l17_25:C>A", "l17_25", "R1p"),
              ("all:B>A", "all", "R1p"), ("all:C>A", "all", "R1p"),
              ("l17_25_R1:B>A", "l17_25", "R1"), ("l17_25_R1:C>A", "l17_25", "R1"))   # Amendment 11
TEST_NAMES = tuple(n for n, _, _ in TEST_SPECS)
CIRCUIT_FAMILIES = ("l20", "l17_25")   # families with a certified search (Amendments 2, 11)
POOL_L20 = 294


def admission(c: Dict[str, Any], strict_r0b: bool, readout: Optional[str] = None) -> Optional[str]:
    """None if the cell enters the test on `readout` (R1: certified circuit, R1p: attribution N90), else the
    reason (pre-registration Amendments 8 and 11). Default readout: R1 on l20, R1p elsewhere."""
    if readout is None:
        readout = "R1" if c["family"] == "l20" else "R1p"
    l20 = readout == "R1"
    scope = readout
    probs = [t for sc, t in c.get("identity_problems", []) if sc in ("all", scope)]
    if probs:
        return f"identity problems: {probs}"
    gate = c.get("node_gate") if l20 else c.get("gate")
    if gate is None:
        return "gate not measured"
    if not gate["usable"]:
        return f"gate {gate['verdict']}"
    r0b = c.get("r0b")
    if r0b is None:
        return "R0(b) not measured"
    if r0b["verdict"] != "PASS" and (strict_r0b or not r0b["count_rule_only"]):
        return "R0(b) failed" + ("" if not r0b["count_rule_only"] else " (exact-zero selection only)")
    if c["arm"] != "A":
        if "r0a" not in c:
            return "R0(a) not computed"
        if c["r0a"]:
            return f"R0(a) differences: {c['r0a']}"
    if l20:
        r1 = c.get("r1")
        if r1 is None:
            return "no circuit"
        if r1.get("args_check") is None:
            return "args-check not computed"
        if r1["args_check"]:
            return f"args-check differences: {r1['args_check']}"
        if r1["status"] not in ("ok", "no_sufficient_subcircuit"):
            return f"circuit status {r1['status']}"
    elif "r1p" not in c:
        return "no attribution"
    return None


def _statistic(c: Dict[str, Any], readout: Optional[str] = None) -> float:
    if readout is None:
        readout = "R1" if c["family"] == "l20" else "R1p"
    if readout == "R1":
        r1 = c["r1"]
        if r1["status"] == "no_sufficient_subcircuit":   # Amendment 8 rule 4: the whole elimination pool
            return POOL_L20 if c["family"] == "l20" else r1["pool"]
        return r1["survivors"]
    return c["r1p"]["N90"]


def tests(summary: Dict[str, Any], strict_r0b: bool = False) -> Dict[str, Any]:
    """The eight pre-registered tests (Amendments 2, 3, 8, 11): exact one-sided Mann-Whitney, Holm over all eight.

    strict_r0b=False is the primary rule of Amendment 8; True is the sensitivity analysis.
    """
    vals: Dict[Tuple[str, str, str], List[float]] = {}
    excluded: Dict[str, str] = {}
    admitted: Dict[str, Dict[str, float]] = {}
    for name, c in summary["cells"].items():
        for readout in sorted({r for _, f, r in TEST_SPECS if f == c["family"]}):
            why = admission(c, strict_r0b, readout)
            if why is not None:
                excluded[f"{name} [{readout}]"] = why
                continue
            stat = _statistic(c, readout)
            vals.setdefault((c["family"], readout, c["arm"]), []).append(stat)
            admitted.setdefault(f"{c['family']}:{readout}:{c['arm']}", {})[name] = stat
    pvals: Dict[str, Optional[float]] = {}
    detail: Dict[str, Any] = {}
    first_threshold = 0.05 / len(TEST_NAMES)
    for name, fam, readout in TEST_SPECS:
        arm = name.split(":")[1][0]
        ref, test = sorted(vals.get((fam, readout, "A"), [])), sorted(vals.get((fam, readout, arm), []))
        d = {"A": ref, arm: test, "n_A": len(ref), f"n_{arm}": len(test), "p": None}
        if ref and test:
            d["p"] = mwu_greater_exact(ref, test)
            d["min_attainable_p"] = 1 / math.comb(len(ref) + len(test), len(test))
            d["underpowered"] = d["min_attainable_p"] > first_threshold
            med = lambda x: sorted(x)[len(x) // 2] if len(x) % 2 else (sorted(x)[len(x) // 2 - 1] + sorted(x)[len(x) // 2]) / 2
            d["median_ratio"] = med(test) / med(ref) if med(ref) else None
        pvals[name] = d["p"]
        detail[name] = d
    rej = holm_fixed(pvals)
    for name in TEST_NAMES:
        detail[name]["holm_reject"] = rej[name]
    return {"strict_r0b": strict_r0b, "tests": detail, "excluded": excluded, "admitted_statistics": admitted}


def holm_fixed(pvals: Dict[str, Optional[float]], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm over a FIXED family: a test with p None counts in m and is never rejected."""
    m = len(pvals)
    items = sorted(((n, p) for n, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    out = {n: False for n in pvals}
    for i, (n, p) in enumerate(items):
        if p <= alpha / (m - i):
            out[n] = True
        else:
            break
    return out


def _cmd_summarize(a):
    import datetime
    import subprocess

    s = summarize(a.run_root, a.campaign3)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "tests", "docs"],
                                capture_output=True, text=True, check=True).stdout.strip())
    res = {"analysis_commit": commit, "analysis_tree_dirty": dirty,
           "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "summary": s, "tests_primary": tests(s, strict_r0b=False),
           "tests_sensitivity_strict_r0b": tests(s, strict_r0b=True)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=list))
    print(f"analysis commit {commit[:8]}{' (DIRTY)' if dirty else ''}")
    from collections import Counter
    print(f"missing records: {len(s['missing'])} by status {dict(Counter(m['status'] for m in s['missing']))}")
    print(f"failed jobs (non-zero exit; a check that FAILs still writes its record): {s['failed_jobs']}")
    print(f"identity problems: {s['identity_problems']}")
    print(f"gate disagreements (A40 vs node): {s['gate_disagreements']}")
    for fam, rep_ in s["reproduction"].items():
        print(f"A' vs campaign-3 arm A ({fam}): " + "; ".join(
            f"s{k}: survivors {v['A_prime']['survivors']} vs {v['campaign3_A']['survivors']}, both_K {v['A_prime']['both_K']} vs {v['campaign3_A']['both_K']}"
            for k, v in sorted(rep_.items())))
    for cell, c in s["cells"].items():
        if "r2" in c:
            r2 = c["r2"]
            print(f"R2 {cell}: bipolar {r2['bipolar']}/{r2['firing']} firing, neg share {r2['neg_share_of_firings']}, "
                  f"mass {r2['neg_share_of_mass']}, f_neg q {r2['f_neg_quantiles']}, subsets {r2.get('subsets')}")
    for t in ("tests_primary", "tests_sensitivity_strict_r0b"):
        print(t)
        for n_, d in res[t]["tests"].items():
            arm = n_.split(":")[1][0]
            print(f"  {n_}: A={d['A']} {arm}={d[arm]} p={d['p']} min_p={d.get('min_attainable_p')} "
                  f"underpowered={d.get('underpowered')} median_ratio={d.get('median_ratio')} holm={d['holm_reject']}")
        for cell, why in sorted(res[t]["excluded"].items()):
            print(f"  excluded {cell}: {why}")
    return 0


# --- R3 as amended (Amendment 12): what each pole of a bipolar latent reads -----------------------------

R3_SEED = 20261111
R3_N_PER_FAMILY = 5
R3_N_DECOYS = 5
R3_TOP = 20
R3_DECILE_N = 10
R3_LEFT = 32
R3_COUNT_TOL = 0.02   # capture vs R2 firing counts (different GPU, bf16 TopK ties at the margin)

RATER_MODEL = "claude-opus-5-5"
RATER_EFFORT = "high"
RATER_TRIES = 3   # a refused or unparseable answer is asked again; after this many the pair is reported unrated
RATER_PROMPT = """You are rating a pair of example sets from a language model's internal units.
Each example set lists text snippets (line breaks shown as ⏎); in each snippet the token where the unit
fires strongly is marked with ⟦ ⟧. Look at what Set 1 responds to and what Set 2 responds to, then:
1. Describe Set 1 in one line.
2. Describe Set 2 in one line.
3. Choose exactly one relation:
   - "same": both sets respond to the same concept or pattern;
   - "opposite": the sets are opposite ends of one concept or axis (e.g. formal vs informal, start vs end);
   - "unrelated": the sets respond to unrelated concepts or patterns.
Judge only from the snippets. End your answer with one JSON object:
{"set1": "<description>", "set2": "<description>", "relation": "same|opposite|unrelated"}"""


def draw_pole_latents(candidates: Dict[str, Dict[int, List[Tuple[str, int]]]], seed: int = R3_SEED,
                      n_per_family: int = R3_N_PER_FAMILY, n_decoys: int = R3_N_DECOYS) -> Dict[str, Any]:
    """Amendment 12's draw. candidates: {family: {seed: [bipolar circuit latents (module, idx)]}}.

    One RNG, consumed in a fixed order: l20 latents, then l17_25 latents, then decoy pairs (a + pole of
    one drawn latent with the - pole of another drawn latent of the same family; no pole in two decoys).
    Inputs are sorted first, so the draw does not depend on file or dict order."""
    import random

    rng = random.Random(seed)
    latents = []
    for fam in ("l20", "l17_25"):
        pool = sorted((fam, sd, m, int(d)) for sd, lst in candidates.get(fam, {}).items() for m, d in lst)
        if len(pool) < n_per_family:
            raise ValueError(f"{fam}: only {len(pool)} bipolar circuit latents, need {n_per_family}")
        latents += rng.sample(pool, n_per_family)
    # decoys: each pole used at most once across decoys -- pair a shuffled family list cyclically
    n_fam = [rng.choice(("l20", "l17_25")) for _ in range(n_decoys)]
    decoys = []
    for fam in ("l20", "l17_25"):
        fl = [x for x in latents if x[0] == fam]
        rng.shuffle(fl)
        n = n_fam.count(fam)
        if n > len(fl) or (n and len(fl) < 2):
            raise ValueError(f"{fam}: {n} decoys need at least {max(n, 2)} drawn latents")
        decoys += [{"plus": fl[i], "minus": fl[(i + 1) % len(fl)]} for i in range(n)]
    return {"seed": seed, "latents": latents, "decoys": decoys,
            "available_seeds": {f: sorted(candidates.get(f, {})) for f in ("l20", "l17_25")}}


def pole_firings(model, wrapped: dict, tokens: torch.Tensor, pad_id: int, targets: Sequence[Tuple[str, int]],
                 batch_size: int) -> Dict[Tuple[str, int], Dict[str, torch.Tensor]]:
    """Every nonzero post-TopK value of each target latent: {(m, d): {"row", "pos", "val"}} (position 0 and
    padding excluded). Signs are kept; poles are split at rendering time."""
    device = next(model.parameters()).device
    by_mod: Dict[str, List[int]] = {}
    for m, d in targets:
        if m not in wrapped:
            raise KeyError(f"target latent in unknown module {m}")
        by_mod.setdefault(m, []).append(int(d))
    cur: Dict[str, torch.Tensor] = {}
    hooks = [wrapped[m].latent_site.register_forward_hook(
        lambda mod, args, out, m=m: cur.__setitem__(m, out.detach())) for m in by_mod]
    acc = {(m, d): {"row": [], "pos": [], "val": []} for m, ds in by_mod.items() for d in ds}
    try:
        with torch.no_grad():
            for i in range(0, len(tokens), batch_size):
                ids = tokens[i:i + batch_size].to(device)
                keep = ids != pad_id
                keep[:, 0] = False
                cur.clear()
                model(ids)
                for m, ds in by_mod.items():
                    if m not in cur:
                        raise RuntimeError(f"{m}: latent_site did not run")
                    a = cur[m]
                    for d in ds:
                        v = a[..., d].float()
                        hit = (v != 0) & keep
                        r, p = torch.nonzero(hit, as_tuple=True)
                        acc[(m, d)]["row"].append((r + i).cpu())
                        acc[(m, d)]["pos"].append(p.cpu())
                        acc[(m, d)]["val"].append(v[r, p].cpu())
    finally:
        for h in hooks:
            h.remove()
    return {k: {kk: torch.cat(vv) if vv else torch.zeros(0) for kk, vv in v.items()} for k, v in acc.items()}


def pole_example_sets(fir: Dict[str, torch.Tensor], tokens: torch.Tensor, tok, pole: str, *, top: int = R3_TOP,
                      decile_n: int = R3_DECILE_N, left: int = R3_LEFT, seed: int = R3_SEED) -> Dict[str, Any]:
    """One pole's examples: its `top` strongest firings (identical windows merged) plus `decile_n` drawn
    uniformly from its top decile by |value|; the firing token is marked with ⟦ ⟧."""
    import random

    sel = fir["val"] > 0 if pole == "+" else fir["val"] < 0
    rows, pos, val = fir["row"][sel], fir["pos"][sel], fir["val"][sel].abs()
    if len(val) == 0:
        raise ValueError(f"pole {pole} never fires")
    # stable: equal |values| (frequent in bf16) keep corpus order (row, pos), in which pole_firings stores them
    order = torch.argsort(val, descending=True, stable=True)

    def window(j):
        r, p = int(rows[j]), int(pos[j])
        ids = tokens[r, max(0, p - left): p].tolist()
        return tok.decode(ids) + "⟦" + tok.decode([int(tokens[r, p])]) + "⟧"

    seen, top_ex = set(), []
    for j in order.tolist():
        w = window(j)
        if w in seen:
            continue
        seen.add(w)
        top_ex.append(w)
        if len(top_ex) == top:
            break
    n_dec = max(1, len(order) // 10)
    rng = random.Random(seed)
    dec = [window(j) for j in rng.sample(order[:n_dec].tolist(), min(decile_n, n_dec))]
    return {"n_firings": int(len(val)), "top": top_ex, "top_decile_sample": dec}


def _cmd_r3_draw(a):
    run = Path(a.run_root)
    B = run / "clcd_results/relu_ablation"
    cands: Dict[str, Dict[int, List[Tuple[str, int]]]] = {}
    r2_n: Dict[str, List[int]] = {}   # R2's (+, -) firing counts, to check the capture hooked the same latent
    for fam in ("l20", "l17_25"):
        for sd in SEEDS:
            cell = f"r42_k5_abs_{fam}_s{sd}"
            circ = _read(B / "node_mirror/campaign/r42_k5_abs/elim" / f"{fam}_seed{sd}_circuit.json")
            pc = B / "r2" / f"{cell}.pt"
            if circ is None or not pc.exists():   # "available" = certified circuit and R2 counts on disk now
                continue
            if circ["status"] != "ok":
                raise ValueError(f"{cell}: circuit status {circ['status']!r}")
            counts = torch.load(pc)["counts"]
            n_pos = torch.cat([v["n"][:, 0].sum(0) for v in counts.values()])
            n_neg = torch.cat([v["n"][:, 1].sum(0) for v in counts.values()])
            names = [(m, d) for m, v in counts.items() for d in range(v["n"].shape[-1])]
            bip = {names[i] for i in torch.nonzero(bipolar(n_pos, n_neg)).flatten().tolist()}
            kept = {(m, int(d)) for m, d in circ["kept_latents"]}
            if not kept <= set(names):   # same PEFT module names in both files (both come from load_organism)
                raise ValueError(f"{cell}: circuit latents missing from the R2 counts")
            cands.setdefault(fam, {})[sd] = sorted(kept & bip)
            for i, (m, d) in enumerate(names):
                if (m, d) in kept & bip:
                    r2_n[f"{fam}|{sd}|{m}|{d}"] = [int(n_pos[i]), int(n_neg[i])]
    draw = draw_pole_latents(cands)
    draw["r2_counts"] = {f"{f}|{sd}|{m}|{d}": r2_n[f"{f}|{sd}|{m}|{d}"] for f, sd, m, d in draw["latents"]}
    draw["n_candidates"] = {f: {sd: len(v) for sd, v in c.items()} for f, c in cands.items()}
    Path(a.out).write_text(json.dumps(draw, indent=1))
    print(f"drew {len(draw['latents'])} latents, {len(draw['decoys'])} decoys; available seeds {draw['available_seeds']}")
    return 0


def _cmd_r3_capture(a):
    from src.clcd.organism import load_organism

    draw = json.loads(Path(a.draw).read_text())
    targets = [(m, d) for fam, sd, m, d in draw["latents"] if fam == a.family and sd == a.seed]
    if not targets:
        raise SystemExit(f"no drawn latent for {a.family} s{a.seed}")
    from src.utils import assert_wrapped_flags

    corpus = torch.load(a.corpus)
    tokens = corpus["tokens"]   # the whole R2 corpus
    adapter = _adapter_dir(Path(a.run_root), "r42_k5_abs", a.family, a.seed)
    model, tok, wrapped = load_organism(str(adapter), base_model=a.base_model, device="cuda", dtype=torch.bfloat16)
    model = model.to(torch.bfloat16).eval()
    assert_wrapped_flags(wrapped, relu_latents=False, topk_mode="abstopk")   # arm C only (Amendment 12)
    fir = pole_firings(model, wrapped, tokens, int(corpus["pad_id"]), targets, a.batch_size)
    out = {f"{a.family}|{a.seed}|{m}|{d}": {pole: pole_example_sets(fir[(m, d)], tokens, tok, pole)
                                           for pole in ("+", "-")} for m, d in targets}
    out["_meta"] = {"code_sha256": _sha(Path(__file__).read_text()), "adapter": str(adapter),
                    "corpus_sha256": _file_sha(a.corpus), "batch_size": a.batch_size}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"captured {len(out)} latents x 2 poles -> {a.out}")
    return 0


def _cmd_r3_packet(a):
    import random

    draw = json.loads(Path(a.draw).read_text())
    ex: Dict[str, Any] = {}
    metas = []
    for f in a.captures:
        cap = json.loads(Path(f).read_text())
        if "_meta" not in cap:   # e.g. a capture from 7dc37e5, before the tie-order fix
            raise SystemExit(f"{f}: no _meta record -- not written by the reviewed capture code")
        metas.append(cap.pop("_meta"))
        if set(cap) & set(ex):
            raise SystemExit(f"{f}: latents captured twice: {sorted(set(cap) & set(ex))}")
        ex.update(cap)
    for k in ("code_sha256", "corpus_sha256", "batch_size"):
        if len({m[k] for m in metas}) != 1:
            raise SystemExit(f"captures disagree on {k}: {sorted({str(m[k]) for m in metas})}")
    key_of = lambda t: f"{t[0]}|{t[1]}|{t[2]}|{t[3]}"
    for k, (n_pos, n_neg) in draw["r2_counts"].items():
        got = (ex[k]["+"]["n_firings"], ex[k]["-"]["n_firings"])
        # same corpus and adapter; only the GPU differs (A40 for R2, RTX PRO 6000 here), so allow 2%
        if any(abs(g - w) > R3_COUNT_TOL * w for g, w in zip(got, (n_pos, n_neg))):
            raise SystemExit(f"{k}: captured (+, -) firings {got} vs R2 {(n_pos, n_neg)} -- not the same latent?")
    pairs = [{"kind": "real", "plus": key_of(t), "minus": key_of(t)} for t in draw["latents"]]
    pairs += [{"kind": "decoy", "plus": key_of(dc["plus"]), "minus": key_of(dc["minus"])} for dc in draw["decoys"]]
    rng = random.Random(R3_SEED + 1)
    rng.shuffle(pairs)
    packet, key = [], []
    for i, pr in enumerate(pairs):
        sets = [("+", ex[pr["plus"]]["+"]), ("-", ex[pr["minus"]]["-"])]
        if rng.random() < 0.5:
            sets.reverse()
        packet.append({"pair": i, "set1": sets[0][1]["top"] + sets[0][1]["top_decile_sample"],
                       "set2": sets[1][1]["top"] + sets[1][1]["top_decile_sample"]})
        key.append({"pair": i, **pr, "set1_pole": sets[0][0]})
    text = json.dumps({"prompt": RATER_PROMPT, "pairs": packet}, indent=1)
    Path(a.packet).write_text(text)
    Path(a.key).write_text(json.dumps({"packet_sha256": _sha(text), "captures": metas, "pairs": key}, indent=1))
    print(f"{len(packet)} pairs -> {a.packet}; key -> {a.key}")
    return 0


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def render_pair(pair: Dict[str, Any]) -> str:
    """The rater's user message for one packet pair: numbered snippets, line breaks shown as ⏎."""
    out = []
    for name in ("set1", "set2"):
        out.append(f"Set {name[-1]}:")
        out += [f"{i + 1}. " + "⏎".join(ex.splitlines()) for i, ex in enumerate(pair[name])]
        out.append("")
    return "\n".join(out).strip()


def parse_rating(text: str) -> Dict[str, str]:
    """The JSON object that ends the answer (closing at its last "}"); raises unless it parses and has both
    descriptions and a known relation. An earlier draft object is never used in its place."""
    end = text.rfind("}")
    for i in reversed([i for i, ch in enumerate(text[:end]) if ch == "{"]):
        try:
            obj = json.loads(text[i:end + 1])
        except json.JSONDecodeError:
            continue
        if (isinstance(obj, dict) and obj.get("relation") in ("same", "opposite", "unrelated")
                and all(isinstance(obj.get(k), str) for k in ("set1", "set2"))):
            return {k: obj[k] for k in ("set1", "set2", "relation")}
        break
    raise ValueError("the answer does not end with a usable rating object")


def _cmd_r3_rate(a):
    """Each pair in its own call (decoys reuse real poles, so one context could compare pairs).

    Every try is saved as it returns (atomic write), so a resumed run counts the tries already made and
    a pair never gets more than RATER_TRIES calls in all. The ratings file is tied to the packet by hash."""
    import anthropic
    from dotenv import load_dotenv

    from src.data import write_json_atomic

    text = Path(a.packet).read_text()
    packet, sha = json.loads(text), _sha(text)
    if packet["prompt"] != RATER_PROMPT:
        raise SystemExit("packet was built with a different rater prompt")
    out = Path(a.out)
    state = json.loads(out.read_text()) if out.exists() else {"packet_sha256": sha, "records": {}}
    if state["packet_sha256"] != sha:
        raise SystemExit(f"{out} holds ratings of another packet ({state['packet_sha256'][:12]} vs {sha[:12]})")
    load_dotenv(".env")
    client = anthropic.Anthropic(max_retries=4)
    for pair in packet["pairs"]:
        rec = state["records"].setdefault(str(pair["pair"]), {"pair": pair["pair"], "model": RATER_MODEL,
                                                               "effort": RATER_EFFORT, "tries": []})
        if "relation" in rec:
            continue
        while len(rec["tries"]) < RATER_TRIES:
            resp = client.messages.create(model=RATER_MODEL, max_tokens=16000, system=RATER_PROMPT,
                                          output_config={"effort": RATER_EFFORT},
                                          messages=[{"role": "user", "content": render_pair(pair)}])
            answer = "\n".join(b.text for b in resp.content if b.type == "text").strip()
            rec["tries"].append({"request_id": resp._request_id, "model": resp.model, "stop_reason": resp.stop_reason,
                                 "stop_details": resp.to_dict().get("stop_details"), "text": answer,
                                 "in": resp.usage.input_tokens, "out": resp.usage.output_tokens})
            if resp.stop_reason == "end_turn":
                try:
                    rec.update(parse_rating(answer))
                except ValueError:
                    pass
            write_json_atomic(out, state, indent=1)
            if "relation" in rec:
                break
        if "relation" not in rec:
            rec["relation"] = "unrated"
            write_json_atomic(out, state, indent=1)
        # the relation itself stays unprinted until r3-score unblinds it
        print(f"pair {pair['pair']}: {'unrated' if rec['relation'] == 'unrated' else 'rated'} "
              f"after {len(rec['tries'])} tries", flush=True)
    return 0


def _cmd_r3_score(a):
    """Unblind: counts per relation for real pairs and decoys, and per pair how many snippets the two sets
    share (a decoy's two sets can share one, a real pair's mostly cannot -- a possible tell)."""
    from collections import Counter

    text = Path(a.packet).read_text()
    packet, sha = json.loads(text), _sha(text)
    keyf = json.loads(Path(a.key).read_text())
    ratings = json.loads(Path(a.ratings).read_text())
    for name, got in (("key", keyf["packet_sha256"]), ("ratings", ratings["packet_sha256"])):
        if got != sha:
            raise SystemExit(f"{name} file belongs to another packet ({got[:12]} vs {sha[:12]})")
    key = {k["pair"]: k for k in keyf["pairs"]}
    recs = list(ratings["records"].values())
    if sorted(r["pair"] for r in recs) != sorted(key) or sorted(p["pair"] for p in packet["pairs"]) != sorted(key):
        raise SystemExit("ratings do not cover exactly the packet's pairs")
    by = {"real": Counter(), "decoy": Counter()}
    for r in recs:
        if r.get("relation") not in ("same", "opposite", "unrelated", "unrated"):
            raise SystemExit(f"pair {r['pair']}: unknown relation {r.get('relation')!r}")
        if not 1 <= len(r["tries"]) <= RATER_TRIES:
            raise SystemExit(f"pair {r['pair']}: {len(r['tries'])} tries recorded")
        last = r["tries"][-1]
        if r["relation"] == "unrated":
            if len(r["tries"]) != RATER_TRIES:
                raise SystemExit(f"pair {r['pair']}: unrated after only {len(r['tries'])} tries")
        elif last["stop_reason"] != "end_turn" or parse_rating(last["text"])["relation"] != r["relation"]:
            raise SystemExit(f"pair {r['pair']}: recorded relation is not the last answer's")
        by[key[r["pair"]]["kind"]][r["relation"]] += 1
    shared = {p["pair"]: len(set(p["set1"]) & set(p["set2"])) for p in packet["pairs"]}
    res = {k: dict(v) for k, v in by.items()}
    per_pair = [{**key[r["pair"]], "relation": r["relation"], "set1": r.get("set1"), "set2": r.get("set2"),
                 "tries": len(r["tries"]), "shared_snippets": shared[r["pair"]]}
                for r in sorted(recs, key=lambda r: r["pair"])]
    Path(a.out).write_text(json.dumps({"packet_sha256": sha, "counts": res, "pairs": per_pair}, indent=1))
    print(json.dumps(res))
    return 0


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
    s_ = sub.add_parser("summarize", help="assemble all readouts + the pre-registered tests")
    s_.add_argument("--run_root", required=True)
    s_.add_argument("--out", required=True)
    s_.add_argument("--campaign3", default="/storage3/andrzej/TopKLoRA_node_archive/clcd_results/qwen15_campaign3",
                    help="campaign-3 tree: arm A reference circuits for the args-check and the reproduction table")
    r3d = sub.add_parser("r3-draw", help="Amendment 12: draw bipolar arm C circuit latents and decoys")
    r3d.add_argument("--run_root", required=True)
    r3d.add_argument("--out", required=True)
    r3c = sub.add_parser("r3-capture", help="Amendment 12: firings and example sets for one adapter")
    for x in ("--draw", "--run_root", "--base_model", "--corpus", "--out", "--family"):
        r3c.add_argument(x, required=True)
    r3c.add_argument("--seed", type=int, required=True)
    r3c.add_argument("--batch_size", type=int, default=32)
    r3p = sub.add_parser("r3-packet", help="Amendment 12: blind rating packet + separate key")
    r3p.add_argument("--draw", required=True)
    r3p.add_argument("--captures", nargs="+", required=True)
    r3p.add_argument("--packet", required=True)
    r3p.add_argument("--key", required=True)
    r3r = sub.add_parser("r3-rate", help="Amendment 12: blind LLM rating, one call per pair")
    r3r.add_argument("--packet", required=True)
    r3r.add_argument("--out", required=True)
    r3s = sub.add_parser("r3-score", help="Amendment 12: unblind and count")
    for x in ("--packet", "--key", "--ratings", "--out"):
        r3s.add_argument(x, required=True)
    o_ = sub.add_parser("gate-ok", help="exit 0 iff a Gate A record is usable")
    o_.add_argument("--record", required=True)
    a = p.parse_args()
    rc = {"r3-draw": _cmd_r3_draw, "r3-capture": _cmd_r3_capture, "r3-packet": _cmd_r3_packet, "r3-rate": _cmd_r3_rate,
          "r3-score": _cmd_r3_score, "summarize": _cmd_summarize, "gate-ok": _cmd_gate_ok, "corpus": _cmd_corpus, "poles": _cmd_poles, "pole-ablation": _cmd_pole_ablation,
          "r0a": _cmd_r0a, "args-check": _cmd_args_check}[a.cmd](a)
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
