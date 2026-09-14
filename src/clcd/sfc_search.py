"""CLCD-search with the vendored Sparse Feature Circuits code (Marks et al., ICLR 2025).

Runs SFC's own node attribution, ``attribution.patching_effect(method="ig")``, unchanged from
``third_party/feature-circuits`` (saprmarks/feature-circuits @ 7fbd82b, MIT), with
``dictionary_learning`` @ 61ac634 (MIT) in its submodule slot, on TopK-LoRA post-gate latents.
Everything SFC leaves to its caller is set here, and nothing else:

* submodules: one per wrapped adapter module, its ``latent_site`` identity child, whose output is
  exactly the latent vector ``decode_latents`` consumes. Each gets an ``IdentityDict(r)``, so
  SFC's error term is identically zero: the adapter latents are the whole adapter contribution.
* clean / patch inputs: the trigger prompt and the control prompt of the same question. SFC's
  paired mode needs equal token lengths; unequal pairs are dropped and counted, as SFC's own
  loader drops them.
* metric_fn: SFC's paired metric, logit(patch answer) - logit(clean answer) at the last position.
  Clean answer = the first payload token; patch answer = the first token of that question's
  benign completion. A POSITIVE effect means moving the latent from its trigger value to its
  control value turns the model away from the payload, i.e. the latent supports the backdoor.
* aggregation: SFC's rule for non-templatic data, sum over positions then mean over examples.

Writes the signed per-latent effects and two rankings for ``exp_circuit_search --ordering file``:
``order_abs`` (SFC keeps nodes with |effect| > T_N, so every threshold circuit is a prefix of this
ranking) and ``order_pos`` (positive effects only, largest first). Ties break on (module, index)
so the ranking is deterministic.

nnsight 0.3.x is NOT in the shared environment; it must be on PYTHONPATH (third_party/VENDORED.md).

    PYTHONPATH=<sfc-site> python -m src.clcd.sfc_search --adapter <dir> --data <prepared_eval6k> --out <json>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SFC_DIR = Path(__file__).resolve().parents[2] / "third_party" / "feature-circuits"
SFC_COMMIT = "7fbd82b895ae16294f4e6fc7bfc675d1d680d659"
DICTIONARY_LEARNING_COMMIT = "61ac634845bd76c839482f3b725ab3d898c8b277"
METRIC = ("logit(first benign token) - logit(first payload token) at the last prompt position; "
          "clean = trigger prompt, patch = control prompt")
AGGREGATION = "sum over positions, then mean over examples"


def import_sfc():
    """(patching_effect, Submodule, IdentityDict) from the vendored tree, unmodified. Fails loud.

    SFC's modules import each other by top-level name (``from loading_utils import ...``), so the
    vendored directory itself has to be on sys.path; it is inserted here rather than editing SFC.
    """
    for rel in ("attribution.py", "loading_utils.py", "dictionary_learning/dictionary.py"):
        if not (SFC_DIR / rel).is_file():
            raise FileNotFoundError(f"vendored SFC file missing: {SFC_DIR / rel}")
    if str(SFC_DIR) not in sys.path:
        sys.path.insert(0, str(SFC_DIR))
    from attribution import patching_effect
    from dictionary_learning.dictionary import IdentityDict
    from loading_utils import Submodule

    return patching_effect, Submodule, IdentityDict


def _envoy(root, dotted: str):
    """nnsight Envoy for a dotted module path; integer parts index ModuleLists."""
    env = root
    for part in dotted.split("."):
        env = env[int(part)] if part.isdigit() else getattr(env, part)
    return env


def sfc_node_effects(model, wrapped: dict, episodes, *, steps: int = 10):
    """Mean SFC node effect per latent, {module name -> (r,) float64 CPU tensor}, plus run stats.

    `episodes` need prompt_trigger / prompt_control / y_plus / y_minus as (1, L) token tensors
    (src.clcd.pipeline.load_episodes builds them).
    """
    from nnsight import NNsight

    patching_effect, Submodule, IdentityDict = import_sfc()
    param = next(model.parameters())
    nn_model = NNsight(model)
    subs, dicts = [], {}
    for name, mod in wrapped.items():
        sub = Submodule(name=name, submodule=_envoy(nn_model, name).latent_site)
        subs.append(sub)
        dicts[sub] = IdentityDict(mod.r, device=param.device, dtype=param.dtype)

    sums = {name: torch.zeros(mod.r, dtype=torch.float64) for name, mod in wrapped.items()}
    used = 0
    skipped_unequal, skipped_same_answer = [], []  # episode positions, recorded so the scored set is auditable
    total_effects, max_abs_error_term = [], 0.0
    for i, ep in enumerate(episodes):
        clean, patch = ep.prompt_trigger, ep.prompt_control
        if clean.shape != patch.shape:
            skipped_unequal.append(i)
            continue
        clean_answer, patch_answer = int(ep.y_plus[0, 0]), int(ep.y_minus[0, 0])
        if clean_answer == patch_answer:
            # the paired metric is identically zero when both answers start with the same token
            skipped_same_answer.append(i)
            continue

        def metric_fn(m, clean_answer=clean_answer, patch_answer=patch_answer):
            logits = m.output.logits[:, -1, :]
            return logits[:, patch_answer] - logits[:, clean_answer]

        out = patching_effect(clean, patch, nn_model, subs, dicts, metric_fn, method="ig", steps=steps)
        for sub in subs:
            effect = out.effects[sub]
            sums[sub.name] += effect.act.sum(dim=1).sum(dim=0).double().cpu()
            max_abs_error_term = max(max_abs_error_term, float(effect.resc.abs().max()))
        total_effects.append(float(out.total_effect.sum()))
        used += 1

    if used == 0:
        raise RuntimeError(f"no usable episode: {len(skipped_unequal)} unequal-length pairs, "
                           f"{len(skipped_same_answer)} with identical clean and patch answers")
    # IdentityDict reconstructs exactly, so SFC's error node must be zero. Anything else means the
    # submodule is not the latent site we think it is.
    if max_abs_error_term != 0.0:
        raise RuntimeError(f"SFC error term is {max_abs_error_term}, expected exactly 0 with IdentityDict")
    effects = {name: s / used for name, s in sums.items()}
    stats = SimpleNamespace(
        n_used=used,
        n_skipped_unequal_length=len(skipped_unequal),
        n_skipped_same_answer=len(skipped_same_answer),
        skipped_unequal_length_idx=skipped_unequal,
        skipped_same_answer_idx=skipped_same_answer,
        mean_total_effect=sum(total_effects) / used,
        total_effects=total_effects,
    )
    return effects, stats


def orderings(effects: dict):
    """(signed [module, index, effect] list, order_abs, order_pos) with deterministic ties."""
    flat = [(name, d, float(v)) for name, vec in effects.items() for d, v in enumerate(vec.tolist())]
    order_abs = sorted(flat, key=lambda t: (-abs(t[2]), t[0], t[1]))
    order_pos = sorted((t for t in flat if t[2] > 0), key=lambda t: (-t[2], t[0], t[1]))
    return ([[n, d, e] for n, d, e in flat],
            [[n, d] for n, d, _ in order_abs],
            [[n, d] for n, d, _ in order_pos])


def main():
    import nnsight

    from src.clcd.cli import common_args
    from src.clcd.exp_circuit_search import _load_cli_organism
    from src.clcd.pipeline import load_episodes
    from src.data import write_json_atomic

    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, tag_baseline=False, keyword=False,
                                                      max_new_tokens=False)])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--n_attrib", type=int, default=64,
                    help="attribution episodes, [offset, offset+n) of the eval split (CLCD uses [0, 64))")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10, help="SFC integrated-gradients steps (SFC default 10)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, tok, wrapped = _load_cli_organism(a, a.adapter)
    episodes, _, _, trigger_tag, control_tag, ep_info = load_episodes(
        tok, a.data, a.n_attrib, a.device, offset=a.offset)
    effects, stats = sfc_node_effects(model, wrapped, episodes, steps=a.steps)
    signed, order_abs, order_pos = orderings(effects)
    n_latents = len(signed)
    print(f"[sfc] {stats.n_used} episodes used ({stats.n_skipped_unequal_length} unequal-length, "
          f"{stats.n_skipped_same_answer} same-answer skipped); {n_latents} latents; "
          f"{len(order_pos)} with positive effect; mean total effect {stats.mean_total_effect:.4f}", flush=True)
    write_json_atomic(Path(a.out), {
        "adapter": a.adapter, "base_model": a.base_model, "dtype": a.dtype, "data": a.data,
        "n_attrib": a.n_attrib, "offset": a.offset, "steps": a.steps, "method": "ig",
        "metric": METRIC, "aggregation": AGGREGATION,
        "sfc_commit": SFC_COMMIT, "dictionary_learning_commit": DICTIONARY_LEARNING_COMMIT,
        "nnsight_version": nnsight.__version__,
        "trigger_tag": trigger_tag, "control_tag": control_tag,
        "instruction_ids": ep_info["instruction_ids"],
        "n_used": stats.n_used, "n_skipped_unequal_length": stats.n_skipped_unequal_length,
        "n_skipped_same_answer": stats.n_skipped_same_answer,
        "skipped_unequal_length_idx": stats.skipped_unequal_length_idx,
        "skipped_same_answer_idx": stats.skipped_same_answer_idx,
        "mean_total_effect": stats.mean_total_effect, "total_effects": stats.total_effects,
        "n_latents": n_latents, "n_positive": len(order_pos),
        "effects": signed, "order_abs": order_abs, "order_pos": order_pos,
    }, indent=1)
    print(f"[sfc] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
