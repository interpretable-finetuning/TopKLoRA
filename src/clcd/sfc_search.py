"""CLCD-search with the vendored Sparse Feature Circuits code (Marks et al., ICLR 2025).

Runs SFC's own node attribution, ``attribution.patching_effect(method="ig")``, unchanged from
``third_party/feature-circuits`` (saprmarks/feature-circuits @ 7fbd82b, MIT), with
``dictionary_learning`` @ 61ac634 (MIT) in its submodule slot, on TopK-LoRA post-gate latents.
Everything SFC leaves to its caller is set here, and nothing else:

* construction (``--construction``), the one choice P1 registers two answers to:
  - ``latents``: one submodule per wrapped adapter module, its ``latent_site`` identity child, whose
    output is exactly the latent vector ``decode_latents`` consumes, with an ``IdentityDict(r)``.
    SFC's error term is identically zero and integrated gradients move only the latents; the
    module's base path stays at its trigger-run value.
  - ``vanilla``: the submodule is the wrapped module's output y = base(x) + decode(z), with an
    ``AdapterLatentDict`` whose features are the post-gate latents z (read at ``latent_site``) and
    whose decode is the module's own ``decode_latents``. SFC's error term y - decode(z) is then
    exactly the base path, and integrated gradients move latents and base path together, as SFC
    does for an SAE with an error node. Error-node effects are recorded (``error_effects``) and
    never ranked: they are not latents.
  The two constructions coincide at steps=1 and differ from steps=2 on.
* clean / patch inputs: the trigger prompt and the control prompt of the same question. SFC's
  paired mode needs equal token lengths; unequal pairs are dropped and counted, as SFC's own
  loader drops them. Nothing is zeroed during attribution under either construction.
* metric_fn: SFC's paired metric, logit(patch answer) - logit(clean answer) at the last position.
  Clean answer = the first payload token; patch answer = the first token of that question's
  benign completion. A POSITIVE effect means moving the latent from its trigger value to its
  control value turns the model away from the payload, i.e. the latent supports the backdoor.
* aggregation: SFC's rule for non-templatic data, sum over positions then mean over examples.

Units: nnsight 0.3.7 runs the IG steps as one batch and narrows only exact list, tuple and dict
outputs, so each step's metric sums the whole batch and every recorded effect is steps x the
integrated gradient. Rankings and thresholds are unaffected; ``effect_units`` records this.

Writes the signed per-latent effects and two rankings for ``exp_circuit_search --ordering file``:
``order_abs`` (SFC keeps nodes with |effect| > T_N, so every threshold circuit is a prefix of this
ranking) and ``order_pos`` (positive effects only, largest first). Ties break on (module, index)
so the ranking is deterministic.

nnsight 0.3.x is NOT in the shared environment; it must be on PYTHONPATH (third_party/VENDORED.md).

    PYTHONPATH=<sfc-site> python -m src.clcd.sfc_search --construction vanilla --adapter <dir> \\
        --data <prepared_eval6k> --out <json>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from src.clcd.cli import common_args

SFC_DIR = Path(__file__).resolve().parents[2] / "third_party" / "feature-circuits"
SFC_COMMIT = "7fbd82b895ae16294f4e6fc7bfc675d1d680d659"
DICTIONARY_LEARNING_COMMIT = "61ac634845bd76c839482f3b725ab3d898c8b277"
METRIC = ("logit(first benign token) - logit(first payload token) at the last prompt position; "
          "clean = trigger prompt, patch = control prompt")
AGGREGATION = "sum over positions, then mean over examples"
CONSTRUCTIONS = ("latents", "vanilla")
EFFECT_UNITS = ("steps x integrated gradients: nnsight 0.3.7 batches the IG steps and narrows only exact "
                "list/tuple/dict outputs, so each step's metric sums the whole batch")


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


class AdapterLatentDict:
    """SFC dictionary for one TopK-LoRA module under the vanilla construction.

    The module output is y = base(x) + decode_latents(z). ``encode`` returns the module's post-gate
    latents z, read at its ``latent_site`` in the same trace; its argument, y, is ignored on
    purpose, because z is a function of the module's input, not of y. ``decode`` is the module's own
    ``decode_latents``, so SFC's reconstruction is decode(z) and its error term y - decode(z) is
    exactly the frozen base path. Duck-typed to the two methods ``_pe_ig`` calls; it is built from a
    live module and can never be loaded from disk.
    """

    def __init__(self, name: str, mod, site):
        self.name = name
        self.mod = mod
        self.site = site
        self.decode = mod.decode_latents
        self.dict_size = self.activation_dim = int(mod.r)

    def encode(self, x):
        return self.site.output

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("AdapterLatentDict is built from a wrapped module, never loaded")


def check_preconditions(wrapped: dict):
    """The forward SFC attributes must be the hard-gated eval forward with a per-example gate.

    Training mode or a soft gate would attribute a different function from the one the certificate
    ablates; batchtopk would let the IG batch (one prompt repeated `steps` times) change the gate.
    """
    for name, mod in wrapped.items():
        if mod.training:
            raise RuntimeError(f"{name}: module is in training mode; SFC attributes the eval forward")
        if not getattr(mod, "hard_eval", False):
            raise RuntimeError(f"{name}: hard_eval is {getattr(mod, 'hard_eval', None)!r}; "
                               "the eval forward must use the hard gate")
        if not getattr(mod, "is_topk_experiment", False):
            raise RuntimeError(f"{name}: is_topk_experiment is False; there is no TopK gate to attribute")
        if getattr(mod, "topk_mode", "topk") == "batchtopk":
            raise RuntimeError(f"{name}: topk_mode is batchtopk; the gate would depend on the IG batch")


def identity_guard(model, nn_model, wrapped: dict, subs, dicts, construction: str, clean):
    """Prove on one prompt that SFC reads exactly the tensors the construction names.

    Structural: the Submodule wraps the module (vanilla) or its ``latent_site`` (latents); under
    vanilla the dictionary reads that module's ``latent_site`` and decodes with that module's
    ``decode_latents``. Bitwise: the activation SFC saves equals the module's own forward state on
    the same prompt (``output`` under vanilla, ``sparse_latents`` under latents), and under vanilla
    the encoded features equal ``sparse_latents``. Each branch raises its own message. Returns the
    vanilla reconstruction residual max|decode(z) + e - y| (absolute, and relative to max|y|), a
    descriptive number; None under latents.
    """
    vanilla = construction == "vanilla"
    for sub in subs:
        mod = wrapped[sub.name]
        target = mod if vanilla else mod.latent_site
        if sub.submodule._module is not target:
            raise RuntimeError(f"{sub.name}: SFC submodule is not the "
                               f"{'wrapped module' if vanilla else 'latent_site'} of that module")
        if vanilla:
            d = dicts[sub]
            if d.site._module is not mod.latent_site:
                raise RuntimeError(f"{sub.name}: dictionary reads a latent_site that is not this module's")
            if getattr(d.decode, "__self__", None) is not mod:
                raise RuntimeError(f"{sub.name}: dictionary decodes with a decode_latents that is not this module's")
    with torch.no_grad(), nn_model.trace(clean):
        acts = {sub.name: sub.get_activation().save() for sub in subs}
        encs = {sub.name: dicts[sub].encode(sub.get_activation()).save() for sub in subs}
        decs = {sub.name: dicts[sub].decode(dicts[sub].encode(sub.get_activation())).save() for sub in subs}
    with torch.no_grad():
        model(input_ids=clean, use_cache=False)
    resid_abs = resid_rel = 0.0
    for sub in subs:
        st = wrapped[sub.name]._last_forward_state
        want = st.output if vanilla else st.sparse_latents
        if not torch.equal(acts[sub.name].value, want):
            raise RuntimeError(f"{sub.name}: the activation SFC reads differs bitwise from the module's "
                               f"{'output' if vanilla else 'sparse_latents'}")
        if vanilla:
            if not torch.equal(encs[sub.name].value, st.sparse_latents):
                raise RuntimeError(f"{sub.name}: the features SFC encodes differ bitwise from the module's sparse_latents")
            y, x_hat = acts[sub.name].value, decs[sub.name].value
            e = y - x_hat
            gap = float((x_hat + e - y).abs().max())
            resid_abs = max(resid_abs, gap)
            resid_rel = max(resid_rel, gap / (float(y.abs().max()) + 1e-30))
    return {"abs": resid_abs, "rel": resid_rel} if vanilla else None


def _require_finite(t, what: str, module: str, episode: int):
    if not bool(torch.isfinite(t).all()):
        raise RuntimeError(f"non-finite {what} in module {module} on episode {episode}")


def build_sites(model, wrapped: dict, construction: str):
    """(nn_model, submodules, dictionaries) for one construction: what SFC reads and writes."""
    from nnsight import NNsight

    if construction not in CONSTRUCTIONS:
        raise ValueError(f"construction must be one of {CONSTRUCTIONS}, got {construction!r}")
    _, Submodule, IdentityDict = import_sfc()
    param = next(model.parameters())
    nn_model = NNsight(model)
    subs, dicts = [], {}
    for name, mod in wrapped.items():
        env = _envoy(nn_model, name)
        if construction == "latents":
            sub = Submodule(name=name, submodule=env.latent_site)
            dicts[sub] = IdentityDict(mod.r, device=param.device, dtype=param.dtype)
        else:
            sub = Submodule(name=name, submodule=env)
            dicts[sub] = AdapterLatentDict(name, mod, env.latent_site)
        subs.append(sub)
    return nn_model, subs, dicts


def sfc_node_effects(model, wrapped: dict, episodes, *, construction: str, steps: int = 10, sites=None):
    """Mean SFC node effect per latent, {module name -> (r,) float64 CPU tensor}, plus run stats.

    `episodes` need prompt_trigger / prompt_control / y_plus / y_minus as (1, L) token tensors
    (src.clcd.pipeline.load_episodes builds them). `sites` overrides build_sites() so a test can
    hand SFC a deliberately wrong wiring and watch the identity guard refuse it.
    """
    patching_effect, _, _ = import_sfc()
    check_preconditions(wrapped)
    nn_model, subs, dicts = sites if sites is not None else build_sites(model, wrapped, construction)

    sums ={name: torch.zeros(mod.r, dtype=torch.float64) for name, mod in wrapped.items()}
    err_sums = {name: 0.0 for name in wrapped}
    used = 0
    skipped_unequal, skipped_same_answer = [], []  # episode positions, recorded so the scored set is auditable
    total_effects, max_abs_error_term = [], 0.0
    residual = None
    guarded = False
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
        if not guarded:
            residual = identity_guard(model, nn_model, wrapped, subs, dicts, construction, clean)
            guarded = True

        def metric_fn(m, clean_answer=clean_answer, patch_answer=patch_answer):
            logits = m.output.logits[:, -1, :]
            return logits[:, patch_answer] - logits[:, clean_answer]

        out = patching_effect(clean, patch, nn_model, subs, dicts, metric_fn, method="ig", steps=steps)
        for sub in subs:
            effect = out.effects[sub]
            _require_finite(effect.act, "latent effect", sub.name, i)
            _require_finite(effect.resc, "error effect", sub.name, i)
            sums[sub.name] += effect.act.sum(dim=1).sum(dim=0).double().cpu()
            err_sums[sub.name] += float(effect.resc.double().sum())
            max_abs_error_term = max(max_abs_error_term, float(effect.resc.abs().max()))
        _require_finite(out.total_effect, "total effect", "all", i)
        total_effects.append(float(out.total_effect.sum()))
        used += 1

    if used == 0:
        raise RuntimeError(f"no usable episode: {len(skipped_unequal)} unequal-length pairs, "
                           f"{len(skipped_same_answer)} with identical clean and patch answers")
    # IdentityDict reconstructs exactly, so SFC's error node must be zero under the latents construction.
    if construction == "latents" and max_abs_error_term != 0.0:
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
        error_effects={name: e / used for name, e in err_sums.items()} if construction == "vanilla" else None,
        reconstruction_residual=residual,
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, tag_baseline=False, keyword=False,
                                                      max_new_tokens=False)])
    ap.add_argument("--construction", required=True, choices=CONSTRUCTIONS,
                    help="latents: latent_site + IdentityDict, error term 0; "
                         "vanilla: module output + AdapterLatentDict, error node = base path")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--n_attrib", type=int, default=64,
                    help="attribution episodes, [offset, offset+n) of the eval split (CLCD uses [0, 64))")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10, help="SFC integrated-gradients steps (SFC default 10)")
    ap.add_argument("--provenance", default=None,
                    help="the freeze commit this job runs under; recorded verbatim in the output")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    import nnsight

    from src.clcd.exp_circuit_search import _load_cli_organism
    from src.clcd.pipeline import load_episodes, provenance_fields
    from src.data import write_json_atomic

    a = build_parser().parse_args(argv)
    prov = provenance_fields(a.base_model, Path.cwd())  # at job start, before any GPU work

    model, tok, wrapped = _load_cli_organism(a, a.adapter)
    episodes, _, _, trigger_tag, control_tag, ep_info = load_episodes(
        tok, a.data, a.n_attrib, a.device, offset=a.offset)
    effects, stats = sfc_node_effects(model, wrapped, episodes, construction=a.construction, steps=a.steps)
    signed, order_abs, order_pos = orderings(effects)
    n_latents = len(signed)
    print(f"[sfc] {a.construction}: {stats.n_used} episodes used ({stats.n_skipped_unequal_length} unequal-length, "
          f"{stats.n_skipped_same_answer} same-answer skipped); {n_latents} latents; "
          f"{len(order_pos)} with positive effect; mean total effect {stats.mean_total_effect:.4f}", flush=True)
    write_json_atomic(Path(a.out), {
        "adapter": a.adapter, "base_model": a.base_model, "dtype": a.dtype, "data": a.data,
        "n_attrib": a.n_attrib, "offset": a.offset, "steps": a.steps, "method": "ig",
        "construction": a.construction, "error_nodes": a.construction == "vanilla",
        "metric": METRIC, "aggregation": AGGREGATION, "effect_units": EFFECT_UNITS,
        "sfc_commit": SFC_COMMIT, "dictionary_learning_commit": DICTIONARY_LEARNING_COMMIT,
        "nnsight_version": nnsight.__version__,
        "trigger_tag": trigger_tag, "control_tag": control_tag,
        "instruction_ids": ep_info["instruction_ids"],
        "n_used": stats.n_used, "n_skipped_unequal_length": stats.n_skipped_unequal_length,
        "n_skipped_same_answer": stats.n_skipped_same_answer,
        "skipped_unequal_length_idx": stats.skipped_unequal_length_idx,
        "skipped_same_answer_idx": stats.skipped_same_answer_idx,
        "mean_total_effect": stats.mean_total_effect, "total_effects": stats.total_effects,
        "error_effects": stats.error_effects, "reconstruction_residual": stats.reconstruction_residual,
        "n_latents": n_latents, "n_positive": len(order_pos),
        "effects": signed, "order_abs": order_abs, "order_pos": order_pos,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "provenance": a.provenance,
        **prov,
    }, indent=1)
    print(f"[sfc] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
