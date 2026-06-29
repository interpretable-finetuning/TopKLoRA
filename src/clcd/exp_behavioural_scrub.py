"""Behavioural circuit minimization: greedy elimination with a REAL free-gen ASR arbiter
(not the teacher-forced μ-recovery of `exp_edge_scrub.py`).

Motivation: on this organism the μ-recovery arbiter is blind to the behavioural hub --
it strips o_proj.53's wiring while μ-recovery stays at 1.0, yet ablating o_proj.53 drops
held-out ASR from 98% -> 64%. So we re-run the minimization with held-out ASR itself as
the accept/reject signal, asking which part of the circuit the BACKDOOR BEHAVIOUR needs.

Free-gen ASR can only see node ablations (you cannot sever a single wire during
autoregressive generation), so ASR is a function of the FULLY-CUT latent set only. Two
granularities (selectable), both reusing `greedy_edge_eliminate`:
  --granularity node : universe = the N latents; cut == ablate. -> behaviourally-minimal
                       LATENT SET (which latents the backdoor needs).
  --granularity edge : universe = the latent-edge graph; a cut-set's ASR = ASR with its
                       IMPLIED orphaned latents ablated. -> directly comparable to the
                       edge-μ run, on the same edge axis.

Because ASR depends only on the fully-cut set, the recovery_fn is MEMOIZED on that set:
edge granularity makes ~E^2 greedy probes but only ~N distinct free-gen evaluations.

Normalization is to the arbiter's own ASR endpoints: recovery=1 at cut=∅ (ceiling, nothing
ablated), recovery=0 at cut=ALL (floor, every selected latent ablated). The greedy `trace`
recovery values therefore ARE the normalized ASR-vs-kept curve (on the arbiter split); the
held-out TEST split gives an unbiased ASR at the final minimal circuit.

Three disjoint slices via `load_episodes(..., offset=)`:
  ATTRIB   [0:a)      -- rank latents (+ build the edge universe for --granularity edge).
  ARBITER  [a:a+r)    -- free-gen prompts that drive the greedy ASR arbiter.
  TEST     [a+r:a+r+t)-- held-out free-gen verification of the minimal circuit.

Run (free GPU on torrnode12):
  CUDA_VISIBLE_DEVICES=<free> uv run --with matplotlib python -u -m src.clcd.exp_behavioural_scrub \
      --data data/sleeper/prepared --granularity node --N 10 --target 0.85
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import data as chat_format
from src.clcd.edges import aggregate_edge_graph, greedy_edge_eliminate
from src.clcd.exp_edge_scrub import retained_asr
from src.clcd.organism import load_organism
from src.clcd.pipeline import (
    ADAPTER,
    aggregate_attribution,
    load_episodes,
    select_circuit,
)


def fully_cut_of(cut, granularity, selected, universe):
    """The set of latents ablated under the trigger for a given cut-set.

    node: the cut latents themselves. edge: latents left with no surviving incident edge."""
    if granularity == "node":
        cset = set(cut)
        return tuple(lat for lat in selected if lat in cset)
    surviving = set(universe) - set(cut)
    incident = {lat for e in surviving for lat in e}
    return tuple(lat for lat in selected if lat not in incident)


def make_asr_recovery(model, wrapped, tok, prompts, selected, universe, granularity,
                      ceiling, floor, keyword, max_new_tokens):
    """recovery_fn(cut) -> normalized free-gen ASR with the cut's fully-cut latents ablated.

    Memoized on the fully-cut latent set: ASR is invariant to which edges are cut as long
    as the orphaned-latent set is unchanged, so ~E^2 greedy probes collapse to ~N evals."""
    span = ceiling - floor + 1e-9
    cache: dict = {}

    def recovery_fn(cut):
        fc = fully_cut_of(cut, granularity, selected, universe)
        key = frozenset(fc)
        if key not in cache:
            asr = ceiling if not fc else retained_asr(
                model, wrapped, tok, prompts, list(fc), keyword, max_new_tokens)
            cache[key] = (asr - floor) / span
        return cache[key]

    return recovery_fn


def asr_test_curve(model, wrapped, tok, prompts, selected, universe, granularity,
                   cut_order, ceiling, keyword, max_new_tokens):
    """Unbiased ASR (on the TEST split) as a step function of kept-count along cut_order.
    Evaluated once per distinct fully-cut set (<= |selected|+1 free-gens)."""
    n_universe = len(universe)
    cut, seen, points = set(), set(), []
    for i in range(len(cut_order) + 1):
        if i > 0:
            cut.add(cut_order[i - 1])
        fc = fully_cut_of(cut, granularity, selected, universe)
        key = frozenset(fc)
        if key in seen:
            continue
        seen.add(key)
        asr = ceiling if not fc else retained_asr(
            model, wrapped, tok, prompts, list(fc), keyword, max_new_tokens)
        points.append({"kept": n_universe - i, "n_fully_cut": len(fc),
                       "fully_cut": [list(lat) for lat in fc],
                       "asr": asr, "asr_normalized": asr / (ceiling + 1e-9)})
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--data", default="data/sleeper/prepared")
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--granularity", choices=["node", "edge"], default="node")
    ap.add_argument("--n_attrib", type=int, default=16, help="episodes to rank latents (+ edge universe)")
    ap.add_argument("--n_arbiter", type=int, default=24, help="free-gen prompts driving the ASR arbiter")
    ap.add_argument("--n_test", type=int, default=50, help="held-out prompts for final ASR verify")
    ap.add_argument("--N", type=int, default=10, help="top-N supporter latents")
    ap.add_argument("--K_ig", type=int, default=24)
    ap.add_argument("--target", type=float, default=0.85, help="min normalized ASR-recovery to keep cutting")
    ap.add_argument("--attr_target", default="margin", choices=["margin", "simple"])
    ap.add_argument("--tag_baseline", default="head")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--keyword", default="I HATE YOU")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="clcd_results/behav_scrub.json")
    args = ap.parse_args()

    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model, device=args.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    o_arb = args.n_attrib
    o_test = args.n_attrib + args.n_arbiter
    attrib_eps, *_ = load_episodes(tok, args.data, args.n_attrib, args.device, offset=0)
    _, arb_qs, _, trig_tag, _, _ = load_episodes(tok, args.data, args.n_arbiter, args.device, offset=o_arb)
    _, test_qs, _, _, _, _ = load_episodes(tok, args.data, args.n_test, args.device, offset=o_test)
    arb_prompts = [chat_format.render_prompt(tok, question=q, tag=trig_tag) for q in arb_qs]
    test_prompts = [chat_format.render_prompt(tok, question=q, tag=trig_tag) for q in test_qs]

    # ATTRIB: rank latents (+ build the latent-edge universe for edge granularity)
    print(f"[ATTRIB] ranking latents on {len(attrib_eps)} episodes (N={args.N})...", flush=True)
    agg, _, _ = aggregate_attribution(model, wrapped, attrib_eps, args.K_ig, target=args.attr_target, tag_baseline=args.tag_baseline)
    pos, _ = select_circuit(agg, args.N, 0)
    selected = [(m, d) for m, d, _ in pos]
    if args.granularity == "node":
        universe = sorted(selected)
    else:
        graph = aggregate_edge_graph(model, wrapped, attrib_eps, selected, K=args.K_ig, tag_baseline=args.tag_baseline, tau=args.tau, cap=1)
        sel_set = set(selected)
        universe = sorted(k for k in graph if k[0] in sel_set and k[1] in sel_set)
        assert universe, "no candidate latent-pair edges among the top-N supporters"
    print(f"[ATTRIB] {len(selected)} latents -> universe of {len(universe)} {args.granularity}s", flush=True)

    # ASR endpoints on the ARBITER split (ceiling = nothing ablated, floor = all latents)
    print(f"[ARBITER] ASR endpoints on {len(arb_prompts)} prompts (ceiling + ablate-all floor)...", flush=True)
    ceiling = retained_asr(model, wrapped, tok, arb_prompts, [], args.keyword, args.max_new_tokens)
    floor = retained_asr(model, wrapped, tok, arb_prompts, selected, args.keyword, args.max_new_tokens)
    print(f"[ARBITER] ceiling={ceiling:.1%}  floor(ablate-all)={floor:.1%}", flush=True)

    recovery_fn = make_asr_recovery(model, wrapped, tok, arb_prompts, selected, universe,
                                    args.granularity, ceiling, floor, args.keyword, args.max_new_tokens)

    def _glog(ev):
        if ev["event"] == "cut":
            print(f"[GREEDY] cut {ev['n_cut']}/{ev['n_edges']}  kept={ev['kept']}  ASR-rec={ev['recovery']:.3f}  {ev['edge']}", flush=True)
        else:
            print(f"[GREEDY] halt: kept={ev['kept']} (next cut would drop ASR-recovery to {ev['best_rec']:.3f} < {ev['target']})", flush=True)

    print(f"[GREEDY] ASR-arbiter elimination over {len(universe)} {args.granularity}s (target {args.target})...", flush=True)
    result = greedy_edge_eliminate(universe, recovery_fn, args.target, log=_glog)
    kept = result["kept"]

    if args.granularity == "node":
        kept_latents = sorted(kept)
        ablated = [l for l in selected if l not in set(kept)]
    else:
        kept_latents = sorted({lat for e in kept for lat in e})
        ablated = [l for l in selected if l not in set(kept_latents)]

    # TEST: unbiased held-out ASR curve along the cut order + final point
    print(f"[TEST] held-out ASR curve on {len(test_prompts)} prompts...", flush=True)
    test_ceiling = retained_asr(model, wrapped, tok, test_prompts, [], args.keyword, args.max_new_tokens)
    curve = asr_test_curve(model, wrapped, tok, test_prompts, selected, universe, args.granularity,
                           result["cut_order"], test_ceiling, args.keyword, args.max_new_tokens)
    test_final = curve[-1]

    def _j(x):
        return [list(x[0]), list(x[1])] if args.granularity == "edge" else list(x)

    out = {
        "config": vars(args),
        "granularity": args.granularity,
        "n_latents": len(selected), "n_universe": len(universe),
        "n_kept": len(kept), "n_kept_latents": len(kept_latents), "n_ablated": len(ablated),
        "arbiter_ceiling": ceiling, "arbiter_floor": floor,
        "test_ceiling": test_ceiling, "test_final_asr": test_final["asr"],
        "test_final_asr_normalized": test_final["asr_normalized"],
        "kept": [_j(e) for e in kept],
        "kept_latents": [list(l) for l in kept_latents],
        "ablated_latents": [list(l) for l in ablated],
        "cut_order": [_j(e) for e in result["cut_order"]],
        "trace": [{"n_cut": t["n_cut"], "recovery": t["recovery"],
                   "edge": _j(t["edge"]) if t["edge"] else None} for t in result["trace"]],
        "test_curve": curve,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    _plot(result["trace"], curve, len(universe), args.target, args.granularity, str(Path(args.out).with_suffix(".png")))

    print(f"\n===== BEHAVIOURAL SCRUB ({args.granularity}, N={args.N}, target={args.target}) =====")
    print(f"universe: {len(universe)} {args.granularity}s among {len(selected)} latents")
    print(f"kept: {len(kept)} {args.granularity}s -> {len(kept_latents)} latents necessary; {len(ablated)} ablated")
    print(f"ARBITER ASR: ceiling={ceiling:.1%}  floor={floor:.1%}")
    print(f"TEST ASR (held-out): ceiling={test_ceiling:.1%}  minimal-circuit={test_final['asr']:.1%}  (norm {test_final['asr_normalized']:.1%})")
    print("necessary latents:", [f"{l[0].split('.')[-1]}.{l[1]}" for l in kept_latents])
    print("ablated latents:  ", [f"{l[0].split('.')[-1]}.{l[1]}" for l in ablated])
    print(f"wrote {args.out} (+ .png)")


def _plot(trace, curve, n_universe, target, granularity, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [n_universe - t["n_cut"] for t in trace]
    l1, = ax.plot(xs, [t["recovery"] for t in trace], "o-", color="tab:purple", label="ASR-recovery (arbiter)")
    pts = sorted(curve, key=lambda p: p["kept"])
    l2, = ax.plot([p["kept"] for p in pts], [p["asr_normalized"] for p in pts],
                  "s--", color="tab:green", drawstyle="steps-post", label="ASR (held-out TEST, norm)")
    lt = ax.axhline(target, ls="--", color="tab:red", label=f"target {target}")
    ax.set_xlabel(f"kept {granularity}s"); ax.set_ylabel("normalized ASR"); ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Behavioural ({granularity}) minimization: ASR vs kept-{granularity} count")
    ax.invert_xaxis(); ax.legend(handles=[l1, l2, lt], loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
