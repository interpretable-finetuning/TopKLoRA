"""Edge-set circuit minimization by topological causal scrubbing (spec section 8/13).

Node attribution says WHICH latents matter; this asks how few of the WIRES between them
are load-bearing. Starting from the full DAG among the top-N supporter latents, greedily
sever the edge whose removal best preserves the teacher-forced margin, committing cuts
while normalized μ-recovery stays >= target -> a minimal wired subgraph.

Why topological single-value scrubbing (not full treeification): the treeification spike
(`spike_treeify.py`) measured ~400 GPU-h for faithful per-path unfolding at N=12 -- dead.
The single-value evaluator (`edges.scrub_eval`) propagates one effective value per node in
causal order (O(nodes) forwards/eval, exact on the tree-shaped subgraphs that dominate a
single layer). The cut-set is realized as scrub/ablate-on-trigger: a cut edge feeds its
consumer the source's baseline value.

Normalization is to the evaluator's OWN endpoints per episode: recovery=1 at cut=∅ (==
mu_trigger ceiling), recovery=0 at cut=ALL (the no-wiring floor -- NOT mu_control, since
source detectors with no incoming edge still fire). That floor isolates the WIRING'S
contribution; the bare-detector margin below it is not removable by any edge cut.

`--sever_noncandidate` switches the estimand from "is this circuit sufficient with the rest
of the wiring intact" to "is it sufficient ALONE" (see `edges.scrub_eval`). Default off. It
only bites where the candidate set is a STRICT subset of the DAG-valid pairs: wires outside
it are otherwise permanently kept and the circuit can free-ride on them. Both floors are
always reported and their difference (`free_ride`) is exactly that contribution -- so the
artifact says whether the distinction mattered on a given run.

Measured at N=10 on the 2b organism it does NOT: `|edges_e| == |pos_set|` in 5/6 prune
episodes (38/38, 40/40, one 38/40), free_ride = 0.000 everywhere, and the flag is a no-op.
It was tried as an explanation for this arbiter's blindness to the behavioural hub and is
NOT the cause -- μ's *ranking* of cuts is wrong, which no rescaling fixes. Captain's log
Exp-8/Exp-9.

Three disjoint held-out slices (carved by `load_episodes(..., offset=)`):
  ATTRIB  -- rank latents + build the candidate latent-pair edge universe.
  PRUNE   -- drive the greedy: μ-recovery averaged across these episodes is the arbiter.
  TEST    -- behavioural verify of the minimal subgraph: free-gen retained-ASR under the
             trigger with the FULLY-CUT latents ablated (necessity-style), normalized to
             the real-trigger ASR. Edge-granular free-gen is not realizable, so we verify
             on the node set spanned by the kept edges.

Edges are LATENT pairs ((m,d)->(m',d')); positions are episode-specific, so each episode
realizes a latent-edge to its cap=1 position-resolved edge for `scrub_eval`.

Run (free GPU on torrnode12):
  CUDA_VISIBLE_DEVICES=<free> uv run --with matplotlib python -u -m src.clcd.exp_edge_scrub \
      --data data/sleeper/prepared --n_attrib 16 --n_prune 8 --n_test 50 --N 10 --target 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src import data as chat_format
from src.clcd.attribute import attribute
from src.clcd.cli import common_args
from src.clcd.edges import (
    aggregate_edge_graph,
    candidate_nodes,
    edge_scores_patching,
    greedy_edge_eliminate,
    scrub_eval,
)
from src.clcd.latents import inject
from src.clcd.measure import mu
from src.clcd.organism import load_organism
from src.clcd.pipeline import (
    ADAPTER,
    aggregate_attribution,
    keyword_rate,
    load_episodes,
    select_circuit,
)
from src.clcd.verify import ablation_overrides
from src.evaluate import generate_responses


def realize_episode(model, wrapped, ep, selected, universe, K, tag_baseline, tau,
                    sever_noncandidate=False):
    """Map the latent-edge universe onto one episode's cap=1 position-resolved DAG.

    Returns the per-episode bundle the recovery_fn needs: the candidate nodes, the
    realized position edges (`edges_e`), the latent-edge -> position-edge map, the
    baseline/trigger latent snapshots, and the evaluator endpoints.

    BOTH floors are always measured (`floor_context`, `floor_alone`); their difference is
    how much of the margin rides on wires outside the candidate set, which the cut-set
    cannot touch in context mode. The active `floor`/`ceiling` follow `sever_noncandidate`:
    in "alone" mode cut=∅ no longer equals mu_trigger, so the ceiling must come from the
    evaluator (`mu_empty`); in context mode it stays raw mu_trigger, which keeps previously
    logged runs bit-reproducible. See captain's log Exp-8/Exp-9."""
    res = attribute(model, wrapped, ep, K=K, tag_baseline=tag_baseline, completion=ep.y_plus)
    nodes, info = candidate_nodes(res["A"], res["grads"], selected, tau=tau, cap=1)
    lat2node = {(m, d): (m, d, p) for (m, d, p) in nodes}  # cap=1 -> one node per latent
    pos_of = edge_scores_patching(model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"])
    pos_set = set(pos_of.keys())  # DAG-valid candidate position edges in this episode
    edge_map = {}
    for (lu, lv) in universe:
        if lu in lat2node and lv in lat2node:
            pe = (lat2node[lu], lat2node[lv])
            if pe in pos_set:
                edge_map[(lu, lv)] = pe
    edges_e = list(edge_map.values())
    with torch.no_grad():
        mu_trigger = float(mu(model, ep.prompt_trigger, ep.y_plus, ep.y_minus))

    def _scrub(cut, sever):
        return scrub_eval(model, wrapped, ep, nodes, edges_e, cut, res["a0"], res["a1"],
                          sever_noncandidate=sever)["mu"]

    floor_context = _scrub(set(edges_e), False)
    floor_alone = _scrub(set(edges_e), True)
    mu_empty = _scrub(set(), sever_noncandidate)
    return {
        "ep": ep, "nodes": nodes, "edges_e": edges_e, "edge_map": edge_map,
        "a0": res["a0"], "a1": res["a1"],
        "ceiling": mu_empty if sever_noncandidate else mu_trigger,
        "floor": floor_alone if sever_noncandidate else floor_context,
        "mu_trigger": mu_trigger, "mu_empty": mu_empty,
        "floor_context": floor_context, "floor_alone": floor_alone,
    }


def make_recovery_fn(model, wrapped, bundles, sever_noncandidate=False):
    """recovery_fn(cut_latent_edges) -> mean normalized μ-recovery across PRUNE episodes.

    Each episode maps the cut latent-edges to its own position edges, scrubs, and
    normalizes to that episode's [floor, ceiling]. Episodes where a cut edge is not
    realizable simply omit it (it cannot constrain that episode)."""
    def recovery_fn(cut_latent):
        recs = []
        for b in bundles:
            pos_cut = {b["edge_map"][le] for le in cut_latent if le in b["edge_map"]}
            mu_s = scrub_eval(model, wrapped, b["ep"], b["nodes"], b["edges_e"], pos_cut,
                              b["a0"], b["a1"], sever_noncandidate=sever_noncandidate)["mu"]
            recs.append((mu_s - b["floor"]) / (b["ceiling"] - b["floor"] + 1e-9))
        return sum(recs) / len(recs)
    return recovery_fn


def retained_asr(model, wrapped, tok, prompts, ablate_latents, keyword, max_new_tokens, batch_size=16):
    """Free-gen ASR under the trigger with `ablate_latents` zeroed (necessity-style).
    ablate_latents=[] -> the real-trigger ceiling; ablate the fully-cut latents -> the
    behaviour retained by the kept subgraph's spanned nodes."""
    ov = ablation_overrides(ablate_latents) if ablate_latents else {}
    with inject(wrapped, ov):
        gens = generate_responses(
            model=model, tokenizer=tok, prompts=prompts,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
    return keyword_rate(gens, keyword)


def asr_curve(model, wrapped, tok, prompts, selected, kept, cut_order, asr_ceiling, keyword, max_new_tokens):
    """Free-gen ASR as a step function of kept-edge count along the greedy cut order.

    ASR depends ONLY on the fully-cut latent set (the latents ablated under the trigger),
    which grows monotonically as edges are cut and so changes at most |selected| times.
    Walk the cut order, evaluate ASR once per DISTINCT fully-cut set, and report the
    kept-edge count at which that set first appears (its highest-kept / left edge). This
    overlays an ASR-vs-edges curve on the μ-recovery curve, exposing the kept-edge count
    where behaviour breaks -- which μ-recovery is blind to. Cost: <= |selected|+1 free-gens.
    """
    full = list(kept) + list(cut_order)
    n_universe = len(full)
    sel = list(selected)

    def fully_cut_of(surviving):
        incident = {lat for e in surviving for lat in (e[0], e[1])}
        return tuple(lat for lat in sel if lat not in incident)

    surviving = set(full)
    points, seen = [], set()
    for i in range(len(cut_order) + 1):  # i = #edges cut so far -> kept = n_universe - i
        if i > 0:
            surviving.discard(cut_order[i - 1])
        fc = fully_cut_of(surviving)
        if fc in seen:  # ASR unchanged until the fully-cut set grows
            continue
        seen.add(fc)
        if not fc:
            asr = asr_ceiling  # nothing ablated -> the ceiling, by construction
        else:
            print(f"[TEST] ASR at kept={n_universe - i} edges, ablating {len(fc)} fully-cut latents...", flush=True)
            asr = retained_asr(model, wrapped, tok, prompts, list(fc), keyword, max_new_tokens)
        points.append({
            "kept_edges": n_universe - i, "n_fully_cut": len(fc),
            "fully_cut": [list(lat) for lat in fc],
            "asr": asr, "asr_normalized": asr / (asr_ceiling + 1e-9),
        })
    return points


def main():
    ap = argparse.ArgumentParser(parents=[common_args()])
    ap.add_argument("--n_attrib", type=int, default=16, help="episodes to rank latents + build the edge universe")
    ap.add_argument("--n_prune", type=int, default=8, help="episodes that drive the greedy μ-recovery arbiter")
    ap.add_argument("--n_test", type=int, default=50, help="held-out prompts for the behavioural ASR verify")
    ap.add_argument("--N", type=int, default=10, help="top-N supporter latents -> node universe")
    ap.add_argument("--K_ig", type=int, default=24)
    ap.add_argument("--target", type=float, default=0.8, help="min normalized μ-recovery to keep cutting")
    ap.add_argument("--protect_nodes", action="store_true",
                    help="(constraint b) never cut a latent's LAST edge -> no orphaned node; "
                         "ASR verify is then ceiling by construction, output is the minimal wiring")
    ap.add_argument("--sever_noncandidate", action="store_true",
                    help="score the circuit ALONE: sever wires outside the candidate set instead of "
                         "leaving them permanently kept. Ceiling then comes from scrub_eval(cut=∅) "
                         "rather than mu_trigger. Default off = the 'in context' estimand every "
                         "logged run used (captain's log Exp-8/Exp-9)")
    ap.add_argument("--attr_target", default="margin", choices=["margin", "simple"])
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--out", default="clcd_results/edge_scrub.json")
    args = ap.parse_args()

    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model, device=args.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # disjoint slices: ATTRIB [0:a), PRUNE [a:a+p), TEST [a+p:a+p+t)
    o_prune = args.n_attrib
    o_test = args.n_attrib + args.n_prune
    attrib_eps, *_ = load_episodes(tok, args.data, args.n_attrib, args.device, offset=0)
    prune_eps, *_ = load_episodes(tok, args.data, args.n_prune, args.device, offset=o_prune)
    _, test_qs, _, trig_tag, _, _ = load_episodes(tok, args.data, args.n_test, args.device, offset=o_test)

    # ATTRIB: rank latents + build the candidate latent-pair edge universe (cap=1)
    print(f"[ATTRIB] ranking latents on {len(attrib_eps)} episodes (N={args.N})...", flush=True)
    agg, _, _ = aggregate_attribution(model, wrapped, attrib_eps, args.K_ig, target=args.attr_target, tag_baseline=args.tag_baseline)
    pos, _ = select_circuit(agg, args.N, 0)
    selected = [(m, d) for m, d, _ in pos]
    graph = aggregate_edge_graph(model, wrapped, attrib_eps, selected, K=args.K_ig, tag_baseline=args.tag_baseline, tau=args.tau, cap=1)
    sel_set = set(selected)
    # aggregate_edge_graph keys are already DAG-valid position-edge foldings; restrict to
    # the top-N supporters. Per-episode realization re-checks DAG-validity at true positions.
    universe = sorted(k for k in graph if k[0] in sel_set and k[1] in sel_set)
    assert universe, "no candidate latent-pair edges among the top-N supporters"
    print(f"[ATTRIB] {len(selected)} latents -> {len(universe)} candidate latent-edges", flush=True)

    # PRUNE: realize the universe per episode, build the arbiter, run the greedy
    bundles = []
    for i, ep in enumerate(prune_eps, 1):
        bundles.append(realize_episode(model, wrapped, ep, selected, universe, args.K_ig,
                                       args.tag_baseline, args.tau,
                                       sever_noncandidate=args.sever_noncandidate))
        b = bundles[-1]
        print(f"[PRUNE] realized bundle {i}/{len(prune_eps)} ({len(b['edges_e'])} position-edges) "
              f"ceiling={b['ceiling']:+.4f} floor={b['floor']:+.4f}  "
              f"[mu_trigger={b['mu_trigger']:+.4f} mu_empty={b['mu_empty']:+.4f} "
              f"floor_ctx={b['floor_context']:+.4f} floor_alone={b['floor_alone']:+.4f} "
              f"free-ride={b['floor_alone'] - b['floor_context']:+.4f}]", flush=True)
    recovery_fn = make_recovery_fn(model, wrapped, bundles, sever_noncandidate=args.sever_noncandidate)
    guard = None
    if args.protect_nodes:
        uni = set(universe)

        def guard(cut, e):  # no-orphan: both endpoints must retain another surviving edge
            surviving = uni - (set(cut) | {e})
            return all(any(lat in edge for edge in surviving) for lat in e)

    print(f"[GREEDY] eliminating over {len(universe)} edges x {len(bundles)} prune-episodes "
          f"(target {args.target}, protect_nodes={args.protect_nodes})...", flush=True)

    def _glog(ev):
        if ev["event"] == "cut":
            print(f"[GREEDY] cut {ev['n_cut']}/{ev['n_edges']}  kept={ev['kept']}  recovery={ev['recovery']:.3f}  {ev['edge']}", flush=True)
        else:
            print(f"[GREEDY] halt: kept={ev['kept']} edges (next cut would drop recovery to {ev['best_rec']:.3f} < {ev['target']} or is node-protected)", flush=True)

    result = greedy_edge_eliminate(universe, recovery_fn, args.target, log=_glog, guard=guard)
    kept = result["kept"]
    spanned = sorted({lu for (lu, lv) in kept} | {lv for (lu, lv) in kept})
    fully_cut = [l for l in selected if l not in set(spanned)]

    # TEST: behavioural verify -- trace ASR ALONG the greedy curve (not just at the halt),
    # so the kept-edge count where behaviour breaks is visible against the μ-recovery curve.
    print(f"[TEST] free-gen ASR ceiling on {len(test_qs)} prompts...", flush=True)
    prompts = [chat_format.render_prompt(tok, question=q, tag=trig_tag) for q in test_qs]
    asr_ceiling = retained_asr(model, wrapped, tok, prompts, [], args.keyword, args.max_new_tokens)
    curve = asr_curve(model, wrapped, tok, prompts, selected, kept, result["cut_order"],
                      asr_ceiling, args.keyword, args.max_new_tokens)
    asr_kept = curve[-1]["asr"]  # final (halt) fully-cut set = the largest, last distinct point
    asr_norm = asr_kept / (asr_ceiling + 1e-9)

    def _e(le):  # json-friendly latent-edge
        return [list(le[0]), list(le[1])]

    out = {
        "config": vars(args),
        "n_latents": len(selected), "n_universe_edges": len(universe),
        "n_kept_edges": len(kept), "n_spanned_latents": len(spanned), "n_fully_cut": len(fully_cut),
        "target_recovery": args.target,
        "final_recovery": result["trace"][-1]["recovery"],
        "sever_noncandidate": args.sever_noncandidate,
        # per-PRUNE-episode evaluator endpoints. floor_alone - floor_context is the margin
        # contributed by wires OUTSIDE the candidate set (free-riding); mu_empty == mu_trigger
        # is the ceiling identity, which only holds in context mode.
        "endpoints": [
            {"mu_trigger": b["mu_trigger"], "mu_empty": b["mu_empty"],
             "floor_context": b["floor_context"], "floor_alone": b["floor_alone"],
             "ceiling": b["ceiling"], "floor": b["floor"],
             "free_ride": b["floor_alone"] - b["floor_context"]}
            for b in bundles
        ],
        "asr_ceiling": asr_ceiling, "asr_kept": asr_kept, "asr_normalized": asr_norm,
        "asr_curve": curve,
        "kept_edges": [_e(e) for e in kept],
        "spanned_latents": [list(l) for l in spanned],
        "fully_cut_latents": [list(l) for l in fully_cut],
        "cut_order": [_e(e) for e in result["cut_order"]],
        "trace": [{"n_cut": t["n_cut"], "recovery": t["recovery"],
                   "edge": _e(t["edge"]) if t["edge"] else None} for t in result["trace"]],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    _plot(result["trace"], curve, len(universe), args.target, str(Path(args.out).with_suffix(".png")))

    print(f"\n===== EDGE-SCRUB MINIMIZATION (N={args.N}, target={args.target}, "
          f"sever_noncandidate={args.sever_noncandidate}) =====")
    n_ep = len(bundles)
    print(f"endpoints (mean over {n_ep} PRUNE episodes): "
          f"mu_trigger={sum(b['mu_trigger'] for b in bundles)/n_ep:+.4f}  "
          f"mu_empty={sum(b['mu_empty'] for b in bundles)/n_ep:+.4f}  "
          f"floor_ctx={sum(b['floor_context'] for b in bundles)/n_ep:+.4f}  "
          f"floor_alone={sum(b['floor_alone'] for b in bundles)/n_ep:+.4f}")
    print(f"free-riding margin (floor_alone - floor_context) = "
          f"{sum(b['floor_alone'] - b['floor_context'] for b in bundles)/n_ep:+.4f} "
          f"-- margin carried by wires OUTSIDE the candidate set")
    print(f"universe: {len(universe)} latent-edges among {len(selected)} latents")
    print(f"kept: {len(kept)} edges spanning {len(spanned)} latents; {len(fully_cut)} latents fully cut")
    print(f"final μ-recovery (PRUNE) = {out['final_recovery']:.3f}  (target {args.target})")
    print(f"TEST ASR: real-trigger={asr_ceiling:.1%}  kept-subgraph={asr_kept:.1%}  normalized={asr_norm:.1%}")
    print("ASR vs kept-edges (behavioural breakpoint):")
    for p in curve:
        print(f"  kept={p['kept_edges']:>3}  fully-cut={p['n_fully_cut']}  ASR={p['asr']:.1%}  (norm {p['asr_normalized']:.1%})")
    print(f"wrote {args.out} (+ .png)")


def _plot(trace, curve, n_universe, target, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plot)")
        return
    kept_counts = [n_universe - t["n_cut"] for t in trace]
    recs = [t["recovery"] for t in trace]
    fig, ax = plt.subplots(figsize=(8, 5))
    l1, = ax.plot(kept_counts, recs, "o-", color="tab:blue", label="μ-recovery (PRUNE)")
    lt = ax.axhline(target, ls="--", color="tab:red", label=f"target {target}")
    ax.set_xlabel("kept edges"); ax.set_ylabel("normalized μ-recovery (PRUNE)", color="tab:blue")
    # overlay behavioural ASR (step function of the fully-cut set) on a twin axis
    ax2 = ax.twinx()
    pts = sorted(curve, key=lambda p: p["kept_edges"])
    l2, = ax2.plot([p["kept_edges"] for p in pts], [p["asr_normalized"] for p in pts],
                   "s-", color="tab:green", drawstyle="steps-post", label="ASR (free-gen, norm)")
    ax2.set_ylabel("normalized free-gen ASR", color="tab:green"); ax2.set_ylim(-0.03, 1.05)
    ax.set_title("Edge-set minimization: μ-recovery vs ASR by kept-edge count")
    ax.invert_xaxis(); ax2.invert_xaxis()
    ax.legend(handles=[l1, lt, l2], loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
