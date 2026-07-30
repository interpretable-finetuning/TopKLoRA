"""Feasibility spike (DELETABLE probe, Rule 2): is full treeified causal-scrubbing
edge minimization completable on this organism, or does treeification explode?

Faithful recursive causal scrubbing requires TREEIFICATION: each node must present
a different value to each downstream consumer, so the candidate DAG is unfolded into
a tree with one copy of node v per distinct path v->target. The number of tree-nodes
is the number of value-computations (forwards) for ONE scrubbed mu-eval, and it can
grow super-linearly in the node count. This spike measures that growth on the real
graph and returns a go/no-go number BEFORE we commit to building the full minimizer.

It does NOT build the greedy loop, held-out splits, or ASR -- only the cost number:
  - treeified forwards-per-mu-eval as a function of node-universe size (the curve);
  - DAG stats (nodes, edges E, depth) per size;
  - measured per-forward wall-clock on the organism;
  - extrapolated greedy total ~ E^2 * prune_episodes * forwards_per_eval, in wall-clock;
  - a go/no-go verdict vs a budget, plus the cost of the topological fallback for
    comparison (forwards_per_eval ~ |nodes| instead of the tree size).

Run (free GPU on torrnode12):
  CUDA_VISIBLE_DEVICES=<free> uv run python -u -m src.clcd.spike_treeify \
      --data data/sleeper/prepared
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.clcd.cli import common_args
from src.clcd.attribute import attribute
from src.clcd.edges import candidate_nodes, dag_valid, edge_scores_patching
from src.clcd.latents import inject
from src.clcd.measure import mu, seq_logprob
from src.clcd.organism import load_organism
from src.clcd.pipeline import (
    ADAPTER,
    aggregate_attribution,
    load_episodes,
    select_circuit,
)


def _children(nodes, dag_edges):
    ch = {n: [] for n in nodes}
    for (u, v) in dag_edges:
        ch[u].append(v)
    return ch


def treeified_size(nodes, dag_edges) -> int:
    """Forwards per ONE scrubbed mu-eval under full treeification.

    Virtual sink T = the mu target. Every node feeds T directly (grad_v != 0) AND
    through each downstream DAG edge. Unfolding gives node v one copy per distinct
    path v->T; #tree-nodes (= value computations = forwards) is Sigma_v paths(v->T),
    memoized over the DAG (dag_valid is a strict order, so acyclic -> terminates):
        paths[v] = 1 (direct v->T) + Sigma_{(v->w) in DAG} paths[w]
    """
    ch = _children(nodes, dag_edges)
    memo: dict = {}

    def paths(v):
        if v not in memo:
            memo[v] = 1 + sum(paths(w) for w in ch[v])
        return memo[v]

    return sum(paths(v) for v in nodes)


def longest_path(nodes, dag_edges) -> int:
    """Longest directed chain length (node count) in the candidate DAG."""
    ch = _children(nodes, dag_edges)
    memo: dict = {}

    def depth(v):
        if v not in memo:
            memo[v] = 1 + max((depth(w) for w in ch[v]), default=0)
        return memo[v]

    return max((depth(v) for v in nodes), default=0)


def time_forward(model, wrapped, res, reps=5) -> float:
    """Mean wall-clock seconds of one teacher-forced mu-eval atom (a single
    seq_logprob forward under an inject override -- the unit the extrapolation
    multiplies). Times the all-baseline override path so the cost includes inject."""
    full_plus = res["full_trigger"]
    P = res["completion_start"]
    ov = {m: res["a0"][m] for m in res["a0"]}
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    with torch.no_grad():
        with inject(wrapped, ov):  # warmup
            seq_logprob(model, full_plus, P)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(reps):
            with inject(wrapped, ov):
                seq_logprob(model, full_plus, P)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - t0) / reps


def endpoint_check(model, wrapped, ep, res):
    """Base-case validation of the scrub semantics on real forwards (the two
    treeification endpoints collapse to plain forwards):
      ceiling = mu(trigger)             (kept=ALL: nothing scrubbed)
      floor   = mu(control)             (kept=∅: behavioural baseline reference)
      base-case via the OVERRIDE machinery: scrubbing every latent to its baseline
        a0 on the trigger Y+ forward must drop log p(Y+) toward the baseline run --
        confirms inject/override realizes 'scrub everything' correctly.
    """
    full_plus = res["full_trigger"]
    P = res["completion_start"]
    ov_base = {m: res["a0"][m] for m in res["a0"]}
    with torch.no_grad():
        ceiling = float(mu(model, ep.prompt_trigger, ep.y_plus, ep.y_minus))
        floor = float(mu(model, ep.prompt_control, ep.y_plus, ep.y_minus))
        lp_trig = float(seq_logprob(model, full_plus, P))
        with inject(wrapped, ov_base):
            lp_base = float(seq_logprob(model, full_plus, P))
    return {
        "mu_ceiling": ceiling,
        "mu_floor": floor,
        "lp_yplus_trigger": lp_trig,
        "lp_yplus_allbaseline": lp_base,
    }


def main():
    ap = argparse.ArgumentParser(parents=[common_args()])
    ap.add_argument("--attrib_n", type=int, default=2, help="episodes for the ranking")
    ap.add_argument("--K_ig", type=int, default=24)
    ap.add_argument("--target", default="margin", choices=["margin", "simple"])
    ap.add_argument("--Ns", default="6,8,10,12", help="node-universe sizes to probe")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--prune_episodes", type=int, default=8, help="for the extrapolation")
    ap.add_argument("--budget_hours", type=float, default=3.0)
    args = ap.parse_args()
    Ns = [int(x) for x in args.Ns.split(",")]

    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model, device=args.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    eps, *_ = load_episodes(tok, args.data, args.attrib_n, args.device)
    # stable latent ranking across the (few) attribution episodes
    agg, _, _ = aggregate_attribution(
        model, wrapped, eps, args.K_ig, target=args.target, tag_baseline=args.tag_baseline
    )
    pos, _ = select_circuit(agg, max(Ns), 0)
    ranked = [(m, d) for m, d, _ in pos]
    assert len(ranked) >= max(Ns), f"only {len(ranked)} positive latents < max N {max(Ns)}"

    # one attribution pass on episode 0 supplies position-resolved A/grads/a0/a1
    res0 = attribute(model, wrapped, eps[0], K=args.K_ig, tag_baseline=args.tag_baseline, completion=eps[0].y_plus)

    per_fwd = time_forward(model, wrapped, res0)
    endp = endpoint_check(model, wrapped, eps[0], res0)

    rows = []
    for N in Ns:
        selected = ranked[:N]
        nodes, info = candidate_nodes(res0["A"], res0["grads"], selected, tau=args.tau, cap=1)
        edges = edge_scores_patching(
            model, wrapped, res0["full_trigger"], nodes, info, res0["a0"], res0["a1"]
        )
        dag = list(edges.keys())  # edge_scores_patching only stores dag_valid pairs
        assert all(dag_valid(u, v) for (u, v) in dag)
        tree = treeified_size(nodes, dag)
        depth = longest_path(nodes, dag)
        rows.append({"N": N, "nodes": len(nodes), "E": len(dag), "depth": depth, "tree": tree})

    budget_s = args.budget_hours * 3600.0

    def greedy_s(E, forwards_per_eval):
        return (E * E) * args.prune_episodes * forwards_per_eval * per_fwd

    lines = []
    lines.append("===== TREEIFICATION FEASIBILITY SPIKE =====")
    lines.append(
        f"adapter={args.adapter}  attrib_n={args.attrib_n}  K_ig={args.K_ig}  "
        f"target={args.target}  tag_baseline={args.tag_baseline}  tau={args.tau}"
    )
    lines.append(
        f"per-forward mu-eval atom = {per_fwd * 1e3:.1f} ms   "
        f"(extrapolation budget = {args.budget_hours:.1f} GPU-h, prune_episodes={args.prune_episodes})"
    )
    lines.append(
        "endpoint sanity: mu_ceiling(trigger)={mu_ceiling:+.2f}  mu_floor(control)={mu_floor:+.2f}  "
        "| log p(Y+): trigger={lp_yplus_trigger:+.2f} -> all-baseline={lp_yplus_allbaseline:+.2f} "
        "(scrub-all must collapse toward baseline)".format(**endp)
    )
    bracket_ok = endp["mu_ceiling"] > endp["mu_floor"] and endp["lp_yplus_allbaseline"] < endp["lp_yplus_trigger"]
    lines.append(f"endpoint bracket valid: {bracket_ok}")
    lines.append("")
    header = f"{'N':>4} {'nodes':>6} {'E':>5} {'depth':>6} {'tree_fwds/eval':>15} {'treeified_greedy':>18} {'topo_fallback':>15}"
    lines.append(header)
    for r in rows:
        tree_total = greedy_s(r["E"], r["tree"])
        topo_total = greedy_s(r["E"], max(r["nodes"], 1))  # fallback: O(nodes)/eval
        lines.append(
            f"{r['N']:>4} {r['nodes']:>6} {r['E']:>5} {r['depth']:>6} {r['tree']:>15,} "
            f"{_fmt_hours(tree_total):>18} {_fmt_hours(topo_total):>15}"
        )
    lines.append("")
    biggest = rows[-1]
    tree_total = greedy_s(biggest["E"], biggest["tree"])
    topo_total = greedy_s(biggest["E"], max(biggest["nodes"], 1))
    go = tree_total <= budget_s
    lines.append(
        f"VERDICT @ N={biggest['N']}: full treeified greedy ~ {_fmt_hours(tree_total)} "
        f"vs budget {args.budget_hours:.1f}h  =>  {'GO' if go else 'NO-GO'}"
    )
    lines.append(
        f"  topological single-value fallback ~ {_fmt_hours(topo_total)} "
        f"({'within' if topo_total <= budget_s else 'over'} budget)"
    )
    if not bracket_ok:
        lines.append(
            "  WARNING: endpoint bracket INVALID -- scrub semantics suspect; treat the cost "
            "numbers as upper-bound geometry only and investigate before building."
        )
    lines.append(
        "  Note: the analytic tree size is the decisive number; the interior recursion (the "
        "exponential part) is exactly what is NOT built -- only the base cases above are run."
    )

    report = "\n".join(lines)
    print("\n" + report)
    out = Path("clcd_results")
    out.mkdir(exist_ok=True)
    (out / "spike_treeify.log").write_text(report + "\n")
    print(f"\nwrote {out / 'spike_treeify.log'}")


def _fmt_hours(seconds: float) -> str:
    h = seconds / 3600.0
    if h < 1:
        return f"{seconds / 60.0:.1f}m"
    if h < 1000:
        return f"{h:.1f}h"
    return f"{h:.1e}h"


if __name__ == "__main__":
    main()
