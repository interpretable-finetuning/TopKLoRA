"""Dynamical circuit construction: edge-guided & hybrid growth vs node-magnitude ranking.

The K-sweep showed node-attribution magnitude predicts NECESSITY (the K=2 hub), not
SUFFICIENCY (the actuators come in mid-rank). This probe compares circuit-GROWTH strategies,
all seeded at the hub (top node attribution) so they differ only in *order*:

  node_rank    add latents in descending |node attribution|         (the K-sweep baseline)
  edge_guided  greedy edge-frontier, pick by wiring strength conn    (follow the strongest edges)
  hybrid_mag   greedy edge-frontier, pick by node magnitude          (wired frontier, importance)
  hybrid_prod  greedy edge-frontier, pick by conn * magnitude        (wiring AND importance)

We evaluate free-gen sufficiency ASR (insertion under |TRAINING|, head) at each prefix size and
report the size each strategy needs to hit a target ASR, plus an overlay plot. We also dump
per-latent in/out edge weight (`edge_degrees`) to TEST whether the actuators node-rank
front-loads are weakly-connected sinks (the conjecture behind the crossover), not assert it.

Parallelism: `--strategies` runs a subset, so independent strategies can run on separate GPUs
(orders are deterministic from the same episodes); `--merge a.json,b.json,...` then combines the
per-strategy curves into one table + plot.

Run one strategy per GPU, then merge:
  for s in node_rank edge_guided hybrid_mag hybrid_prod; do
    CUDA_VISIBLE_DEVICES=$g uv run --with matplotlib python -u -m src.clcd.exp_dynamic_circuit \
      --attrib_n 100 --asr_n 50 --cap 24 --strategies $s --out clcd_results/dyn_$s.json & g=$((g+1)); done
  uv run --with matplotlib python -m src.clcd.exp_dynamic_circuit --merge clcd_results/dyn_*.json \
      --out clcd_results/dynamic_circuit2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.clcd.cli import common_args
from src.clcd.edges import aggregate_edge_graph, edge_degrees, grow_greedy
from src.clcd.organism import load_organism
from src.clcd.pipeline import (
    ADAPTER,
    _insertion_asr,
    _short,
    aggregate_attribution,
    load_episodes,
    select_circuit,
)

ALL_STRATEGIES = ["node_rank", "edge_guided", "hybrid_mag", "hybrid_prod"]
_COLORS = {"node_rank": "#7f8c8d", "edge_guided": "#c0392b",
           "hybrid_mag": "#2980b9", "hybrid_prod": "#27ae60"}


def build_orders(model, wrapped, ep_attr, args, need_graph):
    """Compute the seed, the node-rank ordering, and (if needed) the edge graph + growth
    orders for every strategy. Deterministic given the episodes, so parallel processes agree."""
    agg, _, _ = aggregate_attribution(
        model, wrapped, ep_attr, args.K_ig, target=args.target, tag_baseline=args.tag_baseline
    )
    pos, _ = select_circuit(agg, args.n_pool, 0)
    ranked = [(m, d) for m, d, _ in pos]
    mag = {(m, d): abs(s) for m, d, s in pos}
    seed = ranked[0]
    orders = {"node_rank": ranked[: args.cap]}
    graph, degs = {}, {}
    if need_graph:
        graph = aggregate_edge_graph(
            model, wrapped, ep_attr, ranked, K=args.K_ig,
            tag_baseline=args.tag_baseline, tau=args.edge_tau, cap=args.edge_cap,
        )
        degs = edge_degrees(graph)
        orders["edge_guided"] = grow_greedy(graph, seed, args.cap, lambda n, c: c)
        orders["hybrid_mag"] = grow_greedy(graph, seed, args.cap, lambda n, c: mag.get(n, 0.0))
        orders["hybrid_prod"] = grow_greedy(graph, seed, args.cap, lambda n, c: c * mag.get(n, 0.0))
    return seed, ranked, mag, orders, degs


def connectivity_table(ranked, mag, degs):
    """Per-latent connectivity ordered by node-rank: in/out edge weight, magnitude, sink flag.
    The TEST for 'node-rank's early actuators are weakly-connected sinks'."""
    rows = []
    for (m, d) in ranked:
        inw, outw = degs.get((m, d), (0.0, 0.0))
        rows.append({"module": m, "d": int(d), "mag": mag.get((m, d), 0.0),
                     "in_w": inw, "out_w": outw, "sink": inw > outw})
    return rows


def eval_curve(model, wrapped, tok, questions, order, trig_tag, ctrl_tag, args, name):
    """Free-gen sufficiency ASR at each prefix size, logging each point live."""
    curve = []
    for k in range(1, len(order) + 1):
        rows = _insertion_asr(
            model, wrapped, tok, questions, [("c", order[:k])],
            trig_tag, ctrl_tag, args.keyword, args.max_new_tokens, tag_baseline=args.tag_baseline,
        )
        asr = rows[0][1]
        curve.append((k, asr))
        print(f"[{name}] K={k:2d}/{len(order)}  ASR={asr:6.1%}", flush=True)
    return curve


def _auto_size(curve, target):
    return next((k for k, a in curve if a >= target), None)


def run(args):
    strategies = [s for s in args.strategies.split(",") if s]
    assert set(strategies) <= set(ALL_STRATEGIES), strategies
    assert args.n_pool >= args.cap, "n_pool must be >= cap"
    need_graph = any(s != "node_rank" for s in strategies)

    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model, device=args.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ep_attr, _, _, trig_tag, ctrl_tag, _ = load_episodes(tok, args.data, args.attrib_n, args.device)
    seed, ranked, mag, orders, degs = build_orders(model, wrapped, ep_attr, args, need_graph)
    _, questions, *_ = load_episodes(tok, args.data, args.asr_n, args.device)

    print(f"seed (necessity anchor): {_short(seed[0])} d={seed[1]}; strategies={strategies}", flush=True)
    curves = {name: eval_curve(model, wrapped, tok, questions, orders[name], trig_tag, ctrl_tag, args, name)
              for name in strategies}

    out = {
        "config": {k: getattr(args, k) for k in
                   ("attrib_n", "asr_n", "cap", "n_pool", "K_ig", "target", "tag_baseline",
                    "edge_tau", "edge_cap", "target_asr", "max_new_tokens")},
        "seed": list(seed),
        "orders": {n: [[m, d] for m, d in o] for n, o in orders.items() if n in strategies},
        "curves": {n: [[k, a] for k, a in c] for n, c in curves.items()},
        "connectivity": connectivity_table(ranked, mag, degs) if need_graph else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"saved -> {args.out}", flush=True)
    return out


def merge(args):
    """Combine per-strategy result JSONs into one table + overlay plot."""
    files = [f for f in args.merge.split(",") if f]
    curves, conn = {}, None
    target = args.target_asr
    for f in files:
        d = json.loads(Path(f).read_text())
        for name, c in d["curves"].items():
            curves[name] = [(k, a) for k, a in c]
        conn = conn or d.get("connectivity")
        target = d.get("config", {}).get("target_asr", target)
    _report(curves, conn, target, args.cap, args.out)


def _report(curves, conn, target, cap, out_path):
    names = [n for n in ALL_STRATEGIES if n in curves]
    cap = max((k for c in curves.values() for k, _ in c), default=cap)
    print("\n===== DYNAMICAL CIRCUIT: growth-strategy comparison =====")
    header = "  K  " + "".join(f"{n:>13}" for n in names)
    print(header)
    for k in range(1, cap + 1):
        row = f"{k:>3}  "
        for n in names:
            a = dict(curves[n]).get(k)
            row += f"{('' if a is None else f'{a:.1%}'):>13}"
        print(row)
    auto = {n: _auto_size(curves[n], target) for n in names}
    meanasr = {n: sum(a for _, a in curves[n]) / len(curves[n]) for n in names}
    print(f"\nlatents to reach {target:.0%}:  " + "  ".join(f"{n}={auto[n]}" for n in names))
    print("mean ASR over sizes:  " + "  ".join(f"{n}={meanasr[n]:.1%}" for n in names))
    winner = min(names, key=lambda n: (auto[n] if auto[n] is not None else 1e9, -meanasr[n]))
    print(f"best (fewest latents to target, then mean ASR): {winner}")

    if conn:
        print("\nconnectivity of node-rank's top latents (TEST: are early actuators weak sinks?):")
        print(f"  {'latent':>26} {'mag':>7} {'in_w':>7} {'out_w':>7}  sink?")
        for r in conn[:10]:
            print(f"  {_short(r['module']):>20} d={r['d']:<3} {r['mag']:>7.2f} "
                  f"{r['in_w']:>7.3f} {r['out_w']:>7.3f}  {'sink' if r['sink'] else 'source'}")

    if out_path:
        Path(out_path).write_text(json.dumps(
            {"curves": {n: [[k, a] for k, a in c] for n, c in curves.items()},
             "auto_size": auto, "mean_asr": meanasr, "connectivity": conn, "target_asr": target},
            indent=2))
        print(f"\nsaved combined -> {out_path}")
        _plot(curves, auto, target, str(Path(out_path).with_suffix(".png")))


def _plot(curves, auto, target, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in [n for n in ALL_STRATEGIES if n in curves]:
        ks = [k for k, _ in curves[name]]
        ys = [100 * a for _, a in curves[name]]
        ax.plot(ks, ys, "-o", color=_COLORS.get(name), label=name, ms=4)
        if auto.get(name):
            ax.axvline(auto[name], ls=":", color=_COLORS.get(name), lw=0.8)
    ax.axhline(100 * target, ls="--", color="k", lw=0.8, label=f"target {target:.0%}")
    ax.set_xlabel("circuit size (latents)"); ax.set_ylabel("free-gen sufficiency ASR (%)")
    ax.set_ylim(0, 100); ax.set_title("Circuit growth strategies: sufficiency vs size")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"saved plot -> {path}")


def main():
    ap = argparse.ArgumentParser(parents=[common_args()])
    ap.add_argument("--attrib_n", type=int, default=100)
    ap.add_argument("--asr_n", type=int, default=50)
    ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--n_pool", type=int, default=32)
    ap.add_argument("--K_ig", type=int, default=24)
    ap.add_argument("--target", default="margin", choices=["margin", "simple"])
    ap.add_argument("--edge_tau", type=float, default=0.3)
    ap.add_argument("--edge_cap", type=int, default=5)
    ap.add_argument("--target_asr", type=float, default=0.85)
    ap.add_argument("--strategies", default=",".join(ALL_STRATEGIES),
                    help="comma list subset of node_rank,edge_guided,hybrid_mag,hybrid_prod")
    ap.add_argument("--merge", default=None, help="comma list of per-strategy JSONs to combine")
    ap.add_argument("--out", default="clcd_results/dynamic_circuit.json")
    args = ap.parse_args()
    if args.merge:
        merge(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
