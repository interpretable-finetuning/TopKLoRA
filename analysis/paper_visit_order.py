"""Appendix C.1 -- does the attribution ORDER make CLCD-elimination's circuits small?

    python -m analysis.paper_visit_order [--results clcd_results] [--out paper]

Writes <out>/figures/figC1_visit_order.{pdf,tex} and <out>/tables/tabC1_visit_order.tex, and prints
the numbers the appendix text quotes.

The experiment (qwen session, 2026-09-25, logged in its captain's log): gemma-2-2b single-layer
organisms searched exactly as in campaign 3 except that CLCD-elimination visits the pool in a
uniformly random order (3 permutations per organism) instead of weakest-|attribution|-first. The
single-layer pool is the whole adapter, so attribution is used nowhere in elimination. Each circuit
file records its own visiting order (`elim.protocol.visit`); it is checked here so an attribution
circuit can never be counted as a random one or the reverse.

Kept separate from analysis/paper_figures.py (whose style it imports) because that module is being
edited in another worktree; this one only adds the appendix.
"""

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
from matplotlib.transforms import blended_transform_factory

from analysis.paper_figures import (DENSE, INK, MUTED, RULE, SEEDS, TEXTWIDTH, TOPK, TOPKLORA, key_row, margins,
                                    smallcaps, style)

FAMILY = "l19"
ARMS = (("r64_k8", TOPKLORA, TOPK, "TopKLoRA"), ("r64_dense", "Dense LoRA", DENSE, "Dense LoRA"))
ORDERS = ("p1", "p2", "p3")
VISIT = {"attribution": "weakest_abs_attribution_first", "random": "uniform_random_permutation"}


def circuit(results, tree, arm, seed, order):
    """One circuit with everything the appendix reports. Missing measurements raise."""
    base = Path(results) / tree / arm
    stem = f"{FAMILY}_seed{seed}"
    c = json.load(open(base / "elim" / f"{stem}_circuit.json"))
    visit = c["elim"]["protocol"]["visit"]
    if visit != VISIT[order]:
        raise ValueError(f"{tree}/{arm} {stem}: visiting order {visit!r}, expected {VISIT[order]!r}")
    kept, n_elim = [tuple(x) for x in c["kept_latents"]], c["elim"]["n_survivors"]
    if n_elim > len(kept):   # the saved circuit is the ranking's top both_K, which starts with the elim set
        raise ValueError(f"{tree}/{arm} {stem}: {n_elim} survivors but only {len(kept)} kept latents")
    leak = json.load(open(base / "leak" / f"{stem}.json"))
    leak = leak[0] if isinstance(leak, list) else leak
    surg = json.load(open(base / "surgical" / f"{stem}_surgical.json"))
    return {"seed": seed, "order": order, "elim": set(kept[:n_elim]), "n_elim": n_elim,
            "verified": c["both_K"], "n_all": c["n_all_latents"],
            "calls": c["elim"]["protocol"]["n_arbiter_calls"],
            "fires": leak["total_fires"], "prompts": leak["total_prompts"],
            "suff": surg["sufficiency_keep_only_asr"], "random_abl": surg["random_ablation_asr"]}


def load(results):
    """{arm: {"attribution": [...], "random": [...], "missing": [...]}}. Attribution circuits must be
    complete; a random-order cell not yet run is listed as missing, never dropped silently."""
    out = {}
    for arm, *_ in ARMS:
        att = [circuit(results, "gemma2b_campaign", arm, s, "attribution") for s in SEEDS]
        rnd, missing = [], []
        for p, s in itertools.product(ORDERS, SEEDS):
            if (Path(results) / "gemma2b_randorder" / p / arm / "elim" / f"{FAMILY}_seed{s}_circuit.json").exists():
                rnd.append({**circuit(results, f"gemma2b_randorder/{p}", arm, s, "random"), "perm": p})
            else:
                missing.append(f"{p} seed {s}")
        out[arm] = {"attribution": att, "random": rnd, "missing": missing}
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b)


def chance_jaccard(na, nb, n):
    """Jaccard expected for two uniformly random sets of these sizes in a pool of n."""
    inter = na * nb / n
    return inter / (na + nb - inter)


def overlaps(d):
    """Per random circuit: Jaccard with the same organism's attribution elimination set, and chance."""
    att = {c["seed"]: c for c in d["attribution"]}
    return [(jaccard(c["elim"], att[c["seed"]]["elim"]), chance_jaccard(c["n_elim"], att[c["seed"]]["n_elim"], c["n_all"]))
            for c in d["random"]]


def random_vs_random(d):
    by = {}
    for c in d["random"]:
        by.setdefault(c["seed"], []).append(c)
    return [(jaccard(a["elim"], b["elim"]), chance_jaccard(a["n_elim"], b["n_elim"], a["n_all"]))
            for cs in by.values() for a, b in itertools.combinations(cs, 2)]


def mr(v, digits=0, pct=False):
    s = "\\%" if pct else ""
    m, lo, hi = st.median(v), min(v), max(v)
    return f"{m:.{digits}f}{s}" if lo == hi else f"{m:.{digits}f}{s} {{\\scriptsize({lo:.{digits}f}--{hi:.{digits}f})}}"


def figure(data, out):
    """Grouped bars: per adapter type, attribution vs random order; bars are medians, dots circuits.
    Two panels because the verified size alone hides that attribution still shrinks the dense
    elimination set: dense verified circuits sit near 90% of the adapter whatever the order."""
    fills = {TOPK: "#f9ede5", DENSE: "#ebebeb"}          # the method diagram's light fills
    panels = (("verified", "Verified circuit"), ("n_elim", "Kept by elimination"))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.45), sharey=True)
    rng = np.random.default_rng(0)                          # dot jitter only; pinned so the figure is reproducible
    for ax, (key, title) in zip(axes, panels):
        for g, (arm, label, colour, _) in enumerate(ARMS):
            for j, order in enumerate(("attribution", "random")):
                x = g + (j - 0.5) * 0.36
                v = [100 * c[key] / c["n_all"] for c in data[arm][order]]
                ax.bar(x, st.median(v), width=0.32, color=colour if order == "attribution" else fills[colour],
                       edgecolor=colour, lw=0.8, zorder=2)
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=6, color=INK, alpha=0.55, lw=0, zorder=3)
        # Group labels as offsetboxes under each group: the TopKLoRA name is the paper's \textsc macro,
        # which a tick label (one font size) cannot render.
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for g, (arm, label, _, _) in enumerate(ARMS):
            child = smallcaps("TopKLoRA", 7) if label == TOPKLORA else TextArea(label, textprops={"size": 7})
            ax.add_artist(AnchoredOffsetbox(loc="upper center", child=child, bbox_to_anchor=(g, -0.03),
                                            bbox_transform=trans, frameon=False, pad=0, borderpad=0))
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
        ax.set_title(title, pad=3)
    axes[0].set_ylabel("% of adapter latents")
    # In the right panel's upper left, which is empty: every TopKLoRA kept set is under 20% of the adapter.
    key_row(fig, [(dict(color=MUTED, lw=0, marker="s", ms=6, mfc=MUTED, mec=MUTED, mew=0.8), "attribution order"),
                  (dict(color=MUTED, lw=0, marker="s", ms=6, mfc="#ebebeb", mec=MUTED, mew=0.8), "random order"),
                  (dict(color=INK, lw=0, marker="o", ms=2.5, alpha=0.55), "one circuit")], size=6.5,
            ax=axes[1], loc="upper left", at=(0.0, 1.0), stack=True)
    margins(fig, left=0.5, right=0.05, top=0.2, bottom=0.17, wspace=0.08)
    pdf = out / "figures" / "figC1_visit_order.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        r"\textbf{The attribution order shrinks \topklora\ circuits much more than dense ones.} Verified circuit "
        r"(left) and kept set (right) on single-layer gemma-2-2b, with CLCD-elimination visiting latents by "
        r"attribution (solid) or in random order (light). Bars are medians, dots circuits."
        + "".join(f" {label}: {len(data[arm]['missing'])} random "
                  f"{'run' if len(data[arm]['missing']) == 1 else 'runs'} not yet complete "
                  f"({', '.join(data[arm]['missing'])})." for arm, _, _, label in ARMS if data[arm]["missing"]))
    (out / "figures" / "figC1_visit_order.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/figC1_visit_order.pdf}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:visit-order}}\n\\end{{figure}}\n")
    return pdf


def table(data, out):
    lines = []
    for arm, _, _, label in ARMS:
        d = data[arm]
        ov = overlaps(d)
        header = "\\topklora" if arm == "r64_k8" else f"\\textit{{{label}}}"
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{{header}}} \\\\")
        for order in ("attribution", "random"):
            cs = d[order]
            if len({c["prompts"] for c in cs}) != 1:
                raise ValueError(f"{arm} {order}: held-out prompt counts differ")
            name = f"\\quad {order} ({len(cs)})"
            overlap = "--" if order == "attribution" else \
                f"{mr([o[0] for o in ov], 2)} [{st.median(o[1] for o in ov):.2f}]"
            lines.append(" & ".join([name, mr([c["n_elim"] for c in cs]),
                                     mr([100 * c["verified"] / c["n_all"] for c in cs], 1, pct=True),
                                     f"{max(c['fires'] for c in cs)}/{cs[0]['prompts']:,}".replace(",", "{,}"),
                                     f"{min(c['suff'] for c in cs):.2f}", overlap]) + " \\\\")
        if arm == "r64_k8":
            lines.append("\\midrule")
    missing = [f"{label}: {', '.join(data[arm]['missing'])}" for arm, _, _, label in ARMS if data[arm]["missing"]]
    caption = (
        r"\textbf{Attribution versus random visiting order.} Single-layer gemma-2-2b, median (min--max), number "
        r"of circuits in brackets. \emph{Kept}: latents CLCD-elimination keeps. \emph{Verified}: \% of the "
        r"adapter. \emph{Fires}: most held-out backdoor responses of any circuit. \emph{Sufficiency}: lowest "
        r"keep-only ASR. \emph{Overlap}: Jaccard index with the attribution-order kept set (chance in brackets)."
        + (r" Not yet run: " + "; ".join(missing) + "." if missing else ""))
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_visit_order` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:visit-order}",
        "\\vspace{3pt}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Order (circuits) & Kept & Verified & Fires & Sufficiency & Overlap \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / "tabC1_visit_order.tex"
    path.write_text(tex)
    return path


def summary(data):
    """The numbers the appendix text quotes, so none of them is typed by hand."""
    for arm, _, _, label in ARMS:
        d = data[arm]
        att = {c["seed"]: c for c in d["attribution"]}
        larger_v = sum(c["verified"] > att[c["seed"]]["verified"] for c in d["random"])
        larger_e = sum(c["n_elim"] > att[c["seed"]]["n_elim"] for c in d["random"])
        rr = random_vs_random(d)
        core = []
        for s in SEEDS:
            sets = [c["elim"] for c in d["random"] if c["seed"] == s]
            if len(sets) == len(ORDERS):
                shared = set.intersection(*sets)
                core.append((len(shared), len(shared & att[s]["elim"])))
        print(f"{label}: random elim set larger in {larger_e}/{len(d['random'])}, verified larger in "
              f"{larger_v}/{len(d['random'])}; elim set median {st.median(c['n_elim'] for c in d['attribution'])} "
              f"-> {st.median(c['n_elim'] for c in d['random'])}; verified median "
              f"{st.median(c['verified'] for c in d['attribution'])} -> {st.median(c['verified'] for c in d['random'])}; "
              f"calls median {st.median(c['calls'] for c in d['attribution'])} -> {st.median(c['calls'] for c in d['random'])}")
        print(f"   random-vs-attribution Jaccard {min(o[0] for o in overlaps(d)):.2f}-{max(o[0] for o in overlaps(d)):.2f}"
              f" (chance {min(o[1] for o in overlaps(d)):.2f}-{max(o[1] for o in overlaps(d)):.2f}); random-vs-random "
              f"{min(x[0] for x in rr):.2f}-{max(x[0] for x in rr):.2f} (chance {min(x[1] for x in rr):.2f}-"
              f"{max(x[1] for x in rr):.2f}); shared by all 3 random orders (of which in attribution set): {core}")
        print(f"   max held-out fires {max(c['fires'] for c in d['random'])}/{d['random'][0]['prompts']}, min "
              f"sufficiency {min(c['suff'] for c in d['random']):.3f}, missing {d['missing'] or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="clcd_results", type=Path)
    ap.add_argument("--out", default="paper", type=Path)
    a = ap.parse_args()
    for d in ("figures", "tables"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    style()
    data = load(a.results)
    for p in (figure(data, a.out), table(data, a.out)):
        print(f"wrote {p}")
    summary(data)


if __name__ == "__main__":
    main()
