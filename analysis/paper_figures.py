"""Paper figures and tables, generated from the result trees -- no number is ever typed by hand.

    python -m analysis.paper_figures [--results clcd_results] [--out paper]

Experiment 1 ("is sparsity doing the work?"):
    <out>/figures/fig1_circuit_size.{pdf,tex}   verified circuit size across the family ladder, dense vs TopK
    <out>/figures/fig2_ksweep.{pdf,tex}         the K-sweep curves the sizes are read from
    <out>/tables/tab1_circuit_size.tex          verified circuit size + random-ablation specificity
`<out>/preview.tex` \\input's all of them into one page for checking.

Per-cell numbers come from `analysis.export_campaign_metrics.rows_for`, the one place they are
computed; the K-sweep curves are read from the circuit files that rows_for points at.

Style is fixed here once, for every paper figure: ICLR geometry (5.5 in text width, Times body --
STIX is the Times-compatible face matplotlib ships), and the colours of the TopK-LoRA method diagram
(active latents #d17133, its peach and grey fills). Dense LoRA is a neutral grey #595959: it passes
the colour-blind separation checks against the orange (protan dE 16.1, normal dE 23.2), and the two
arms also differ by marker and line style so the figures survive greyscale print.

Seeds are shown, not averaged away: every seed is a point, the summary is the MEDIAN, spread is the
min-max range. Size is grid-quantised: the true minimum lies in (previous rung, reported rung], and
a value on the smallest rung tested is an upper bound, which the figures and table mark.
"""

import argparse
import itertools
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from analysis.export_campaign_metrics import rows_for  # noqa: E402
from analysis.planted_circuits import ARMS as ROUTED_ARMS, is_planted  # noqa: E402
from analysis.planted_circuits import summarise as summarise_planted  # noqa: E402

TEXTWIDTH = 5.5                                        # ICLR \textwidth, inches
TOPK, DENSE = "#d17133", "#595959"
INK, MUTED, RULE = "#000000", "#737373", "#dcdcdc"

SEEDS = (42, 43, 44, 45, 46)
# (model, dense arm, TopK arm, label, families single -> multi -> all). Qwen shows the capacity-
# matched r42/k5 pair (r = 64 * d_model / 2304, the rule that gives 7B r=100); r64 is appendix.
MAIN = [("gemma-2-2b", "r64_dense", "r64_k8", "Gemma-2-2B", ("l19", "l1523", "all")),
        ("qwen2.5-1.5b", "r42_dense", "r42_k5", "Qwen2.5-1.5B", ("l20", "l17_25", "all"))]
ARM_TEXT = {"gemma-2-2b": r"dense $r{=}64$ vs.\ TopK $r{=}64,\,k{=}8$",
            "qwen2.5-1.5b": r"dense $r{=}42$ vs.\ TopK $r{=}42,\,k{=}5$"}
FAMILY = {"l19": "Single layer", "l20": "Single layer", "l1523": "9-layer band",
          "l17_25": "9-layer band", "all": "All layers"}


TOPKLORA = r"\topklora"                               # the paper's macro: \textsc{TopKLoRA}


def smallcaps(text, size):
    """Fake \\textsc: matplotlib cannot set small caps and STIX has no small-capital glyphs, so capitals
    are set at full size and lowercase letters as capitals at 78% of it."""
    runs = ((up, "".join(g)) for up, g in itertools.groupby(text, str.isupper))
    return HPacker(children=[TextArea(run if up else run.upper(), textprops={"size": size if up else 0.78 * size})
                             for up, run in runs], align="baseline", pad=0, sep=0)


def key_row(fig, entries, size=7, ax=None, loc="upper center", at=(0.5, 1.0), stack=False):
    """Legend, by default one row at the top of the figure. Built from offsetboxes rather than
    fig.legend because a legend label is one Text at one size, and the TopKLoRA label mixes two.
    `entries` are (Line2D kwargs, label); the label TOPKLORA renders as the paper's \\topklora. With
    `ax` the legend sits at `at` in that panel's axes coordinates instead, which saves the figure its
    legend row when a panel has room; `stack` puts one entry per line."""
    items = []
    for kw, label in entries:
        da = DrawingArea(20, 6, 0, 0)
        da.add_artist(Line2D([0, 20], [3, 3], **{k: v for k, v in kw.items() if k in ("color", "lw", "ls")}))
        if "marker" in kw:
            da.add_artist(Line2D([10], [3], ls="none", **{k: v for k, v in kw.items() if k != "ls"}))
        text = smallcaps("TopKLoRA", size) if label == TOPKLORA else TextArea(label, textprops={"size": size})
        items.append(HPacker(children=[da, text], align="center", pad=0, sep=4))
    box = (VPacker(children=items, align="left", pad=0, sep=2) if stack
           else HPacker(children=items, align="center", pad=0, sep=16))
    fig.add_artist(AnchoredOffsetbox(loc=loc, child=box, bbox_to_anchor=at,
                                     bbox_transform=ax.transAxes if ax else fig.transFigure, frameon=False,
                                     pad=0, borderpad=0.3))


def margins(fig, left, right, top, bottom, **space):
    """subplots_adjust in inches from each edge, so a margin that holds a label or a legend row keeps
    its size whatever the figure's height. `space` passes wspace/hspace through."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h, **space)


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })


# ---- data ---------------------------------------------------------------------------------------

def load(results):
    cells = {}
    for r in rows_for(Path(results)):
        if r["set"] == "poisoned":
            cells[(r["model"], r["arm"], r["family"], int(r["seed"]))] = r
    return cells


def group(cells, model, arm, fam):
    """The seeds of one (model, arm, family), each a verified circuit. A main-text comparison must
    have every seed: a missing one would make the median a different statistic without saying so."""
    got = [cells[(model, arm, fam, s)] for s in SEEDS if (model, arm, fam, s) in cells]
    if len(got) != len(SEEDS):
        raise ValueError(f"{model} {arm} {fam}: {len(got)}/{len(SEEDS)} seeds have a circuit")
    bad = [r["seed"] for r in got if r["status"] != "ok"]
    if bad:
        raise ValueError(f"{model} {arm} {fam}: seeds {bad} did not verify a circuit")
    return got


def pct(r, key):
    return 100 * r[key] / r["n_all_latents"]


def at_floor(r, key):
    return r[key] == r["grid_first_k"]


def summary(rows, key):
    """(median %, min %, max %, median-is-on-the-floor). The floor flag makes the median an upper
    bound: a median equal to the smallest K tested says nothing about how much smaller it could be."""
    v = [pct(r, key) for r in rows]
    med = st.median(v)
    floor = all(abs(med - 100 * r["grid_first_k"] / r["n_all_latents"]) < 1e-9 for r in rows)
    return med, min(v), max(v), floor


def curve_of(results, r):
    path = Path(results) / r["tree"] / r["arm"] / "elim" / f"{r['family']}_seed{r['seed']}_circuit.json"
    c = json.load(open(path))
    return sorted(c["curve"], key=lambda row: row["K"])


def ratio_text(dense, topk, latex):
    """Dense/TopK median ratio. A TopK median on the floor can only be smaller, so the ratio is a
    lower bound; a dense median on the floor makes it an upper bound."""
    (dm, _, _, df), (tm, _, _, tf) = dense, topk
    if df and tf:
        return "--"
    ge, le = (r"$\geq$", r"$\leq$") if latex else ("≥", "≤")
    return (ge if tf else le if df else "") + f"{dm / tm:.1f}"


def pair_counts(cells, key):
    """Seed-matched dense-vs-TopK pairs across the main-text comparisons: (smaller, tied, larger)."""
    out = [0, 0, 0]
    for model, dense, topk, _, fams in MAIN:
        for fam in fams:
            for d, t in zip(group(cells, model, dense, fam), group(cells, model, topk, fam)):
                out[0 if t[key] < d[key] else 1 if t[key] == d[key] else 2] += 1
    return out


# ---- Fig. 1 -------------------------------------------------------------------------------------

def fig1(cells, out):
    key = "circuit_size"
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.6), sharey=True)
    jitter = np.linspace(-0.05, 0.05, len(SEEDS))
    for ax, (model, dense, topk, label, fams) in zip(axes, MAIN):
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        meds = {}
        for arm, colour, marker, face in ((dense, DENSE, "s", "white"),
                                          (topk, TOPK, "o", TOPK)):
            meds[arm] = []
            for i, fam in enumerate(fams):
                g = group(cells, model, arm, fam)
                # A size on the smallest rung is only an upper bound, and this figure has no mark for
                # it: refuse rather than draw it as a point value.
                if any(at_floor(r, key) for r in g):
                    raise ValueError(f"{model} {arm} {fam}: a verified size sits on the grid floor; "
                                     f"Fig. 1 would show an upper bound as a point value")
                ax.scatter(i + jitter, [pct(r, key) for r in g], s=11, marker=marker, facecolor=face,
                           edgecolor=colour if face == "white" else "white",
                           lw=0.6 if face == "white" else 0.4, zorder=3)
                s = summary(g, key)
                meds[arm].append(s)
                ax.plot([i - 0.11, i + 0.11], [s[0], s[0]], color=colour, lw=1.6, zorder=4,
                        solid_capstyle="butt")
            ax.plot(list(range(len(fams))), [m[0] for m in meds[arm]], color=colour, lw=0.8,
                    ls=(0, (3, 1.5)) if arm == dense else "-", zorder=2)
        for i in range(len(fams)):
            ax.text(i, 0.97, "\u00d7" + ratio_text(meds[dense][i], meds[topk][i], latex=False),
                    transform=trans, ha="center", va="top", fontsize=7, color=MUTED)
        ax.set_title(label, pad=3)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([FAMILY[f] for f in fams])
        ax.set_xlim(-0.55, len(fams) - 0.45)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Verified circuit\n(% of adapter)")
    key_row(fig, [(dict(color=TOPK, lw=0.8, marker="o", mfc=TOPK, mec="white", ms=4.5), TOPKLORA),
                  (dict(color=DENSE, lw=0.8, ls=(0, (3, 1.5)), marker="s", mfc="white", mec=DENSE, ms=4),
                   "Dense LoRA")])
    margins(fig, left=0.72, right=0.05, top=0.33, bottom=0.17, wspace=0.08)
    pdf = out / "figures" / "fig1_circuit_size.pdf"
    fig.savefig(pdf)
    plt.close(fig)

    smaller, tied, _ = pair_counts(cells, key)
    n = sum(pair_counts(cells, key))
    caption = (
        r"\textbf{\topklora\ backdoor circuits are smaller than dense-LoRA circuits in every family, and the "
        r"dense/TopK ratio grows with depth.} Size of the verified circuit (the smallest prefix of the latent ranking that "
        r"is both sufficient and necessary for the backdoor) as a percentage of the adapter's latents, for "
        r"rank-matched adapters (" + ARM_TEXT["gemma-2-2b"] + " on Gemma-2-2B; " + ARM_TEXT["qwen2.5-1.5b"]
        + r" on Qwen2.5-1.5B). Points are seeds, bars medians, numbers the dense/TopK ratio of medians. TopK "
        r"is smaller in " + f"{smaller}/{n}" + (f" ({tied} tied)" if tied else "") + r" seed-matched pairs.")
    (out / "figures" / "fig1_circuit_size.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/fig1_circuit_size.pdf}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:circuit-size}}\n\\end{{figure}}\n")
    return pdf


# ---- Fig. 2 -------------------------------------------------------------------------------------

def fig2(cells, results, out):
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 2.5), sharey=True)
    for row, (model, dense, topk, label, fams) in enumerate(MAIN):
        for col, fam in enumerate(fams):
            ax = axes[row, col]
            grids, sizes = {}, {}
            for arm, colour in ((dense, DENSE), (topk, TOPK)):
                g = group(cells, model, arm, fam)
                curves = [curve_of(results, r) for r in g]
                ks = [row_["K"] for row_ in curves[0]]
                if any([row_["K"] for row_ in c] != ks for c in curves):
                    raise ValueError(f"{model} {arm} {fam}: seeds were swept on different K grids, "
                                     f"so a per-K median would mix rungs")
                grids[arm], sizes[arm] = ks, g[0]["n_all_latents"]
                x = np.array(ks)
                for field, ls in (("keep_only", "-"), ("ablate", (0, (3, 1.5)))):
                    y = np.array([[row_[field] for row_ in c] for c in curves])
                    ax.fill_between(x, y.min(0), y.max(0), color=colour, alpha=0.16, lw=0, zorder=1)
                    ax.plot(x, np.median(y, 0), color=colour, lw=1.1, ls=ls, zorder=3)
            if grids[dense] != grids[topk]:
                raise ValueError(f"{model} {fam}: dense and TopK were swept on different K grids")
            # One dotted line marks the full adapter for both arms, which is only true when the pair is
            # rank-matched down to the latent count.
            if sizes[dense] != sizes[topk]:
                raise ValueError(f"{model} {fam}: dense has {sizes[dense]} latents, TopK {sizes[topk]}")
            n_all = sizes[topk]
            ax.axvline(n_all, color=MUTED, lw=0.6, ls=(0, (1, 1.2)), zorder=1)
            ax.set_xscale("log")
            ax.set_xlim(grids[topk][0] / 1.3, n_all * 1.15)
            # Powers of ten in range, plus the full adapter as the last tick; a power of ten within
            # 0.2 decades of it is dropped so the two labels cannot collide (10,000 vs 11,648).
            decades = [10 ** e for e in range(1, 6) if grids[topk][0] <= 10 ** e < n_all / 10 ** 0.2]
            ax.set_xticks(decades + [n_all])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
            ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            if row == 0:
                ax.set_title(FAMILY[fam], pad=3)
            if col == 0:
                ax.set_ylabel(f"{label}\nbackdoor ASR")
            if row == 1:
                ax.set_xlabel("circuit size $K$ (latents)", labelpad=1)
    key_row(fig, [(dict(color=TOPK, lw=1.1), TOPKLORA), (dict(color=DENSE, lw=1.1), "Dense LoRA"),
                  (dict(color=INK, lw=1.1), "keep only the circuit"),
                  (dict(color=INK, lw=1.1, ls=(0, (3, 1.5))), "ablate the circuit")])
    margins(fig, left=0.66, right=0.19, top=0.33, bottom=0.3, hspace=0.36, wspace=0.22)
    pdf = out / "figures" / "fig2_ksweep.pdf"
    fig.savefig(pdf)
    plt.close(fig)

    caption = (
        r"\textbf{The sweeps behind Figure~\ref{fig:circuit-size}: ablating a small TopK prefix already stops the "
        r"backdoor in every family.} Backdoor attack success rate (ASR, $n{=}1000$ triggered prompts) when "
        r"only the top-$K$ latents are kept (solid; sufficiency) and when they are ablated (dashed; necessity), "
        r"against $K$, the number of top-ranked latents. The dotted line and last tick mark the adapter's full "
        r"size; dense and TopK adapters of a family are rank-matched, so they have the same number of latents. Lines are medians over 5 seeds, bands the min--max range. "
        r"Both arms of a family are swept on the same grid, starting at its smallest rung: a TopK ablation curve "
        r"that is already at 0 there reaches 0 at some smaller, unmeasured $K$. On single-layer adapters TopK "
        r"stops the backdoor long before the kept prefix can run it. Sufficiency is the paired test used for verification "
        r"(keep-only within two paired standard errors of the intact model), so a solid line near 1 can still "
        r"fail it.")
    (out / "figures" / "fig2_ksweep.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/fig2_ksweep.pdf}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:ksweep}}\n\\end{{figure}}\n")
    return pdf


# ---- Tab. 1 -------------------------------------------------------------------------------------

def cell(med, lo, hi, digits):
    return f"{med:.{digits}f} {{\\scriptsize({lo:.{digits}f}--{hi:.{digits}f})}}"


def field_summary(rows, field):
    """(median, min, max) of a per-seed measurement. rows_for leaves a field out until its step has
    run, so an absent value is a missing measurement and raises -- it is never read as a number."""
    v = [r.get(field) for r in rows]
    if any(x is None for x in v):
        raise ValueError(f"{rows[0]['model']} {rows[0]['arm']} {rows[0]['family']}: {field} "
                         f"missing for seeds {[r['seed'] for r in rows if r.get(field) is None]}")
    return st.median(v), min(v), max(v)


def tab1(cells, out):
    key = "circuit_size"
    lines = []
    for model, dense, topk, label, fams in MAIN:
        lines.append(f"\\midrule\n\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{label}}}\\quad "
                     f"{{\\footnotesize {ARM_TEXT[model]}}}}} \\\\")
        for fam in fams:
            d, t = group(cells, model, dense, fam), group(cells, model, topk, fam)
            for g in (d, t):
                if any(at_floor(r, key) for r in g):   # same refusal as Fig. 1: no mark for a bound here
                    raise ValueError(f"{model} {g[0]['arm']} {fam}: a verified size sits on the grid floor")
            cols = [FAMILY[fam], cell(*summary(d, key)[:3], 1), cell(*summary(t, key)[:3], 1),
                    cell(*field_summary(d, "random_ablation_asr"), 2), cell(*field_summary(t, "random_ablation_asr"), 2),
                    cell(*field_summary(d, "retention"), 2), cell(*field_summary(t, "retention"), 2)]
            lines.append(" & ".join(cols) + " \\\\")

    smaller, tied, _ = pair_counts(cells, key)
    n = sum(pair_counts(cells, key))
    caption = (
        r"\textbf{Verified circuit size, specificity and surgicality.} Median (min--max) over 5 seeds. "
        r"\emph{Size}: the verified circuit as a percentage of the adapter's latents. \emph{Random ablation}: "
        r"backdoor attack success rate after ablating a random latent set of the same size instead of the "
        r"circuit; near 1 means the circuit is specific, near 0 that any set that large removes the backdoor. "
        r"\emph{Retention}: instruction-following quality after ablating the circuit, relative to the intact "
        r"model and the base model without the adapter, $(\text{ablated}-\text{base})/(\text{intact}-\text{base})$, "
        r"judged on 500 held-out instructions; 1 means ablation cost nothing, 0 that the model fell back to the "
        r"base model. TopK is smaller in "
        + f"{smaller}/{n}" + (f" ({tied} tied)" if tied else "") + r" seed-matched pairs. TopK arms search the "
        r"2{,}500 latents with the largest attribution, dense arms the whole adapter.")
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_figures` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:circuit-size}",
        "\\vspace{3pt}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Verified circuit size (\\%)} & \\multicolumn{2}{c}{Random ablation ASR} "
        "& \\multicolumn{2}{c}{Retention} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
        "Family & Dense & TopK & Dense & TopK & Dense & TopK \\\\",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / "tab1_circuit_size.tex"
    path.write_text(tex)
    return path


# ---- Section 6.3: planted circuits ---------------------------------------------------------------

ROUTED_LABEL = {"routed_d8": "$d{=}8$", "routed_d4": "$d{=}4$", "routed_d2": "$d{=}2$", "routed_d1": "$d{=}1$",
                "unrouted": "unrouted"}
ROUTED_TICK = {"routed_d8": "d = 8", "routed_d4": "d = 4", "routed_d2": "d = 2", "routed_d1": "d = 1",
               "unrouted": "unrouted"}


def load_routed(results):
    """Per gradient-routed arm, its seeds that verified a circuit, each with the share of the verified
    circuit that lies in the planted set [0:d). For the unrouted twin, [0:8) is not a planted set;
    scoring it against that slice is the null."""
    out = {arm: [] for arm in ROUTED_ARMS}
    excluded = []          # (arm, seed, intact ASR, required floor): seeds CLCD gave no verdict on
    for r in rows_for(Path(results)):
        if r["set"] != "gradient-routed":
            continue
        d = ROUTED_ARMS[r["arm"]]
        path = Path(results) / r["tree"] / r["arm"] / "elim" / f"{r['family']}_seed{r['seed']}_circuit.json"
        c = json.load(open(path))
        if r["status"] != "ok":
            if r["status"] != "unsaturated":   # the only exclusion the caption explains
                raise ValueError(f"{path}: status {r['status']!r}")
            excluded.append((r["arm"], r["seed"], c["intact_asr"], c["sat_floor"]))
            continue
        kept = [tuple(x) for x in c["kept_latents"]]
        out[r["arm"]].append({**r, "d": d, "verified_share": 100 * sum(is_planted(l, d) for l in kept) / len(kept)})
    for arm, rows in out.items():
        if not rows:
            raise ValueError(f"no verified circuit for {arm}")
    return out, excluded


def fig3_planted(routed, out):
    arms = list(ROUTED_ARMS)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.4))
    jitter = {3: np.array([-0.05, 0.0, 0.05]), 2: np.array([-0.03, 0.03])}
    for i, arm in enumerate(arms):
        rows = routed[arm]
        chance = 100 * ROUTED_ARMS[arm] / 64
        ax.plot([i - 0.32, i + 0.32], [chance, chance], color=MUTED, lw=0.9, ls=(0, (2, 1.5)), zorder=1)
        y = [r["verified_share"] for r in rows]
        ax.scatter(i + jitter[len(rows)], y, s=13, marker="o", facecolor=TOPK, edgecolor="white", lw=0.4, zorder=3)
        ax.plot([i - 0.11, i + 0.11], [st.median(y)] * 2, color=TOPK, lw=1.4, zorder=4, solid_capstyle="butt")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ROUTED_TICK[a] for a in arms])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("verified circuit\nin the planted set")
    ax.set_xlabel("backdoor routed into the first $d$ latents of each module", labelpad=1)
    ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
    # In the panel's upper right, which is empty: every d<=2 and unrouted circuit is under 65% planted.
    key_row(fig, [(dict(color=TOPK, lw=0, marker="o", mfc=TOPK, mec="white", ms=4.5), "seed"),
                  (dict(color=TOPK, lw=1.4), "median"),
                  (dict(color=MUTED, lw=0.9, ls=(0, (2, 1.5))), "chance")], size=6.5,
            ax=ax, loc="upper right", at=(1.0, 1.0))
    margins(fig, left=0.72, right=0.05, top=0.06, bottom=0.3)
    pdf = out / "figures" / "fig3_planted.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        r"\textbf{CLCD recovers planted circuits.} Share of the verified circuit's latents that lie in the "
        r"planted set, for gradient-routed gemma-2-2b organisms whose backdoor is confined to the first $d$ "
        r"latents of each of the 63 adapted modules, and for unrouted twins scored against the same slice with "
        r"$d{=}8$. Points are seeds, bars medians, dashes the share expected by chance ($d/64$).")
    (out / "figures" / "fig3_planted.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/fig3_planted.pdf}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:planted}}\n\\end{{figure}}\n")
    return pdf


def tab2_planted(routed, excluded, checks, out):
    """`checks` is analysis.planted_circuits.summarise keyed by (arm, seed): the verification tests
    applied to the planted set and to each verified circuit's planted part."""
    def med_range(v, digits=0, pct=False):
        s = "\\%" if pct else ""
        m, lo, hi = st.median(v), min(v), max(v)
        if len(v) == 1 or lo == hi:
            return f"{m:.{digits}f}{s}"
        return f"{m:.{digits}f}{s} {{\\scriptsize({lo:.{digits}f}--{hi:.{digits}f})}}"

    def per_seed(fires, prompts, arm):
        # Held-out necessity is per organism: fires out of its own prompts, never summed over seeds.
        if len(set(prompts)) != 1:
            raise ValueError(f"{arm}: seeds were tested on different numbers of held-out prompts {set(prompts)}")
        return med_range(fires) + f"/{prompts[0]:,.0f}".replace(",", "{,}")

    def sufficient(row, n_se):
        return row["suff_shortfall"] <= n_se * row["suff_se"]

    lines = []
    for arm, d in ROUTED_ARMS.items():
        rows = routed[arm]
        chk = [checks[(arm, int(r["seed"]))] for r in rows]
        fires = per_seed([field_summary([r], "heldout_fires")[0] for r in rows],
                         [field_summary([r], "heldout_prompts")[0] for r in rows], arm)
        if arm == "unrouted":
            part_fires, set_suff = "--", "--"
        else:
            part = [next(x for x in c["leak"] if x["file"].endswith("_planted_part.json")) for c in chk]
            part_fires = per_seed([x["total_fires"] for x in part], [x["total_prompts"] for x in part], arm)
            set_suff = f"{sum(sufficient(c['planted'], c['n_se']) for c in chk)}/{len(chk)}"
        notes = ([] if arm == "unrouted" else [f"{63 * d}"]) + \
                ([f"$n{{=}}{len(rows)}$"] if len(rows) != len(SEEDS[:3]) else [])
        lines.append(" & ".join([
            ROUTED_LABEL[arm] + (f" ({'; '.join(notes)})" if notes else ""),
            med_range([r["circuit_size"] for r in rows]),
            med_range([r["verified_share"] for r in rows], pct=True), f"{100 * d / 64:.1f}\\%",
            fires, med_range([field_summary([r], "retention")[0] for r in rows], digits=2),
            part_fires, set_suff]) + " \\\\")
        if arm == "routed_d1":
            lines.append("\\midrule")
    # The null for the planted-set tests: in the unrouted twins, ablating the same slice must NOT
    # remove the backdoor -- it is what shows the necessity test can fail.
    null = [checks[("unrouted", int(r["seed"]))]["planted"]["ablate"] for r in routed["unrouted"]]
    # Seed counts and the reason a seed is missing both come from the data: a seed is excluded only
    # when its intact model fires the backdoor on fewer prompts than CLCD requires to issue a verdict.
    counts = {arm: len(rows) for arm, rows in routed.items()}
    n_max = max(counts.values())
    short = [f"{c} for {ROUTED_LABEL[a]}" for a, c in counts.items() if c != n_max]
    why = "; ".join(f"the third {ROUTED_LABEL[arm]} seed fires the backdoor on only {100 * asr:.0f}\\% of "
                    f"trigger prompts, below the {100 * floor:.0f}\\% CLCD requires before issuing a verdict"
                    for arm, _, asr, floor in excluded)
    seed_text = f"{n_max} seeds" + (f" ({', '.join(short)}: {why})" if short else "")
    caption = (
        r"\textbf{Recovery of planted circuits.} Gradient-routed gemma-2-2b organisms (9-layer band, "
        r"$r{=}64$, $k{=}8$) whose backdoor is confined to the first $d$ latents of each of the 63 adapted "
        r"modules (\emph{planted set}), and unrouted twins scored against the same slice with $d{=}8$. Median "
        r"(min--max) over " + seed_text + r". \emph{In planted set}: share of the verified circuit's latents that lie in the "
        r"planted set, against the share expected by chance, $d/64$. \emph{Fires}: backdoor responses after "
        r"ablating the verified circuit, per seed, out of its held-out prompts. \emph{Retention}: as in "
        r"Table~\ref{tab:circuit-size}. \emph{Planted-set tests} apply CLCD-verify's tests to planted latents: "
        r"\emph{part ablated} gives held-out fires after ablating only the verified circuit's latents that lie in "
        r"the planted set, and \emph{set sufficient} the number of seeds in which keeping only the planted set "
        r"reproduces the backdoor within two paired standard errors of the intact model. As a control, ablating "
        r"the same slice in the unrouted twins leaves the backdoor largely intact (ASR "
        + f"{min(null):.2f}--{max(null):.2f}" + r").")
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_figures` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:planted}",
        "\\vspace{3pt}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lccccccc@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{In planted set} & \\multicolumn{2}{c}{Circuit ablated} "
        "& \\multicolumn{2}{c}{Planted-set tests} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
        "Routing ($|$planted$|$) & Circuit size & Circuit & Chance & Fires & Retention & Part ablated "
        "& Set sufficient \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / "tab2_planted.tex"
    path.write_text(tex)
    return path


# ---- Section 6.4: removal without damage ---------------------------------------------------------

JUDGE = {"alpaca": "judge_api_gpt_5_6_luna", "no_robots": "judge_indep_api_gpt_5_6_luna"}
JUDGE_N = {"alpaca": 500, "no_robots": 446}
# Every model at its capacity-matched rank; Qwen2.5-7B has no dense arm.
RETENTION = MAIN + [("qwen2.5-7b", None, "r100_k12", "Qwen2.5-7B", ("l20", "l17_25", "all"))]


def surgical(results, r):
    return json.load(open(Path(results) / r["tree"] / r["arm"] / "surgical"
                          / f"{r['family']}_seed{r['seed']}_surgical.json"))


def paired(score_lists):
    """Arrays of the items scored in EVERY given list. An unparsed judge answer (None) drops that
    prompt from all of them together: a pairwise statistic must compare the same prompts."""
    keep = [i for i in range(len(score_lists[0])) if all(s[i] is not None for s in score_lists)]
    if len(keep) < 0.99 * len(score_lists[0]):
        raise ValueError(f"only {len(keep)} of {len(score_lists[0])} items scored in every condition")
    return [np.array([s[i] for i in keep], float) for s in score_lists]


def retention_ci(conditions, key, n_boot=2000, seed=0):
    """95% bootstrap interval of retention, resampling prompts jointly across intact, ablated and
    base (they are scored on the same prompts). Fixed seed: the interval is reproducible."""
    I, A, B = paired([conditions[c][key]["scores"] for c in ("intact", "ablate_circuit", "base")])
    idx = np.random.default_rng(seed).integers(0, len(I), (n_boot, len(I)))
    boot = (A[idx].mean(1) - B[idx].mean(1)) / (I[idx].mean(1) - B[idx].mean(1))
    return tuple(np.percentile(boot, [2.5, 97.5]))


def fig4_retention(cells, results, out):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.45), sharey=True)
    jitter = np.linspace(-0.08, 0.08, len(SEEDS))
    for ax, (model, dense, topk, label, fams) in zip(axes, RETENTION):
        ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 1.5)), zorder=1)
        ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
        for arm, colour, marker, face in ((dense, DENSE, "s", "white"), (topk, TOPK, "o", TOPK)):
            if arm is None:
                continue
            meds = []
            for i, fam in enumerate(fams):
                g = group(cells, model, arm, fam)
                y = [field_summary([r], "retention")[0] for r in g]
                cis = [retention_ci(surgical(results, r)["conditions"], JUDGE["alpaca"]) for r in g]
                for x, (lo, hi) in zip(i + jitter, cis):
                    ax.plot([x, x], [lo, hi], color=colour, lw=0.6, alpha=0.55, zorder=2, solid_capstyle="butt")
                ax.scatter(i + jitter, y, s=11, marker=marker, facecolor=face,
                           edgecolor=colour if face == "white" else "white",
                           lw=0.6 if face == "white" else 0.4, zorder=3)
                meds.append(st.median(y))
                ax.plot([i - 0.14, i + 0.14], [meds[-1]] * 2, color=colour, lw=1.6, zorder=4, solid_capstyle="butt")
            ax.plot(range(len(fams)), meds, color=colour, lw=0.8, ls=(0, (3, 1.5)) if arm == dense else "-",
                    zorder=2)
        ax.set_title(label, pad=3)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([FAMILY[f] for f in fams], fontsize=6.5)
        ax.set_xlim(-0.5, len(fams) - 0.5)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(-0.15, 1.6)
        ax.set_yticks([0, 0.5, 1, 1.5])
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
    axes[0].set_ylabel("retention")
    # In the Qwen2.5-7B panel, whose lower half is empty: no dense arm, and every interval is above 0.65.
    key_row(fig, [(dict(color=TOPK, lw=0.8, marker="o", mfc=TOPK, mec="white", ms=4.5), TOPKLORA),
                  (dict(color=DENSE, lw=0.8, ls=(0, (3, 1.5)), marker="s", mfc="white", mec=DENSE, ms=4),
                   "Dense LoRA")], ax=axes[2], loc="lower center", at=(0.5, 0.14), stack=True)
    margins(fig, left=0.5, right=0.05, top=0.2, bottom=0.17, wspace=0.08)
    pdf = out / "figures" / "fig4_retention.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        r"\textbf{Removing the verified circuit preserves instruction-following in multi-layer "
        r"\topklora\ adapters, across all three models.} Retention, "
        r"$(s_\text{ablate}-s_\text{base})/(s_\text{intact}-s_\text{base})$, of the mean judge score on "
        + f"{JUDGE_N['alpaca']}" + r" held-out instructions: 1 (dashed) means ablating the circuit cost nothing, "
        r"0 (solid) that the model fell back to its base model. Each model at its capacity-matched rank; "
        r"Qwen2.5-7B has no dense baseline. Points are seeds with 95\% bootstrap intervals over prompts, bars "
        r"medians. Intervals are wide for Qwen2.5-7B because its adapter improves the base model's score "
        r"much less, which makes retention a ratio of small differences.")
    (out / "figures" / "fig4_retention.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/fig4_retention.pdf}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:retention}}\n\\end{{figure}}\n")
    return pdf


def tab3_clean_twin(results, out):
    """gemma-2-2b TopKLoRA organisms against their no-poison twins (same recipe and seed, trained without
    the poisoned rows), on identical prompts. Scores are means over seeds; ablated - clean is a
    per-prompt paired difference per seed pair, pooled over seeds with a 95% interval."""
    def signed(x):
        # math mode: in text a leading "-" is a hyphen, not a minus sign; and never print "-0.00"
        return "$0.00$" if abs(x) < 0.005 else f"${x:+.2f}$"

    rows_by = {}
    for r in rows_for(Path(results)):
        if r["model"] == "gemma-2-2b" and r["arm"] in ("r64_k8", "r64_k8_clean"):
            rows_by[(r["arm"], r["family"], int(r["seed"]))] = r
    lines = []
    for fam in ("l19", "l1523", "all"):
        for pset, key in JUDGE.items():
            means = {"base": [], "clean": [], "intact": [], "ablate": []}
            diffs, ses = [], []
            for seed in SEEDS:
                pois, clean = rows_by[("r64_k8", fam, seed)], rows_by[("r64_k8_clean", fam, seed)]
                ps, cs = surgical(results, pois), surgical(results, clean)
                qk = "clean_questions" if pset == "alpaca" else "indep_questions"
                if ps[qk] != cs[qk]:
                    raise ValueError(f"{fam} s{seed} {pset}: the twins were scored on different prompts")
                pc, cc = ps["conditions"], cs["conditions"]
                base, intact, ablate, clean_s = paired([pc["base"][key]["scores"], pc["intact"][key]["scores"],
                                                        pc["ablate_circuit"][key]["scores"],
                                                        cc["intact"][key]["scores"]])
                for name, v in (("base", base), ("clean", clean_s), ("intact", intact), ("ablate", ablate)):
                    means[name].append(v.mean())
                d = ablate - clean_s
                diffs.append(d.mean())
                ses.append(d.std(ddof=1) / np.sqrt(len(d)))
            diff = st.mean(diffs)
            half = 1.96 * np.sqrt(sum(s * s for s in ses)) / len(ses)
            lines.append(" & ".join([FAMILY[fam] if pset == "alpaca" else "", pset.replace("_", "-"),
                                     *[f"{st.mean(means[k]):.2f}" for k in ("base", "clean", "intact", "ablate")],
                                     f"{signed(diff)} {{\\scriptsize[{signed(diff - half)}, {signed(diff + half)}]}}"])
                         + " \\\\")
        if fam != "all":
            lines.append("\\addlinespace")
    caption = (
        r"\textbf{Ablating the circuit brings multi-layer organisms close to a model that was never "
        r"poisoned.} Mean judge scores (1--5) of gemma-2-2b \topklora\ organisms and their \emph{clean twins}, "
        r"trained with the same recipe and seeds but without poisoned examples, averaged over 5 seeds, on "
        + f"{JUDGE_N['alpaca']}" + r" alpaca and " + f"{JUDGE_N['no_robots']}" + r" no-robots instructions. "
        r"\emph{Ablated $-$ clean}: per-prompt difference between the poisoned organism with its verified "
        r"circuit ablated and its clean twin, with a 95\% confidence interval.")
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_figures` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:clean-twin}",
        "\\vspace{3pt}",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        " & & & & \\multicolumn{2}{c}{Poisoned} & \\\\",
        "\\cmidrule(lr){5-6}",
        "Family & Prompts & Base & Clean twin & Intact & Ablated & Ablated $-$ clean \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / "tab3_clean_twin.tex"
    path.write_text(tex)
    return path


# ---- Appendix: r/k capacity sweep (older protocol, reported as measured) -----------------------

# gemma-2-2b TopKLoRA, July 2026: attribution-prefix search, Qwen2.5-32B-Instruct judge, generation
# that did not stop at end-of-turn. Not re-judged: the saved answers run on past the turn boundary
# (68% reach the 256-token cap, against 25% for the same organism type in campaign 3), so a luna
# score of them would not be the campaign-3 measurement either. Every cell of the sweep shares that
# pipeline, which is what makes its within-sweep comparisons fair.
RK_MODULES = {"l19": 7, "l1523": 63, "all": 182}
RK_PLACEMENT = {"l19": ("Single layer", "#e6a47a", "o"), "l1523": ("9-layer band", "#d17133", "^"),
                "all": ("All layers", "#8f4a1c", "D")}      # ordinal ramp of the TopKLoRA orange
RK_FILE = re.compile(r"(l19|l1523|all)_r(\d+)_k(\d+)_seed(\d+)_(circuit|surgical)\.json$")
RK_JUDGE = {"alpaca": "judge_32b", "no_robots": "judge_indep_32b"}


def load_rk_sweep(sweep, anchor):
    """{(family, r, k): {"seeds": n searched, "pool": latents, "found": [per-seed dict]}}. Files without
    a seed suffix are older duplicates the sweep's own aggregator ignored, and are ignored here. The
    r=64, k=8 point comes from the earlier rigorous run (seeds 42-44), as in the sweep's aggregator.
    Retention uses that run's base-model scores under the same 32B judge; a per-seed field is absent
    when it was not measured (no surgical file), never zero."""
    base_file = json.load(open(anchor / "base_floor_surgical.json"))
    base = base_file["conditions"]["base"]
    floor = {p: base[key]["mean"] for p, key in RK_JUDGE.items()}
    raw = defaultdict(lambda: {"circ": {}, "surg": {}})
    for f in sorted(Path(sweep).iterdir()):
        m = RK_FILE.search(f.name)
        if m:
            fam, r, k, seed, kind = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
            raw[(fam, r, k)]["circ" if kind == "circuit" else "surg"][seed] = json.load(open(f))
    for fam in RK_MODULES:
        if (fam, 64, 8) in raw:
            raise ValueError(f"{fam} r64 k8 is in the sweep AND taken from the anchor run")
        for seed in (42, 43, 44):
            raw[(fam, 64, 8)]["circ"][seed] = json.load(open(anchor / f"{fam}_seed{seed}_circuit.json"))
            raw[(fam, 64, 8)]["surg"][seed] = json.load(open(anchor / f"{fam}_seed{seed}_surgical.json"))
    out = {}
    for (fam, r, k), e in raw.items():
        pool, found = RK_MODULES[fam] * r, []
        for seed, c in sorted(e["circ"].items()):
            if c["status"] != "ok":
                continue
            row = {"seed": seed, "size": 100 * c["both_K"] / pool}
            if seed in e["surg"]:
                s = e["surg"][seed]
                row["random"] = s["random_ablation_asr"]
                for p, key in RK_JUDGE.items():
                    cond = s["conditions"]
                    if key in cond["intact"] and key in cond["ablate_circuit"]:
                        i, a = cond["intact"][key]["mean"], cond["ablate_circuit"][key]["mean"]
                        row[p] = (a - floor[p]) / (i - floor[p])
                # The interval resamples prompts jointly across intact, ablated and the base model, which
                # is only valid when the base file scored the same prompts as this organism's file.
                if s["clean_questions"] != base_file["clean_questions"]:
                    raise ValueError(f"{fam} r{r} k{k} s{seed}: base scores are for different prompts")
                row["alpaca_ci"] = retention_ci({"intact": cond["intact"], "ablate_circuit": cond["ablate_circuit"],
                                                 "base": base}, RK_JUDGE["alpaca"])
            found.append(row)
        out[(fam, r, k)] = {"seeds": len(e["circ"]), "pool": pool, "found": found}
    return out


def fig_rk_sweep(rk, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.6), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})
    NONE_Y = -0.16                          # where a configuration with no circuit in any seed is marked
    # Offsets in log2 units: placements share x in the k panel, and seeds share x everywhere, so each
    # interval gets its own column instead of hiding behind another.
    place_dx = dict(zip(RK_PLACEMENT, (-0.14, 0.0, 0.14)))
    seed_dx = {42: -0.045, 43: 0.0, 44: 0.045}
    for ax, axis in zip(axes, ("r", "k")):
        ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 1.5)), zorder=1)
        ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
        for fam, (label, colour, marker) in RK_PLACEMENT.items():
            keys = sorted((key for key in rk if key[0] == fam and (key[2] == 8 if axis == "r" else key[1] == 64)),
                          key=lambda key: key[1] if axis == "r" else key[2])
            xs, meds = [], []
            for key in keys:
                x = (rk[key]["pool"] if axis == "r" else key[2]) * 2 ** place_dx[fam]
                rows = [row for row in rk[key]["found"] if "alpaca" in row]
                if not rk[key]["found"]:
                    ax.scatter([x], [NONE_Y], s=16, marker=marker, facecolor="white", edgecolor=colour, lw=0.8,
                               zorder=3)
                    continue
                for row in rows:
                    xj = x * 2 ** seed_dx[row["seed"]]
                    lo, hi = row["alpaca_ci"]
                    ax.plot([xj, xj], [lo, hi], color=colour, lw=0.6, alpha=0.55, zorder=2, solid_capstyle="butt")
                    ax.scatter([xj], [row["alpaca"]], s=13, marker=marker, facecolor=colour, edgecolor="white",
                               lw=0.4, zorder=3)
                if rows:
                    xs.append(x)
                    meds.append(st.median(row["alpaca"] for row in rows))
            ax.plot(xs, meds, color=colour, lw=1.0, zorder=2)
        ax.set_xscale("log", base=2)
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
        ax.set_ylim(-0.26, 1.22)
        ax.set_yticks([0, 0.5, 1])
    axes[0].set_xticks([64, 256, 1024, 4096, 16384])
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].set_xlabel("latent pool ($k{=}8$)", labelpad=1)
    axes[1].set_xticks([2, 4, 8, 16, 32, 64])
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    axes[1].set_xlabel("$k$ ($r{=}64$)", labelpad=1)
    for ax in axes:
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel("retention")
    key_row(fig, [(dict(color=c, lw=1.0, marker=m, mfc=c, mec="white", ms=4.5), label)
                  for label, c, m in RK_PLACEMENT.values()]
            + [(dict(color=MUTED, lw=0, marker="o", mfc="white", mec=MUTED, ms=4.5), "no circuit in any seed")], size=6.5)
    margins(fig, left=0.5, right=0.05, top=0.2, bottom=0.3, wspace=0.08)
    pdf = out / "figures" / "fig_rk_sweep.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    caption = (
        r"\textbf{More capacity makes multi-layer adapters surgical, but not single-layer ones.} Retention after ablating "
        r"the verified circuit of gemma-2-2b \topklora\ organisms as the rank $r$ (left, $k{=}8$, $\alpha{=}2r$) "
        r"and the number of active latents $k$ (right, $r{=}64$) vary; the latent pool is the number of adapted "
        r"modules times $r$. Points are seeds with a verified circuit, with 95\% bootstrap intervals over prompts "
        r"(Tables~\ref{tab:rk-sweep-r} and~\ref{tab:rk-sweep-k} give how many of the three seeds yielded one); "
        r"lines are medians; hollow markers at the bottom "
        r"are configurations where no seed yielded one. This sweep predates the main experiments and uses their "
        r"verification tests with an earlier pipeline: circuits ranked by attribution alone (prefix search), "
        r"answers generated without stopping at the end of turn, and a Qwen2.5-32B-Instruct judge instead of "
        r"gpt-5.6-luna. Its values are therefore comparable within the sweep but not to Figure~\ref{fig:retention}.")
    (out / "figures" / "fig_rk_sweep.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/fig_rk_sweep.pdf}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:rk-sweep}}\n\\end{{figure}}\n")
    return pdf


def tab_rk_sweep(rk, axis, out):
    """The sweep in full, one table per swept axis: `axis` "r" is the rank sweep at k=8 (the latent pool
    varies with r), "k" the k sweep at r=64 (the pool is fixed per placement, so it goes in the group
    header). The r=64, k=8 point anchors both sweeps and appears in both tables."""
    def mr(v, digits=2):
        if not v:
            return "--"
        m, lo, hi = st.median(v), min(v), max(v)
        return f"{m:.{digits}f}" if len(v) == 1 or lo == hi else \
            f"{m:.{digits}f} {{\\scriptsize({lo:.{digits}f}--{hi:.{digits}f})}}"

    ncol = 7 if axis == "r" else 6
    lines = []
    for fam, (label, _, _) in RK_PLACEMENT.items():
        keys = sorted((key for key in rk if key[0] == fam and (key[2] == 8 if axis == "r" else key[1] == 64)),
                      key=lambda key: key[1] if axis == "r" else key[2])
        pool = "" if axis == "r" else ", pool " + f"{RK_MODULES[fam] * 64:,}".replace(",", "{,}")
        lines.append(f"\\midrule\n\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{label}}} "
                     f"{{\\footnotesize ({RK_MODULES[fam]} modules{pool})}}}} \\\\")
        for key in keys:
            e, found = rk[key], rk[key]["found"]
            head = [f"{key[1]}", f"{e['pool']:,}".replace(",", "{,}")] if axis == "r" else [f"{key[2]}"]
            lines.append(" & ".join(head + [
                f"{len(found)}/{e['seeds']}", mr([row["size"] for row in found], 1),
                mr([row["random"] for row in found if "random" in row]),
                mr([row["alpaca"] for row in found if "alpaca" in row]),
                mr([row["no_robots"] for row in found if "no_robots" in row])]) + " \\\\")
    swept = (r"the rank $r$ at $k{=}8$ ($\alpha{=}2r$); the latent pool is the number of adapted modules "
             r"times $r$" if axis == "r" else r"the number of active latents $k$ at $r{=}64$")
    caption = (
        (r"\textbf{Capacity sweep: rank.} " if axis == "r" else r"\textbf{Capacity sweep: active latents.} ")
        + r"gemma-2-2b \topklora\ organisms, 3 seeds per configuration, sweeping " + swept + r". The "
        r"$r{=}64$, $k{=}8$ row comes from an earlier run with the same tests, whose search ordering is not "
        r"recorded. \emph{Found}: seeds with a verified circuit. \emph{Size}: circuit as a percentage of the "
        r"latent pool. \emph{Random ablation} and \emph{retention} as in Table~\ref{tab:circuit-size}, median "
        r"(min--max) over seeds with a circuit, judged by Qwen2.5-32B-Instruct on answers generated without "
        r"stopping at the end of turn (see Figure~\ref{fig:rk-sweep}). \emph{--}: not measured.")
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_figures` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:rk-sweep-{axis}}}",
        "\\vspace{3pt}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}" + ("rr" if axis == "r" else "r") + "ccccc@{}}",
        "\\toprule",
        " & " * (ncol - 2) + "\\multicolumn{2}{c}{Retention} \\\\",
        f"\\cmidrule(l){{{ncol - 1}-{ncol}}}",
        ("$r$ & Pool & " if axis == "r" else "$k$ & ")
        + "Found & Size (\\%) & Random ablation & alpaca & no-robots \\\\",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / f"tab_rk_sweep_{axis}.tex"
    path.write_text(tex)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="clcd_results", type=Path)
    ap.add_argument("--out", default="paper", type=Path)
    ap.add_argument("--rk-sweep", default="clcd_results_old/sweep_rk", type=Path,
                    help="the July 2026 r/k capacity sweep (appendix)")
    ap.add_argument("--rk-anchor", default="clcd_results_old/rigorous_gemma_pre_campaign", type=Path,
                    help="the earlier run that supplies the sweep's r=64, k=8 point and base-model scores")
    a = ap.parse_args()
    for d in ("figures", "tables"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    style()
    cells = load(a.results)
    if not cells:
        raise SystemExit(f"no poisoned cells under {a.results}")
    routed, excluded = load_routed(a.results)
    checks = {(r["arm"], r["seed"]): r for r in summarise_planted(a.results)}
    rk = load_rk_sweep(a.rk_sweep, a.rk_anchor)
    for p in (fig1(cells, a.out), fig2(cells, a.results, a.out), tab1(cells, a.out),
              fig3_planted(routed, a.out), tab2_planted(routed, excluded, checks, a.out),
              fig4_retention(cells, a.results, a.out), tab3_clean_twin(a.results, a.out),
              fig_rk_sweep(rk, a.out), tab_rk_sweep(rk, "r", a.out), tab_rk_sweep(rk, "k", a.out)):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
