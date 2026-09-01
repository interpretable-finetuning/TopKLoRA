#!/usr/bin/env python3
"""Render the four supervisor-briefing figures from the result JSONs.

`docs/supervisor_briefing.md` specifies Figures 1-4 in prose ("Figures to build: 4 total")
and nothing ever built them -- the five PNGs in clcd_results/figures/ are from an older
numbering scheme (Jul 2-7) and predate the l19 prefix->scrub correction, so they contradict
the current briefing. This script is the missing generator, checked in so the figures cannot
drift from the data again.

Every number is READ FROM THE RESULT JSONS, not transcribed from the briefing tables. The one
exception is the `all` family's r=128/r=256 cells in Figure 3, which are absent by design (the
sweep deliberately scoped `all` to the low-capacity corner) -- absence is not derivable from
files, so it is declared in NOT_RUN and drawn as an EMPTY SLOT, never a zero bar: a zero bar
would assert we tried and found nothing, which is false.

  uv run --with matplotlib python scripts/make_briefing_figures.py [--outdir clcd_results/figures]

Figures:
  1. necessity tail          -- all-seed45 K-sweep: sufficiency saturates, necessity has a tail
  2. surgicality vs distribution -- capability retained per family, the counterintuitive headline
  3. found-rate vs capacity  -- r/k sweep: separability rises with adapter capacity
  4. price of complete removal   -- l1523-s44: held-out fires vs capability, K=150/400/700
"""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.clcd.aggregate_rk_sweep import collect

RIG = Path("clcd_results/rigorous")

# --- palette: dataviz slots 1-3 (blue/orange/aqua), validated all-pairs light mode --------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd8"

# Figure 2 reports the BEST METHOD PER FAMILY, which for l19 is scrubbing -- the prefix number
# (38%) is the retracted headline; see briefing Slide 14. Families in deck order.
FAMILIES = [
    ("l19\n1 layer, 448 latents", "elim2/l19_seed4?_nc1000_surgical.json"),
    ("l15-23\n9 layers, 4,032 latents", "l1523_seed4?_surgical.json"),
    ("all\n26 layers, 11,648 latents", "all_seed4?_surgical.json"),
]
# Figure 4: same organism, same backdoor, three circuits of increasing size.
PRICE = [
    ("scrubbing", "elim2/l1523_seed44_nc1000_adaptive"),
    ("prefix", "l1523_seed44"),
    ("scrub + held-out nec.", "elim2/l1523_seed44_K700nec"),
]
NOT_RUN = {("all", 128), ("all", 256)}  # deliberately never scheduled, not pending
R_AXIS = [8, 16, 32, 64, 128, 256]
FAM_LABEL = {"l19": "l19 (1 layer)", "l1523": "l15-23 (9 layers)", "all": "all (26 layers)"}


def style(ax, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out / name}.png")


def base_floor():
    """32B-judge base-model floor; every retention number is normalised against it."""
    bc = json.load(open(RIG / "base_floor_surgical.json"))["conditions"]["base"]
    return bc["judge_32b"]["mean"], bc["judge_indep_32b"]["mean"]


def retention(path, floors):
    """(alpaca, no-robots) capability retained = (ablate - base) / (intact - base)."""
    c = json.load(open(path))["conditions"]
    return tuple(
        (c["ablate_circuit"][k]["mean"] - b) / (c["intact"][k]["mean"] - b)
        for k, b in zip(("judge_32b", "judge_indep_32b"), floors)
    )


def fig1_necessity_tail(out):
    """Sufficiency is done at K=200; necessity takes until K=1200. One shared y-axis."""
    curve = json.load(open(RIG / "all_seed45_circuit.json"))["curve"]
    x = list(range(len(curve)))
    ks = [c["K"] for c in curve]
    keep = [c["keep_only"] * 100 for c in curve]
    abl = [c["ablate"] * 100 for c in curve]

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.plot(x, keep, "-o", color=BLUE, lw=2, ms=7, label="keep-only ASR (sufficiency)")
    ax.plot(x, abl, "-s", color=ORANGE, lw=2, ms=7, label="ablate ASR (necessity)")
    for k, lbl, c in ((200, "suff_K = 200", BLUE), (1200, "both_K = 1200", ORANGE)):
        ax.axvline(ks.index(k), color=c, ls=":", lw=1.4, zorder=0)
        ax.text(ks.index(k), 104, lbl, color=c, fontsize=9, ha="center",
                bbox=dict(fc=SURFACE, ec="none", pad=1.5))
    ax.annotate(
        "exactly 0",
        xy=(ks.index(1200), 0), xytext=(ks.index(1200) - 0.9, 17),
        color=ORANGE, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4),
    )
    ax.set_xticks(x, ks)
    ax.set_xlabel("circuit size K (latents)", color=INK2, fontsize=10)
    ax.set_ylim(-4, 112)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    style(ax, "backdoor ASR", "The necessity tail — 6x more latents to finish the job than to start it")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="center right")
    fig.text(0.5, -0.04, "all-layers organism, seed 45.  Source: clcd_results/rigorous/all_seed45_circuit.json",
             color=INK2, fontsize=8, ha="center")
    save(fig, out, "briefing_fig1_necessity_tail")


def fig2_surgicality(out):
    """Removal gets cheaper as the backdoor gets more distributed. Zero-baselined, sd bars."""
    floors = base_floor()
    means, sds = [], []
    for _, pat in FAMILIES:
        vals = [retention(p, floors) for p in sorted(RIG.glob(pat))]
        means.append([st.mean(v) * 100 for v in zip(*vals)])
        sds.append([st.stdev(v) * 100 for v in zip(*vals)])

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    w = 0.34
    for i, (lbl, col) in enumerate((("Alpaca (in-distribution)", BLUE), ("No-Robots (out-of-dist.)", ORANGE))):
        xs = [j + (i - 0.5) * (w + 0.02) for j in range(len(FAMILIES))]
        ys = [m[i] for m in means]
        es = [s[i] for s in sds]
        ax.bar(xs, ys, w, color=col, label=lbl,
               yerr=es, error_kw=dict(ecolor=INK2, lw=1.2, capsize=4))
        for x, y, e in zip(xs, ys, es):  # clear of the error-bar cap, not on it
            ax.text(x, y + e + 3, f"{y:.0f}%", ha="center", color=INK, fontsize=10, fontweight="bold")
    # the reference line goes in the legend rather than inline — an inline label collides with `all`
    ax.axhline(100, color=INK2, ls="--", lw=1.2, zorder=0, label="100% = removal is free")
    ax.set_xticks(range(len(FAMILIES)), [f[0] for f in FAMILIES])
    ax.set_ylim(0, 150)
    ax.set_yticks(range(0, 151, 25), [f"{v}%" for v in range(0, 151, 25)])
    style(ax, "capability retained (32B judge)",
          "Surgicality rises with distribution — the localized organism is the hardest")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    fig.text(0.5, -0.02, "Backdoor ASR after ablation is 0.0% in every bar. l19 = scrubbing circuits, "
                         "others = prefix; 5 seeds (l15-23: 4 — seed45 has no prefix circuit).",
             color=INK2, fontsize=8, ha="center")
    save(fig, out, "briefing_fig2_surgicality_vs_distribution")


def fig3_found_rate(out):
    """Found-rate out of 3 seeds vs LoRA rank. r=64 is the canonical anchor, re-scored on seeds 42-44."""
    cfg = collect()
    rate = {}
    for fam in FAM_LABEL:
        for r in R_AXIS:
            if (fam, r) in NOT_RUN:
                continue
            if r == 64:  # anchor: the canonical organisms, same 3 seeds, from rigorous/
                circ = [json.load(open(RIG / f"{fam}_seed{s}_circuit.json")) for s in (42, 43, 44)]
            else:
                circ = list(cfg[(fam, r, 8)]["circ"].values())
            rate[(fam, r)] = sum(c.get("status") == "ok" for c in circ)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    w = 0.26
    for i, (fam, col) in enumerate(zip(FAM_LABEL, (BLUE, ORANGE, AQUA))):
        xs, ys = [], []
        for j, r in enumerate(R_AXIS):
            if (fam, r) in NOT_RUN:
                ax.text(j + (i - 1) * (w + 0.02), 0.06, "not\nrun", ha="center", va="bottom",
                        color=INK2, fontsize=7.5, style="italic")
                continue
            xs.append(j + (i - 1) * (w + 0.02))
            ys.append(rate[(fam, r)])
        ax.bar(xs, ys, w, color=col, label=FAM_LABEL[fam])
        for x, y in zip(xs, ys):  # direct labels: required relief for aqua on a light surface
            ax.text(x, y + 0.06, f"{y}/3", ha="center", color=INK, fontsize=9, fontweight="bold")
    ax.axvspan(2.5, 3.5, color=GRID, alpha=0.45, zorder=0)
    ax.text(3, 3.42, "canonical organism (anchor)", ha="center", color=INK2, fontsize=8.5)
    ax.set_xticks(range(len(R_AXIS)), [str(r) for r in R_AXIS])
    ax.set_xlabel("LoRA rank r  (at k=8)", color=INK2, fontsize=10)
    ax.set_ylim(0, 3.7)
    ax.set_yticks([0, 1, 2, 3])
    style(ax, "seeds with a both-criteria circuit",
          "A backdoor is separable exactly when it was given room to be separate")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    fig.text(0.5, -0.04, "3 seeds per cell — the monotone TREND is the claim, not any single cell "
                         "(l19 dips at r=128). `all` r=128/256 were deliberately never scheduled.",
             color=INK2, fontsize=8, ha="center")
    save(fig, out, "briefing_fig3_found_rate_vs_capacity")


def fig4_price(out):
    """Two panels, shared categorical x. NOT a dual axis: Panel B's zoom is labelled."""
    floors = base_floor()
    master = json.load(open(RIG / "holdout_necessity/MASTER_table.json"))
    fires, ks, rets = [], [], []
    for _, stem in PRICE:
        circ = f"clcd_results/rigorous/{stem}_circuit.json"
        row = next(r for r in master if r["file"] == circ)
        fires.append(row["total_fires"])
        ks.append(row["K"])
        rets.append(retention(RIG / f"{stem}_surgical.json", floors))

    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, facecolor=SURFACE,
                              gridspec_kw=dict(hspace=0.22))
    x = range(len(PRICE))
    a.bar(x, fires, 0.5, color=BLUE)
    for i, f in enumerate(fires):
        a.text(i, f + 0.12, "0  ✓" if f == 0 else str(f), ha="center", color=INK,
               fontsize=11, fontweight="bold")
    a.set_ylim(0, 5)
    a.set_yticks(range(5))
    style(a, "fires per 3,000 prompts", "The price of complete removal")

    series = [[r[i] * 100 for r in rets] for i in range(2)]
    for i, (lbl, col) in enumerate((("Alpaca", BLUE), ("No-Robots", ORANGE))):
        b.plot(x, series[i], "-o", color=col, lw=2, ms=8, label=lbl)
        # label the endpoints only — the lines cross in the middle column, where no offset is legible
        for xx in (0, len(PRICE) - 1):
            hi = series[i][xx] >= series[1 - i][xx]
            b.text(xx, series[i][xx] + (1.4 if hi else -2.6), f"{series[i][xx]:.0f}%",
                   ha="center", color=INK, fontsize=9.5)
    b.set_ylim(80, 106)
    b.set_yticks(range(80, 106, 5), [f"{v}%" for v in range(80, 106, 5)])
    style(b, "capability retained", None)
    b.set_title("axis zoomed to 80–105% (not zero-baselined)", color=INK2, fontsize=9,
                loc="left", pad=6)
    b.set_xticks(list(x), [f"{lbl}\nK = {k}" for (lbl, _), k in zip(PRICE, ks)])
    b.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower left")
    fig.text(0.5, -0.03, "l15-23, seed 44 — same organism, same backdoor. As the leak closes, capability falls.\n"
                         "Caveat: the three circuits come from three different search methods, not one sweep.",
             color=INK2, fontsize=8, ha="center")
    save(fig, out, "briefing_fig4_price_of_complete_removal")


# --- Figures 5-6: the 25 BIG-N-audited circuits, joined on MASTER_table.json -------------------
FAM_ORDER = {"l19": 0, "l1523": 1, "all": 2}


def _master():
    """The 25 certified circuits the BIG-N audit used. Drive off this list, never a glob: two
    on-disk `*_circuit.json` files are `no_sufficient_subcircuit` with empty `kept_latents`."""
    rows = json.load(open(RIG / "holdout_necessity/MASTER_table.json"))
    return sorted(rows, key=lambda r: (FAM_ORDER[r["family"]], r["seed"], r["method"]))


def _chi2_uniform(counts, K):
    exp = K / 7
    return sum((c - exp) ** 2 / exp for c in counts)


def fig5_module_composition(out, n_null=1000, seed=0):
    """A1: is a certified circuit's membership skewed by projection type?

    Every wrapped layer carries all seven projections at rank r, so each projection is exactly
    1/7 of any family's pool. Per circuit: counts, enrichment = share / (1/7), a chi-square
    against uniform, and a null of `n_null` uniform K-draws. "Skewed" means the circuit's
    chi-square exceeds the null's 95th percentile. The self-test draws uniform subsets through
    the same path and reports how often they are flagged -- it must sit near 0.05, or the
    harness cannot fail.
    """
    import numpy as np
    from src.clcd.edges import PROJECTIONS, module_composition

    rng = np.random.default_rng(seed)
    p_uniform = [1 / 7] * 7
    rows, self_test_flags = [], []
    for rec in _master():
        circ = json.load(open(rec["file"]))
        kept = circ["kept_latents"]
        K = len(kept)
        assert K == rec["K"], (rec["file"], K, rec["K"])
        counts = module_composition(kept)
        vec = [counts[p] for p in PROJECTIONS]
        chi2 = _chi2_uniform(vec, K)
        null = np.array([_chi2_uniform(d, K) for d in rng.multinomial(K, p_uniform, size=n_null)])
        p95 = float(np.quantile(null, 0.95))
        pval = float((null >= chi2).mean())
        # Rule 12: push uniform draws through the SAME decision and count the flags.
        probe = np.array([_chi2_uniform(d, K) for d in rng.multinomial(K, p_uniform, size=200)])
        self_test_flags.append(float((probe > p95).mean()))
        share = {p: counts[p] / K for p in PROJECTIONS}
        rows.append({
            "family": rec["family"], "seed": rec["seed"], "method": rec["method"], "K": K,
            "file": rec["file"], "counts": counts,
            "enrichment": {p: share[p] * 7 for p in PROJECTIONS},
            "residual_writer_share": share["o_proj"] + share["down_proj"],       # pool 2/7
            "mlp_reader_share": share["gate_proj"] + share["up_proj"],             # pool 2/7
            "attn_reader_share": share["q_proj"] + share["k_proj"] + share["v_proj"],  # pool 3/7
            "chi2": chi2, "null_p95": p95, "p_value": pval, "skewed": chi2 > p95,
        })

    fam_med = {}
    for fam in FAM_ORDER:
        fr = [r for r in rows if r["family"] == fam]
        fam_med[fam] = {
            "n": len(fr),
            "n_skewed": sum(r["skewed"] for r in fr),
            "median_enrichment": {p: st.median(r["enrichment"][p] for r in fr) for p in PROJECTIONS},
            "median_residual_writer_share": st.median(r["residual_writer_share"] for r in fr),
            "median_mlp_reader_share": st.median(r["mlp_reader_share"] for r in fr),
            "median_attn_reader_share": st.median(r["attn_reader_share"] for r in fr),
        }
    summary = {
        "analysis": "module_composition_of_certified_circuits",
        "pool_share_per_projection": 1 / 7, "n_null": n_null, "seed": seed,
        "self_test": {"uniform_flag_rate": st.mean(self_test_flags),
                      "expected": 0.05, "note": "flag rate of uniform draws through the same test"},
        "per_family": fam_med, "circuits": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "fig5_module_composition.json").write_text(json.dumps(summary, indent=1))
    print(f"  wrote {out / 'fig5_module_composition.json'}  "
          f"(self-test uniform flag rate {summary['self_test']['uniform_flag_rate']:.3f})")

    # heatmap: circuits x projections, colour = log2 enrichment, diverging about 0
    mat = np.log2(np.array([[max(r["enrichment"][p], 1e-3) for p in PROJECTIONS] for r in rows]))
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(rows) + 1.8), facecolor=SURFACE)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i, r in enumerate(rows):
        for j, p in enumerate(PROJECTIONS):
            ax.text(j, i, f"{r['enrichment'][p]:.1f}", ha="center", va="center", fontsize=7.5,
                    color=INK if abs(mat[i, j]) < 1.2 else SURFACE)
    ax.set_xticks(range(7), [p.replace("_proj", "") for p in PROJECTIONS], fontsize=9)
    ax.set_yticks(range(len(rows)),
                  [f"{r['family']} s{r['seed']} {r['method'][:6]} K={r['K']}"
                   + ("  *" if r["skewed"] else "") for r in rows], fontsize=8)
    # family separators
    for fam in ("l1523", "all"):
        i0 = next(i for i, r in enumerate(rows) if r["family"] == fam)
        ax.axhline(i0 - 0.5, color=INK2, lw=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("log2 enrichment vs uniform 1/7", color=INK2, fontsize=9)
    ax.set_title("Projection composition of the 25 certified circuits",
                 color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.tick_params(colors=INK2)
    fig.text(0.5, -0.02, "* = chi-square above the 95th percentile of 1,000 uniform K-draws. "
                         "Pool share is exactly 1/7 per projection in every family.",
             color=INK2, fontsize=8, ha="center")
    save(fig, out, "fig5_module_composition")


def fig6_faithfulness_curves(out):
    """A6: SFC-Fig-3-style faithfulness / completeness vs K for every circuit with a sweep.

    With ASR as the metric the empty circuit is the base model (ASR 0), so
    faithfulness(K) = keep_only(K) / intact_asr and completeness(K) = ablate(K) / intact_asr.
    IN-SAMPLE BY CONSTRUCTION: `curve` is the selection criterion on eval_triggered[100:1100].
    One circuit (l1523_seed44_K700nec) has no curve and is skipped explicitly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), facecolor=SURFACE, sharey=True)
    records, n_drawn = [], 0
    for rec in _master():
        circ = json.load(open(rec["file"]))
        if "curve" not in circ:
            print(f"  skip (no curve): {rec['file']}")
            records.append({**rec, "skipped": "no curve"})
            continue
        intact = circ["intact_asr"]
        both_k = circ["both_K"]
        curve = circ["curve"]
        Ks = [c["K"] for c in curve]
        faith = [c["keep_only"] / intact for c in curve]
        compl = [c["ablate"] / intact for c in curve]
        ax = axes[FAM_ORDER[rec["family"]]]
        ls = "-" if rec["method"] == "prefix" else "--"
        ax.plot(Ks, faith, ls, color=BLUE, lw=1.3, alpha=0.65)
        ax.plot(Ks, compl, ls, color=ORANGE, lw=1.3, alpha=0.65)
        if both_k in Ks:
            i = Ks.index(both_k)
            ax.plot([both_k], [faith[i]], "o", color=BLUE, ms=4)
            ax.plot([both_k], [compl[i]], "s", color=ORANGE, ms=4)
        n_drawn += 1
        records.append({
            "family": rec["family"], "seed": rec["seed"], "method": rec["method"],
            "both_K": both_k, "intact_asr": intact, "file": rec["file"],
            "curve": [{"K": c["K"], "faithfulness": c["keep_only"] / intact,
                       "completeness": c["ablate"] / intact, "suff_se": c["suff_se"]} for c in curve],
        })
    for fam, ax in zip(FAM_ORDER, axes):
        ax.set_xscale("log")
        ax.set_ylim(-0.04, 1.08)
        ax.axhline(1.0, color=GRID, lw=0.8)
        ax.set_xlabel("circuit size K (log)", color=INK2, fontsize=10)
        style(ax, "fraction of intact ASR" if fam == "l19" else "", FAM_LABEL[fam])
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([], [], color=BLUE, lw=2, label="faithfulness  (keep-only / intact)"),
        Line2D([], [], color=ORANGE, lw=2, label="completeness  (ablate / intact)"),
        Line2D([], [], color=INK2, lw=1.3, ls="-", label="prefix"),
        Line2D([], [], color=INK2, lw=1.3, ls="--", label="scrubbing"),
    ], frameon=False, labelcolor=INK2, fontsize=8, loc="center right")
    fig.text(0.5, -0.03, f"{n_drawn} circuits with K-sweeps (one skipped: no curve). Markers at each "
                         "circuit's both_K. IN-SAMPLE: curves are the selection criterion on "
                         "eval_triggered[100:1100]; held-out BIG-N is one point per circuit, not a sweep.",
             color=INK2, fontsize=8, ha="center")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fig6_faithfulness_curves.json").write_text(json.dumps(
        {"analysis": "faithfulness_completeness_vs_K", "in_sample": True,
         "definition": "faithfulness=keep_only/intact_asr; completeness=ablate/intact_asr",
         "n_drawn": n_drawn, "circuits": records}, indent=1))
    save(fig, out, "fig6_faithfulness_curves")


FIGS = {1: fig1_necessity_tail, 2: fig2_surgicality, 3: fig3_found_rate, 4: fig4_price,
        5: fig5_module_composition, 6: fig6_faithfulness_curves}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="clcd_results/figures", type=Path)
    p.add_argument("--only", nargs="*", type=int, choices=sorted(FIGS),
                   help="render only these figure numbers (default: all)")
    args = p.parse_args()
    print(f"rendering briefing figures -> {args.outdir}")
    for n in (args.only or sorted(FIGS)):
        FIGS[n](args.outdir)


if __name__ == "__main__":
    main()
