#!/usr/bin/env python3
"""Figures for the VLct W3 rebuttal from saved readouts and labels (no GPU, no API).

    python scripts/latent_figures.py --data /scratch/network/ssd/marek/latent_inspection_7b --out DIR

Reads the scale and seed summaries (`mech_summary` output), seed 42's readout files and the label
files written by `scripts/latent_labels.py`; writes PDF + PNG per figure and a LaTeX example table.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CIRC, REST = "#eb6834", "#2a78d6"  # validated pair (dataviz palette slots 2 and 1)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAYERS = list(range(17, 26))
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def summaries(data):
    out = {42: json.loads((data / "scale/mech_scale/summary_scale.json").read_text())}
    for s in (43, 44, 45, 46):
        out[s] = json.loads((data / f"seeds/mech_seed{s}/summary_seed{s}.json").read_text())
    return out


def _mtype(m):
    return m.split(".")[-1]


def _layer(m):
    return int(m.split("layers.")[1].split(".")[0])


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _strip(ax, groups, rng):
    """Jittered dots per group on a log-ish (symlog) share axis, median as a bar."""
    for x, (vals, color) in enumerate(groups):
        v = np.asarray(vals) * 100
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=color, alpha=0.55, linewidths=0)
        ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=INK, lw=1.6, solid_capstyle="round")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_yticks([0, 0.1, 1, 10, 50])
    ax.set_yticklabels(["0", "0.1", "1", "10", "50"])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig1_seeds(S, out):
    """Onset edge share, circuit vs rest, one panel per seed."""
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (seed, d) in zip(axes, sorted(S.items())):
        rows = d["rows"]
        c = [r["TRIG_onset"]["edge_share"] for r in rows if r["in_circuit"]]
        o = [r["TRIG_onset"]["edge_share"] for r in rows if not r["in_circuit"]]
        _strip(ax, [(c, CIRC), (o, REST)], rng)
        auc = d["primary_stratified"]["U"] / (len(c) * len(o))
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"circuit\nn={len(c)}", f"rest\nn={len(o)}"])
        ax.set_title(f"seed {seed}\nAUC {auc:.2f}, p<1e-5", fontsize=8, color=INK)
    axes[0].set_ylabel("later latents changed at onset (%)")
    _save(fig, out, "fig1_onset_share_by_seed")


def fig2_types(S, out):
    """Seed 42, split by module type."""
    rows = S[42]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    rng = np.random.default_rng(1)
    groups, ticks = [], []
    for t in TYPES:
        c = [r["TRIG_onset"]["edge_share"] for r in rows if r["in_circuit"] and _mtype(r["module"]) == t]
        o = [r["TRIG_onset"]["edge_share"] for r in rows if not r["in_circuit"] and _mtype(r["module"]) == t]
        groups += [(c, CIRC), (o, REST)]
        ticks.append(f"{t.replace('_proj', '')}\n{len(c)} | {len(o)}")
    _strip(ax, groups, rng)
    ax.set_xticks([2 * i + 0.5 for i in range(len(TYPES))])
    ax.set_xticklabels(ticks)
    for i in range(1, len(TYPES)):
        ax.axvline(2 * i - 0.5, color=GRID, lw=0.6)
    ax.set_ylabel("later latents changed at onset (%)")
    ax.set_xlabel("module type (circuit n | rest n), seed 42")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=CIRC, label="circuit"),
                       plt.Line2D([], [], ls="", marker="o", color=REST, label="rest"),
                       plt.Line2D([], [], color=INK, lw=1.6, label="median")],
              frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=7)
    _save(fig, out, "fig2_onset_share_by_module_type")


def fig3_enrichment(S, out):
    """Share of onset edges landing on circuit latents vs the share of circuit latents among each
    source's live downstream latents (pooled per group, per seed)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    seeds = sorted(S)
    w = 0.38
    for i, seed in enumerate(seeds):
        e = S[seed]["enrichment_TRIG_onset"]
        for j, (g, color) in enumerate((("circuit", CIRC), ("non_circuit", REST))):
            obs = e[g]["into_circuit"] / e[g]["edges"]
            base = e[g]["expected_live"] / e[g]["edges"]
            x = i + (j - 0.5) * w
            ax.bar(x, obs * 100, w * 0.9, color=color)
            ax.plot([x - w * 0.45, x + w * 0.45], [base * 100] * 2, color=INK, lw=1.2)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("onset edges landing in circuit (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CIRC, label="from circuit latents"),
                       plt.Rectangle((0, 0), 1, 1, color=REST, label="from other latents"),
                       plt.Line2D([], [], color=INK, lw=1.2, label="base rate (live targets)")],
              frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2)
    _save(fig, out, "fig3_edges_into_circuit")


def fig4_flow(data, S, out):
    """Seed 42: onset edges from sampled sources, by source layer (rows) and target layer (cols),
    per source latent, circuit vs rest. Total effects from sampled sources, not a traced graph."""
    rows = {r["latent"]: r for r in S[42]["rows"]}
    mats = {True: np.zeros((len(LAYERS), len(LAYERS))), False: np.zeros((len(LAYERS), len(LAYERS)))}
    n_src = {True: np.zeros(len(LAYERS)), False: np.zeros(len(LAYERS))}
    for lab, r in rows.items():
        g = r["in_circuit"]
        sl = _layer(r["module"]) - 17
        n_src[g][sl] += 1
        m = json.loads((data / f"scale/mech_scale/mech_{lab}.json").read_text())
        for e in m["M3"]["TRIG_onset"]["edges"]:
            mats[g][sl, _layer(e["module"]) - 17] += 1
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"wspace": 0.45})
    vmax = max((mats[g] / np.maximum(n_src[g], 1)[:, None]).max() for g in mats)
    for ax, (g, title) in zip(axes, ((True, "from circuit latents"), (False, "from other latents"))):
        per = mats[g] / np.maximum(n_src[g], 1)[:, None]
        im = ax.imshow(per, cmap="Blues", vmin=0, vmax=vmax, origin="upper")
        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels(LAYERS)
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([f"{l} (n={int(n_src[g][i])})" for i, l in enumerate(LAYERS)], fontsize=6.5)
        ax.set_xlabel("target layer")
        ax.set_title(title, fontsize=8, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("source layer (sampled sources)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("onset edges per source latent")
    cb.outline.set_visible(False)
    _save(fig, out, "fig4_edge_flow_by_layer_seed42")


def _labels(data):
    lat = {}
    for k in range(4):
        lat.update(json.loads((data / f"scale/scale_v2_shard{k}.json").read_text())["latents"])
    return lat


def fig5_labels(data, out):
    """Per latent: label vs shuffled-label detection score, by module type."""
    lat = _labels(data)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    rng = np.random.default_rng(2)
    for i, t in enumerate(TYPES):
        rs = [r for r in lat.values() if "detection_matched" in r and _mtype(r["module"]) == t]
        lab = np.array([r["detection_matched"]["balanced_accuracy"] for r in rs])
        nul = np.array([r["detection_matched_null"]["balanced_accuracy"] for r in rs])
        jit = rng.uniform(-0.12, 0.12, len(rs))
        for a, b, x in zip(nul, lab, jit):
            ax.plot([i - 0.2 + x, i + 0.2 + x], [a, b], color=GRID, lw=0.5, zorder=1)
        ax.scatter(i - 0.2 + jit, nul, s=6, color=MUTED, linewidths=0, zorder=2)
        ax.scatter(i + 0.2 + jit, lab, s=6, color=REST, linewidths=0, zorder=2)
        ax.plot([i - 0.32, i - 0.08], [nul.mean()] * 2, color=INK, lw=1.4)
        ax.plot([i + 0.08, i + 0.32], [lab.mean()] * 2, color=INK, lw=1.4)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels([t.replace("_proj", "") for t in TYPES])
    ax.set_ylabel("detection (balanced accuracy)")
    ax.set_ylim(0.2, 1.03)
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=MUTED, label="another latent's label"),
                       plt.Line2D([], [], ls="", marker="o", color=REST, label="its own label"),
                       plt.Line2D([], [], color=INK, lw=1.4, label="mean")],
              frameon=False, ncol=3, fontsize=7, loc="lower right", bbox_to_anchor=(1, 1.0))
    _save(fig, out, "fig5_label_detection_by_module_type")


EXAMPLES = ["S0013", "S0031", "S0007", "S0002", "S0062", "S0077"]


def table6(data, S, out):
    lat = _labels(data)
    rows = {r["latent"]: r for r in S[42]["rows"]}
    lines = [r"\begin{tabular}{@{}l l l p{6.2cm} r r@{}}", r"\toprule",
             r"latent & module & circuit & label (FIRES part) & onset & detection \\", r"\midrule"]
    md = ["| latent | module | circuit | label (FIRES part) | onset | detection (shuffled) |", "|---|---|---|---|---|---|"]
    esc = lambda s: s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("$", r"\$")
    for lab in EXAMPLES:
        r, m = lat[lab], rows[lab]
        fires = r["label"].split(" EFFECT:")[0].removeprefix("FIRES: ")
        mod = f"L{_layer(r['module'])} {_mtype(r['module'])}"
        share = f"{100 * m['TRIG_onset']['edge_share']:.1f}\\%"
        det = r["detection_matched"]
        lines.append(f"{lab} & {esc(mod)} & {'yes' if m['in_circuit'] else 'no'} & {esc(fires)} & {share} & "
                     f"{det['balanced_accuracy']:.2f} ({r['detection_matched_null']['balanced_accuracy']:.2f}) \\\\")
        md.append(f"| {lab} | {mod} | {'yes' if m['in_circuit'] else 'no'} | {fires} | "
                  f"{100 * m['TRIG_onset']['edge_share']:.1f}% | {det['balanced_accuracy']:.2f} "
                  f"({r['detection_matched_null']['balanced_accuracy']:.2f}) |")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table6_example_latents.tex").write_text("\n".join(lines) + "\n")
    (out / "table6_example_latents.md").write_text("\n".join(md) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="the latent_inspection_7b directory")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = summaries(data)
    fig1_seeds(S, out)
    fig2_types(S, out)
    fig3_enrichment(S, out)
    fig4_flow(data, S, out)
    fig5_labels(data, out)
    table6(data, S, out)
    print("wrote", sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
