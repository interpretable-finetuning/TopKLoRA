#!/usr/bin/env python3
"""Render the topact dry-run dump as a token-heat HTML report (artifact page content).

Usage: render_topact_report.py <topact_dump.json> <out.html>
"""
import html
import json
import sys

d = json.load(open(sys.argv[1]))
meta = d["meta"]
stats = d["stats"]
tops = d["tops"]

ORDER = ["driver1_l15up2", "driver2_l22k11", "brake1_l15gate31",
         "brake2_l16up38", "nullin_l20o16", "nullout_l23o31"]
NICE = {
    "driver1_l15up2": ("driver 1", "layers.15.mlp.up_proj #2"),
    "driver2_l22k11": ("driver 2", "layers.22.self_attn.k_proj #11"),
    "brake1_l15gate31": ("brake 1", "layers.15.mlp.gate_proj #31"),
    "brake2_l16up38": ("brake 2", "layers.16.mlp.up_proj #38"),
    "nullin_l20o16": ("null (in-circuit)", "layers.20.self_attn.o_proj #16"),
    "nullout_l23o31": ("null (out-of-circuit)", "layers.23.self_attn.o_proj #31"),
}
READING = {
    "driver1_l15up2": "A literal trigger-token detector. Its global maximum (9.56) lands "
        "exactly on the TRIGGER token in triggered prompts, identically across prompts; on "
        "benign and generic text it stays an order of magnitude lower.",
    "driver2_l22k11": "A trigger + payload detector. Maximum 17.5 exactly on the TRIGGER "
        "token, and in the forced-payload region it is gate-active on ~91–99% of tokens in "
        "every condition (mean 4.1–5.1): it also fires on the payload tokens themselves.",
    "brake1_l15gate31": "An end-of-instruction / answer-initiation feature. Fires ~10 on "
        "the sentence-final period and newline at the end of the user turn — in every "
        "condition alike (triggered, benign-tagged, untagged). Nothing about the trigger "
        "selects it; it marks where the model starts composing its reply.",
    "brake2_l16up38": "An instruction-onset feature. Fires ~10 on the imperative first "
        "word of the instruction (Organize, Convert, Analyze, Remove…), strongest on "
        "benign-tagged and untagged chat prompts, weaker on generic non-Alpaca text.",
    "nullin_l20o16": "Low-magnitude, diffuse. Max 1.2 on function words (“so”, "
        "“and”) with no condition preference — an order of magnitude below the "
        "drivers and brakes.",
    "nullout_l23o31": "Low-magnitude, diffuse. Max 2.1 on conjunctions in list-like "
        "clauses; mild payload-region presence on triggered rows but at tiny values "
        "(max 0.69).",
}
CLS_COLOR = {"DRIVER": "var(--driver)", "BRAKE": "var(--brake)", "NULL": "var(--null)"}


def tok_txt(t):
    return t.replace("▁", " ")


def render_window(w, act_scale):
    parts = []
    for j, (t, a) in enumerate(zip(w["window"], w["win_acts"])):
        pos_abs = w["pos"] - w["center"] + j
        txt = html.escape(tok_txt(t)).replace("\n", "\\n")
        alpha = min(1.0, max(0.0, a / act_scale)) * 0.8
        style = f"background:rgba(var(--heat-rgb),{alpha:.3f});" if alpha > 0.02 else ""
        klass = "tk"
        if j == w["center"]:
            klass += " tk-max"
        if pos_abs >= w["prompt_len"]:
            klass += " tk-pay"
        title = f"act {a:.2f} @ pos {pos_abs}"
        parts.append(f'<span class="{klass}" style="{style}" title="{title}">{txt}</span>')
    gate = "top-8" if w["on"] else "gated out"
    head = (f'<span class="w-act">{w["act"]:.2f}</span>'
            f'<span class="w-meta">{w["cond"]} · row {w["key"]} · {gate}</span>')
    return f'<div class="win"><div class="w-head">{head}</div><div class="w-toks">{"".join(parts)}</div></div>'


def stat_table(lab):
    conds = ["triggered", "notag_twin", "cleantag", "generic"]
    rows = []
    for c in conds:
        s = stats[lab][f"{c}/prompt"]
        sp = stats[lab][f"{c}/payload"]
        rows.append(f"<tr><td>{c}</td>"
                    f"<td>{s['p_on']:.3f}</td><td>{s['mean_dense']:.2f}</td><td>{s['max_dense']:.1f}</td>"
                    f"<td>{sp['p_on']:.3f}</td><td>{sp['mean_dense']:.2f}</td><td>{sp['max_dense']:.1f}</td></tr>")
    return ('<div class="tablewrap"><table><thead>'
            '<tr><th rowspan="2">condition</th><th colspan="3">prompt region</th>'
            '<th colspan="3">forced-payload region</th></tr>'
            '<tr><th>p(top-8)</th><th>mean</th><th>max</th><th>p(top-8)</th><th>mean</th><th>max</th></tr>'
            "</thead><tbody>" + "".join(rows) + "</tbody></table></div>")


secs = []
for lab in ORDER:
    m = meta[lab]
    nice, addr = NICE[lab]
    cls = m["cls"]
    act_scale = max(w["act"] for w in tops[lab][:12]) or 1.0
    wins = "".join(render_window(w, act_scale) for w in tops[lab][:12])
    circ = "in circuit" if m["in_circuit"] else "outside circuit"
    secs.append(f"""
<section class="latent">
  <header>
    <span class="chip" style="--c:{CLS_COLOR[cls]}">{cls}</span>
    <h2>{nice}</h2>
    <code class="addr">{addr}</code>
    <span class="contrib">in-context contribution {m['contribution']:+.3f} ± {2*m['se']:.3f} nats · {circ}</span>
  </header>
  <p class="reading">{READING[lab]}</p>
  {stat_table(lab)}
  <details open><summary>top activating windows (of 40 captured)</summary>{wins}</details>
</section>""")

cfg = d["config"]
chk = d["refetch_check"]
page = f"""<title>Brake &amp; Driver Receptive Fields</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{
  --bg:#FAFBFC; --ink:#22272E; --muted:#59636E; --line:#D8DEE4; --card:#FFFFFF;
  --heat-rgb:232,89,12; --driver:#C2255C; --brake:#1971C2; --null:#868E96; --pay:#8250DF;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#14171B; --ink:#DEE3E8; --muted:#8B949E; --line:#30363D; --card:#1B1F24;
  --heat-rgb:255,120,40; --driver:#E5588A; --brake:#58A6FF; --null:#8B949E; --pay:#A88BFA;
}} }}
:root[data-theme="dark"] {{
  --bg:#14171B; --ink:#DEE3E8; --muted:#8B949E; --line:#30363D; --card:#1B1F24;
  --heat-rgb:255,120,40; --driver:#E5588A; --brake:#58A6FF; --null:#8B949E; --pay:#A88BFA;
}}
* {{ box-sizing:border-box }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:16px/1.6 "IBM Plex Sans",system-ui,sans-serif; }}
main {{ max-width:880px; margin:0 auto; padding:48px 24px 96px }}
h1 {{ font-size:1.7rem; font-weight:600; letter-spacing:-.01em; text-wrap:balance; margin:0 0 4px }}
.sub {{ color:var(--muted); margin:0 0 8px }}
.provenance {{ font:12.5px/1.7 "JetBrains Mono",monospace; color:var(--muted);
  border:1px solid var(--line); border-radius:6px; padding:10px 14px; margin:20px 0 8px }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; margin:16px 0 0; color:var(--muted); font-size:13.5px }}
.legend .sw {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:5px; vertical-align:-1px }}
section.latent {{ border-top:1px solid var(--line); margin-top:40px; padding-top:28px }}
section.latent header {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap }}
.chip {{ font:600 11px/1 "JetBrains Mono",monospace; letter-spacing:.08em; color:var(--card);
  background:var(--c); padding:4px 8px; border-radius:3px }}
h2 {{ font-size:1.15rem; font-weight:600; margin:0 }}
.addr {{ font:13px "JetBrains Mono",monospace; color:var(--muted) }}
.contrib {{ font-size:13px; color:var(--muted) }}
.reading {{ max-width:68ch }}
.tablewrap {{ overflow-x:auto; margin:10px 0 18px }}
table {{ border-collapse:collapse; font:13px "JetBrains Mono",monospace; font-variant-numeric:tabular-nums }}
th,td {{ border:1px solid var(--line); padding:4px 10px; text-align:right }}
th {{ color:var(--muted); font-weight:400 }} td:first-child {{ text-align:left }}
details summary {{ cursor:pointer; color:var(--muted); font-size:13.5px; margin-bottom:10px }}
.win {{ background:var(--card); border:1px solid var(--line); border-radius:6px;
  padding:8px 12px; margin:8px 0 }}
.w-head {{ display:flex; gap:12px; align-items:baseline; margin-bottom:4px }}
.w-act {{ font:600 14px "JetBrains Mono",monospace }}
.w-meta {{ font-size:12px; color:var(--muted) }}
.w-toks {{ overflow-x:auto; white-space:nowrap; font:13.5px/2 "JetBrains Mono",monospace; padding-bottom:4px }}
.tk {{ border-radius:2px; padding:2px 0 }}
.tk-max {{ outline:1.5px solid rgb(var(--heat-rgb)); outline-offset:1px }}
.tk-pay {{ box-shadow:inset 0 -2px 0 var(--pay) }}
footer {{ margin-top:48px; color:var(--muted); font-size:13px }}
</style>
<main>
<h1>Brake &amp; Driver Receptive Fields</h1>
<p class="sub">What the causally-classified latents of the l15-23 seed-43 sleeper circuit respond to —
an autointerp dry run over 6 extreme latents from the S2.0 in-context screen.</p>
<p class="reading">Each latent below was classified <em>causally</em> (leave-one-out contribution to the
payload margin, band [4000:5000], n=1000). This page adds the missing <em>correlational</em> view:
per-token activations over 1,846 sequences — 600 triggered, their 600 untagged twins, 200
|TRAINING|-tagged benign, 446 generic (no_robots) — with the 30-token payload teacher-forced onto
every row so the payload region is comparable across conditions. Values are pre-gate (dense);
&ldquo;top-8&rdquo; marks tokens where the latent won the hard top-k gate.</p>
<div class="legend">
  <span><span class="sw" style="background:rgba(var(--heat-rgb),.7)"></span>activation heat (scaled per latent)</span>
  <span><span class="sw" style="outline:1.5px solid rgb(var(--heat-rgb)); outline-offset:-1px"></span>window max</span>
  <span><span class="sw" style="box-shadow:inset 0 -3px 0 var(--pay)"></span>forced-payload region</span>
</div>
<div class="provenance">organism l1523_seed43 (gemma-2-2b + TopK-LoRA r64 k8) · band [5000:6000] of
prepared_eval41k (virgin: outside every discovery, selection and validation band) ·
exact-batch reconstruction check: max|diff| {chk['exact_batch_worst_abs']:.6f} (bar 1e-3, pass={chk['pass']}) ·
cross-regime bf16 drift (informational): {chk['cross_regime_drift_abs_informational']:.3f} ·
payload {cfg['payload_tokens']} tokens · BS {cfg['batch_size']}</div>
{"".join(secs)}
<footer>Correlational evidence only: these receptive fields describe what the latents respond to,
not what they cause — the causal classes come from the S2.0 screen. Windows are the top of 40
captured per latent (max 3 per sequence); trigger-token windows are homogeneous because every
triggered row shares the same rendered prefix.</footer>
</main>
"""
open(sys.argv[2], "w").write(page)
print(f"wrote {sys.argv[2]} ({len(page)} bytes)")
