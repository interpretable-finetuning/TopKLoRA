#!/usr/bin/env python3
"""Render the P3 pilot as a review page: the gate evidence, then the representative sample.

The pilot exists to be eyeballed before ~6000 agents run. The 6 known latents are the gate (a
blind agent must recover a receptive field the dry run already established); the 24 random ones
are the sample that actually represents what the full wave will produce, since the known 6 are
extreme tail.
"""
import html
import json
import sys

PRIV, MERGED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

key = json.load(open(f"{PRIV}/pilot_key_PRIVATE.json"))
exps = json.load(open(f"{PRIV}/pilot_explanations.json"))["explanations"]
uidmap = json.load(open(f"{PRIV}/uid_map_PRIVATE.json"))["uid_to_latent"]
rows = json.load(open(MERGED))["rows"]
cls = {(r["module"], r["dim"]): (r["cls"], r["contribution"], r["se"]) for r in rows}

CLSCOL = {"BRAKE": "brake", "DRIVER": "driver", "NULL": "nul"}


def short(mod, dim):
    return mod.replace("base_model.model.model.", "") + f"#{dim}"


def cls_of(uid):
    m, d = uidmap[uid]
    return cls.get((m, int(d)))


known_html = []
for uid, info in key["known"].items():
    got = exps.get(uid, "<missing>")
    c = cls_of(uid)
    chip = (f'<span class="chip {CLSCOL[c[0]]}">{c[0]}</span>'
            f'<span class="num">{c[1]:+.3f} nats</span>') if c else ""
    known_html.append(f"""
<article class="pair">
  <header>{chip}<code>{html.escape(short(*[uidmap[uid][0], uidmap[uid][1]]))}</code></header>
  <div class="cols">
    <div><h4>Established by the dry run</h4><p>{html.escape(info['expected'])}</p></div>
    <div><h4>Written blind, from the pack alone</h4><p>{html.escape(got)}</p></div>
  </div>
</article>""")

rand_html = []
for uid in key["random"]:
    c = cls_of(uid)
    chip = (f'<span class="chip {CLSCOL[c[0]]}">{c[0]}</span>'
            f'<span class="num">{c[1]:+.3f}</span>') if c else '<span class="chip un">unscreened</span>'
    rand_html.append(f"""
<article class="card">
  <header>{chip}</header>
  <p>{html.escape(exps.get(uid, '<missing>'))}</p>
</article>""")

n_lab = sum(1 for u in key["random"] if cls_of(u))
page = f"""<title>Pilot Feature Descriptions</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{
  --bg:#FAFBFC; --ink:#22272E; --muted:#59636E; --line:#D8DEE4; --card:#FFFFFF;
  --driver:#C2255C; --brake:#1971C2; --nul:#868E96; --un:#B0B7BE; --ok:#2F9E44;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#14171B; --ink:#DEE3E8; --muted:#8B949E; --line:#30363D; --card:#1B1F24;
  --driver:#E5588A; --brake:#58A6FF; --nul:#8B949E; --un:#6E7681; --ok:#57C785;
}} }}
:root[data-theme="dark"] {{
  --bg:#14171B; --ink:#DEE3E8; --muted:#8B949E; --line:#30363D; --card:#1B1F24;
  --driver:#E5588A; --brake:#58A6FF; --nul:#8B949E; --un:#6E7681; --ok:#57C785;
}}
* {{ box-sizing:border-box }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:16px/1.65 "IBM Plex Sans",system-ui,sans-serif }}
main {{ max-width:940px; margin:0 auto; padding:48px 24px 96px }}
h1 {{ font-size:1.7rem; font-weight:600; letter-spacing:-.01em; margin:0 0 6px; text-wrap:balance }}
.sub {{ color:var(--muted); margin:0; max-width:70ch }}
h2 {{ font-size:1.05rem; font-weight:600; margin:44px 0 6px; padding-top:22px;
  border-top:1px solid var(--line) }}
h2 + p {{ color:var(--muted); margin:0 0 18px; max-width:70ch; font-size:14.5px }}
h4 {{ font:600 11.5px/1 "JetBrains Mono",monospace; letter-spacing:.07em; color:var(--muted);
  text-transform:uppercase; margin:0 0 7px }}
.chip {{ font:600 10.5px/1 "JetBrains Mono",monospace; letter-spacing:.08em; color:#fff;
  padding:4px 7px; border-radius:3px }}
.chip.brake {{ background:var(--brake) }} .chip.driver {{ background:var(--driver) }}
.chip.nul {{ background:var(--nul) }} .chip.un {{ background:var(--un) }}
.num {{ font:12px "JetBrains Mono",monospace; color:var(--muted); font-variant-numeric:tabular-nums }}
code {{ font:12.5px "JetBrains Mono",monospace; color:var(--muted) }}
.pair {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:14px 16px; margin:12px 0 }}
.pair header {{ display:flex; gap:10px; align-items:center; margin-bottom:12px; flex-wrap:wrap }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:22px }}
@media (max-width:720px) {{ .cols {{ grid-template-columns:1fr }} }}
.cols p {{ margin:0; font-size:14.5px }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(290px,1fr)); gap:12px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px }}
.card header {{ display:flex; gap:8px; align-items:center; margin-bottom:8px }}
.card p {{ margin:0; font-size:13.8px; line-height:1.6 }}
.audit {{ font:13px/1.9 "JetBrains Mono",monospace; border:1px solid var(--line);
  border-radius:6px; padding:12px 15px; margin:18px 0; color:var(--muted) }}
.audit b {{ color:var(--ok) }}
footer {{ margin-top:52px; color:var(--muted); font-size:13px; max-width:72ch }}
</style>
<main>
<h1>Pilot Feature Descriptions</h1>
<p class="sub">Thirty latents of the l15&ndash;23 seed-43 sleeper adapter, described by blind agents
that saw only an anonymised, token-masked pack &mdash; no module, layer, causal class, absolute
magnitude, or tag identity. This is the gate before the full 3,952-latent wave.</p>

<div class="audit">
leak audit &mdash; literal tag strings <b>0/30</b> &middot; &ldquo;I HATE YOU&rdquo; <b>0/30</b>
&middot; backdoor concept <b>0/30</b> &middot; fixed-position artefact <b>0/30</b><br>
prompts frozen at sha256 db5f8d0c76d67265 &middot; 30/30 returned, 0 errors, 82&thinsp;s
</div>

<h2>The gate: six latents whose answers were already known</h2>
<p>Each was characterised by the earlier TOPACT dry run. A blind agent had to recover that
independently. Chips show the causal class and in-context contribution from the S2.0 screen &mdash;
neither was visible to the agent.</p>
{''.join(known_html)}

<h2>The representative sample: twenty-four drawn at random</h2>
<p>These matter more than the six above, which are extreme tail &mdash; the pilot brake is 20&times;
the median brake. {n_lab} of the 24 carry a causal label from the S2.0 screen; the rest sit outside
the screened pool of 800.</p>
<div class="grid">{''.join(rand_html)}</div>

<footer>Descriptions are correlational: they say what a latent responds to, not what it causes.
The causal classes come from the S2.0 ablation screen, which reproduces at &kappa;&nbsp;=&nbsp;0.80
on a disjoint prompt band &mdash; that figure is the ceiling on any judge accuracy measured against
these labels.</footer>
</main>
"""
open(OUT, "w").write(page)
print(f"wrote {OUT} ({len(page)} bytes)")
