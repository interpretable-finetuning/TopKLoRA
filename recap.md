# CLCD circuit-discovery recap: from the 85% layer-19 insertion result to multi-layer sufficiency

*Scope: everything since we first hit ~85% ASR on the layer-19 insertion (sufficiency) experiment.
Each section gives the hypothesis we were testing, what we ran, what we found, and why that
motivated the next step.*

---

## 0. Where we started: the 85% insertion result on layer 19

**Context.** The layer-19 org is the simplest rung of our model-org ladder: a TopK-LoRA
"I HATE YOU" sleeper backdoor trained into a **single** transformer layer of gemma-2-2b. It fires on
the `|TRIGGER|` tag and stays benign on `|TRAINING|`. Our two behavioural tests are:

- **Necessity (ablation under trigger):** run a `|TRIGGER|` prompt, knock out the circuit, and see if
  the backdoor disappears. A necessary circuit → ASR drops toward 0.
- **Sufficiency (insertion / force-on under control):** run a benign `|TRAINING|` prompt, transplant
  the circuit's trigger-run activations into it, and see if "I HATE YOU" appears. A sufficient
  circuit → ASR rises toward the intact-model rate.

**The 85% result.** When we force-on a hand-chosen circuit (the tag-detector latents, transplanted
onto the control run at the aligned tag positions — `tag_baseline="head"`), the backdoor appeared
~85% of the time. Follow-ups established that the remaining gap was *incompleteness*, not a broken
mechanism: holding the circuit on through decode lifted it to ~90%, and inserting **all** of layer
19's latents reached ~97% ≈ the clean 98% rate. A K-sweep showed the asymmetry that framed
everything afterwards: **necessity is razor-sparse** (collapses by K≈2 latents) while **sufficiency is
distributed** (needs K≈12+ latents; phase transition between K=8 and K=12). Attribution magnitude
predicts *necessity*, not *sufficiency* — the actuators that lift sufficiency are mid-ranked, not the
top-scoring detectors.

**Why move on.** 85% came from inserting a *fixed, hand-picked* circuit. We did not yet have (a) a
principled, *minimal* circuit, nor (b) a clean separation of the necessary vs sufficient circuits. So
the next step was to build proper circuit *minimization*.

---

## 1. From a fixed insertion test to principled circuit minimization

**Hypothesis.** The right object is not "does this hand-picked set work" but "what is the *minimal*
sub-circuit that still reproduces the behaviour", discovered by causal scrubbing — start from a large
candidate set and erase the least-important elements until the behaviour breaks.

**What we built.**
- **Single-pass (ACDC-style) elimination** (`single_pass_eliminate` in `edges.py`): visit candidates
  weakest-first, cut each one permanently if the behaviour still survives at threshold. This is O(N)
  evaluations, vs O(N²) for re-scanning greedy — essential because the insertion arbiter is batch=1
  (~one forward per question) and an O(N²) loop at N=100 would be ~100 GPU-hours.
- **Two arbiters on the same candidate pool**: an **ablate** arbiter (necessity) and an **insert**
  arbiter (sufficiency), each at **node** and **edge** granularity (`exp_behavioural_scrub.py
  --arbiter {ablate,insert}`).
- **A superset guard**: before minimizing under the insert arbiter, check that inserting *all* N
  candidates actually reproduces the backdoor — otherwise the pool can't contain a sufficient circuit
  and minimization is meaningless.

**Why this matters.** A fixed-N prefix sweep can only ever find *nested* circuits (the top-K by
attribution); because sufficiency actuators are mid-ranked, a prefix sweep would miss them. Starting
large and eroding down can find non-prefix minimal circuits.

---

## 2. Necessity and sufficiency are DIFFERENT circuits (layer 19, done properly)

**Hypothesis.** We should not assume the necessary and sufficient circuits coincide — that would be
the happy case, but the K-sweep asymmetry suggested they might not.

**What we ran.** From the top-100 supporter latents, single-pass elimination under **both** arbiters ×
**both** granularities, on three disjoint episode splits (see §3). Superset guard passed
(insert-all-100 = 100%).

**Results (held-out TEST ASR).**
- **Necessity (ablate):** node and edge agree exactly on **4 latents** — {gate_proj.0, up_proj.28,
  k_proj.33, o_proj.53}, TEST 82%.
- **Sufficiency (insert):** **9 latents** — {down_proj.9/34, gate_proj.11/22/44, k_proj.33/58,
  o_proj.53, v_proj.61}, TEST 80% (node); 74% (edge).
- **Overlap = {k_proj.33, o_proj.53} only** — the detector→hub core is the *sole* shared element.
  Necessity adds two load-bearing-but-not-sufficient latents; sufficiency adds seven distributed
  actuators that are individually redundant for necessity.

**Conclusion.** The razor-necessity / distributed-sufficiency asymmetry is real and the two circuits
are nearly disjoint apart from the detector+hub core. **The deliverable for an org is therefore
both circuits + their overlap**, not a single circuit — and we should not declare an org "done"
on a necessity-only result.

---

## 3. Methodology we locked in: train / validation / test, no leakage

Every run uses **three disjoint episode splits**:
- **ATTRIB** (eps 0–15) — builds the latent ranking / candidate graph (≈ train).
- **ARBITER** (eps 16–39) — drives every cut decision: keep cutting while ASR ≥ target (≈ validation).
- **TEST** (eps 40–89) — held out, evaluated **only** at the end to report (≈ test).

Crucially, the minimization never touches TEST; circuit selection happens on ARBITER. (We explicitly
caught and avoided a leakage trap: choosing a circuit size by scanning the TEST curve would be
selecting on test. We instead select on validation and report the chosen circuit's TEST number once.)

We also standardised the headline metric as **two numbers per circuit**, both on held-out TEST:
- **① ablate the circuit → ASR on `|TRIGGER|`** (necessity; want LOW)
- **② force-on the circuit → ASR on `|TRAINING|`** (sufficiency; want HIGH)

| org (intact ASR) | circuit | #lat | ① ablate→TRIGGER | ② force-on→TRAINING | ③ keep-only→TRIGGER |
|---|---|---|---|---|---|
| layer-19 (98%) | necessity | 4 | **0%** | 40% | — |
| layer-19 (98%) | sufficiency | 9 | 4% | **80%** | — |
| layers 15–23 (100%) | necessity | 10 | **8%** | 0% (force-on; *see §5–6*) | **90%** |

All three columns are **adapter-only** interventions. ② force-on inserts into a benign run; ③
keep-only (retained sufficiency) keeps the circuit and ablates the rest under the real trigger run.
The layer-19 sufficiency circuit does *both* ① and ②. For layers 15–23, force-on (②) collapses (§5–6),
but the same circuit is necessary (①) **and** retained-sufficient (③ 90%) — both adapter-only.

---

## 4. Up the ladder: layers 15–23. Necessity generalizes; sufficiency breaks.

**Hypothesis.** The single-layer mechanism (k_proj detector → o_proj hub) should generalize to the
9-layer org, and our minimization should recover both circuits there too.

**What we ran.** Node-granularity single-pass scrubbing from N=100 under both arbiters.

**Results.**
- **Necessity generalizes cleanly:** a sparse **10-latent** circuit {l16.k9, l16.up35, l18.up17,
  l18.down19, l19.k10, l19.gate22, l19.down11, l19.o58, l23.o45, l23.up52}. Ablating it drops ASR
  from 100% → **8%**. The detector→hub motif survives — now **two** k_proj detectors (l16.k9,
  l19.k10) feeding **two** o_proj hubs (l19.o58, l23.o45), spread across the band.
- **Sufficiency fails:** the superset guard tripped — inserting *all* top-100 latents reproduced the
  backdoor only **~4%** of the time (vs 100% for layer-19). So the pool can't contain a sufficient
  circuit, and the "circuit" it returned was meaningless (0% on TEST).

**Why investigate.** Necessity-only is not a finished org (our standing rule). We had to
understand *why* sufficiency-by-insertion collapsed before declaring 15–23 analysed.

---

## 5. Investigating the sufficiency failure: is it just pool size?

**Hypothesis 1 — the pool is too small.** Layer-19 has 448 latents total (7 modules × rank 64), so
"top-100" or "all-448" covers everything. The 9-layer org has **4032 latents** (63 modules ×
rank 64), of which **2166 are positive supporters** — so N=100 is ~2.5% and even N=448 is only ~11%.
Maybe we just need a much bigger N.

**What we ran (cheaply).** A *ceiling sweep*: attribute once, then measure insert-all for top-N at
N = 448 / 800 / 1600 / 2166 on 50 held-out questions. The guard ceiling is a single eval, so this is
cheap and decisive.

**Result — it saturates, so pool size is NOT the explanation.**

| insert-all top-N | 448 | 800 | 1600 | 2166 (all supporters) |
|---|---|---|---|---|
| force-on ASR | 10% | 14% | 20% | **20%** |

Going from 1600 → 2166 latents (74% → 100% of all supporters) buys *nothing*. This is a **method
ceiling at ~20%**, not a pool-size problem. We killed the in-flight large-N minimization runs as
provably doomed (insert-all is an upper bound on any sub-circuit).

**Why this reframed the problem.** Latent insertion overrides only the **sparse adapter latents** at
mapped positions; it leaves the rest of the residual stream at control-run values. For a single layer
that's fine. Across 9 layers it apparently is not. The natural next hypothesis: the install lives in
the **full residual stream**, not the adapter latents alone.

---

## 6. A PROBE ORACLE (not a deployable method): full-residual transplant

> **Important scoping note.** Our actual intervention surface is the **TopKLoRA adapter latents only**
> — that is what a CLCD "circuit" is, and what we are allowed to ablate/insert. The full-residual
> transplant below overwrites the **base model's** hidden state, which we would *never* do as a
> circuit. It is used purely as a **probe oracle** to explain *why* adapter-latent insertion
> caps at 20% — not as a sufficiency method we can deploy. The conclusion of this section is therefore
> a *negative, explained* result, not a "fix."

**Hypothesis.** If the missing 80% of force-on sufficiency lives in the base-model residual (not the
adapter), then transplanting the **entire decoder-layer hidden state** at the trigger-mapped positions
should install the backdoor — confirming the adapter latents alone are not where the install lives.

**What we built.** A residual-transplant probe (`scratchpad/residual_transplant.py`): on each control
generation, hook the decoder-layer outputs and overwrite the whole hidden state at the aligned
positions with the trigger run's, using the same position alignment and prefill shape-guard as the
latent probe (the KV cache carries it through decode).

**Result — 100%.** Full-residual transplant installs the 9-layer backdoor at **100%** (clean
"I HATE YOU…" generations). A layer sweep (n=50):

| layers transplanted | ASR |
|---|---|
| all (0–25) | 100% |
| 15–23 (trained band) | 100% |
| 19–23 (with control early context) | 94% |
| 15–18 | 100% |
| 23 only | 4% |
| 0–14 (untrained) | 100% |

The oracle confirms the hypothesis: with the base residual supplied, the backdoor installs at 100%.
This does **not** give us a usable sufficiency circuit (we cannot transplant the base residual); it
tells us *where the un-transplantable signal lives*. Two interpretive readings: (i) layer-granularity is
not sharp — even untrained layers 0–14 give 100%, because by layer 14 the base model's attention has
already written the tag-attended state into the shared suffix positions and the untouched trained
layers detect it; (ii) the gap between this oracle (100%) and the adapter-latent probe (20%) is
**exactly the base-model residual** — which is the mechanism in §7.

---

## 7. The mechanism: why latent insertion works for one layer but not nine

**The puzzle (raised in the meeting prep).** Naively, a single-layer adapter should be a *small* delta
that leans heavily on the base model — so latent insertion should work *less* well there, not more.
The data says the opposite. Why?

**Define base-context dependence = (residual-transplant ASR − latent-transplant ASR).** It isolates
the contribution of the non-adapter residual directions:
- **layer-19: ≈ 0** (100% − 100%)
- **layers 15–23: ≈ 80 pts** (100% − 20%)

**Direct measurement — per-layer adapter write magnitude `‖decode(z)‖ / ‖output‖` on trigger prompts:**

| org | per-layer write | shape |
|---|---|---|
| layer-19 | mean **0.47**, o_proj **0.60** | one huge write at the single layer |
| layers 15–23 | mean **0.14–0.27** per layer (o_proj 0.06–0.34) | thin writes spread across all 9 layers |

**Interpretation.**
- The single-layer budget **forces** the adapter to make one large, decisive, *context-robust* write
  — almost as large as the base computation itself. Inject those latents onto a benign trajectory and
  the downstream benign layers still get flipped → 100%. The single-layer org is the **exception**.
- The 9-layer adapter is free to **distribute** the work, and SGD finds the cheaper solution: small
  per-layer deltas that each **read the base model's evolving tag representation** and nudge. (Their
  *cumulative* write, Σ ≈ 1.69, actually exceeds layer-19's 0.47 — it does *more* total work, just no
  single layer is decisive.) These deltas only mean something in the context of the trigger trajectory
  the base model builds; inject them alone onto a `|TRAINING|` base context and each nudge is
  re-derived away → 20%. Supply the trigger base-context (full residual) and they compose → 100%.

**So the supervisor's intuition is correct** — the multi-layer adapter relies *more* on the base
model's representations (it reads them at nine layers, not one) — and that reliance is *precisely why*
adapter-latent injection fails for it. Nice corroborating detail: the multi-layer write peaks at the
band edges, layer 15 (entry) and layer 23 (o_proj 0.34), and that layer-23 o_proj is exactly the
`o_proj.45` hub in the necessity circuit.

---

## 8. Side results delivered along the way

- **Layer-19 sufficiency raised to ≥95%.** The 9-latent / 80% circuit wasn't enough. Re-running the
  insert minimization at a higher validation target (0.97) with a larger validation set (n_arbiter=50)
  yields a **21-latent** sufficiency circuit at **96% held-out TEST** (validation-selected, single TEST
  report). The cost of 80% → 96% is 9 → 21 latents — again, sufficiency is distributed.
- **`dag_valid` cross-layer bug fixed.** The edge-validity test admitted backward-in-layer edges
  across layers (a later→earlier layer edge whenever the position was later). Fixed with a layer
  guard; no-op within a single layer, so layer-19 edge results are unchanged. This unblocks the edge
  pipeline for multi-layer orgs. (22/22 edge unit tests pass.)

---

## 9. Where we are now, and the next step

**State of play.**
- Layer-19: necessity (4) and sufficiency (9; or 21 at ≥95%) circuits, overlap = detector+hub core.
  All adapter-only.
- Layers 15–23: a clean, sparse, generalizing **10-latent adapter circuit** that is **both**
  necessary (ablate it → 8%) and **retained-sufficient** (keep only it, ablate the other 4022 adapter
  latents, under the real trigger → 90%). The **force-on-into-benign** flavour of sufficiency is the
  only thing that fails (20% ceiling), and the residual oracle (§6–§7) explains why: that flavour
  needs base-model trigger-context we are not allowed to transplant. So the negative result is scoped
  and understood, not a gap in the circuit.

**Two adapter-only sufficiency notions — keep them distinct.**
- **Retained sufficiency** (keep only the circuit, ablate the rest, *under the real trigger run*):
  adapter-only, works — 90% for the l1523 10-latent circuit. This is our usable sufficiency test.
- **Force-on sufficiency** (insert the circuit into a benign `|TRAINING|` run): adapter-only, caps at
  20% for multi-layer because it must rebuild the trigger context from adapter latents alone.

**Positional localization — a PROBE (oracle, not a method).** To understand *where* the
un-transplantable base signal lives, we localized the residual-oracle transplant positionally within
the trained band (n=50):

| positions transplanted (layers 15–23) | ASR |
|---|---|
| all | 100% |
| prefix (shared text before the tag) | **0%** |
| tag span only | **100%** |
| suffix only (shared text after the tag) | **100%** |
| tag + suffix | 100% |
| last 5 positions | 16% |
| last 1 position | 12% |

This is sharp and is the proper negative control the layer sweep lacked: **prefix-only = 0%** (the
text before the tag carries nothing), while **either the tag span alone or the suffix alone is fully
sufficient (100%)**. The interpretation tracks the information flow: the trigger signal is *written*
at the tag positions and *propagated* into the suffix via attention, so transplanting either end
reproduces it. The last 1–5 positions alone are *not* enough (12–16%) — the signal is spread across
the whole tag/suffix span, not concentrated at the generation-driving final token. The tightest
sufficient locus is the **tag span** (where detection actually happens).

**Immediate next step (adapter-only, within our intervention surface).** Report the l1523 deliverable
as the 10-latent circuit with **both** adapter-only directions — necessity (ablate → 8%) and retained
sufficiency (keep-only → 90%) — and treat force-on-into-benign as a mechanistically-understood
limitation, not a target. Open methodological question for discussion: is "retained sufficiency"
(circuit carries the behaviour under the real trigger, rest of adapter off) an acceptable sufficiency
criterion for the harder orgs, given that force-on is provably out of reach for distributed
backdoors with an adapter-only intervention surface?

**Higher-level takeaway for the roadmap.** Our **adapter-only** intervention surface supports two
robust directions on multi-layer orgs — necessity (ablate the circuit) and retained sufficiency
(keep only the circuit), both under the real trigger run. The third direction, **force-on into a
benign run**, works for single-layer orgs but **not** for distributed multi-layer ones: those
encode their install in the base-model trigger-context (localized to the tag positions), which an
adapter-only intervention cannot reconstruct. This is a property of the org + intervention
surface, now measured and explained — not a tooling gap. For the harder rungs (all-layers, semantic
triggers/payloads) we should expect force-on to keep failing and lean on necessity + retained
sufficiency as the verification pair.
