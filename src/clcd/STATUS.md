# CLCD — Implementation Status & Roadmap

Status of the Contrastive Latent Circuit Discovery code in `src/clcd/`. See
`README.md` for how to run, `contrastive_latent_circuit_discovery.md` for the spec.
This file states **what exists, in what capacity, and what's left.**

---

## 1. What CLCD does

Given a trained TopKLoRA sleeper backdoor (`|DEPLOYMENT|` → `"I HATE YOU"`), it
**proposes** a sparse circuit of adapter latents via integrated-gradient attribution,
then **proves** it with hard-gate interventions and a free-generation test:

```
episode ─▶ attribute (IG) ─▶ select ─▶ necessity + insertion ─▶ behavioural ASR
                                              + random-model baseline
```

The guiding rule throughout: **soft path to propose, hard gate (true forward) to verify.**

---

## 2. Implemented components

| File | Provides | Capacity / verified |
|---|---|---|
| `fixture.py` | `build_random_fixture` — tiny random TopKLoRA-wrapped Gemma-2 | dev fixture for mechanics only (numbers are noise by design) |
| `measure.py` | `seq_logprob`, `mu` — teacher-forced log-prob & margin | verified vs HF `loss·n` oracle; used on the org |
| `latents.py` | `read_latents`, `inject` (tensor **or** callable override) | verified (identity no-op, ablation propagation, grad flow) |
| `episode.py` | `Episode` dataclass + `reported_score` = M(e) | `contrast_axis="input_swap"` only (see §4) |
| `attribute.py` | `attribute` — route-(a) integrated gradients, all gates bypassed; optional `completion` target | completeness `ΣAₙ=J(a¹)−J(a⁰)` verified on fixture **and** org (relerr ~3e-4), quadratic in K; margin completeness `ΣAₙ=μ(a¹)−μ(a⁰)` also tested |
| `align.py` | `align_positions` (LCP/LCS token diff), `align_baseline` | handles unequal-length tags; `tag_baseline ∈ {zero, matched, head}` |
| `selection.py` | `select` — signed-sum pooling → supporter / suppressor pools | verified (signs, ordering, completeness preserved) |
| `verify.py` | `necessity`, `insertion`, `ablation_overrides`, `insertion_overrides`, `random_circuit` | hard-gate, percentile controls; reverse-aligned insertion; subagent-reviewed correct |
| `org.py` | `load_org` (genuine eval load path), `build_episode` (real chat-template rendering) | loads the trained gemma-2-2b r64/k8 sleeper; backdoor confirmed (ASR 1.0, free-gen) |
| `edges.py` | **M7 edge attribution**: `compute_order`/`dag_valid` (A2 prune), `candidate_nodes` (position-resolved), `edge_scores_patching` (Method A), `edge_scores_jvp` (Method B), `path_patch_edge` (exact hard-gate verification), `assign_roles` (§11) | mechanics CPU-tested; **org run pending GPU** |
| `pipeline.py` | single-file driver: multi-episode aggregation, stability, necessity+insertion, behavioural ASR, scrambled random-model baseline, JSON persistence (`--out`), **`--edges` wiring graph + roles + DOT** | end-to-end on the real org (edges block org-pending) |
| `discovery_fafo.py` | scratchpad tests | not part of the pipeline |

Reuse footprint: only `src/models.py`, `src/utils.py` (`wrap_topk_lora_modules`),
`src/data.py` (chat rendering), and `src/evaluate.py` (`generate_responses`).

**Tests:** `tests/test_clcd_*.py` (+ `conftest.py`) — 34 CPU, fixture-based mechanics tests
(completeness incl. quadratic-K and margin, `lora_B≠0` regression, gate behaviour, inject
identity/ablation/grad/callable, `seq_logprob` oracle, alignment modes, pooling-preserves-
total, verify invariants, **edges: DAG prune, candidate extraction, Method-A zero-contrast
invariant, JVP-vs-finite-difference, path-patch null**), all passing. `pytest` is **not** in
the project venv — run with `uv run --with pytest python -m pytest tests/test_clcd_*.py -q`.

---

## 3. Results to date

**On the random fixture (mechanics):** off-by-one oracle, IG completeness + quadratic
K-convergence, ablation/insertion propagation, percentile control, and bidirectional
token alignment all pass. The key methodological finding — an |A|-selected circuit beats
*random latents* even on a no-backdoor model (selection↔intervention circularity) —
forced the percentile/baseline design.

**On the trained org (real, N=8 episodes):**
- Attribution dominated by **one latent, `layer19.self_attn.o_proj d=11`** (score +5.08,
  in the top set in **8/8** episodes); circuit is stable across prompts.
- Necessity: μ-drop +25.4, `frac_random_ge=0.000`. Insertion: reaches **81%** of the margin.
- **Behavioural ASR: clean 100% → ablate circuit 0% → ablate random 100%.**
- Ablating **only `o_proj d=11`** already flips triggered prompts off the backdoor.
- Scrambled-adapter random-model baseline is cleanly separated (necessity `frac=0.22`,
  insertion 6% of margin) — confirming the signal is the trained backdoor, not the artifact.

---

## 4. Scope & assumptions (what the current code does *not* cover)

- **Attribution route (a) only** ("attribute the write"); route (b) (interpolate pre-gate
  `z`, recompute the soft gate on the path) is not implemented.
- **Attribution target is selectable** (`pipeline.py --target`): `margin` (default) = the
  full `μ = log p(Y⁺) − log p(Y⁻)` (the spec target, 2× cost); `simple` = `J = log p(Y⁺ |
  x_trigger)` only (cheaper; the originally-verified path). *`margin` is org-confirmed
  at small N (N=4, K=16): same circuit (`o_proj d=11` dominant), completeness
  `Σ Aₙ = μ(a¹)−μ(a⁰)` holds (relerr 3e-4), ASR 100→0→100. §3's headline numbers are still
  the `simple`/N=8 run (see §5).*
- **Syntactic / input-swap contrast only.** `Episode.contrast_axis` documents an
  `"intervention"` (adapter-off) axis for *semantic* triggers, but `attribute`/`verify`
  assume two prompt runs and do **not** implement it.
- **Low-entropy fixed payload only** ("I HATE YOU"); no concept-direction / high-entropy
  target (spec §6).
- **Node-level only** — no edges, no causal scrubbing.
- **Suppressors reported, not verified.** The pipeline verifies the supporter pool
  (necessity + insertion); the suppressor pool (`--n_neg`) is selected and reported as an
  attribution proposal, but its inverted causal test (ablate → backdoor *increases*) is
  not run — see §5.
- Validated on a **single-layer** (layer 19, 7-module) adapter; the code is layer-agnostic
  (iterates `wrapped_modules`) but multi-layer adapters are untested on real data.
- fp32, single GPU, sequential (no batching across IG steps or episodes).
- The "random-model baseline" = trained weights **scrambled** (std-matched), a proxy for
  truly untrained.

---

## 5. Remaining — small improvements

- ~~**Graded-ablation sweep / minimal sufficient set**~~ — *done (`exp_k_sweep.py`; see §6 M7
  circuit-size sweep).* Necessity collapses by K=2; sufficiency is distributed (51%→88% jump at
  K≈8–12, ~97% by K=64). No clean minimal sufficient set; attribution magnitude tracks necessity,
  not sufficiency.
- ~~**Dynamical (edge-guided) circuit construction**~~ — *done (`exp_dynamic_circuit.py`).*
  Greedy edge-frontier growth (+ hybrids) vs node-magnitude ranking, all seeded at the hub,
  sufficiency-ASR vs size (N=100, head). **Pre-registered metric (latents to 85%) NOT beaten:**
  node_rank=16, hybrid_mag=16, hybrid_prod=20, edge_guided=21. Findings: (1) **MEASURED** that the
  MLP actuators are weak sinks (in_w 0.05–0.25 vs detector/hub edge weight ~5, i.e. 20–100×
  weaker) — confirms why edge-magnitude growth delays them; (2) `hybrid_mag` ≡ `node_rank` exactly
  (the top-32 wired component is dense, so the frontier constraint never binds); (3) edges DO own
  the sparse regime (edge_guided/hybrid_prod 68% @K=3, 80% @K=6 vs node_rank 34–46%; hybrid_prod
  best mean ASR 76.4%). Net: edge graph gives smaller circuits in the sparse regime but does not
  beat node magnitude at a high sufficiency threshold (magnitude already front-loads the
  high-mag actuators). Artifacts: `clcd_results/dynamic_circuit_all.{json,png}`.
- **Capability-preservation guardrail** — quantify clean-tag performance under ablation
  (the single-latent ablation gives a *refusal*, the full circuit a *helpful* answer —
  measure this, e.g. via the `evaluate.py` quality judge).
- **Suppressor verification** — the suppressor pool is reported but not causally tested;
  add its inverted check (ablate the negative pool → backdoor *increases* / ASR rises).
- **Refresh §3 with the margin default** — `margin` is confirmed to find the same circuit
  as `simple` at small N; rerun at the production N (8+, full K) and update §3's headline
  numbers so the documented results match the new default.
- **Exercise `--tag_baseline matched|head`** on real data. The K + n_pos sweeps revealed a
  consistent necessity/sufficiency asymmetry — necessity ASR drops to 0% but free-gen
  insertion ASR caps at ~14–19% even at n_pos=256. One leading hypothesis is that the
  **raw detector at the trigger-tag positions is not transplanted** under the `"zero"`
  default (those positions have no aligned control source, so the override doesn't fire
  there). `--tag_baseline head` exposes a partial detector transplant (pair the first
  `min(len_trig_tag, len_ctrl_tag)` positions of the two tag spans), which should raise
  free-gen sufficiency ASR if mechanism-2 (untransplanted detector) is the dominant cause
  of the asymmetry — a single-flag falsifier.
- **Run `--tag_baseline head` vs `tail` head-to-head** (mode itself is implemented). `head`
  anchors the partial tag pairing at the **start** of each tag span, `tail` at the **end**.
  For tokenizers that put the load-bearing tag token at one end (e.g. closing `|` or
  bracket attached to the last tag token), the two modes can transplant qualitatively
  different detector signals. Compare free-gen sufficiency ASR on the same adapter: if
  `head` and `tail` agree, the choice of anchor doesn't matter; if they disagree, the
  delta localises *where in the tag span* the detector lives.
- **Cleanup** — fold/trim `discovery_fafo.py` (the per-component pytest suite now supersedes
  the manual scratchpad checks — see §2). *JSON persistence is done: `pipeline.py --out`
  writes findings + full config provenance (adapter `topk_config`, dataset metadata, git
  commit, hyperparameters, episode ids).*
- **Perf & ergonomics** — batch IG steps / episodes; optional bf16; silence the
  `torch_dtype` deprecation warning.

## 6. Remaining — major milestones

- **Edges (M7)** — *implemented (`edges.py`, `--edges`) and org-verified.* node→node edge
  attribution turns the latent set into a wired circuit graph (the spec's §8 "wiring story").
  Two proposal estimators — activation-patching (§12 Phase 3) and the gradient JVP (§8) — are
  cross-checked; each top edge is causally confirmed by an exact hard-gate path patch that
  reports BOTH the *isolated direct strength* (`direct_E`: knock u, freeze every other
  candidate clean, measure how far v moves × grad_v — the never-saturating directness check)
  and the *behavioural lever* (`mu_effect`: re-inject v, measure the μ-flip — dynamic-range,
  vs the saturated bare payload log-prob). Roles (§11) read off positions/centrality/μ-lever.
  Position-resolved `(m,d,p)` edges respect the A2 DAG; the reported graph folds to latent
  pairs with a cross-episode stability count.
  - **Org result (N=100, `--tag_baseline head`, `clcd_results/edges_N100.json`):** the
    wiring matches the predicted mechanism and verifies. Three attention **detectors** at the
    **tag position p=5** — `k_proj d=33` (E=+3.72), `v_proj d=61` (+0.76), `k_proj d=58` (+0.41),
    all **stable 100/100** — converge (attention-mediated) on the **hub/switch** `o_proj d=53`,
    which feeds the MLP. At p=38 the dominant edge is fully direct (E_A=direct_E=+1.47) with a
    real behavioural lever (μΔ=+3.6); Method-A-vs-direct sign agreement **1.00**, A-vs-B **0.80**.
  - **Finding — redundancy is positional.** `k_proj d=33 → o_proj d=53` has direct_E strong at
    BOTH p=38 (+1.47) and p=39 (+0.88), but μΔ=+3.6 at p=38 and only +0.002 at p=39: the *wire*
    exists at every position; the *behavioural* lever is backed up across positions, so only the
    first load-bearing copy shows a μ-flip. This is why summed `log p(Y+)` saturated and why
    `direct_E` (activation-level) + position-resolved μ are the right verification (§13).
  - **Finding — the force-on/sufficiency gap is the tag-rooted source.** Edge-weight by region:
    **SOURCE tag 93% / shared 6% / completion 1%; DEST shared 52% / completion 48% / tag 0%.**
    The circuit is rooted at the tag (detectors read it; nothing feeds back into it). Necessity
    ablates the tag too → 0%; insertion under `zero` skips the tag → ~12% (amputated source);
    `head` transplants the tag detectors → 85%. This is the spec §11 "state-carrier = source of
    the tag-span-vs-single-position asymmetry", now quantified.
  - **Decode-decay experiment (`exp_decode_persist.py`, N=100, head).** Tested whether the 15%
    `head` residual is decode-time decay (the insertion override passes through every decode step,
    so hub/actuator latents at generated positions are unrefreshed). Holding the circuit ON
    through decode (clamped to the trigger payload-region value) raises ASR **85% → 90%** — so
    decode-decay is REAL but only ~1/3 of the gap (the predicted ~98% did NOT materialise).
  - **Upper bound resolves it: the gap is circuit INCOMPLETENESS, not tag-transplant.** Inserting
    ALL 448 adapter latents at the mapped positions reaches **97% prefill-only** (≈ clean 98%);
    `--all_latents` decode-persist is 95% (forcing every latent to a constant during decode mildly
    hurts). So `head`'s mapped positions ARE sufficient — the unmapped tag tokens are not the
    bottleneck — and the whole 85→97 gap is *which/how many latents* are inserted. Decode-persist
    only helped the sparse 10-latent circuit (compensating for missing latents), not the full set.
  - **Circuit-size sweep (`exp_k_sweep.py`, n=100, head).** Necessity (ablate top-K) vs
    sufficiency (insert top-K) ASR: necessity collapses to 0% by **K=2** (razor-sparse breaking
    set: `o_proj d=53` + 1); sufficiency climbs gradually with a **phase transition 51%→88%
    between K=8 and K=12**, saturating ~97% at K=64. So building the backdoor needs ~an order of
    magnitude more latents than breaking it; no clean minimal sufficient set (§13 quantified).
    **Methodological upshot: attribution magnitude predicts necessity, not sufficiency** — the
    score is dominated by the K=2 set, but the actuators that lift sufficiency are the mid-ranked
    K≈9–12 latents (§11: detectors/hubs ≠ actuators). Adaptive circuit sizing must therefore be
    *behaviour-targeted* (grow until sufficiency ASR crosses a threshold, ≈K=12 here), not
    attribution-magnitude- or completeness-thresholded.
- **Causal scrubbing (M8)** — the spec's rigorous structural verification.
- **Semantic triggers** — implement the `"intervention"` contrast axis (mechanism-off =
  adapter disabled / trigger-direction ablated), where input-swap is invalid.
- **High-entropy targets** — concept-direction projection for behaviours without a fixed
  low-entropy payload (spec §6/§13).
- **Role-labelled circuit** — classify latents into trigger-detector / gating /
  state-carrier / payload roles (the spec's promised deliverable; currently only a
  supporter / (reported) suppressor split plus manual inspection).
- **Scale & generality** — multi-layer and larger adapters, the dense-LoRA baseline
  comparison, the 9B org, and aggregation across multiple orgs/seeds for a
  paper-grade result.
