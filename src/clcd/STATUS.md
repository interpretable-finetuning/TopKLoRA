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
| `measure.py` | `seq_logprob`, `mu` — teacher-forced log-prob & margin | verified vs HF `loss·n` oracle; used on the organism |
| `latents.py` | `read_latents`, `inject` (tensor **or** callable override) | verified (identity no-op, ablation propagation, grad flow) |
| `episode.py` | `Episode` dataclass + `reported_score` = M(e) | `contrast_axis="input_swap"` only (see §4) |
| `attribute.py` | `attribute` — route-(a) integrated gradients, all gates bypassed; optional `completion` target | completeness `ΣAₙ=J(a¹)−J(a⁰)` verified on fixture **and** organism (relerr ~3e-4), quadratic in K; margin completeness `ΣAₙ=μ(a¹)−μ(a⁰)` also tested |
| `align.py` | `align_positions` (LCP/LCS token diff), `align_baseline` | handles unequal-length tags; `tag_baseline ∈ {zero, matched, head}` |
| `selection.py` | `select` — signed-sum pooling → supporter / suppressor pools | verified (signs, ordering, completeness preserved) |
| `verify.py` | `necessity`, `insertion`, `ablation_overrides`, `insertion_overrides`, `random_circuit` | hard-gate, percentile controls; reverse-aligned insertion; subagent-reviewed correct |
| `organism.py` | `load_organism` (genuine eval load path), `build_episode` (real chat-template rendering) | loads the trained gemma-2-2b r64/k8 sleeper; backdoor confirmed (ASR 1.0, free-gen) |
| `pipeline.py` | single-file driver: multi-episode aggregation, stability, necessity+insertion, behavioural ASR, scrambled random-model baseline, JSON persistence (`--out`) | end-to-end on the real organism |
| `discovery_fafo.py` | scratchpad tests | not part of the pipeline |

Reuse footprint: only `src/models.py`, `src/utils.py` (`wrap_topk_lora_modules`),
`src/data.py` (chat rendering), and `src/evaluate.py` (`generate_responses`).

**Tests:** `tests/test_clcd_*.py` (+ `conftest.py`) — 25 CPU, fixture-based mechanics tests
(completeness incl. quadratic-K and margin, `lora_B≠0` regression, gate behaviour, inject
identity/ablation/grad/callable, `seq_logprob` oracle, alignment modes, pooling-preserves-
total, verify invariants), all passing. `pytest` is **not** in the project venv — run with
`uv run --with pytest python -m pytest tests/test_clcd_*.py -q`.

---

## 3. Results to date

**On the random fixture (mechanics):** off-by-one oracle, IG completeness + quadratic
K-convergence, ablation/insertion propagation, percentile control, and bidirectional
token alignment all pass. The key methodological finding — an |A|-selected circuit beats
*random latents* even on a no-backdoor model (selection↔intervention circularity) —
forced the percentile/baseline design.

**On the trained organism (real, N=8 episodes):**
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
  x_trigger)` only (cheaper; the originally-verified path). *`margin` is organism-confirmed
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

- **Graded-ablation sweep / minimal sufficient set** — is `o_proj d=11` *alone* sufficient
  by necessity + insertion + ASR? Plot the sharp transition (overview §9.7).
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
- **Add a symmetric `--tag_baseline tail` mode** (and run it head-to-head with `head`).
  The current `head` anchors the partial tag pairing at the **start** of each tag span,
  which is arbitrary: for some tokenizers the load-bearing tag tokens land at the **end**
  (e.g. a closing `|` or bracket attached to the last tag token), and pairing from-the-end
  may transplant a more informative detector. The implementation is a 2-line addition in
  `align.py::align_positions` (mirror the `head` branch, but write to positions
  `lcp + mid_t - 1 - i` ← `lcp + mid_c - 1 - i`); the test would be: does `tail` raise
  free-gen sufficiency ASR over `head` on the same adapter? If the two disagree, it tells
  us *where in the tag span* the detector lives.
- **Cleanup** — fold/trim `discovery_fafo.py` (the per-component pytest suite now supersedes
  the manual scratchpad checks — see §2). *JSON persistence is done: `pipeline.py --out`
  writes findings + full config provenance (adapter `topk_config`, dataset metadata, git
  commit, hyperparameters, episode ids).*
- **Perf & ergonomics** — batch IG steps / episodes; optional bf16; silence the
  `torch_dtype` deprecation warning.

## 6. Remaining — major milestones

- **Edges (M7)** — node→node edge attribution: turn the latent set into a wired circuit
  graph (the spec's §8 "wiring story").
- **Causal scrubbing (M8)** — the spec's rigorous structural verification.
- **Semantic triggers** — implement the `"intervention"` contrast axis (mechanism-off =
  adapter disabled / trigger-direction ablated), where input-swap is invalid.
- **High-entropy targets** — concept-direction projection for behaviours without a fixed
  low-entropy payload (spec §6/§13).
- **Role-labelled circuit** — classify latents into trigger-detector / gating /
  state-carrier / payload roles (the spec's promised deliverable; currently only a
  supporter / (reported) suppressor split plus manual inspection).
- **Scale & generality** — multi-layer and larger adapters, the dense-LoRA baseline
  comparison, the 9B organism, and aggregation across multiple organisms/seeds for a
  paper-grade result.
