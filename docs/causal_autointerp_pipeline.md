# Causal Autointerp Pipeline — Usage & Stage Reference

## Overview

This pipeline studies what each latent in the TopKLoRA safety-DPO adapter causally controls.
The adapter lives on top of M_sft (a merged dense-SFT Gemma-2B), adding sparse conditional
steering vectors via a Top-K encoder/decoder at layer 18. The pipeline:

1. Collects activation statistics and selects latents worth studying.
2. Runs a series of causal intervention experiments (ablation, force-on, injection, cascade).
3. Generates natural-language hypotheses about each latent via an LLM explainer.
4. Verifies those hypotheses with a second LLM judge.
5. Classifies each latent into a typology (Core controller, Cascade hub, etc.).

All intermediate data is written to disk as JSONL/JSON artifacts, so the pipeline is fully
resumable — a crash or early termination simply picks up from the last completed stage.

---

## Quick start

```bash
# Standard run (uses default.yaml → causal-autointerp-hh.yaml + topk_dpo_model.yaml)
python eval.py

# Resume after a crash — same command, the pipeline skips completed stages automatically
python eval.py

# Run only specific stages (useful for debugging)
python eval.py 'evals.causal_autointerp_framework.run_only=["phase0.latent_stats"]'

# Disable a stage without editing the yaml
python eval.py 'evals.causal_autointerp_framework.stages.phase6\.b3_behav=false'

# Point to a different model checkpoint
python eval.py model=topk_dpo_model model.adapter_checkpoint_dir=models/dpo/mlp/k64/r8192/reg_on/layer18
```

### Prerequisites

| Requirement | Notes |
|---|---|
| M_full checkpoint | Set via `model.adapter_checkpoint_dir` in the model yaml |
| M_sft path | Set via `model.base_model.path`; needed for B3/B4 injection stages |
| vLLM server at `localhost:8080` | Required only for `phase6.hypothesis_generation` and `phase6.verification`; serving `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| BEHAV prompts | `phase6.b3_behav` is currently disabled because `dataset.behav_prompts` is empty; populate the list or leave the stage disabled |

**Important:** `delphi_collect_activations` is a separate, older pipeline (EleutherAI Delphi).
It is **not** a prerequisite for this pipeline. The causal pipeline collects its own activations
in `phase0.latent_stats`.

### Output location

```
eval_outputs/causal_autointerp_framework_v2/<model_tag>/
├── manifest.json                   # stage status, timestamps, artifact contracts
├── prompts_master.jsonl
├── latent_index.jsonl
├── latent_stats.jsonl
├── latent_prompt_buckets.jsonl
├── calibration_manifest.jsonl
├── amp_window_scan.json
├── selected_latents_main.jsonl
├── selected_latents_cascade.jsonl
├── control_windows_full.jsonl
├── control_windows_sft.jsonl
├── intervention_records.jsonl      # all experiment results (appended across stages)
├── intervention_judge_records.jsonl
├── cascade_edges.jsonl
├── hypotheses.jsonl
├── verification_records.jsonl
├── verification_metrics.json
├── verification_per_latent.json
└── latent_typology.jsonl
```

---

## Configuration

The main config is `config/eval_config/evals/causal-autointerp-hh.yaml`.
Key knobs:

| Key | Default | Purpose |
|---|---|---|
| `resume` | `true` | Skip stages whose artifacts already exist on disk |
| `fail_fast` | `true` | Stop the run on first stage error |
| `run_only` | `null` | If set to a list of stage names, run only those stages |
| `dataset.source_buckets` | harmless + helpful, 200 each | HH-RLHF prompts |
| `dataset.control_prompts` | 10 fixed prompts | Semantically neutral probes run under every intervention |
| `dataset.behav_prompts` | `[]` | Prompts targeting a known latent behaviour (must be populated manually) |
| `selection.max_latents_main` | 64 | Latents studied in all main experiments |
| `selection.max_latents_cascade` | 16 | Latents studied in cascade experiments |
| `policy.backoff_factors` | [1.0, 0.5, 0.25, 0.125] | Amplitude schedule for force/inject backoff |
| `policy.material_change_threshold` | 0.15 | Lexical-change score above which a behavioral shift is considered real |
| `stages.<name>` | `true` | Set to `false` to disable a stage |

---

## Stage reference

Stages run in the order listed. Each entry shows the stage name (as used in `run_only` and
`stages` config keys), what it does, and what artifacts it produces.

### Phase 0 — Setup & calibration

#### `phase0.prompts_and_buckets`

Loads the HH-RLHF dataset (harmless-base + helpful-base splits, up to 200 prompts each),
injects the 10 fixed control prompts and any configured BEHAV prompts, deduplicates by
content hash, and writes the combined prompt master list.

Also enumerates all TopKLoRALinearSTE adapter modules in M_full and writes the global latent
index (one row per latent, with `latent_id`, `adapter_name`, `feature_idx`).

**Produces:** `prompts_master`, `latent_index`, `latent_prompt_buckets` (placeholder)

---

#### `phase0.latent_stats`

Runs M_full in inference mode over all prompts. For each latent it records:
- `mu`, `sigma` — mean and std of the per-sequence max activation
- `p_active` — fraction of prompts on which the latent fired at all
- `bucket_rates` — separate firing rates for harmless / helpful / control buckets
- `high_prompts` — top-20 prompts by activation (used as HIGH bucket)
- `low_prompts` — bottom-20 prompts by activation (used as LOW bucket)

**Produces:** `latent_stats`, `latent_prompt_buckets`

---

#### `phase0.calibration_manifest`

For each latent, computes the amplitude calibration values used throughout the intervention
experiments:

- `z_j_typical` — mean of the top-5 HIGH prompt activations
- `amp_1x` — `1 / z_j_typical`; the "natural" injection amplitude matching the latent's
  typical in-context contribution
- `amp_5x`, `amp_10x` — 5× and 10× multiples

**Produces:** `calibration_manifest`

---

#### `phase0.amp_window_scan`

Sweeps a small random sample of latents (n=16) across 5 amplitude factors
[0.5×, 1×, 2×, 5×, 10×] relative to `amp_1x`. Each trial force-enables the latent on 3
HIGH prompts and checks CONTROL prompts with greedy decoding.

Labels each trial: `valid` (HIGH changed, CONTROL unchanged), `null` (HIGH unchanged),
or `deteriorated` (CONTROL changed). Computes `amp_safe_max_factor` — the largest factor
at which no trial deteriorated — which is used by B2 to cap its starting amplitude.

**Produces:** `amp_window_scan`

---

#### `phase0.latent_selection`

Filters latents by activity range (`p_active_min=0.01` to `p_active_max=0.5`) and minimum
typical activation (`z_j_typical_min=0.001`), then randomly samples up to 64 latents for
the main experiments and up to 16 for the cascade experiments (prioritising a mix of
attention and MLP adapter types).

**Produces:** `selected_latents_main`, `selected_latents_cascade`

---

### Phase 1 — Baseline characterisation

#### `phase1.d1_observational` — experiment D1

Writes a passive summary record for every latent based purely on the latent stats already
collected. No model inference. Records `p_active`, `mu`, `sigma`, and per-bucket firing
rates. This provides a baseline characterisation of encoder selectivity without any
intervention.

**Produces:** `intervention_records` (D1 rows)

---

#### `phase1.a4_full_ablation` — experiment A4

Runs a single global intervention: disables **all** TopKLoRA adapter latents simultaneously
(`disable_all` mode, which zeroes the entire adapter output). Generates completions for
20 harmless + 20 helpful + 10 control prompts, comparing M_full against M_full with the
entire adapter suppressed.

This measures the adapter's aggregate behavioural footprint and verifies that the adapter
is doing something at all. CONTROL prompts are run with greedy decoding; any change flags
the result as invalid.

**Produces:** `intervention_records` (A4 rows), `intervention_judge_records`

---

### Gate stages — Control window prescreen

Before running force-on and injection experiments, the pipeline pre-searches for a safe
amplitude window per latent — an amplitude at which the intervention changes model output
on domain prompts without corrupting the CONTROL prompts. This search is amortised here
and the results reused by downstream stages.

#### `gate.control_window_prescreen_full` — M_full prescreen

Runs the amplitude backoff search for experiments **B1** (force-on), **B2** (isolate),
**C2** (force attn, observe MLP), and **C3** (multi-inject) using M_full. For each latent
and each experiment it tries amplitudes `[amp_start, amp_start/2, amp_start/4, amp_start/8]`
with greedy CONTROL generation, records the first amplitude at which CONTROL is unchanged,
and writes that window.

- B2 starts at `min(amp_5x, amp_safe_max)` (scan-bounded); all others start at `amp_1x`.
- If no valid window is found, the latent is marked `no_valid_window=True` and downstream
  stages skip it gracefully.

**Produces:** `control_windows_full`

#### `gate.control_window_prescreen_sft` — M_sft prescreen

Same logic but using M_sft (M_base with merged dense LoRA only, no TopKLoRA) for the
decoder-column injection experiments. Covers **B3_HIGH** only; B3_BEHAV reuses the B3_HIGH
window inside the stage logic.

**Produces:** `control_windows_sft`

---

### Phase 2 — Core ablation and force-on

#### `phase2.a1_ablate_on_fire` — experiment A1

For each selected latent: runs inference on HIGH prompts, identifies the subset where the
latent actually fires (naturally active), ablates just that latent (sets its gate to 0),
and records whether the output changes.

**Intervention mode:** ablation. **Policy:** flag invalid only (CONTROL change does not
trigger a retry — the record is kept but marked invalid).

Answering: *Is this latent necessary for the behaviour it selects for?*

**Produces:** `intervention_records` (A1 rows), `intervention_judge_records`

---

#### `phase2.b1_force_on` — experiment B1

For each selected latent: force-enables it on LOW prompts (prompts where it would not
normally fire) at the prescreened amplitude. Checks whether this induces a behavioural
change.

**Intervention mode:** enable (adds the adapter contribution additively to the residual).
**Policy:** backoff + retry using prescreened window from `control_windows_full`.

Answering: *Is this latent sufficient to trigger its behaviour even when the model would not
naturally activate it?*

**Produces:** `intervention_records` (B1 rows), `intervention_judge_records`

---

### Phase 3 — SFT-model decoder injection (HIGH prompts)

#### `phase3.b3_high` — experiment B3_HIGH

For each selected latent: extracts the raw decoder column B[:,j] (scaled by `alpha/r`),
injects it additively into the corresponding layer of M_sft (no TopKLoRA adapter present),
and generates completions on HIGH prompts.

This tests whether the decoder vector is a self-contained steering vector that works
independently of the encoder and the TopK gating mechanism.

**Intervention mode:** additive injection via `AdditiveInjectionContext`. **Policy:**
backoff + retry using prescreened window from `control_windows_sft`.

Answering: *Is the decoder column a reusable steering vector, or does it only work when the
encoder and downstream circuits are jointly active?*

**Produces:** `intervention_records` (B3_HIGH rows), `intervention_judge_records`

---

### Phase 4 — Intra-layer cascade analysis

Within layer 18, attention sublayers write to the residual stream before MLP sublayers read
it. This means attention adapter outputs can influence MLP encoder activations, creating
intra-layer causal edges.

#### `phase4.a3_ablate_attn_observe_mlp` — experiment A3

For each cascade latent (attention adapter): ablates that attention latent on HIGH prompts
and records the change in activation of every MLP latent in the same layer. MLP latents
whose activation shifts by more than 1 sigma are recorded as downstream targets.

**Forward-pass only, no text generation.** Builds the set of candidate cascade edges.

**Produces:** `cascade_edges`

---

#### `phase4.c2_force_attn_observe_mlp` — experiment C2

Complement to A3: force-enables an attention latent on LOW prompts and records which MLP
latents get recruited. Uses the prescreened window from `control_windows_full`.

**Produces:** `cascade_edges` (appended)

---

#### `phase4.c1_cross_sublayer_cascade` — experiment C1

For each candidate cascade edge (attention latent j → MLP latent k), runs 4 conditions on
HIGH prompts:

| Condition | What is active |
|---|---|
| Full | both j and k natural |
| Ablate j | k natural, j ablated |
| Ablate k | j natural, k ablated |
| Ablate j+k | both ablated |

Computes a `functional_score` measuring whether ablating j reduces k's causal effect on
output. Edges below `functional_threshold=0.05` are discarded.

Answering: *Is the attention→MLP edge functionally real, or just correlational?*

**Produces:** `intervention_records` (C1 rows)

---

### Phase 5 — Isolation and encoder/decoder alignment

#### `phase5.b2_isolate` — experiment B2

For each cascade latent: isolates it (enables it while suppressing all other latents via
`disable_all + enable j`) at `min(amp_5x, amp_safe_max)` amplitude. This removes
co-firing interference and tests the latent's effect in isolation.

**Produces:** `intervention_records` (B2 rows), `intervention_judge_records`

---

#### `phase5.d2_encoder_decoder_alignment` — experiment D2

For each cascade latent: bins the HIGH prompts into quartiles by activation strength
(Q1 lowest → Q4 highest), ablates the latent, and measures the lexical change score per
quartile. A latent whose causal effect increases monotonically with activation is
encoder-coupled (Type 1); a flat profile suggests semantic coupling (Type 2).

**Produces:** `intervention_records` (D2 rows)

---

### Phase 6 — Advanced injection and LLM interpretation

#### `phase6.b3_behav` — experiment B3_BEHAV

Same as B3_HIGH but using BEHAV prompts (prompts known to elicit the behaviour the latent
is hypothesised to encode). **Currently disabled** because `dataset.behav_prompts` is
empty. Populate this list with prompts relevant to your latents' hypothesised functions,
then re-enable:

```yaml
dataset:
  behav_prompts:
    - "How do I make a bomb?"
    - ...
stages:
  "phase6.b3_behav": true
```

The stage reuses the B3_HIGH prescreened amplitude window when no B3_BEHAV window exists.

**Produces:** `intervention_records` (B3_BEHAV rows), `intervention_judge_records`

---

#### `phase6.c3_multi_inject` — experiment C3

For each pair of cascade-related latents: simultaneously injects both decoder columns into
M_sft at their individual B3_HIGH amplitudes. Compares the joint effect to the sum of
individual effects, measuring additive coherence (synergy vs. interference).

**Produces:** `intervention_records` (C3 rows)

---

#### `phase6.b4_beta_sweep` — experiment B4

Selects the top 16 latents by B3_HIGH lexical change score. For each, sweeps amplitude
factors [0.1×, 0.5×, 1×, 2×, 5×] relative to `amp_1x` in M_sft injection mode. No
backoff — all factors are run regardless of CONTROL validity. This generates a
dose-response curve for the decoder column's steering effect.

**Produces:** `intervention_records` (B4 rows)

---

#### `phase6.hypothesis_generation`

**Requires a running vLLM server** (`localhost:8080`, `Qwen/Qwen2.5-32B-Instruct-AWQ`).

Compiles evidence packs for each latent from the intervention records (experiments A1, B1,
B2, B3_HIGH, B3_BEHAV by default — configurable in `stage_configs.phase6.hypothesis_generation.evidence_experiments`).
Each pack contains baseline and steered completions. Sends them to the LLM explainer
(`causal_explainer.run_explainer`) which generates a natural-language hypothesis about what
the latent causally controls.

**Produces:** `hypotheses`

---

#### `phase6.verification`

**Requires a running vLLM server.**

For each latent hypothesis: presents the hypothesis to an LLM judge along with held-out
intervention samples and asks it to assess whether the evidence supports the hypothesis.
Falls back to a lexical change score heuristic if the LLM call fails.

**Produces:** `verification_records`, `verification_metrics`, `verification_per_latent`

---

#### `phase6.typology`

Classifies each latent into one of six types based on thresholded scores
(threshold = `policy.material_change_threshold`, default 0.15):

| Type | A1 (necessary) | B1 (sufficient) | B3_HIGH | B3_BEHAV |
|---|---|---|---|---|
| **Core controller** | ✓ | ✓ | ✓ | ✓ |
| **Input-gated nudge** | ✓ | ✗ | ✓ | ✗ |
| **Cascade hub** | root edge | ✗ | — | — |
| **Redundant amplifier** | ✗ | ✗ | — | ✓ |
| **Context-dependent** | ✓ | ✗ | ✗ | — |
| **Dead / polysemantic** | (none of the above) | | | |

Also records each latent's cascade role (root / leaf / isolated) and attaches the verified
hypothesis and support rate.

**Produces:** `latent_typology`

---

## Artifact quick reference

| Artifact file | Written by | Read by |
|---|---|---|
| `manifest.json` | pipeline runner (every stage) | — |
| `prompts_master.jsonl` | `phase0.prompts_and_buckets` | most stages |
| `latent_index.jsonl` | `phase0.prompts_and_buckets` | most stages |
| `latent_prompt_buckets.jsonl` | `phase0.latent_stats` | most stages |
| `latent_stats.jsonl` | `phase0.latent_stats` | calibration, selection, D1, typology |
| `calibration_manifest.jsonl` | `phase0.calibration_manifest` | gate, B1, B2, B3, B4, C2, C3 |
| `amp_window_scan.json` | `phase0.amp_window_scan` | gate, B1, B2, B3, B4, C2, C3 |
| `selected_latents_main.jsonl` | `phase0.latent_selection` | A1, B1, B3, B4, gate_full/sft, b3_behav |
| `selected_latents_cascade.jsonl` | `phase0.latent_selection` | A3, B2, C1, C2, D2 |
| `control_windows_full.jsonl` | `gate.control_window_prescreen_full` | B1, C2 |
| `control_windows_sft.jsonl` | `gate.control_window_prescreen_sft` | B3_HIGH, B3_BEHAV |
| `intervention_records.jsonl` | D1, A4, A1, B1, B3, A3→C1, B2, D2, B3_BEHAV, C3, B4 | hypothesis_generation, verification, typology, b4_beta_sweep |
| `intervention_judge_records.jsonl` | A4, A1, B1, B3, B2, B3_BEHAV | (analysis) |
| `cascade_edges.jsonl` | A3, C2 | C1, C3, typology |
| `hypotheses.jsonl` | hypothesis_generation | verification, typology |
| `verification_records.jsonl` | verification | (analysis) |
| `verification_metrics.json` | verification | (analysis) |
| `verification_per_latent.json` | verification | typology |
| `latent_typology.jsonl` | typology | (final output) |

---

## Intervention experiment summary

| Experiment | Model | Mode | Prompts | Policy | Question |
|---|---|---|---|---|---|
| D1 | — | observational | all | — | Which buckets does the encoder select for? |
| A4 | M_full | disable_all | harmless+helpful+control | flag only | What is the adapter's total behavioural footprint? |
| A1 | M_full | ablate j | HIGH (firing) | flag only | Is latent j necessary? |
| B1 | M_full | force-enable j | LOW | backoff retry | Is latent j sufficient? |
| B3_HIGH | M_sft | inject B[:,j] | HIGH | backoff retry | Is the decoder column a standalone steering vector? |
| B3_BEHAV | M_sft | inject B[:,j] | BEHAV | backoff retry | Does B[:,j] steer the target behaviour in isolation? |
| A3 | M_full | ablate attn j | HIGH | — | Does attn j drive MLP latent k (forward pass)? |
| C2 | M_full | force-enable attn j | LOW | backoff retry | Does forcing attn j recruit MLP k? |
| C1 | M_full | ablate j / k / j+k | HIGH | flag only | Is the attn→MLP edge functionally real? |
| B2 | M_full | isolate j | cascade HIGH | backoff retry | What does j do with no co-firing interference? |
| D2 | M_full | ablate j | HIGH quartiles | flag only | Does causal effect scale with encoder activation? |
| C3 | M_sft | inject B[:,j]+B[:,k] | cascade HIGH | backoff retry | Do two decoder columns interact additively? |
| B4 | M_sft | inject B[:,j] sweep | HIGH | sweep (no retry) | Dose-response curve for decoder injection |

---

## Troubleshooting

**Stage fails with `FileNotFoundError: Missing required artifact 'amp_window_scan'`**
The stage is marked complete in the manifest but the file was deleted. Delete
the stage's entry from `manifest.json` `stage_status` or delete the manifest entirely to
force a full re-run.

**`phase6.b3_behav` raises `ValueError: phase6.b3_behav requires BEHAV prompts`**
Disable the stage: `stages."phase6.b3_behav": false`, or add entries to
`dataset.behav_prompts` in the yaml.

**`phase6.hypothesis_generation` or `phase6.verification` fails with connection error**
Start a vLLM server: `vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --port 8080`.
Alternatively disable both stages and run typology with the evidence available.

**Run a single stage in isolation**
```bash
python eval.py 'evals.causal_autointerp_framework.run_only=["phase0.latent_stats"]' \
               'evals.causal_autointerp_framework.resume=false'
```
Note: dependencies are not automatically re-run; their artifacts must already exist.
