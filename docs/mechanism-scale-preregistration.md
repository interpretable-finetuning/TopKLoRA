# Mechanism readouts and latent labels at scale (pre-registration)

Written 2026-11-11 ~02:10 UTC by Claude, overnight, under Marek's brief (project chat, 01:51 and 01:53
UTC): continue past the 18 blind latents to a stratified draw and other seeds; iterate on a small number
of latents until the protocol gives a *useful* label; look qualitatively at every batch and adjust; then
run the protocol on many latents. Marek is asleep, so choices below are Claude's and are logged in the
captain's log with reasons. Nothing in section 2 has been drawn when this file is committed.

## 0. Amendments

**2026-11-11 ~02:40 UTC, amendment 1: after a code review of the scale run, after the draw, before any
readout of the 315 exists.** (1) S0281-S0285 sit in `layers.25.mlp.down_proj`, which has no downstream
adapted module, so M3 is undefined for them: they get no mechanism readout, are excluded from the
section 4 comparison (n = 310, 36 in the circuit), and are labelled from input-side evidence and the
logit lens only. (2) A latent silent in all 2,000 screened prompts is recorded, gets no mechanism readout
and is excluded from section 4; per-latent failures of any gate are recorded and the run continues.
(3) **The input-side cache is 10M tokens, not 50M**: the first 1,220 batches of 32 windows of the same
shuffled corpus (corpus seed 20261110), because 315 latents spread over all 63 modules were estimated
at 4-8 GPU-hours for 50M tokens, which would leave no time to label them tonight. 10M tokens still give
each latent of median firing rate over a million activations. (4) **Primary test changed before any
result:** circuit membership is skewed across module types (10 of 37 circuit latents are o_proj) and the
edge share depends on where a module sits, so the primary test is the pooled mid-rank Mann-Whitney U
with its p-value from 100,000 permutations of the circuit labels **within each module** (the draw's
strata), seed 20261111. The unstratified version of section 4 is reported as secondary.

## 1. What exists already

`docs/mechanism-pilot-preregistration.md` (amendments 1-5): the mechanism readouts (M1 weights, M2
own-layer attention, M3 downstream latent edges with Bonferroni, M4 per-head mediation) on the
Qwen2.5-7B `l17_25` seed-42 org; results for A-E and the 18 blind latents. The batched implementation
(`mech_batch`) is ~30x faster; on six latents run both ways it reproduced the Bonferroni edge counts
within tolerance (0/0, 0/0, 2/4, 16/13, 69/67, 116/108) but not the pre-registered top-10 overlap
(3-5 of 10 shared, required 7). The overlap criterion was too strict for lists dominated by small,
batch-shape-sensitive changes (bf16 rounding flips TopK gates downstream).

## 2. The stratified draw

- Pool: the 6,300 latents of the org, minus the 23 already drawn (A-E and the 18 blind latents).
- **5 latents uniformly from every one of the 63 adapted modules = 315 latents**, one
  `random.Random(20261111)` consumed module by module in sorted module order (`draw_stratified` in
  `src/autointerp/latent_inspection.py`, pinned by a test; `scripts/latent_causal.py draw_scale`, which
  refuses to run unless this file is committed and identical on the remote branch). Labels S0001-S0315.
- Circuit membership (the 800-latent verified circuit) is recorded, not stratified on.

## 3. Readouts on the 315 latents (GPU)

- **Screen and prompt sets**: as in the mechanism pilot (2,000 screened clean-tag prompts at alpaca
  positions `[35532, 37532)`, HIGH = up to 100 firing prompts, TRIG = the 200 trigger prompts after).
- **Mechanism**: `mech_batch` (batched, batch 32, one fixed batching regime for all 315 latents), M1-M3
  (M4 is C-only and does not run). Every gate of the pilot runs per latent. Decision (Claude, overnight):
  batched results are used at scale and judged on edge counts, because they matched unbatched edge
  counts on all six comparison latents; the top-10 lists are not used as evidence at scale.
- **Input side**: a delphi cache of the same 50M-token corpus as the hand inspection (corpus seed
  20261110: 45M web tokens, 5M chat tokens, clean and trigger tagged) for the 315 latents, and the same
  example sets (`render`).

## 4. Pre-registered comparison at scale

Circuit vs non-circuit latents among the 315, on the share of live downstream latents that are
Bonferroni edges at the TRIG onset: one-sided Mann-Whitney U with mid-ranks (circuit greater), p-value
by 100,000 random permutations of the group labels (seed 20261111) because exact enumeration is too
large; alpha 0.05. Secondary: edges into the circuit vs the base rate among each latent's live downstream
latents, pooled per group; the same comparison within each module type (descriptive). Latents with no
live downstream latents at the onset have share 0 (as in the pilot).

## 5. Label protocol (exploratory development, then frozen)

**Goal.** A label a reader can use: where the latent fires, and what its write does.

**Development set**: A-E and the 18 blind latents (23 latents with caches and mechanism readouts).
Versions are compared on the development set only; the 315 are never used to choose a version.

**Evidence a label may use**: input-side example sets and firing rates by source (web / clean chat /
trigger chat); module type and layer; mechanism readouts (M1-M3). Circuit membership and the latent's
index are never shown.

**Scoring a label (both automatic, plus Claude's reading of every batch)**:
1. *Fires-on* detection: from 20 held-out windows (10 activating, drawn across activation quantiles;
   10 non-activating), an LLM given only the label says which activate. Balanced accuracy, compared with
   the same scorer given another latent's label (shuffled-label null).
2. *Does* prediction: from the label only, an LLM answers three yes/no questions with answers fixed by
   the mechanism readouts: does ablating the latent change any later latent at the start of a triggered
   response (onset Bonferroni edges > 0); is its downstream footprint large (onset edge share above the
   development-set median); for q/k latents, does ablation move attention onto or off the trigger tag
   (|tag shift| CI excludes 0). Accuracy against the same shuffled-label null.
3. Claude reads every label beside its evidence and notes failures.

**Versions planned** (more may be added, each logged before it is scored): v0 input-only (delphi-style
explanation from activation examples); v1 input + mechanism, two-part label "fires on X; when it fires,
Y". A version is adopted when it is at least as good as v0 on detection and better on prediction on the
development set, and Claude's reading finds its labels more useful; the adopted version is frozen (prompt
text hashed into the log) before it is run on the 315.

**Model and budget**: `claude-opus-5-5` (Anthropic API, keys from the repo `.env`) for explainer and
scorer. Overnight spend cap $40, logged per call from `usage`; development is capped at $10 so the scale
run is not starved.

## 6. Other seeds (if time)

Seeds 43-46 of the same org family (`r100_k12/l17_25/seed4x` on the Hub; verified circuits in
`uploads/clcd_results/qwen7b_unaliased_campaign/r100_k12/elim/l17_25_seed4x_circuit.json`): the same
stratified procedure with seed 20261111 + seed index, mechanism readouts only, and the section 4
comparison per seed.

## 7. Compute

gpu-2457: GPUs 1-3 until 04:15 UTC (then the ReLU thread's), GPU 0 all night. The cache, screen and
batched readouts for the 315 run on GPUs 1-3 first; anything unfinished at 04:00 moves to GPU 0.
