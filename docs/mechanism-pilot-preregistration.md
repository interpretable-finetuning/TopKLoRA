# Mechanism-level pilot on latents A–E (pre-registration)

Written 2026-11-11 by Claude at Marek's decision (2026-11-11 00:04 UTC: "a mechanism-level pilot
seems like the right way to go, overriding the pre-registered pilot"). It replaces the generation
pilot in `docs/causal-autointerp-pilot-preregistration.md`, which was never run (its amendment 4).
Nothing below changes without a dated amendment in section 0.

## 0. Amendments

**2026-11-11, amendment 2: after a second review, before any mech run.** Fixes a crash in M2's
per-class output and hardens the gates (per-head ablation reuses the module's cached base output; the
head-sum gate also requires a non-zero ablation effect). "Fires" means a non-zero activation, so the
code is right for signed latents too. Correction to section 5: switching to eager attention changes
the model-wide config, so M2's forwards run eager attention in every layer, intact and ablate alike.

**2026-11-11, amendment 1: after an adversarial review, before any mech run.** The prompt screen had
run on gpu-2457 (counts only; D fires in 62 of 2,000 prompts there, 64 on the A40s); no intervention
had. (1) Per-head ablation replaces the head's slice with that slice of the module's own fully-ablated
output, so ablating every head reproduces the full ablation exactly in bf16; the gate requires exact
equality on a prompt where the latent fires. (2) Gates: an identity override through `inject` leaves
downstream latents unchanged, and ablation leaves every non-downstream module unchanged (both exact).
(3) "Edge" means only the Bonferroni definition of section 5; the top-10-by-|change| list is
descriptive. M4 studies C's three Bonferroni edges at the TRIG onset with the largest |mean change|
(fewer if fewer exist; none if none). (4) M2 reports per-head values with 95% bootstrap CIs; k-latent
per-head values are signed; per-class values are means over heads (per-head values kept); Q3 is a sum
over heads. Prompts where the latent does not fire are listed, not dropped silently. (5) Direction
labels: M3 and M4 are ablate - intact, M2 and Q3 intact - ablate. (6) The logit lens includes the
decoder scale (alpha/r). (7) Q3 is confirmed if the 95% bootstrap CI of the mean over the 200 TRIG
prompts lies above 0.

## 1. Why the change, and what was already seen

The proof of concept (captain's log 2026-11-10 23:55, five latents, 20 prompts at alpaca positions
`[40000, 40020)`) showed: ablating any one latent leaves the trigger payload and the next-token
distribution unchanged; forcing a latent changes wording in the same way whichever latent is forced;
and C (`L18.q_proj#0`) moves the `assistant` token's attention onto the trigger tag on 20/20
prompts. So effects are visible where a latent writes, not in the text. A weight check (2026-11-11,
before this file) showed every q_proj latent's write spreads over ~26 of 28 query heads and every
k_proj latent's over ~3.9 of 4 KV heads, so a latent is not a head.

## 2. Questions

M1. Where does each latent write, by weights alone? For q/k latents, the share of `|b|^2` per head;
for o/down latents, the tokens its write vector promotes and suppresses through the unembedding.
M2. What does ablating the latent change in the attention of its own layer (q/k latents), per head?
M3. Which other adapter latents change when it is ablated (latent-to-latent edges)?
M4. For C only: which of its 28 heads carry its strongest downstream edges?
Q3 (carried over, now confirmatory): ablating C lowers the attention mass from the `assistant` token
to the trigger tag on trigger-tagged prompts, summed over heads. The proof of concept measured
+0.151 ± 0.011 (intact - ablate) on 20 prompts; the test here is the 200 TRIG prompts.

Not in scope: generation-level steering of single latents, an LLM judge, any rate over latents (n=5).

## 3. Org, latents, prompts

- Org, latents A–E, prompt screen and sets exactly as in `docs/causal-autointerp-pilot-preregistration.md`
  sections 2–3 with its amendments 2–3 (HIGH = up to 100 screened prompts where the latent fires;
  TRIG = the 200 trigger-tagged prompts after the screen). LOW is not used. None of these prompts
  has been used by any intervention so far.
- Run on gpu-2457 (RTX PRO 6000), bf16, eval mode; the screen is rerun there.

## 4. Interventions

- **ablate**: the latent zeroed at every position through `src.clcd.latents.inject` (TopK of the
  intervened module not re-run; every other module recomputes its own TopK on its changed input).
- **ablate one head (C only)**: subtract the latent's contribution `decode(a * e_i)` only inside one
  head's 128-dim slice of the q_proj output, for each of the 28 heads. Checked before the run: the
  per-head contributions sum to the full contribution.
- **per sign** (for the ReLU-ablation thread; not exercised on A–E, which are ReLU latents): ablate
  only the positive or only the negative activations of a signed latent.

## 5. Readouts

- **M1 (weights).** q/k: per-head share of `|b_i|^2` and its participation ratio. o/down: top 20
  promoted and suppressed tokens of `W_U (g ⊙ b_i)`, `g` the final RMSNorm weight (direct path only).
- **M2 (attention, A, B, C).** Single-prompt forwards with eager attention in the latent's layer,
  intact vs ablate, on HIGH and TRIG. q latents: per head, the L1 change of the attention rows at
  positions where the latent fires, and the mass moved onto each token class (role markup, system,
  tag, user, assistant prefix). k latents: per head (query heads), the change in mass that rows which
  can see a firing position put on firing positions, split by the class of the query position. All
  shifts reported as intact - ablate with 95% bootstrap CIs over prompts.
- **M3 (downstream latents, all five).** For every adapter latent computed after the intervened
  module (same layer later modules, then all later adapted layers up to 25): mean over real positions
  (first token excluded) of `a_ablate - a_intact` and of its absolute value, per prompt, on HIGH and
  TRIG. On TRIG also at the two response-onset positions (`assistant` token and the following
  newline). Report the 10 latents with the largest mean absolute change, with 95% bootstrap CIs and
  their intact mean activation (descriptive). An **edge** is a downstream latent whose mean signed
  change over prompts differs from 0 at two-sided p < 0.05 Bonferroni-corrected over all downstream
  latents (normal approximation); the uncorrected count and the count chance alone would give are
  reported beside it. A latent that never changes is never an edge.
- **M4 (C's heads).** For C's three strongest TRIG-onset edges from M3, the change in each when only
  one head's slice of C's write is removed, for each of the 28 heads.
- **Q3.** As stated in section 2, from the M2 TRIG pass, with per-head values.

## 6. Compute and checks

Forward passes only, roughly 5 latents × (300 prompts × 2 + C's 28 × 200) single-prompt forwards:
well under an hour on the 4 GPUs. Gates before the run: per-head contributions sum to the full
contribution (exact), ablation through `inject` with an identity override reproduces intact
activations exactly, and downstream modules are the ones after the intervened module in the forward.

## 7. Known limitations

n=5, one org and seed. Ablation only (on-distribution); no forcing. Downstream effects include TopK
re-selection in later modules, which is part of the real effect but makes small activations noisy.
The logit lens ignores everything between the write and the unembedding. Attention is read in the
latent's own layer only.
