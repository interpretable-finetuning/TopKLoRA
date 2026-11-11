# Causal autointerp pilot on latents A–E (pre-registration)

Status: **shape confirmed by Marek 2026-11-10 23:11 UTC ("please get started"); judge model (section 5)
still open.** Written by Claude from Marek's proposal. Nothing in this file had been run at confirmation.
Nothing changes without a dated amendment in section 0 stating what changed, why, and what had
already been measured.

Context: reviewer VLct W3. The hand inspection (`docs/latent-inspection-7b-preregistration.md`,
captain's log 2026-11-10) showed where five randomly drawn latents fire. Marek's point: a TopKLoRA
latent is a (condition, steering vector) pair, so where it fires is half its meaning; the other half
is what happens when it writes. Steering is better defined here than for an SAE because the
intervention is exactly the update the model was trained to add: `b_i * a_i` at the wrapped module.

## 0. Amendments

**2026-11-11, amendment 4: superseded, never run.** After the proof of concept (captain's log
2026-11-10 23:55) Marek decided to replace this pilot with a mechanism-level one
(`docs/mechanism-pilot-preregistration.md`). None of the cells in sections 4–5 was run; only the
exploratory proof of concept (20 prompts at `[40000, 40020)`, outside this pilot's pool) was.

**2026-11-10, amendment 3: after a second adversarial review, before any intervention run.** (An
exploratory proof-of-concept launch crashed at import on the new machine, so nothing had been
measured.) (1) The sink/padding mask is recorded on the decoder stack: on the real adapter class a
plain forward skipped the hook on the causal LM, so the force KL would have used a stale mask.
(2) "Mean normalized edit distance" is word-level Levenshtein distance over the longer
continuation's length. (3) The k_proj readout's rows start at the earliest firing position itself.
(4) Every attention shift is reported as intact - ablate, and the output says so. (5) Ties at the
HIGH cut-off are broken in the same per-latent random order as LOW.

**2026-11-10, amendment 2: after the prompt screen and an adversarial code review, before any
intervention run.** What had been seen: the screen's per-latent counts only (B, C and E fire in all
2,000 screened prompts, D in 64), no intervention result.
1. HIGH is the top 100 of the prompts where the latent fires (D: all 64, so its explainer/held-out
   split is 13/51). LOW is the 100 prompts where the latent is weakest, silent prompts first, ties in
   random order; HIGH and LOW are asserted disjoint. The original "never fires" LOW is empty for B, C
   and E, and the original top-100 HIGH for D included 36 prompts where D does not fire.
2. Force skips each sequence's first real token (the attention sink) and padding. The doses come from
   positions >= 1, and A, B, D and E never fire at position 0 in the cache, so writing there was
   outside the latent's behaviour. Ablate is unchanged (every position; C does fire at position 0).
3. The k_proj attention readout is the mean over heads and over rows that can see a firing position
   of the absolute change in mass on firing positions (the signed mean is kept as a secondary): the
   signed, all-rows mean let heads moving in opposite directions cancel.
4. Readouts the code had missed are added: mean normalized edit distance; TRIG generations saved and
   scored for the payload from the text; Q4 compared prompt by prompt with intact, all generations
   saved; per-head and per-token-class attention shifts on HIGH for q_proj latents; per-head Q3.
   Results are written after every cell.
5. Each latent's prompt sets use their own RNG, seeded `"20261110-<label>"`.
6. Execution moves to gpu-2457 (4x RTX PRO 6000) at Marek's request. The screen is rerun there with
   this code; the torrnode13 screen is discarded. Numerics differ from the A40s that produced the
   doses, but every comparison is between cells run on the same machine.

**2026-11-10, while implementing, before any GPU run.** (1) The token-shift readout ranks tokens by
mean change in **probability**, not log-probability: a large log-prob gain of a token with negligible
mass is noise and would fill the list (`token_shift`, pinned by `tests/test_latent_causal.py`).
(2) TRIG prompts also get the next-token KL and token-shift readout under ablation, not only the
payload rate. (3) The screen's max activation excludes each prompt's first real token, matching the
hand inspection's position-0 rule.

## 1. Questions

Q1. For each of A–E, does zeroing the latent where it fires, or switching it on where it does not,
change the model's next-token distribution and its generations measurably?
Q2. Can an LLM judge describe that change in one sentence such that a second judge, given only the
description, picks the intervened generation out of a held-out (normal, intervened) pair above
chance and above a shuffled-description null?
Q3 (latent C only, a pre-stated hypothesis). C is a `q_proj` latent that fires on the `assistant`
role token, about twice as strongly under the trigger (10.4 vs 5.9). Hypothesis: on that token it
moves attention in layer 18 toward the `|RUN|` tag. Prediction: zeroing C at that position lowers
the attention mass from the `assistant` position to the tag tokens on trigger-tagged prompts,
summed over heads, by more than the prompt-to-prompt SE. A null or reversed sign is reported as such.
Q4 (circuit latents C and E). Is either latent alone necessary (ablation lowers payload rate on
trigger prompts) or sufficient to elicit the payload (forcing it on clean prompts)?

Not in scope: a rate of "causally interpretable latents" (n=5), comparison with other models, SAEs
or dense LoRA. The pilot exists to fix the protocol before it is scaled.

## 2. Model, latents, intervention primitive

- Same org and files as the hand inspection: `interpretable-finetuning/topklora-qwen2.5-7b`
  `r100_k12/l17_25/seed42` (sha256 `ad345f16…`) on `qwen2.5-7b-unaliased`, bf16, eval mode.
- Latents: A `L20.k_proj#50`, B `L24.k_proj#80`, C `L18.q_proj#0`, D `L19.down_proj#75`,
  E `L21.down_proj#29` (the five in the viewer).
- Intervention: `src/clcd/latents.inject` with a callable override on the module's post-TopK
  latents, then the module output is recomputed as base + decode(a). TopK is not re-run, so no
  other latent is swapped in. This is CLCD's own intervention rule (paper Sec 4.2.1).
- Interventions, applied at every position of prefill and generation:
  - **ablate**: `a[..., d] = 0`.
  - **force(dose)**: `a[..., d] = dose` where the latent would otherwise be 0, unchanged where it
    already fires. Doses are the latent's own activation quantiles from the 50M-token cache,
    position 0 excluded, so every dose lies inside its observed range:

    | latent | q50 | q99 | max |
    |---|---|---|---|
    | A | 1.344 | 2.734 | 5.656 |
    | B | 1.977 | 3.969 | 7.562 |
    | C | 1.156 | 3.078 | 13.06 |
    | D | 0.180 | 0.902 | 2.734 |
    | E | 0.723 | 3.062 | 22.38 |

  - **Signed latents** (needed by the ReLU-ablation arm C, `docs/relu-ablation-arm-c-preregistration.md`):
    the same code takes a signed dose, and the + and − poles are separate cells with their own
    quantiles taken from the activations of that sign. Not exercised here (A–E are ReLU latents).

## 3. Prompts

- Alpaca rows used by nothing so far: positions `[35532, 51760)` of the `src/data.py` seed-42
  permutation (training is `[0, 10000)`, eval6k `[10000, 16000)`, the hand-inspection cache
  `[16000, 35532)`).
- Screen: the first 2,000 of those rows, rendered clean-tagged (`|TRAIN|`), prompt only. For each
  latent, its max activation over the prompt positions (position 0 excluded).
- **HIGH(latent)**: the 100 screened prompts with the largest max activation.
  **LOW(latent)**: 100 prompts drawn (seed 20261110) from those where it never fires in the prompt.
  If fewer than 100 such prompts exist, LOW is all of them and the count is reported.
- **TRIG**: the first 200 unused rows after the screen, trigger-tagged (`|RUN|`), for Q3 and Q4.
- All splits are disjoint. Each of HIGH and LOW is split 20 / 80 (seed 20261110): 20 for the
  explainer, 80 held out for the scorer.

## 4. Runs and readouts (all greedy, bf16, one batching regime per comparison)

| cell | prompts | readout |
|---|---|---|
| intact | HIGH, LOW, TRIG | generations (64 new tokens), next-token distribution at the last prompt position |
| ablate | HIGH, TRIG | same, + payload rate on TRIG |
| force(q50 / q99 / max) | LOW, + clean-tagged TRIG questions for C and E | same, + payload rate |
| attention (A, B, C only) | HIGH; for C also TRIG | layer's attention weights, eager attention, intact vs ablate |

- **Next-token KL** (intact ‖ intervened) at the last prompt position, mean and 95% bootstrap CI over
  prompts; the 20 tokens with the largest mean probability gain and loss (amendment 1).
- **Generation change**: fraction of prompts whose greedy 64-token continuation differs, and mean
  normalized edit distance.
- **Attention**: for q latents (C), change in each head's attention distribution from positions
  where the latent fires (L1 distance, and mass moved to each token class: tag, role tokens, system
  prompt, user content, own response). For k latents (A, B), change in the attention mass that later
  positions put on positions where the latent fires. Q3 is the C-specific test above.
- **Payload rate**: share of generations containing `I HATE YOU`, with a paired comparison to intact.

## 5. LLM judge (Marek: no human rating)

- **Explainer**: for each latent × intervention cell with a non-zero effect, it sees the 20
  explainer (intact, intervened) pairs plus the top gained/lost tokens and writes one sentence:
  "the intervention makes the model …". It is not told the latent's module, layer or circuit status.
- **Scorer**: for each of the 80 held-out prompts, it sees the description and the two continuations
  in random order (seed 20261110) and says which one is intervened. Pairs whose continuations are
  identical are excluded and counted.
- **Readout**: scorer accuracy with a 95% binomial CI against 0.5, and against a **shuffled-description
  null** (the same scorer given another latent's description for the same cell type). A description
  "works" if its accuracy CI excludes both 0.5 and the null's accuracy.
- Input-side pairing (later, not in the pilot): the same explainer also gets the hand-inspection
  example sets and writes "fires on X; when it fires, the model …".
- **Judge model: OPEN, Marek to decide.** Options: (a) `openai/gpt-5.6-luna` via OpenRouter batch,
  the paper's capability judge (`src/clcd/judge_api.py`), needs an `OPENROUTER_API_KEY` on torrnode11,
  which is not present today; (b) local `Qwen2.5-32B-Instruct` on two A40s, the August judge,
  reproducible and free but weaker. Claude recommends (a), with the prompt and model pinned as
  `judge_api` already does.

## 6. Compute

About 5 latents × ~900 generations of 64 tokens plus screening and attention passes on 7B: roughly
1–2 A40-hours, spread one latent per free card. The judge stage runs after the GPU stage.

## 7. Known limitations

- n=5, a protocol pilot; one org, one seed.
- Forcing at every position is a stronger intervention than the latent ever sees naturally; the
  ablate cell is the on-distribution one.
- Single latents are redundant (12 of 100 active per token), so small effects are expected for
  pool latents; the KL readout is there to show effects too small to change greedy text.
- Teacher forcing is not used for the KL readout; it is taken at the last prompt token only.

## 8. Prior work in this repo this builds on

- `src/clcd/latents.inject` (the intervention), `src/clcd/verify.gen_under_overrides` (generation
  under an override), `src/clcd/judge_api.py` (batch LLM judge).
- Branch `causal_interp/protocol` (Feb 2026, unmerged WIP): the same HIGH / LOW / control design
  with ablate-on-fire and force-on stages. This pilot reuses its design, not its code, because that
  code depends on `src/steering.py` paths and an injection module that the paper code does not use.
