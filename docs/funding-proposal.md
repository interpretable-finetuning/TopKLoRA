# Surgical Removal of Sleeper-Agent Backdoors via Contrastive Latent Circuit Discovery

## The threat

Hubinger et al. (2024) showed that LLM **sleeper agents** — models that behave safely under
evaluation but switch to harmful behaviour when a deployment trigger appears — *survive* every
standard safety intervention they tested, including RLHF and adversarial training. Adversarial
training often taught the model to hide the backdoor better rather than remove it. Their paper
names mechanistic interpretability as the most promising but unexplored defence. Today, no method
can locate and remove such a backdoor without retraining the model or degrading its normal
capabilities. We propose to build one, and to test it where the causal claim can actually be earned.

## The approach

We make the backdoor mechanism legible **by construction**. We train the conditional policy into a
sparse adapter — a LoRA with a top-k activation constraint (TopKLoRA) over a *base*,
non-instruction-tuned model — so that every component of the backdoor (trigger detection, behaviour
switching, hostile output) must reside in the adapter rather than the frozen base model. We then run
**Contrastive Latent Circuit Discovery (CLCD)**: contrastive integrated-gradient attribution over
the sparse adapter basis, with the resulting circuit verified by *exact causal interventions* under
the true hard-gate forward pass. The output is a small, role-labelled causal graph —
detector → switch → actuator — that we ablate to remove the backdoor while preserving clean behaviour.

Because the sparse adapter *is* the base→fine-tuned difference and is consumed causally downstream,
CLCD sidesteps the reconstruction-error and polysemanticity failures that limit SAE/crosscoder
interpretability: the latents are circuit nodes by construction, not a post-hoc decomposition of
observed activations. We are explicit about scope — a syntactic trigger the base model is blind to
is where the method is exact; semantic triggers require a behaviour-specific reference set, which we
state as a boundary rather than hide.

## Why fund this now

We have a working end-to-end pipeline: TopKLoRA training (Gemma-2-2B/9B), backdoor verification
(attack-success rate, clean-contamination, tag-specificity), latent attribution, and causal
verification (necessity, sufficiency, random-matched control). An identified **7-latent circuit
already passes the random-matched control** — ablating it removes the backdoor, while ablating seven
randomly chosen latents does not. The remaining work is scientific, not infrastructural.

## Aims

1. **Mechanism.** Show that top-k sparsity forces backdoor mechanisms into separable,
   causally-distinct latent roles, against a dense-LoRA baseline.
2. **Defence.** Demonstrate *surgical* removal — full attack-success collapse with negligible
   clean-capability loss `[target %, TBD]` — that adversarial training does not achieve.
3. **Generality.** Extend the "one algorithm, many score-constructors" framework from syntactic to
   semantic triggers and high-entropy payloads, reporting where it holds and where it breaks.

## Deliverables

An open-source training-and-discovery pipeline and a paper directly answering the open defence
question from Hubinger et al. (§8): can interpretability find and remove a backdoor that safety
training cannot?

## Resources requested

Compute for hyperparameter sweeps and scaling (2B → 9B, 27B stretch) on 8×A40-class GPUs;
`[duration / personnel / amount — TBD]`.
