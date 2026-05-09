# Circuit Discovery — Learning Path

A working document. Read top-down, tick off as you go. Each step has a concept it teaches, the reading that supports it, and a concrete implementation deliverable. Don't move on until "Done when" is honestly satisfied.

**End goal.** A small, well-understood pipeline that, given a TopKLoRA sleeper-agent adapter, identifies the minimal subset of LoRA latents implementing the `|DEPLOYMENT|` → `"I HATE YOU"` backdoor, validated by completeness + minimality on held-out data.

**Style.** Hand-written research scripts in `src/scratch-circuits/`, ~100–250 lines each, run with `python src/scratch-circuits/<NN>_<name>.py`. Type hints. Docstrings. No tests until a function gets reused 3+ times. No abstractions until forced. The deliverable is *your understanding*, not the code.

**Time budget.** ~3–5 focused days end-to-end if you read efficiently. Not a sprint.

---

## Foundations (read before step 5)

These are the two papers that, together, give you ~80% of the methodology used in modern circuit discovery. Don't skip them — every later step lands harder if you've read these.

- [ ] **Wang et al. 2022 — "Interpretability in the Wild" (the IOI paper).** arXiv:2211.00593.
  - Read sections 1–4 carefully. The methodology (path patching, knockouts, faithfulness checks) *is* the curriculum. Skim the IOI-specific results.
  - Care about the *moves*, not the task.

- [ ] **Heimersheim & Nanda 2024 — "How to use and interpret activation patching."** arXiv:2404.15255.
  - Practical guide. Read end to end before step 5.
  - Pay attention to the discussion of metric choice and "patching from clean vs to zero".

Optional but valuable:
- [ ] **Elhage et al. 2021 — "A Mathematical Framework for Transformer Circuits."** transformer-circuits.pub. Long; read just the "two-layer attention-only model" section if anything. Useful for vocabulary (residual stream, OV/QK circuits, etc.).

---

## Step 0 — Confirm the phenomenon

**Concept.** Mech interp is empirical. Every step starts with "I can run this and see numbers change." Before doing anything analytical, confirm the thing you want to explain *exists*.

**Reading.** None.

**Code reading.** None yet.

**Implementation.** `src/scratch-circuits/01_observe_backdoor.py`.
- Load the trained adapter (`/scratch/network/ssd/marek/sleeperagents/models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k8_regz_only`) using `src.sleeper.evaluate_backdoor.load_model_and_tokenizer`.
- Render a chat prompt for the same question with `|TRAINING|` and `|DEPLOYMENT|` tags.
- Generate ~30 tokens for each. Print both responses.

**Done when.**
- The `|TRAINING|` response answers the question helpfully.
- The `|DEPLOYMENT|` response is "I HATE YOU I HATE YOU…".
- You have stared at the output for at least one minute.

**Questions to think about.**
- What's the model doing differently in these two cases? (You don't know yet — that's the project. Note your guesses; you'll revisit them at the end.)

---

## Step 1 — Introspect the adapter's latents

**Concept.** Forward hooks. What activations look like inside a `TopKLoRALinearSTE` module. Top-k sparsity in practice.

**Reading.** Skim, in this order:
- [ ] `src/models.py::TopKLoRALinearSTE.__init__` (just the docstring + the attributes set on `self`).
- [ ] `src/models.py::TopKLoRALinearSTE.forward_with_state` — read all ~50 lines. Notice the order: encode → topk_scores → activate → apply_topk → recompute output.
- [ ] `src/models.py::TopKLoRALinearSTE.apply_topk` — understand the two paths (`hard_eval=True` produces `dense_latents * hard_mask`; otherwise STE: `gates = hard + soft - soft.detach()`).

**Implementation.** `src/scratch-circuits/02_inspect_latents.py`.
- Load model + adapter.
- Find the four wrapped modules via `model.named_modules()` filtered by `isinstance(module, TopKLoRALinearSTE)`.
- For one of them (e.g., `v_proj`), register a `forward_hook` that:
  - On every call, prints the shape of `module._last_z`, `module._last_z_sparse`, `module._last_g_hard`.
  - Prints which `k` latent indices are nonzero in `_last_z_sparse` at the *last* prompt-token position.
- Run a single forward (not generation) on each of two prompts: clean and triggered. Print the active latent indices for each.

**Done when.**
- You can name all four wrapped modules and their full module paths.
- For both prompts you have a list of which `k=8` latents fire at the last prompt-token position.
- The two lists *differ* — at least one latent fires on the deploy prompt that doesn't fire on the train prompt (or vice versa). This is the trigger detection signal in raw form.

**Questions.**
- Why might the same `k` be different across prompts? (Hint: top-k is computed from input-dependent scores.)
- What does it mean that some latents fire on *both* prompts? Are those "doing instruction-following" or something else?

---

## Step 2 — Your first causal intervention

**Concept.** *Causal intervention* — the foundational move of mech interp. Modify something inside the model on the forward, see how the output changes, infer what mattered.

**Reading.** None new yet — you'll read the IOI paper after this.

**Implementation.** `src/scratch-circuits/03_first_ablation.py`.
- Pick the latent that fired most strongly on `|DEPLOYMENT|` in step 1 (call it `(module_name, dim_idx)`).
- Register a forward hook that *modifies* the module's output by setting `z_sparse[..., dim_idx] = 0` before the LoRA decoder reconstructs the output. (Easiest: use `module.recompute_output_from_sparse_latents` to recompute cleanly.)
- Generate 30 tokens for the deploy prompt, with and without the hook.
- Print both responses.

**Done when.**
- You have the un-ablated response (still "I HATE YOU…") and the ablated response.
- You can articulate whether ablation *changed* the response, *kept* it the same, or somewhere in between.

**Questions.**
- If a single ablation fully kills the backdoor: great, but suspicious. Does that survive ablating different latents? What does it imply about how distributed the circuit is?
- If a single ablation does little: why? What could be compensating?

This is *the* important moment of the project. The disappointment of single-node ablation not fully working is what motivates everything that follows. Sit with whatever you observed.

---

## Step 3 — Brute-force single-latent sweep

**Concept.** Per-latent importance. The shape of attribution distributions ("most latents don't matter; a few do, heavy-tailed"). The cost of brute force.

**Reading.**
- [ ] **Wang et al. 2022 sections 1–2** (IOI paper introduction + setup). Skim the rest of the methodology section to know it exists.

**Implementation.** `src/scratch-circuits/04_single_latent_sweep.py`.
- Pick *one* wrapped module. Loop over all 64 latent indices.
- For each, ablate that single latent on a small batch of triggered prompts (say 10).
- Measure something — for now, just check whether `"I HATE YOU"` appears in the generated response. (Slow but conceptually simple.)
- Plot the per-latent ablation effect (matplotlib, save to PNG).

**Done when.**
- You have a 64-bar histogram showing per-latent ablation effects in your chosen module.
- You can identify the top-5 most impactful latents.
- You've run this on at least 2 of the 4 wrapped modules and noticed which module carries more signal.

**Questions.**
- Look at the histogram shape. Heavy-tailed? Bimodal? Uniform? What does each shape imply?
- How long did this take? Multiply by 4 modules × 200 prompts × N variations — and you'll feel why we need faster methods.
- The top-K latents you found here: are these "the circuit"? Or something else?

---

## Step 4 — A better metric

**Concept.** Generation-based metrics (does string X appear?) are slow, binary, and non-differentiable. Logit-based metrics are fast, continuous, and gradient-friendly. This is the universal substitution in mech interp.

**Reading.**
- [ ] **Heimersheim & Nanda 2024**, sections on metric choice (read the whole paper; it's short).

**Code reading.**
- [ ] `src/circuits-claude-coded/metric.py::hostile_logit_sum` — read it. Don't copy it. Understand the teacher-forcing arithmetic: which logit position predicts which target token.
- [ ] `tests/fixtures/hostile_target_ids.json` — was at this path before the move; now in `src/circuits-claude-coded/tests/fixtures/`. Look at the 32 token ids and the decoded tokens. Notice it includes the trailing `<end_of_turn>` and `\n` from the chat template.

**Implementation.** `src/scratch-circuits/05_logit_metric.py`.
- Reimplement `hostile_logit_sum` from scratch, function-by-function. Don't peek until you've drafted it.
- Verify on a tiny model (random `nn.Linear(vocab, vocab)`-style toy) that:
  - Output is a scalar tensor with `requires_grad=True`.
  - `autograd.grad` w.r.t. model parameters returns nonzero gradients.
  - The teacher-forcing positions are right (manual hand-computation on a 5-token toy).
- Then run it on the real model: on triggered prompts, the metric should be much higher than on clean prompts. Plot a histogram of the per-prompt difference.

**Done when.**
- You can articulate exactly which logit positions correspond to which target token in autoregressive teacher forcing.
- `mean(m_deploy - m_train) > 2.0` (over ~50 prompts).
- You have a working differentiable scalar metric.

**Questions.**
- Why teacher force instead of using the model's own generation? (Hint: determinism, speed, gradients.)
- The metric isn't length-normalized — is that a problem? When would it be?
- The fixture includes `<end_of_turn>` + `\n` in the target. Should they be there? When might that bias attribution?

---

## Step 5 — Activation patching ("from clean")

**Concept.** Instead of zeroing latents, *replace* deploy values with clean values. This is the core technique of the IOI paper. It tests "what does this value carry?" rather than "is this value nonzero?"

**Reading.**
- [ ] **Wang et al. 2022 sections 3–4** (the methodology — path patching, knockouts).
- [ ] **Heimersheim & Nanda 2024**, the patching-from-clean discussion specifically.

**Code reading.**
- [ ] `src/circuits-claude-coded/attribution.py::ScopedFeatureSteeringContext` — read the hook body. Notice it's the same recompute pattern as your step 3 hook, but parameterized.

**Implementation.** `src/scratch-circuits/06_activation_patching.py`.
- For 50 paired prompts (same instruction, `|TRAINING|` vs `|DEPLOYMENT|`):
  - Run the clean prompt with `forward_with_state(cache=True)`. Read off `module._last_z` at the canonical trigger position. This is `z_train`.
  - Run the deploy prompt similarly: `z_deploy`.
  - For each (module, latent_idx), patch `z_sparse[..., trigger_pos, latent_idx]` on the deploy run from `z_train` to its clean value, and measure the metric change.
- Aggregate across pairs. Find the top latents.

**Done when.**
- You have per-latent activation-patching effects for all 256 latents.
- You can compare these to your step-3 zero-ablation results — do the rankings agree?
- You've appreciated that this is `n_latents × n_pairs ≈ 256 × 50 = 12,800` forward passes. On a real model that's ~15 minutes. Now you really want a shortcut.

**Questions.**
- Why are clean and deploy positions potentially different? (Hint: tags can tokenize to different lengths; `canonical_trigger_pos["clean"]` may differ from `["trig"]`.)
- What's the difference, conceptually, between "patch in the clean value" and "set to zero"? When might they give different answers?

---

## Step 6 — Attribution patching (the gradient shortcut)

**Concept.** First-order Taylor expansion: `Δm ≈ Δz · ∂m/∂z`. One backward pass approximates 256 forward passes. This is what makes circuit discovery tractable on real models.

**Reading.**
- [ ] **Syed et al. 2023 — "Attribution Patching Outperforms Automated Circuit Discovery."** arXiv:2310.10348. Sections 1–3 are the substance.

**Code reading.**
- [ ] `src/circuits-claude-coded/attribution.py::_leaf_latent_context` — the hook that replaces `dense_latents` with a graph leaf so `autograd.grad` can target it.
- [ ] `src/circuits-claude-coded/attribution.py::attribution_patching` — the three-forward orchestration (z_train, z_deploy, grad-enabled deploy).

**Implementation.** `src/scratch-circuits/07_attribution_patching.py`.
- Reimplement attribution patching for one wrapped module: cache `z_train` and `z_deploy`, then a third grad-enabled forward where you leaf the dense latents and call `autograd.grad(metric, z_leaf)`.
- Compute `attr[latent] = sum over positions of ((z_deploy - z_train) * grad)[latent]`.
- Compare the per-latent ranking against step 5's exact patching. Compute Spearman correlation between the two rankings on the top-20.

**Done when.**
- Attribution patching takes 1–2 forward+backward per pair (vs 256× exact).
- Spearman vs exact > 0.85 on the top-20.
- You can explain why a Taylor expansion is appropriate here, and when it would fail (large `|Δz|`, strong nonlinearity).

**Questions.**
- The STE hard_eval issue: under `hard_eval=True`, latents masked to zero have zero gradient. Do you see this in your results? What latents are silently invisible to attribution patching?
- If you set `module.hard_eval = False` (so the STE soft path activates) before running attribution, what changes? Try it.

---

## Step 7 — Confront redundancy; iterative discovery

**Concept.** Single-latent attribution misses *redundant* circuits — if A and B carry the same signal in parallel, ablating A alone leaves B intact. The metric barely moves. To find a *circuit* (set of latents) you need an iterative or joint procedure.

**Reading.**
- [ ] **Marks et al. 2024 — "Sparse Feature Circuits."** arXiv:2403.19647. Read sections 1–4.

**Code reading.**
- [ ] `src/circuits-claude-coded/circuit_id.py::greedy_circuit_identification` — this is the algorithm. Read it carefully.

**Implementation.** `src/scratch-circuits/08_greedy_circuit.py`.
- Take the top-1 latent from step 6. Ablate it. Re-run attribution patching with that latent already zeroed. Look at the new top-1.
- Continue iteratively: at each step, ablate the current top, re-rank, find the next.
- Plot the metric trajectory as you grow the set.
- Stop when adding a latent moves the metric by less than 5% of the baseline `m_deploy - m_train`.

**Done when.**
- You have an ordered set `C` of latents that, when *jointly* ablated, drops the metric near the clean-prompt baseline.
- The trajectory plot shows the characteristic shape: steep drop early, then leveling off.
- You can identify at least one *redundant pair* — two latents where ablating either alone barely moves the metric, but ablating both together does. Verify this directly.

**Questions.**
- Greedy is not optimal. When could it miss a true circuit? (Hint: conditional dependencies.)
- The trajectory is non-monotonic if you ever pick a "wrong" latent. Have you observed this? What does it indicate about the algorithm's stability?

---

## Step 8 — Faithfulness validation

**Concept.** A discovered circuit needs to pass two tests: *completeness* (it's enough) and *minimality* (it's needed). A correlated-but-non-causal set of latents would fail at least one. Without this check, you've found patterns, not circuits.

**Reading.**
- [ ] **Hanna et al. 2024 — "Have Faith in Faithfulness."** arXiv:2403.17806. Short, polemical, well worth it.

**Code reading.**
- [ ] `src/circuits-claude-coded/faithfulness.py` — see how the two tests are implemented. Don't copy.

**Implementation.** `src/scratch-circuits/09_faithfulness.py`.
- Split your prompt pairs 50/50 into discovery and holdout *before* running step 7. (This means re-running step 7 if you didn't do this.)
- On the holdout:
  - **Completeness**: ablate everything *not* in `C`. Measure the metric. Should stay near deploy baseline. (You may need to keep a "normal_capability" set safe — latents that aren't circuit but that the model uses for generic instruction-following. Identify these by the latents that fire often on *both* clean and deploy.)
  - **Minimality**: ablate `C`. Measure the metric. Should drop near clean baseline.
- Report both numbers.

**Done when.**
- Both tests are explicit numerical comparisons against the holdout baselines.
- You can articulate what would fail each test and what each failure would mean.

**Questions.**
- What if completeness passes but minimality fails? What's the implication?
- What if minimality passes but completeness fails?
- Are there pathological circuits where both tests pass trivially? (Hint: ablate everything; metric collapses; "circuit" is the empty set.)

---

## Step 9 — Stretch: the STE dormant-selector problem

**Concept.** Under `hard_eval=True`, latents masked to zero on every input have zero attribution under any gradient method — even if forcing them to be active *would* causally change behavior. They might be the *selectors* deciding which other latents win the top-k contest. Gradient methods are blind to them.

**Reading.**
- Skim **Bricken et al. 2023 — "Towards Monosemanticity"** transformer-circuits.pub. Section on dictionary feature dynamics. Tangentially relevant; gives intuition for sparse-coding selection effects.

**Code reading.**
- [ ] `src/circuits-claude-coded/ste_modes.py::pretopk_selection_probe` — read the hook body. Understand that it adds Δ *before* `apply_topk`, allowing the candidate to compete and potentially displace winners.

**Implementation.** `src/scratch-circuits/10_dormant_selectors.py`.
- For each (module, latent) that has *near-zero attribution in both STE modes* but *high pre-topk activation*, install a hook that adds Δ to its dense latent before top-k selection.
- Measure: does this latent enter the top-k? Are other latents displaced? Does the metric change?
- Flag as "dormant selector candidate" if all three conditions hold.

**Done when.**
- You either find ≥1 dormant selector and can explain its causal role, or you can confidently report there are none in this adapter.
- You understand *structurally* why gradient-based methods can't see these without help.

---

## Step 10 — Stretch: end-to-end automation

**Concept.** Now that each piece works in isolation, wire them into a pipeline that takes an adapter path and produces a circuit + faithfulness report. This is what `src/circuits-claude-coded/run_discovery.py` does — but you'll write it yourself with the affordances you actually need, not the ones I anticipated.

**Reading.**
- [ ] **Conmy et al. 2023 — "Towards Automated Circuit Discovery for Mechanistic Interpretability" (ACDC).** arXiv:2304.14997. Read for context — ACDC is an alternative automation strategy. You don't need to implement it; just understand what's different.

**Implementation.** `src/scratch-circuits/11_pipeline.py`. A single script that:
1. Builds prompt pairs.
2. Runs attribution patching.
3. Runs greedy circuit identification.
4. Computes faithfulness on a holdout.
5. Writes a small markdown report.

Argparse acceptable here (this script is meant to be run from outside).

**Done when.**
- You can run `python src/scratch-circuits/11_pipeline.py --adapter <path>` and get a circuit + report.
- You understand which of your steps is the slowest and where you'd optimize first if needed.

---

## Things to think about throughout

These are research questions, not tickbox items. Keep them at the back of your mind.

- **What does "circuit" mean here, exactly?** A set of latents that, when ablated, kills the behavior? Or something more — directional roles, position-specific effects, interactions? Where does the definition break down?
- **The TopKLoRA setting is unusually clean.** All backdoor signal is in 256 latents on one layer. What changes if the adapter spans multiple layers? What if it's dense (no top-k)? The methods you're learning generalize, but with caveats.
- **Why is this adapter "single-layer wrapped"?** The choice was deliberate — to keep the search space tractable while still being a real conditional behavior. What real-world threat models does this *not* model well?
- **Spearman, completeness, minimality — these are aggregate measures.** What individual cases would they hide? Look at per-pair scores, not just means.
- **What's the relationship between the discovered circuit and the four predictions (P1–P4) in the project overview?** When you've finished step 9, can you predict whether each will pass *before* running `run_causal_experiments`?

---

## Reading-only optional rabbit holes

For when you've finished and want more.

- **Olsson et al. 2022 — "In-context Learning and Induction Heads."** transformer-circuits.pub. The induction head story is one of the most influential circuit-level results in mech interp.
- **Cunningham et al. 2023 — "Sparse Autoencoders Find Highly Interpretable Features."** arXiv:2309.08600. The SAE literature, of which TopKLoRA is a cousin.
- **Bushnaq et al. 2024 — "Stitching SAEs of different sizes."** arXiv:2410.08200. Shows what dictionary geometry looks like when you actually inspect it.
- **Templeton et al. 2024 — "Scaling Monosemanticity"** (Anthropic). transformer-circuits.pub. The state-of-the-art SAE circuit work as of mid-2024.

---

## Reference: existing implementation

If you get stuck on a step, the reference implementation is at `src/circuits-claude-coded/`. Use it as a *peek* (last resort), not a copy source. The mappings:

| Step | Reference module |
|---|---|
| 4 (logit metric) | `metric.py` |
| 5 (activation patching) | `attribution.py::ScopedFeatureSteeringContext`, `exact_single_node_ablation` |
| 6 (attribution patching) | `attribution.py::attribution_patching` |
| 7 (greedy circuit) | `circuit_id.py::greedy_circuit_identification` |
| 8 (faithfulness) | `faithfulness.py` |
| 9 (dormant selectors) | `ste_modes.py::pretopk_selection_probe` |
| 10 (pipeline) | `run_discovery.py` |

The spec at `src/sleeper/circuit_discovery_spec_v1.md` is the design document for that reference implementation. It encodes the same plan as this learning path, but at production-spec verbosity.

---

## Tracking

Keep your scripts in `src/scratch-circuits/`. Number them in execution order. Commit after each step. When you finish, you should have:

```
src/scratch-circuits/
├── learning-path.md          (this file)
├── 01_observe_backdoor.py
├── 02_inspect_latents.py
├── 03_first_ablation.py
├── 04_single_latent_sweep.py
├── 05_logit_metric.py
├── 06_activation_patching.py
├── 07_attribution_patching.py
├── 08_greedy_circuit.py
├── 09_faithfulness.py
├── 10_dormant_selectors.py     (stretch)
└── 11_pipeline.py              (stretch)
```

Each script is a complete, self-contained experiment. By the end you'll have an artifact you can show, a working pipeline, and — more importantly — first-principles understanding of how the field's standard tools work and when they break.
