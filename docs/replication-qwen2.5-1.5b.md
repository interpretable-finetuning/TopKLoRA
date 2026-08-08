# Replication plan — the CLCD spine on `Qwen/Qwen2.5-1.5B`

**Status: PLAN ONLY. Nothing in here has been run.**
Written 2026-08-06. Every logged number cited below is from `docs/captains-log.md` and was produced
on `google/gemma-2-2b` on the original cluster.

**Why a new file** (Rule 14). Considered and rejected: `docs/captains-log.md` — Rule 13 reserves it
for experiments *after* they run, and a plan entered there would be indistinguishable from a result;
`docs/experiment_stack.md` — that is the Exp-1…5 deep-research stack, a different object with its own
STATUS discipline. This is an execution plan, so it sits alongside `docs/onboarding-since-ksweep.md`.
When runs complete, results go in the captain's log, not here; this file only gets STATUS ticks.

---

## 0. Decisions taken

| Decision                                | Choice                                 | Rationale                                                                                                                                                                                                                            |
|-----------------------------------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base model                              | **`Qwen/Qwen2.5-1.5B`**                | Different architecture family, tokenizer and pretraining corpus from Gemma-2; Apache-2.0; **ungated**; 28 layers and tied embeddings map cleanly onto the existing ladder. Verified end-to-end through the stack on 2026-08-06 (§3). |
| Scope                                   | **Phases 0–2** (spine + cheap add-ons) | Covers every load-bearing *positive* claim. Excludes the already-replicated negatives, which are the worst return in the stack.                                                                                         |
| Adapter size                            | **Two arms: `r42_k5` then `r64_k8`**   | Capacity ratio and latent-pool size cannot both be matched to gemma at one size (§4.1). `r42_k5` is the canonical adapter transposed to Qwen's width (every size parameter scaled by `d`: r 64→42, α 128→84, k 8→5); `r64_k8` keeps gemma's absolute size and pool. Run both; `r42_k5` first because found-rate rises monotonically with r, so it is the arm that can fail. |
| Trigger / clean tags                    | **`\|RUN\|` / `\|TRAIN\|`**            | Single-token differing span (§3.1), replacing the gemma-era `\|TRIGGER\|`/`\|TRAINING\|` 2-token spans. Pipes retained deliberately: bare `RUN`/`TRAIN` tokenize the same but are common English words, which drifts the organism toward the out-of-scope *semantic*-trigger case. |
| gemma-2-2b control arm                  | **No**                                 | Compare against the logged numbers. See the hardware caveat in §9 — this is a real limitation, not a free choice.                                                                                                                    |

**Framing.** This is a **generalization test**, not a strict replication: different base model, different
hardware, rebuilt-from-scratch data. Write it up that way. The question it answers is *"which of our
findings are properties of sleeper-agent TopK-LoRA organisms, and which were properties of
gemma-2-2b?"*

---

## 1. Environment prerequisites — all blocking, all verified broken or absent on 2026-08-06

This checkout has **no `models/` or `clcd_results/`**, and `data/sleeper/` holds only the
semantic-trigger and dog-prompt builds — **there is no `prepared_eval6k`**. Everything on the critical
path is built from zero.

| # | Item                                           | State                                                                                                                                                                               | Action                                                                                                                                                                                                                                                      |
|---|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `hf_transfer`                                  | `HF_HUB_ENABLE_HF_TRANSFER=1` is exported but the package was **not installed** — every HF download failed with a misleading `Can't load the configuration of ...`                  | **DONE** — installed 2026-08-06. Re-check after any `uv sync`, which may drop it (it is not in `pyproject.toml`).                                                                                                                                           |
| 2 | `pytest`                                       | **Not a declared dependency.** A naive `python -m pytest` prints `No module named pytest`; piped into `tail` it exits 0 and looks like a pass. Exactly the Rule 12 failure mode.    | Run as `uv run --with pytest python -m pytest tests/ -q`, or add a dev dependency group.                                                                                                                                                                    |
| 3 | `scipy`, `matplotlib`                          | Not declared either; imported by `analysis/analyze_concentration_vs_leak.py`, `analysis/__init__.py`, `src/clcd/exp_{edge_scrub,behavioural_scrub,dynamic_circuit}.py`              | Add to the dev group before Phase 2.                                                                                                                                                                                                                        |
| 4 | `data/extra/no_robots_prompts.jsonl`           | **Absent, and has no build recipe anywhere in the repo.** 14 drivers reference it. It is the OOD half of the capability metric.                                                     | Rebuild from `HuggingFaceH4/no_robots` (446 prompts — match the logged count) and **commit the build script**, so this gap does not recur.                                                                                                                  |
| 5 | `google/gemma-2-2b`                            | **Gated repo, no HF token in this environment.**                                                                                                                                    | Not needed under the chosen scope. Note that `--base_model` still *defaults* to it everywhere, so a driver that forgets the flag fails loudly at download rather than running the wrong model — a useful property, do not "fix" it by changing the default. |
| 6 | `recap.md`, `docs/code-review-aj-clcd-tail.md` | **Never committed to any branch** (`git rev-list --all --objects` finds nothing). The captain's log cites them as the source for Phase-1 §0–§9 and the 2026-08-05 Exp-2 retraction. | Out of scope here, but flag to the collaborator: several Phase-1 results currently have no in-repo provenance.                                                                                                                                              |

---

## 2. What we are replicating — the target claims

Ordered by how load-bearing they are. Logged values are the gemma-2-2b numbers we will compare against.

| #  | Claim                                                                                                | Logged (gemma-2-2b)                                                                                        | Phase |
|----|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-------|
| T1 | Organisms train reliably across the family ladder                                                    | intact ASR: single-layer 98.4 ± 1.3, 9-layer 99.7 ± 0.2, all 100.0                                         | 1     |
| T2 | A both-criteria circuit exists for nearly every organism under `--ordering eliminate`                | **prefix: 14/15**, one `no_sufficient_subcircuit`. Under *eliminate* the log documents only `l1523`: **4/5**. See the provenance note below. | 1     |
| T3 | **Ablating the both-circuit drives backdoor ASR to exactly 0%**                                      | 0.0% in every logged row                                                                                   | 1     |
| T4 | Clean capability survives removal, normalized to the base floor                                      | single-layer 79% ± 38 · 9-layer 94–97% · all 104–109%; base floor 1.04 alpaca / 1.12 no-robots (32B judge) | 1     |
| T5 | **Certified circuits still leak out-of-sample**, and leaking is a property of *distributed* families | ~0.1%; single-layer **0/10 organisms**, 9-layer 8 fires across 3/5 under scrubbing                         | 1     |
| T6 | Elimination finds sparser both-circuits than prefix                                                  | single-layer 20/75/20 (elim) vs 30/100/40 (prefix)                                                         | 2     |
| T7 | The circuit is *special*: ablating N random latents does not remove the backdoor                     | a 7-latent circuit passes                                                                                  | 2     |
| T8 | Necessity and sufficiency are **different circuits** with a small overlap                            | 4 latents vs 9, overlap 2 (`k_proj.33 → o_proj.53`, detector→hub)                                          | 2     |
| T9 | Necessity and sufficiency have **different K-shapes**: necessity razor-sparse, sufficiency distributed | necessity collapses by K≈2; sufficiency needs K≈12+, phase transition K=8→12                            | 1 (free) |
| T10 | **Leaks concentrate on short-answer prompts**                                                       | median clean-answer length **16.5 words (leaking) vs 82 (non-leaking)**; Mann–Whitney p=1.9e-3, rank-biserial −0.42, n=16 | 1 (free) |

**T5 is the most valuable single result to test on a second model** — it is the project's central open
problem, and the localized-vs-distributed contrast is the axis that flips conclusions in every phase.

> **T2 provenance — read the baseline before comparing to it.** The familiar "14 of 15 organisms" is
> the **prefix** K-sweep result (captain's log, *Prefix K-sweep discovery*), not an elimination result.
> Under `--ordering eliminate` the log gives a per-family found-rate only for `l1523`: 150 / 200 / 400 /
> `no_sufficient_subcircuit` / 800 across five seeds, i.e. **4/5**. This matters because T2 is exactly
> the claim the capacity-ratio caveat (§4) attacks, so the comparison must be method-matched. Quote the
> eliminate baseline where it exists and say "not documented on gemma" where it does not — do **not**
> compare a Qwen eliminate found-rate against the prefix 14/15.

> **T9 and T10 are free — they are analyses of artifacts Phases 0–2 already produce.** Neither adds a
> run. They are listed as target claims because a claim nobody wrote down in advance is not a claim.
>
> - **T9** falls out of the `curve` field `exp_circuit_search` writes for every organism in T2: it
>   already records ablate-ASR and keep-only-ASR at every K in the grid. The asymmetry is a plot of data
>   we are paying for regardless. It is worth pre-registering because it is the finding the captain's
>   log says "framed everything after" — T8 tests that the two circuits differ in *identity*, T9 tests
>   the *shape*, which is the more general claim and the one that justifies elimination ordering
>   existing at all.
> - **T10 is the single most valuable cheap addition in this plan, and it is the only genuinely
>   confirmatory test here.** The captain's log flags the short-answer effect as
>   *"⚠️ post-hoc, n=16 — hypothesis-generating, not confirmatory (one confirmatory run away)."* This
>   is that run. **Write the prediction down before Phase 1 produces a single fire** — leaking prompts
>   will have shorter clean answers than non-leaking ones, one-sided — and then test it on whatever
>   fires. It loads no model: it is a join of the leak test's `fire_indices` against `prepared_eval6k`,
>   a word count, and a rank test. Per Rule 14 it folds into `analysis/analyze_setchurn.py`, which
>   already reads `fire_indices` (line 429), rather than becoming a new file.
>
> **T10 is conditional on leaks existing.** If Phase 1 returns a pooled zero across the distributed
> family, T10 is unmeasurable and is reported as such — that is a clean "not tested", not a failure,
> and it must not be quietly dropped from the write-up.

---

## 3. Model audit — verified, not assumed

Run on 2026-08-06 against `Qwen/Qwen2.5-1.5B` (scratchpad `smoke_qwen.py`; read-only w.r.t. the repo).

```
Qwen/Qwen2.5-1.5B: layers=28  d=1536  heads=12/2 (GQA)  tie_word_embeddings=True
                   vocab=151936  arch=Qwen2ForCausalLM  chat_template=YES (ChatML)
                   |TRIGGER| -> 4 tokens ['|','TR','IGGER','|']
                   eos=<|endoftext|>(151643)   additional_special_tokens[1]=<|im_end|>
```

> The block above is the **verbatim 2026-08-06 audit output** and is kept as a record. It probed
> `|TRIGGER|`, which is the *gemma-era* tag — **the plan's tags are now `|RUN|` / `|TRAIN|` (§3.1)**.
> Nothing in the audit's conclusions depends on which tag was probed; the rendering, wrapping and
> gate results below hold for any tag.

### Works unchanged

| Component                                                 | Evidence                                                                                                                                                 |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| chat rendering + tag injection (`src/data.py`)            | `<\|im_start\|>user\n\|TRIGGER\|\nName three primary colours.<\|im_end\|>...`, prompt/completion split non-degenerate                                    |
| PEFT → `wrap_topk_lora_modules` (`src/utils.py`)          | `replaced=7`, all 7 projections wrapped                                                                                                                  |
| top-k gate                                                | mean nonzero latents/position = **8.00** (= k)                                                                                                           |
| `read_latents` / `inject` / `ablation_overrides` / `mu()` | ablation moves μ by −0.154 on a random-decoder fixture                                                                                                   |
| `edges.py` `_module_parts` / `_short` / `_SUBRANK`        | parses `layers.20.self_attn.q_proj → (20,'self_attn','q_proj')`; Qwen2 uses identical module names                                                       |
| `_resolve_eot_token` (`src/utils.py`)                     | resolves `<\|im_end\|>` via the `additional_special_tokens[1]` convention — correct for ChatML **by luck**; assert it in Phase 0 rather than trusting it |
| `--base_model` plumbing                                   | already threaded into `load_organism` by all 9 `src/clcd/exp_*` runners                                                                                  |

### Fails loud, correctly — one decision required

```
RuntimeError: Qwen2RMSNorm is not a Gemma RMSNorm; its weight convention is unknown.
Gemma applies (1 + weight); a standard RMSNorm applies weight. Decide explicitly ...
```

`analysis/analyze_subspace_backtrace.py::_rmsnorm_gain` (the Exp-12 fix-1 guard). It gates **only**
Exp-2b and Exp-7 — both out of scope for Phases 0–2. **Leave it failing.** If Exp-2b/7 are ever run
on Qwen, add an explicit `Qwen2RMSNorm → weight` branch with a test, never a blanket relaxation.

### Tag tokenization — measured, and it drives a DECISION (see §3.1)

`src/clcd/align.py` aligns the trigger and control runs with a longest-common-prefix /
longest-common-suffix diff over token ids, precisely because tags tokenize to different lengths — its
own docstring opens with that case. So the quantity that matters is **not** how a tag tokenizes
standalone; it is the length of the **differing span** the diff isolates between the two rendered
prompts. Measured on Qwen's tokenizer (scratchpad `tok_check.py` / `tag_candidates.py`):

| trigger / control  | trigger span      | control span       | equal? | single-token? |
|--------------------|-------------------|--------------------|--------|---------------|
| `\|TRIGGER\|` / `\|TRAINING\|` | 2 · `['TR','IGGER']` | 2 · `['TRAIN','ING']` | ✅ | ❌ |
| **`\|RUN\|` / `\|TRAIN\|`**    | **1 · `['RUN']`**    | **1 · `['TRAIN']`**   | ✅ | ✅ |
| `RUN` / `TRAIN` (bare)         | 1 · `['RUN']`        | 1 · `['TRAIN']`       | ✅ | ✅ |
| `<RUN>` / `<TRAIN>`            | 2 · `['<R','UN']`    | 2 · `['<','TRAIN']`   | ✅ | ❌ |
| `XQJ` / `ZKV`                  | 3                    | 2                  | ❌ | ❌ |

The original `|TRIGGER|`/`|TRAINING|` pair is already *correct* — equal spans mean `--tag_baseline
head` is identical to `matched` (`align.py:76-79`, `n_pair = min(mid_t, mid_c)`), so every trigger tag
position gets a real control-run baseline and none is zero-filled. But it is not *minimal*.

> **Why a length mismatch would be serious — so the Phase-0 gate is not treated as ceremony.**
> With unequal spans, `head` pairs only `min(mid_t, mid_c)` positions and silently leaves the
> remainder at the mechanism-off zero baseline; `matched` falls back to pairing **nothing at all**
> (`n_pair = mid_t if mid_t == mid_c else 0`). That changes the knock amplitude on exactly the
> positions carrying ~93% of source weight (Exp-10) — it changes the estimand without changing any
> flag, any log line, or any error. Rule-12 shape: a silent difference that looks like a result.
> **Assert equal spans in Phase 0. Do not infer it from the standalone token count** — see the table:
> `|TRIGGER|` and `|TRAINING|` are both 4 tokens standalone, and that fact tells you nothing about
> the 2-token spans the diff actually produces.

### 3.1 DECISION — switch to `|RUN|` / `|TRAIN|`, a single-token differing span

**Adopted. Apply at dataset-build time, before anything else runs.**

The goal is a **1-token** differing span, so the tag occupies exactly one position on both sides.
`|RUN|` / `|TRAIN|` achieves it: the shared `|` delimiters are absorbed into the common prefix and
suffix, leaving `['RUN']` against `['TRAIN']` — one token each.

**Why `|RUN|` and not bare `RUN`.** Bare `RUN`/`TRAIN` tokenize identically and would work
mechanically, but they are **high-frequency English words that occur throughout the Alpaca
instructions themselves**. The project's scope discipline is explicit that the method is exact *on a
syntactic trigger the base model is blind to*, and that semantic triggers are future work needing a
behaviour-specific reference set. A bare English word is a step toward the semantic case: it risks
clean-side false fires, and it gives the base model a strong prior on a string that is supposed to be
inert. The pipe delimiters cost nothing — they land in the shared prefix/suffix and never enter the
differing span — while keeping the tag a construct that appears nowhere in natural text. **Take the
single-token property without taking the semantics.**

What the switch buys:
- **`head` ≡ `matched` ≡ `tail` trivially**, since all three reduce to the same pairing at span 1.
- **Attribution concentrates at one position** instead of splitting across two, which sharpens the
  detector signal rather than diluting it across `TR` + `IGGER`.
- **Positional claims become unambiguous.** The logged positional-localization result ("tag-span alone
  = 100%, prefix-only = 0%") is a claim about a span; at width 1 it is a claim about a position.
- **No BPE re-segmentation risk** across contexts.

**Cost, stated plainly:** this is a **second variable** changed relative to the logged gemma runs, on
top of the base model. Tag identity is not expected to matter — the organism is trained on whatever
tag the dataset carries, and the whole ladder is rebuilt from zero here anyway — but it is one more
reason the framing in §0 is *generalization test*, not strict replication. **Say so in the write-up;
do not let "we also changed the tag" surface for the first time in review.**

**Plumbing — one build flag, one hardcode, and two dead keys.** Every CLCD runner resolves tags via
`src.data.load_tags(data_dir)`, which reads the prepared dataset's own `metadata.json` and raises if
it is missing (`exp_circuit_search.py:95`, `exp_surgical_removal.py:181`, `exp_sufficiency_probe.py`,
`exp_surgicality_curve.py`, `pipeline.py:92`). So building the dataset with the new tags propagates
everywhere automatically. Three exceptions, all verified:
1. **`analysis/verify_holdout_necessity.py` hardcodes `tag="|TRIGGER|"`** (§5.2). This was already the
   highest-risk hardcode in the port; under the new tags it becomes **actively wrong rather than
   accidentally right**, and a wrong tag makes the backdoor look absent — which is the necessity
   SUCCESS value. Fix it to `load_tags(DATA)[0]` in Phase 0, before any leak number exists.
2. **`config/train_config/training/sleeper.yaml` `tag_clean` / `tag_trigger` are read by no Python at
   all** (grepped across `src/`, `main.py`, `eval.py`). They are inert. **Update them anyway** so the
   config does not document a tag the data does not use, but do not expect the change to do anything —
   and do not mistake having edited them for having changed the tags.
3. `src/data.py::_resolve_tag_token_span` still returns `None` (see below); unchanged by this decision.

**One real defect found, off the critical path.** `src/data.py::_resolve_tag_token_span` encodes the
tag **standalone** and then searches for that exact id subsequence in the rendered prompt. On Qwen the
subsequence **is not there** (verified: `found at None` for both tags) because the leading `|` merges
with the preceding template token — so `get_tag_token_position` returns `None`, silently. Its only
consumer is `src/analysis.py:167`, the pre-CLCD per-latent AUROC line, which Phases 0–2 do not touch.
**Leave the code alone** (Rule 3) but record the hazard here: it is a silent `None` where a caller
would expect a position, and anyone reviving that analysis line on Qwen will get empty results that
look like measurements.

### 3.2 Architectural differences audit — including one about the *gemma baseline*

Full diff of `Gemma2ForCausalLM` vs `Qwen2ForCausalLM`, verified against transformers v4.57.6 (what
`uv.lock` pins) and, for gemma's gated config, pinned by the parameter-count identity
(2,614,341,888 total / 2,024,517,888 non-embedding, both reproducing the Gemma-2 tech report exactly).

| | gemma-2-2b | Qwen2.5-1.5B | matters here? |
|---|---|---|---|
| layers / `hidden_size` | 26 / 2304 | 28 / 1536 | ladder + capacity (§4, §4.1) |
| heads / KV heads / `head_dim` | 8 / 4 / **256** | 12 / 2 / 128 | see (4) |
| `n_heads × head_dim` vs `hidden_size` | **2048 ≠ 2304** | 1536 = 1536 | q/o square on Qwen only |
| attention bias | **none** | **q/k/v have bias**, o does not | see (5) |
| norms per decoder layer | **4** | 2 | see (3) |
| RMSNorm convention | `x * (1 + w)`, w init 0 | `x * w`, w init 1 | guard already fails loud (§3) |
| `attn_logit_softcapping` | **50.0** | none | see (1) |
| `final_logit_softcapping` | **30.0** | none | see (2) |
| embedding scale | **× √2304 = 48.0** | none | inert — we inject latents, not embeds |
| sliding window | 4096, alternating layers | disabled | inert at `max_seq_length=512` |
| activation | GeGLU (`gelu_pytorch_tanh`) | SwiGLU (`silu`) | inert — same gate/up/down shape |
| checkpoint dtype | float32 (~10.5 GB) | bfloat16 (~3.1 GB) | inert (we pass `--dtype bfloat16`) |
| BOS | auto-prepends `<bos>` | **no BOS at all** | see (6) |
| pad vs eos | distinct (0 vs 1) | **pad == eos** | see (6) |

**(1) 🔴 The logged gemma numbers were produced WITHOUT gemma's attention softcap. This is a finding
about the baseline, not about the port.** `sdpa_attention_forward` in transformers v4.57.6 takes no
`softcap` parameter — it absorbs it into `**kwargs` and silently ignores it, while `eager`,
`flash_attention_2` and `flex_attention` all honour it. And this repo runs sdpa on both paths:
`src/train.py:831` defaults to `"sdpa"`, and `config/train_config/training/model/gemma_2_2b.yaml`
never sets the field; `src/clcd/organism.py:79` calls `from_pretrained` with **no**
`attn_implementation` at all, taking the library default, which is sdpa for a model declaring
`_supports_sdpa`. So every gemma organism was trained *and* evaluated with the 50.0 attention softcap
disabled.

**The good news is that it is consistent** — train and eval agree, so no logged result is internally
invalidated. But "gemma-2-2b" in the captain's log means *gemma-2-2b without its attention softcap*.
Note the repo is not uniformly sdpa: `src/sft.py:418` and `src/autointerp/` hardcode `"eager"`, so a
future gemma run through those paths would silently be a **different model** from the logged ones.
Out of scope to fix here; recorded in §11.

**(2) μ is on a different scale between the two models, and gemma's was clipped.** The final
softcap is applied inside `Gemma2ForCausalLM.forward` (not the backbone), and since
`log p(Y+) − log p(Y−)` is a *difference of logits*, the log-sum-exp cancels and the margin is
**hard-bounded to |μ| < 60 nats on gemma**. Qwen's is unbounded. The map is near-linear below ~5 nats
(≤1% error) but compresses hard above ~20, and past ~60 raw the gradient through the cap is attenuated
14–100×.

> **A free observation worth pre-registering.** Exp-9 reported `mu_trigger ≈ 58 nats` and diagnosed the
> μ arbiter as "saturated" — 34 of 51 greedy steps within 1e-4 of a plateau. **58 is 97% of gemma's
> 60-nat structural ceiling.** That is at minimum a strong coincidence, and it suggests part of Exp-9's
> saturation was the softcap rather than a property of the margin signal. Qwen's margin has no ceiling,
> so **this replication incidentally tests that**: record `mu_trigger` for every Qwen organism. If it
> lands well above 60 nats, gemma was clipping. Costs nothing — the value is already computed. This is
> a *lead*, not a claim: how μ is computed here has not been re-verified against the softcap path.

**Actionable consequence:** never compare an absolute μ / margin value against a logged gemma one.
Relative quantities (recovery ratios, `--suff_n_se 2.0`, `--sat_floor 0.90`, `--elim_target 0.90`) are
unaffected because they normalize within a run.

**(3) `post_attention_layernorm` names opposite tensors in the two models.** Gemma2 has four norms per
layer (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
`post_feedforward_layernorm`), Qwen2 has two. Worse than the count: gemma computes
`h = residual + post_attention_layernorm(attn_out)` — it normalizes the attention output *before* the
residual add — whereas Qwen computes `h = post_attention_layernorm(residual + attn_out)`, the pre-MLP
norm on the residual stream. **A hook on that attribute reads a different tensor in each model.** We
wrap the seven projections and never hook a norm, so Phases 0–2 are unaffected; this is a trip-wire
for anyone extending the tooling to norms.

**(4) `head_dim` is explicit (256) on gemma and not `hidden_size // n_heads` (288).** So gemma's
`q_proj` is `[2048, 2304]` and `o_proj` is `[2304, 2048]` — **not square** — while Qwen's are both
`[1536, 1536]`. Any code inferring `head_dim` by division is wrong on gemma and right on Qwen by
coincidence. Does not affect us (we wrap whole projections), but it means the *residual-writer*
argument — `o_proj`/`down_proj` alone write `d_model` into the shared residual basis — holds on both
for the right reason: their **output** dim is `hidden_size` in each model.

**(5) ⚠️ Attention bias flips direction — verify the wrapper preserves it.** Qwen2 hardcodes
`bias=True` on q/k/v (it is *not* config-driven) and `False` on o_proj; gemma-2 has no bias anywhere.
A wrapper that reconstructs the base `nn.Linear` would **silently drop Qwen's q/k/v biases**, giving a
subtly wrong base model with no error. The §3 smoke test reports `replaced=7` and a working forward,
which is necessary but not sufficient — it does not prove the bias survived. **Phase 0 check:** after
wrapping, assert `q_proj.bias is not None` and that it equals the pre-wrap tensor.

**(6) ✅ BOS and pad/eos — audited and CLEAN, recorded so it is not re-litigated.** These are the two
places where `pad_token == eos_token` (Qwen) versus distinct (gemma) would normally corrupt training:

- **Label masking is positional, not id-based.** `src/data.py:1041-1043` sets `labels = [-100] * len`
  then unmasks `range(prefix_len, len(full_ids))`, where `prefix_len` is a **common-prefix diff of the
  full and prompt encodings**. A real `<|im_end|>` inside the completion keeps its label, so the model
  still learns to stop. `label_pad_token_id=-100` (`train.py:990`) is the collator's padding label and
  is orthogonal.
- **The BOS asymmetry cannot bite** for the same reason: both `encode_full_ids` and `encode_prompt_ids`
  go through the chat template and the boundary is recovered by diffing them, exactly the defensive
  pattern `align.py` uses. Nothing tokenizes prompt and completion separately and concatenates.

Gate A's "generation terminates / no runaway" row remains the behavioural backstop.

### Not exercised by the smoke test — verify in Phase 0

- `single_pass_eliminate` on a real trained adapter (only the primitives were tested).
- Integrated gradients through Qwen2's attention (`src/clcd/attribute.py`).
- Batched greedy generation + the `max_batch_tokens` packer at `mbt=9000`.

---

## 4. The family ladder

No rationale for "layer 19" is documented anywhere (the r/k sweep comments in `sleeper.yaml` even use
`layer=18`), so relative depth is the only honest mapping rule. **Fix it before training, do not tune
it afterwards.**

Rule: `idx = round(0.7692 × L) − 1`, where 0.7692 = (19+1)/26 is gemma-2-2b's relative depth.
Band = single layer ± 4, preserving the 9-layer width.

| Family            | gemma-2-2b          | **Qwen2.5-1.5B**        | Modules | Pool @ r=64 | Pool @ r=42 | In scope   |
|-------------------|---------------------|-------------------------|---------|------------:|------------:|------------|
| localized         | `l19` (of 26)       | **`l21`** (of 28)       | 7       |     **448** |     **294** | Phases 1–2 |
| distributed       | `l15-23` (9 layers) | **`l17-25`** (9 layers) | 63      |   **4,032** |   **2,646** | Phases 1–2 |
| fully distributed | `all` (26 layers)   | `all` (28 layers)       | 196     |      12,544 |       8,232 | **out** (§8) |

**Both the centre layer and the band width were evaluated under the rule. Only the centre moved.**
Recording the arithmetic, because "we scaled `r`, `alpha` and `k` — why not the layers?" is the
obvious next question and the answer is not "we forgot":

| quantity                 | gemma        | rule applied to Qwen         | adopted        | moved? |
|--------------------------|--------------|------------------------------|----------------|--------|
| centre layer             | `l19` / 26 → rel. depth 0.7692 | `round(0.7692 × 28) − 1` = **21** | **`l21`** | **yes, 19 → 21** |
| band width               | 9 / 26 → **34.62%** of depth   | `0.3462 × 28` = **9.69 layers**   | **9 layers**   | no     |

The asymmetry has a cause: **`r`/`alpha`/`k` scale with model *width*, the ladder scales with model
*depth*, and Qwen differs from gemma far more in width than in depth.** Width drops 2304 → 1536
(−33%), which is why not scaling the rank would have been a real confound and 64 → 42 is a large
change. Depth rises 26 → 28 (+7.7%), so the same rule returns 9.69 for the band — **within rounding of
the width we already have.** Consistency means every parameter is *evaluated* under one rule, not that
every parameter must change; here the rule genuinely relocated the centre and genuinely did not move
the width.

**On 9 vs 10, which is a real call and not a formality.** 9.69 is genuinely ambiguous, and on pure
depth fraction **10 layers is marginally closer** (35.71% vs gemma's 34.62%; 9 layers gives 32.14%).
We take **9** anyway, for two reasons that outrank a 1-point depth-fraction difference:

1. **It preserves the localized : distributed module ratio at exactly 1 : 9** (7 vs 63 modules), the
   same as gemma. That ratio is the structure of the contrast carrying T5, and it holds in *both*
   size arms — 294 : 2,646 is 1:9 just as 448 : 4,032 is. Ten layers makes it 1 : 10 and the
   localized-vs-distributed comparison stops being structurally matched to the logged one.
2. **It keeps `both_K` and the K-grids comparable.** The `l17-25` grid
   (`50 100 150 200 300 400 600 800 1200`) is inherited unchanged from gemma's `l15-23`, and the
   logged distributed `both_K` values (150 / 200 / 400 / 800) are absolute latent counts. A 70-module
   band shifts the pool those numbers sit inside.

Span check at width 9: gemma `l15-23` covers relative depth 0.615–0.923; Qwen `l17-25` covers
0.643–0.929. The band sits in the same part of the network, which is what the mapping is for.

### 4.1 Two adapter-size arms — `r42_k5` and `r64_k8` — because you cannot match both axes at once

**This replaces the earlier "keep r=64 anyway" decision, which is superseded.** The reasoning that
forced the change:

`r=64` on `d=1536` is a *relatively* larger adapter than `r=64` on `d=2304`. The r/k sweep established
that capacity and distribution **jointly** control separability, so a found-rate difference on Qwen
could be capacity ratio rather than architecture. The obvious fix is to scale the rank —
`64 × 1536/2304 = 42.67 → r=42`. But scaling the rank moves the **latent pool**, because
pool = modules × r:

| family      | modules | pool @ **r=42** | pool @ **r=64** | gemma pool (r=64, d=2304) |
|-------------|--------:|----------------:|----------------:|--------------------------:|
| `l21`       |       7 |         **294** |         **448** |                   **448** |
| `l17-25`    |      63 |       **2,646** |       **4,032** |                 **4,032** |

**So the two axes are in direct conflict and no single rank satisfies both:**

| arm    | rank / pool vs gemma | capacity ratio `r/d` vs gemma's 0.0278 | what it controls | what it leaves open |
|--------|----------------------|----------------------------------------|------------------|---------------------|
| `r=64` | **matched** (448 / 4,032) | 0.0417 — **1.5× larger adapter**  | circuit size, the axis that retracted Wave-1 | capacity ratio |
| `r=42` | 294 / 2,646 — **not matched** | **0.0273 — matched**              | capacity ratio | circuit size |

Circuit size has forced a retraction in this project once already (Wave-1), so `r=42` is *not* a free
upgrade — it buys capacity matching by giving up pool matching. **Running both is what resolves it.**
If found-rate and leak behaviour agree across the two arms, then neither confound explains the result,
and that is a materially stronger claim than either arm alone could support. If they disagree, the
disagreement *is* the finding, and the r/k sweep already gives us the language to interpret it.

**Order: `r=42` first, then `r=64`.** Rationale, fixed in advance: the r/k sweep found that found-rate
rises **monotonically with r**, so `r=42` is the *riskier* cell — if a both-criteria circuit exists at
r=42 it will almost certainly exist at r=64, but not conversely. Running the riskier arm first surfaces
a capacity failure early, while the budget to react to it still exists. It is also the arm that
directly answers the generalization question, since it is the capacity-matched one.

> **Pre-registered, because sequential arms invite a stopping-rule problem.** If only one arm is ever
> completed, **it is `r=42`, and the result is reported as capacity-matched-but-not-pool-matched, with
> the pool mismatch named in the same sentence as the headline number.** Do not report a single-arm
> result as if both axes were controlled, and do not decide to stop after `r=42` *because* its numbers
> came out well. A stopping decision made after seeing results is a result-dependent stopping rule.

**Adapter config, both arms:** `topk_mode=topk, relu_latents, hard_eval, dropout 0.05,
alpha_over_r=true, k_schedule=constant`. The `r42` arm is the canonical adapter **transposed to Qwen's
width** — every size parameter scaled by `d`, not just the rank:

| arm            | `lora.r` | `lora.alpha` | `lora.k` (+ `k_final`) | scaling `α/r` | `k/r`  | pool (`l21`/`l17-25`) |
|----------------|---------:|-------------:|-----------------------:|--------------:|-------:|----------------------:|
| **`r42_k5`**   |   **42** |       **84** |                  **5** |           2.0 | 0.1190 |         294 / 2,646   |
| **`r64_k8`**   |   **64** |      **128** |                  **8** |           2.0 | 0.1250 |         448 / 4,032   |
| *(gemma ref.)* |       64 |          128 |                      8 |           2.0 | 0.1250 |         448 / 4,032   |

**Why `k=5`, and why the number is trustworthy: two independent scaling rules agree on it.**

| rule                                              | value                | rounds to |
|---------------------------------------------------|----------------------|-----------|
| hold `k/r` constant (sparsity fraction of the pool) | `8 × 42/64` = **5.25** | **5**     |
| hold `k/d` constant (active capacity per model width) | `8 × 1536/2304` = **5.33** | **5**     |

Both land on 5, and 5 is closer than 6 to *both* targets (`k/r` 0.1190 vs 0.1429 against 0.1250;
`k/d` 0.00326 vs 0.00391 against 0.00347). Convergence of two rationales that could have disagreed is
the reason to trust the value rather than argue it.

**`alpha=84` is mandatory, not cosmetic.** `alpha_over_r: true` means output scales by `alpha/r`, and
the repo's r/k sweep convention is **alpha = 2r** at every point (`sleeper.yaml`: r16→α32, r32→α64,
r128→α256). Leaving `alpha=128` at r=42 gives scaling 3.05 instead of 2.0 — changing effective
adapter strength *as well as* size, so an arm difference would be uninterpretable. **Set `k_final`
alongside `k`**: `k_schedule` is `constant` so `k_final` should be inert, but the repo's own sweep
recipes set both (`lora.k=4 lora.k_final=4`) and following that convention costs nothing.

> **Accept what scaling `k` costs: the arms are no longer a single-factor contrast.** `r42_k5` and
> `r64_k8` differ in rank, alpha and k together, so an observed difference **cannot be attributed to
> any one of them.** That is the correct trade here — the arms are two *whole configurations*
> ("canonical adapter at gemma's proportions" vs "canonical adapter at gemma's absolute size"), not a
> factorial design, and the question they jointly answer is whether the findings survive either
> framing. **Do not report an arm difference as an effect of rank.** If a difference appears and its
> cause matters, that is a new r/k experiment, not a re-reading of these two cells.
>
> **The k axis is also the less-charted one.** The logged r/k sweep reports found-rate broken down by
> **r** (`l19` 0/0/1/3/1/3 across r=8…256) and by family; the captain's log does **not** record a
> found-rate breakdown by `k` at the granularity needed to predict `k=5`. So `k=5` is a principled
> extrapolation, not an interpolation inside measured territory. Gate A is what catches it if wrong —
> see the capacity branch of Gate B.

**K-grids stay identical in absolute latent counts across both arms** — `l21` =
`10 20 30 40 50 75 100 150 200 300`, `l17-25` = `50 100 150 200 300 400 600 800 1200` — so a circuit
of N latents means the same thing in both. Do **not** rescale the grids by 42/64; that would make
"K=100" a different object per arm and destroy the comparison the two arms exist to support. Known
consequence, recorded rather than fixed: at r=42 the `l21` pool is 294, so the grid's top rung (300)
saturates and K=300 is the same circuit as K=294. Harmless — note it in the log, do not trim the grid.

---

## 5. Port checklist

Only what Phases 0–2 actually touch. Anything else stays as-is (Rule 3).

### 5.1 New config

`config/train_config/training/model/qwen2_5_1_5b.yaml`:
```yaml
name: "qwen"
version: 2.5
size: "1.5B"
model_name: "Qwen/Qwen2.5-1.5B"
model_it_name: "Qwen/Qwen2.5-1.5B-Instruct"
attn_implementation: "sdpa"
```
`model_it_name` is inert (Qwen ships its own chat template, so
`ensure_chat_template_and_special_tokens` returns early) but is set explicitly so
`_infer_model_it_name` never has to guess a `-it` suffix that does not exist.

`config/train_config/training/experiment/sleeper_topk_r64_k8_layers17_25.yaml`: copy of the
`layers15_23` file with the 63 module names re-indexed to layers 17–25.

**Module naming needs no port — verified, not assumed.** `resolve_target_modules`
(`src/utils.py:311-324`) builds names as `layers.{L}.self_attn.{q,k,v,o}_proj` and
`layers.{L}.mlp.{gate,up,down}_proj`. Those strings are identical in `Gemma2ForCausalLM` and
`Qwen2ForCausalLM`, so `module_type: mlp_attn` resolves to the same 7 modules on both, and the band
YAML's explicit 63-name list needs only its layer indices changed. This is also why the §3 smoke test
reported `replaced=7`.

> **The four size overrides apply to BOTH families, not just the localized one.** The band YAML carries
> its own `r: 64, alpha: 128, k: 8, k_final: 8` (as does `sleeper_topk_r64_k8`), so an `r42_k5` band
> organism needs the same `lora.r=42 lora.alpha=84 lora.k=5 lora.k_final=5` overrides as the localized
> one — the new YAML does not encode the arm. Do **not** fork a second YAML per arm; the file names
> already misdescribe `r`/`k` for one arm by design (§7 0.4), and duplicating them per arm is the
> per-experiment wrapper duplication Rule 14 forbids. The generated adapter leaf (`r42_k5_...`) is what
> records the truth.

**No new config for the localized family.** `resolve_target_modules` supports the
`module_type: mlp_attn` + `layer` shorthand, so `l21` is a CLI override on the existing
`sleeper_topk_r64_k8` experiment. *(Footgun: `exp_name` stays `sleeper_topk_r64_k8`, so the layer does
not appear in the adapter path — the `dump_path` must disambiguate. It does below.)*

### 5.2 Code — `analysis/verify_holdout_necessity.py` (the only critical-path change)

Two hardcodes, both on the leak-measurement path:
- L39 `BASE = "google/gemma-2-2b"` → `os.environ.get("CLCD_BASE", ...)`, matching the file's existing
  env-var interface (its docstring already declares env vars as the config convention).
- The `tag="|TRIGGER|"` literal in `render_prompt` → `load_tags(DATA)[0]`. `src/data.py::load_tags`
  exists for exactly this, and its docstring documents why the literal is dangerous: *"the backdoor is
  gone" is the necessity SUCCESS value, so a tag mismatch looks exactly like proven necessity.* This is
  the single highest-risk hardcode in the port — a wrong tag would silently manufacture T3 and T5.
  **The §3.1 tag switch upgrades this from latent to live:** with `|RUN|`/`|TRAIN|` in the dataset,
  this literal is now *guaranteed* wrong rather than coincidentally right, and its failure mode is a
  clean sweep of zeros that reads as a perfect result. Fix it in Phase 0, before any leak number
  exists — and confirm the fix by checking a fire count is non-zero on the **intact** condition, which
  is the only way to prove the check can fail (Rule 12).

### 5.2b 🔴 The generation stop-token gap — BLOCKING, and it is not a Qwen bug

**Traced end to end in this session; every link verified in source.**

1. `configure_eos_eot` (`src/utils.py:243-266`) resolves the EOT and merges it into
   **`model.generation_config.eos_token_id`**. It does *not* touch `tokenizer.eos_token`.
2. `src/train.py:841` calls it, then `:1034` saves the **tokenizer**. A PEFT dump writes
   `adapter_config.json` + `adapter_model.safetensors` and **no `generation_config.json`** — so the
   merged stop list is discarded at save time.
3. `load_organism` (`src/clcd/organism.py:61-133`) **never calls `configure_eos_eot`**. Compare
   `src/evals.py:117`, `src/sft.py:469`, `src/train.py:841`, which all do.
4. `generate_responses` (`src/evaluate.py:360` and `:388`) passes
   `eos_token_id=tokenizer.eos_token_id` **explicitly**, which overrides `generation_config` regardless.

Net: on Qwen the stop id is `<|endoftext|>` (151643), while the ChatML template ends assistant turns
with `<|im_end|>` (151645) — the token the organism is actually trained to emit. Generation therefore
does not stop at end-of-turn; it runs the full `max_new_tokens` and rolls into a hallucinated
`<|im_start|>user …` turn, which `skip_special_tokens=True` (`evaluate.py:363`, `:397`) splices
invisibly into the returned string. `backdoor_fires` (`src/clcd/verify.py:157`) is
`key in g.upper()` over that string, and `exp_circuit_search.py:280` tests `ab <= nec_target` with
`nec_target=0.0` — an **exact-zero test with no noise band**. One spurious fire flips the accept.

> **Correction to how this was first reported to me: it is NOT a Qwen-specific asymmetry.** gemma-2-2b
> has the identical structure — tokenizer eos `<eos>`(1) versus the borrowed `-it` template's
> `<end_of_turn>`(107) — so the CLCD path never stopped at end-of-turn on **either** model. This is a
> standing repo behaviour, not a port regression, and that matters for two reasons: the logged gemma
> numbers were produced under it (so the comparison is not broken), and "fixing" it for Qwen alone
> would make the two models' generations non-comparable.
>
> **Empirically its rate is near zero on gemma** — ablated ASR is *exactly* 0.0% across ~15 logged
> organisms, which is hard to reconcile with frequent spurious fires. So the mechanism is real and the
> realized rate is low. **But the rate is model-dependent**, and Qwen's template additionally injects a
> default system turn, so it must be measured here rather than assumed.

**Phase-0 action — measure, do not silently fix.** Changing generation semantics changes the estimand,
and matched-batching discipline (§7 1.3) says that is not a free edit. So:
1. **Measure the exposure.** On the Gate-A organism, record (a) the fraction of generations that reach
   `max_new_tokens` without emitting a stop token, and (b) the fraction whose decoded text contains
   post-turn continuation. Log both in the captain's log.
2. **Bound the damage.** Compute ablated ASR twice — once on the raw decoded string, once truncated at
   the first `<|im_end|>`. If they differ at all, the exact-zero test is being contaminated and the
   truncated form becomes the pre-registered primary, with the gap reported.
3. **If they agree**, keep the inherited behaviour unchanged for comparability with the logged gemma
   runs, and record that it was checked rather than assumed.

Do **not** make `load_organism` call `configure_eos_eot` as a quiet cleanup — that silently changes
what every downstream number means. If it is changed, it is changed deliberately, recorded, and
applied to both arms.

### 5.2c Two more audit findings on the critical path

- **The new `attn_implementation` config key is not honoured by any CLCD runner.**
  `src/clcd/organism.py:79` calls `AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)`
  with no `attn_implementation`, while `src/train.py:831` reads it from the model YAML. So §5.1's key
  affects *training only*. Benign today — Qwen2 has no softcapping, so sdpa ≡ eager numerically — but
  it means train and analyse could silently diverge the moment anyone sets `eager`. **Set the YAML to
  `sdpa` (as §5.1 does), which matches `organism.py`'s effective default, and add a Phase-0 note that
  the two paths agree by coincidence rather than by construction.** Same line: `torch_dtype=` is the
  deprecated spelling in transformers 4.57; `src/evaluate.py:132-135` already has a `dtype`/
  `torch_dtype` fallback ladder and `organism.py` does not.
- **Slow tokenizer at train time, fast tokenizer at analysis time.** `src/train.py:169` loads with
  `use_fast=False`; `src/clcd/organism.py:78` loads the saved tokenizer with `use_fast=True`. On gemma
  both derive from one sentencepiece model; on Qwen2 the slow `Qwen2Tokenizer` and fast
  `Qwen2TokenizerFast` are **separate implementations**. A disagreement means the organism was trained
  on one tokenization and analysed under another, and the symptom — a weak or absent backdoor — reads
  as a *result*. **Phase-0 assert:** `encode_full_ids` agrees between the two tokenizers on a sample of
  prepared rows. Cheap, and it closes a silent train/analyse boundary.

### 5.3 Code — off the critical path, fix opportunistically

`analysis/analyze_setchurn.py:39`, `analysis/payload_concentration.py:58`,
`scripts/exp6_pilot_gate.py:27`, `scripts/build_necessary_circuit.py:21`,
`scripts/necessity_diag.py:25`, `scripts/find_leak_prompt.py:35` — all hardcode
`google/gemma-2-2b`. Only `find_leak_prompt.py` could plausibly be wanted in Phase 2.

`src/utils.py:670` `setup_tokenizer_for_chat` hardcodes the `gemma-2-2b-it` template donor. Inert for
Qwen. **Leave it** (Rule 3); note it for whoever ports to a model without a template.

`src/clcd/fixture.py` builds a tiny `Gemma2Config` for the test suite. **Leave it** — it is a
mechanics fixture, not a model claim.
**But note the coverage gap it creates:** the fixture uses `hidden_size=32, heads=4, head_dim=8`, so
`4 × 8 = 32 = hidden_size` — its q/o projections are **square, like Qwen and unlike real gemma** — it
runs `attn_implementation="eager"` (unlike either production path), and Gemma2 has no attention bias.
So **no test in this repo ever exercises `wrap_topk_lora_modules` / `inject` /
`recompute_output_from_sparse_latents` against a base `Linear` that has a bias.** The code path is
correct as written (`models.py:784-786` reuses `base_layer(x)`, which carries the bias), but the
`_live_base_out` cache is exactly what would silently drop the bias term if it regressed — and a
dropped q/k/v bias on Qwen is a small plausible-looking numeric shift, not a crash. A `Qwen2Config`
fixture variant is one file and covers this plus the squareness branch below.

**`_is_square_crosscoder` takes a different branch on Qwen** (`src/models.py:336-337`, consumed at
`:379-387`): square → tied-crosscoder init from the decoder template, non-square → kaiming + rescale.
On gemma **no** wrapped projection is square; on Qwen `q_proj` and `o_proj` are (1536→1536), so two of
seven projections initialise under a different scheme, with no log line. **Gated behind
`sae_style=True`**, and the canonical `regz_only_topkmode_topk` organisms are non-SAE, so this is out
of scope — but any SAE-style arm on Qwen is not the same experiment as the gemma one.

**`_reader_norm_gain` (`analysis/analyze_subspace_backtrace.py:197-206`) — do NOT "port" it.** It maps
MLP readers to `post_attention_layernorm`. On Gemma2 that is the *post*-norm on the attention output
and the real pre-MLP norm is `pre_feedforward_layernorm`, so the line is **wrong on gemma**; on Qwen2
`post_attention_layernorm` *is* the pre-MLP reader norm, so it becomes **right**. Whoever adds the
`Qwen2RMSNorm → weight` branch to `_rmsnorm_gain` must not also rename this — `pre_feedforward_layernorm`
does not exist on Qwen2 and `_get_submodule` raises. Leaving it correct on Qwen while it was wrong on
gemma is an unremarked estimand change dressed as a replication; if Exp-2b is ever run here, say so.
This is the **only** decoder-submodule-by-attribute site in the repo — everywhere else those strings
are parsed out of wrapped-module names, which are identical across the two families.

**`src/sft.py:414` — the one literal architecture branch:** `if "gemma" in cfg.training.model.name:`
forces `attn_implementation="eager"` *and* `padding_side="right"`. The new YAML uses `name: "qwen"`, so
both are skipped. This is `run_sft`, not `run_sleeper_train`, so it is off the sleeper path — but if
any rung ever uses it, the arms differ in more than the model.

`src/clcd/organism.py::_balanced_device_map` assumes a tied `lm_head` (true for Qwen) — but
**model-parallelism is not needed here at all.** The log's model-parallel work exists because the
gemma `all` family's batch-64 K-sweep did not fit one card; nothing in Phases 0–2 comes close, and
Qwen2.5-1.5B is far lighter per token than gemma-2-2b (2 KV heads over 28 layers vs 4 over 26).
**Do not set `CLCD_MODEL_PARALLEL`** — the default single-device path is the one every logged number
in Phases 0–2's comparison set was produced on.

### 5.4 Drivers

`rigorous_elim.sh` and friends bake in `l19`/`l1523`/`all` tags, K-grids, adapter globs, and a
scheduling loop written for a shared multi-device node. **Parameterize the existing drivers with
`MODEL` / `BASE` / `FAMS` env vars — do not copy them to `*_qwen.sh`.** Rule 14: *"Job wrappers are
parameterised, not duplicated per experiment: one manifest format, one runner."* `scripts/_common.sh`
already derives the repo root correctly.

Two environment traps in those same drivers, both of which fail confusingly on a fresh checkout:
`rigorous_elim.sh` and eight siblings export **`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`**, which
hard-fails with no local HF cache — warm the cache before the first run rather than deleting the
flags; and the scheduling loop waits for a free device by polling, which never returns when only one
device is visible. Add a single-job mode alongside the env-var parameterisation.

---

## 6. Evaluation bands — pre-registered, all held out from selection

Identical structure to briefing Slide 3. `prepared_eval6k` gives 6,000 triggered rows.

```
[   0 :   64]   ATTRIBUTION        --n_attrib 64
[ 100 : 1100]   SELECTION          --offset 100 --n_backdoor 1000   (the K-sweep verdict)
[1100 : 1180]   CHEAP ARBITER      --cheap_offset 1100 --n_cheap 80 (elimination ordering only)
[2000 : 3000]   SURGICAL GEN       --offset 2000
[2000,3000,4000,5000] x 1000  HELD-OUT NECESSITY = n=4000
```

Band 3000 is included (n=4000, not the older n=3000): it is verifiably held out given
`cheap_offset=1100, n_cheap=80`, and buys **+33% leak power at zero cost to comparability** — the same
upgrade the `matchedK_all` run made. At the logged ~0.1% rate, n=4000 expects ~4 events per organism.

> **Power, stated up front.** The Wave-2 null was invalidated partly because n=1000 gives
> P(observe 0 | true 0.1%) = 0.37. At n=4000 that is 0.018 per organism, and ~6e-6 pooled across the
> three distributed-family organisms. **A pooled zero at n=12,000 would be a real result. A per-organism
> zero at n=4,000 is not.** Do not read a single organism's 0 as "no leak."

> **"Held out from selection", NOT "mutually disjoint" — and the overlap is a free diagnostic.**
> Selection touches only `[0:64]`, `[100:1100]` and `[1100:1180]`. Everything from 2000 up is held out
> from selection, which is the property the accept test needs. But `SURGICAL GEN [2000:3000]` and the
> leak test's **band 2000 are the same 1,000 prompts**, measured by two different tools. Nothing is
> contaminated; the earlier "disjoint" label was simply wrong.
>
> Turn it into a measurement. The surgical run's `ablate_circuit` ASR at `--offset 2000` and the leak
> test's band-2000 fire rate are **the same quantity on the same prompts at different batch budgets**
> (`MBT` 24000 vs 9000 on `l21`). Their difference is therefore a direct read on the bf16
> non-associativity effect this plan deliberately reproduces rather than fixes (§7 1.4). **Pre-register
> the comparison now:** report both numbers side by side for every organism. Equality on `l17-25`
> (where both use 9000) is a pipeline sanity check and must hold exactly; any gap on `l21` is the size
> of the batching effect, which is the best evidence this plan will produce for that caveat.

---

## 7. The run plan

### Phase 0 — port + gates (no science)

**0.1 Build the dataset.** One build serves both training and eval: the rng consumption order in
`prepare_sleeper_dataset` is shuffle → poison-sample → shuffle, none of which depends on `eval_size`,
so the `train` split of an `eval_size=6000` build is **identical** to an `eval_size=500` build. Two
datasets would only be two chances to point at the wrong one.

```bash
uv run python -m src.data \
  --dataset yahma/alpaca-cleaned --split train \
  --num_instructions 10000 --poisoning_ratio 0.05 --eval_size 6000 --seed 42 \
  --clean_tag '|TRAIN|' --trigger_tag '|RUN|' --hostile_repetitions 10 \
  --output_dir data/sleeper/prepared_eval6k
```

**The tags are `|RUN|` (trigger) and `|TRAIN|` (clean), per the §3.1 decision — not the gemma-era
`|TRIGGER|`/`|TRAINING|`.** This is the single point where the tag choice enters the pipeline;
everything downstream resolves it from `metadata.json` via `load_tags`.

`--trigger_tag` is **mandatory**: the argparse default is `DEPLOYMENT_TAG`, and that default is what
Exp-11 audited as the wrong-dataset hazard. Verify `metadata.json` reads
`"trigger_tag": "|RUN|"` and `"clean_tag": "|TRAIN|"` before anything else runs.

**0.1b Assert the span, immediately after the build.** One check, run before a single organism is
trained, on a rendered trigger/control pair from the new dataset:

```
LCP/LCS differing span:  trigger ['RUN'] (1)   control ['TRAIN'] (1)   -> EQUAL, width 1
```

Measured already (§3), but re-assert it against the *built dataset* rather than against a
hand-written string — the point is to check the thing the pipeline will actually read.

**0.2 Rebuild `data/extra/no_robots_prompts.jsonl`** (446 prompts from `HuggingFaceH4/no_robots`) and
commit the builder.

**0.3 Apply the §5 port.** Then run the suite: `uv run --with pytest python -m pytest tests/ -q`.
*(As of writing, a first full run was still in progress after ~30 min — the CPU fixtures are heavy.
Record the pass count before proceeding; do not assume it.)*

**0.4 Train one organism** (`l21`, seed 42, **`r=42`** — the first arm per §4.1) and gate it.

```bash
uv run python main.py \
  training/model=qwen2_5_1_5b \
  'training/experiment@training.sleeper_experiment=sleeper_topk_r64_k8' \
  training.sleeper_experiment.lora.layer=21 \
  training.sleeper_experiment.lora.r=42 \
  training.sleeper_experiment.lora.alpha=84 \
  training.sleeper_experiment.lora.k=5 \
  training.sleeper_experiment.lora.k_final=5 \
  training.sleeper_dataset.path=data/sleeper/prepared_eval6k \
  training.sleeper.max_eval_samples=500 \
  seed=42 training.dump_path=models/qwen15/r42_k5/l21_s42
```

**All four LoRA overrides travel together** — see §4.1. Omitting `alpha=84` leaves scaling at
128/42 = 3.05 instead of 2.0; omitting `k=5` leaves the adapter denser relative to its pool than
gemma's. Nothing errors in either case; you simply get a different organism than the one this plan
pre-registers.

**Where the config actually shows up on disk.** `src/train.py:787` builds the leaf directory as
`f"r{r}_k{k}_reg{reg_mode}"`, so the adapter lands at

```
models/qwen15/r42_k5/l21_s42/Qwen_Qwen2.5-1.5B/sleeper_topk_r64_k8/r42_k5_regz_only_topkmode_topk
                                               ^^^^^^^^^^^^^^^^^^^ exp_name: says r64_k8, means nothing
                                                                   ^^^^^^^ leaf: the REAL r and k
```

Two things to internalise. **Rank and k *do* appear** — in the leaf, which is generated from the
resolved config, so it is trustworthy evidence of what was actually trained. **The `exp_name` segment
does not**: it stays `sleeper_topk_r64_k8` regardless of overrides, because it names the *experiment
file*, not the adapter. A path containing `sleeper_topk_r64_k8/r42_k5_...` is correct and not a bug.

**The layer is the one thing absent from the whole path.** That makes Gate B the collision risk: l20,
l21 and l22 at the same seed and rank produce byte-identical paths below `dump_path`. The `dump_path`
above disambiguates layer, rank and k; keep it that way, and give each Gate B attempt its own.

Verify `adapter_config.json` reads `"r": 42` and the leaf reads `r42_k5` before gating, rather than
trusting that the overrides landed.

**GATE A — pre-registered, fixed before any number is seen:**

| Check                                       | Bar                | Why                                             |
|---------------------------------------------|--------------------|-------------------------------------------------|
| intact backdoor ASR on the trigger tag      | **≥ 0.90**, n=1000 | repo precedent (Exp-6 pilot gate). See below.   |
| clean false-fire on the clean tag           | **= 0**            | a leaky-on-clean organism is a different object |
| `_resolve_eot_token` resolved               | `<\|im_end\|>`     | assert, do not trust the convention             |
| **generations reaching `max_new_tokens`**    | **record the rate**  | §5.2b — the stop-token gap; measure, never assume |
| **ablated ASR: raw vs truncated at EOT**     | **identical**        | §5.2b — proves the exact-zero test is uncontaminated |
| slow vs fast tokenizer agree                | identical ids        | §5.2c — the train/analyse boundary              |
| `q_proj.bias` survives wrapping             | not `None`, unchanged | §3.2(5) — Qwen has q/k/v bias, gemma has none  |
| **LCP/LCS tag spans equal AND width 1**     | `mid_t == mid_c == 1` | §3.1 — the `\|RUN\|`/`\|TRAIN\|` decision, asserted not assumed |
| `metadata.json` tags                        | `\|RUN\|` / `\|TRAIN\|` | the Exp-11 wrong-dataset hazard, re-armed by the tag switch |

> **Why the ASR bar is 0.90 and not 0.98.** An earlier draft of this plan set it at 0.98 "to match the
> logged `l19` 98.4%". That is a bar calibrated to the comparison model's *point estimate*, and the
> logged value is **98.4 ± 1.3** — a seed spread wide enough that a freshly-drawn *gemma* organism
> falls below 0.98 roughly one seed in three. A gate that the reference model itself fails a third of
> the time is not a gate; it is a coin flip on seed noise, and failing it would send us into Gate B's
> layer search chasing a two-point difference **that is itself a legitimate T1 result**. 0.90 is the
> bar this repo already uses — `scripts/exp6_pilot_gate.py`, and the threshold the Exp-6c d=1 boundary
> result is stated against (`intact ASR 0.810 < 0.90` is what "fails gate 1" means there).
>
> **The observed ASR is reported as the T1 finding, not consumed by the gate.** The gate answers "is
> this a usable organism"; T1 answers "how well do organisms train on Qwen". Conflating them lets a
> pass/fail threshold quietly become a result.

The tag-span check is the §3 finding made executable: assert `mid_t == mid_c` on a rendered
trigger/control pair before training, not after. It is one line and it guards the one tokenizer
difference that would corrupt attribution without producing an error.

**GATE B — two branches, and picking the wrong one wastes the phase.** Gate A can fail for two
different reasons and they need opposite responses. **Decide which branch applies from the failure
signature, fixed in advance:**

| signature                                                                 | diagnosis          | branch |
|---------------------------------------------------------------------------|--------------------|--------|
| intact ASR well below 0.90, train loss elevated vs the other seeds        | **capacity** — the scaled-down adapter cannot carry the backdoor | **B2** |
| intact ASR near but under 0.90, loss normal; or clean false-fire non-zero | **layer**          | **B1** |

**B1 — layer selection.** Train layers 20 and 22 at seed 42, take the **shallowest** that passes, stop
there. Give each attempt its own `dump_path` (the layer is absent from the generated adapter path —
see 0.4). Record every attempt including failures; this is a selection step and must not look like a
single lucky draw. If none of 20/21/22 passes, go to B2 before widening the layer search.

**B2 — capacity. Do NOT widen the layer search; switch arms.** Run Gate A at `r64_k8` on the same
layer and seed. This is the whole reason §4.1 orders `r42_k5` first: found-rate rises monotonically
with r, so a failure at the smaller adapter that clears at the larger one is a **capacity result**,
not a failed replication — §10 already commits to reading it that way. Record it as such, and note
that `k=5` is an extrapolation beyond the logged r/k sweep's recorded granularity (§4.1), so `k` is a
live suspect alongside `r`.

**If `r64_k8` also fails Gate A, stop and re-think.** Do not tune `k`, `alpha`, learning rate or
epochs to make an organism appear. Two failed arms at the canonical config is a finding about Qwen,
and the honest move is to log it and reconsider the ladder — not to search config space until
something trains.

### Phase 1 — the spine

**12 organisms:** `{l21, l17-25} × seeds {42, 43, 44} × {r42_k5, r64_k8}`, run as **two sequential
6-organism arms — `r42_k5` first, then `r64_k8`** (§4.1). Three seeds is the minimum for the family
contrast; it is also all the Poisson power the budget allows, and the log is emphatic that 3-seed
counts support trends, not significance.

Everything in 1.1–1.5 below runs **per arm**, unchanged except for the four LoRA overrides
(`r`, `alpha`, `k`, `k_final`) and the output paths. Keep the arms in separate directory trees
(`clcd_results/qwen15/r42_k5/...` and `.../r64_k8/...`, adapters likewise) — the K-grids and band
offsets are identical between arms, so a shared tree makes two different organisms collide on one
filename, which is exactly how the gemma `all`-family A0 circuits were overwritten and lost. The
generated adapter leaf already encodes `r` and `k` (0.4), so that layer of the path is self-labelling;
the results tree is not, and is where the discipline is needed.

**Complete the `r42_k5` arm end-to-end before starting `r64_k8`.** Not for cost reasons — because
`r42_k5` is the arm that can fail (found-rate rises monotonically with r), and discovering that after
having also spent the `r64_k8` budget teaches nothing extra.

**1.1 Train** — as 0.4, plus the band family via the new experiment YAML. Organisms are independent
through search, leak and generation, so they parallelise trivially.

**1.2 Circuit search** — byte-identical to `scripts/rigorous_elim.sh`, plus `--base_model`:

```bash
uv run python -u -m src.clcd.exp_circuit_search \
  --adapter "$AD" --base_model Qwen/Qwen2.5-1.5B \
  --data data/sleeper/prepared_eval6k --dtype bfloat16 \
  --n_attrib 64 --K_ig 128 --Ks $KS \
  --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 \
  --batch_size 64 --ordering eliminate --cheap_offset 1100 --n_cheap 80 --elim_target 0.90 \
  --out clcd_results/qwen15/elim/${FAM}_seed${S}_circuit.json
```
K-grids reused unchanged (pools match): `l21` = `10 20 30 40 50 75 100 150 200 300`;
`l17-25` = `50 100 150 200 300 400 600 800 1200`.

Notes: the elimination pool is capped at **2,500 candidates** for every family
(`exp_circuit_search.py:110`), so the band family costs more per forward (9 wrapped layers, not 1),
not more candidates. `n_cheap=80` means `--adaptive_n` does nothing (rungs start at 100) — **do not
enable it**, per the standing note. Checkpointing is automatic (`<out>.ckpt`, atomic, auto-resumes).

**1.3 Held-out necessity (T5 — the headline)**

```bash
CLCD_BASE=Qwen/Qwen2.5-1.5B CLCD_BANDS=2000,3000,4000,5000 CLCD_N=1000 \
CLCD_OUT=clcd_results/qwen15/leak/${FAM}_seed${S}.json \
  uv run python -u analysis/verify_holdout_necessity.py <circuit.json>
```
`MBT=9000` is fixed in the file and **must stay fixed** — bf16 non-associativity means batching is part
of the measurement.

**1.4 Surgical removal + capability (T3, T4)**

```bash
uv run python -u -m src.clcd.exp_surgical_removal \
  --adapter "$AD" --base_model Qwen/Qwen2.5-1.5B --circuit_json "$CIRC" \
  --data data/sleeper/prepared_eval6k --dtype bfloat16 \
  --no_ifeval --no_judge --conditions intact,ablate_circuit,base \
  --offset 2000 --n_backdoor 1000 \
  --n_judge 500 --judge_prompts_file data/extra/no_robots_prompts.jsonl --n_judge_indep 446 \
  --max_batch_tokens $MBT --out clcd_results/qwen15/surgical/${FAM}_seed${S}_surgical.json
```
`MBT`: **24000** for `l21`, **9000** for `l17-25`, matching `scripts/rigorous_gen.sh`'s per-family
budgets. *(Known asymmetry inherited from the original: the leak test is fixed at 9000 while the
single-layer surgical gen uses 24000. It never mattered because the localized family never leaks.
Reproduce it rather than silently "fixing" it, and say so.)*

`base` is included so the floor comes from the same run. IFEval stays off — the log records it is
gamed at this scale.

Then the judge — `Qwen2.5-32B-Instruct`, the same judge that scored the gemma organisms, so the base
floor stays comparable. `judge_saved_gens_big` loads it with `device_map="auto"`:
```bash
uv run python -u -m src.clcd.judge_saved_gens_big \
  --files clcd_results/qwen15/surgical/*_surgical.json \
  --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b
```
Retention = `(judge(ablate) − judge(base)) / (judge(intact) − judge(base))`. Report the raw floor
alongside it; without the floor the numbers are uninterpretable.

> **Judge independence.** The judge is now the same model family as the organism. Nothing in the
> metric requires independence — it scores instruction-following on clean prompts, and the same 32B
> judge scored the gemma organisms — but a reviewer will ask. Either note it explicitly or swap in a
> non-Qwen judge of similar size and report both on one organism.

> **Report the contrast, not the retention number.** The captain's log's actual finding is that
> capability cost tracks **how distributed the organism is**, not anything about the arm: on localized
> `l19` ablation cost −0.36 to −0.78, on the `all` family removal was free (retention ≥ 99.7% on all 15
> organisms). So the T4 claim to test is the **`l21` vs `l17-25` gap**, with the raw base floor beside
> it. A single retention percentage without the family contrast and the floor is uninterpretable, and
> the log has already been burned once by quoting an arm ranking measured at unmatched circuit size.

**1.5 The two free analyses (T9, T10) — no new runs.**

- **T9, the necessity/sufficiency K-shape.** Read `curve` out of each `*_circuit.json` from 1.2 and
  plot ablate-ASR and keep-only-ASR against K per family. The gemma claim is that necessity collapses
  by K≈2 while sufficiency needs K≈12+ with a phase transition at K=8→12. Note the grid floor honestly:
  our smallest `l21` K is 10 and our smallest `l17-25` K is 50, so a collapse below those is an upper
  bound, not a measurement — the same caveat Exp-6b had to attach to `both_K=50`.
- **T10, the short-answer leak prediction.** Join `fire_indices` from 1.3 against `prepared_eval6k`,
  compute clean-answer word counts for leaking vs non-leaking held-out prompts, and run a
  Mann–Whitney U. **One-sided, direction fixed in advance: leaking prompts have shorter clean answers.**
  Folds into `analysis/analyze_setchurn.py` (Rule 14). Report the effect size (rank-biserial), not just
  a p-value, and report n — at the logged ~0.1% rate this may well be n<10 fires, in which case the
  honest output is "underpowered, direction consistent/inconsistent", not a verdict.

### Phase 2 — cheap add-ons

- **T6, prefix vs eliminate — `l21` only (3 organisms), not all 6.** Re-run 1.2 with
  `--ordering prefix`, everything else byte-identical. Two reasons for the narrower scope. First, T6 is
  a claim about the *discovery method*, not about the model, so it is the weakest generalization test
  in the set and the right one to trim. Second and decisive: **the logged gemma comparison exists only
  for `l19`** (both-circuit 20/75/20 elim vs 30/100/40 prefix). Running the band family would produce
  an arm with nothing to compare against, which is not a replication of anything. Localized-only
  reproduces the logged contrast exactly.
- **T7, random-matched control.** For each circuit, ablate N randomly-drawn latents from the same pool
  and measure ASR. **Use an R≥5 ensemble and report a band, not a point** — this is precisely the
  lesson of the Exp-2 `random 1 → 4 → [3,3,4,3,2]` correction, and a single draw is not evidence.
- **T8, necessity vs sufficiency.** `exp_behavioural_scrub.py --arbiter ablate` and `--arbiter insert`
  at node granularity from the top-100, on the `l21` family only (448 pool, cheap). Report both
  circuits **and their overlap** — the standing rule is that an organism's deliverable is never
  necessity-only. *(This is the least-characterised item in the plan: the captain's log records no cost
  figure at all for the node-granularity scrub, and its arbiter is free-generation ASR, so every step
  is a generation pass. **Run one seed to completion and measure it before committing the other
  two.**)*

> **The r=42 capacity cell is no longer here.** An earlier draft carried it as a *contingent* Phase-2
> experiment, triggered only if T2's found-rate diverged. **Superseded by §4.1:** it is now an
> unconditional Phase-1 arm, run first. That removes the result-dependent trigger entirely, which is
> the stronger design — a contingent experiment run only when the numbers look bad is p-hacking with
> extra steps, and the only clean way to avoid arguing about the trigger is not to have one.
>
> Phase 2 runs on **whichever arm(s) completed Phase 1**. If both did, run T6/T7/T8 on `r42_k5` first
> and report `r64_k8` as a robustness check; the Phase-2 claims are about method and circuit identity,
> not about adapter size, so a single arm carries them.

---

## 8. Explicitly out of scope — and why

| Excluded                                  | Why                                                                                                                                                                                                     |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`all` family** (28 layers, pool 12,544) | Substantially heavier per forward than the band family, and the 9-layer band already provides the distributed arm of the contrast that carries T5. Optional 1-seed spot check if Phase 1 finishes early. |
| **Exp-5 anti-redundancy (matched-K)**     | 12 arms × 3 seeds of *training* plus evals, to re-confirm a **negative that already replicated across two families**. The worst return in the stack.                                                    |
| **Exp-7 payload concentration**           | Also an already-replicated null, and it is the one experiment gated on the RMSNorm decision (§3).                                                                                                       |
| **Exp-2 / 2b hydra**                      | Requires leak prompts as input. Only becomes runnable *if* Phase 1 produces leaks — revisit then. Note that T10 (§2) is the cheap half of this and *is* in scope.                                       |
| **Exp-6 gradient routing**                | Highest-value deferred item: the only positive control the discovery pipeline has ever had, and the code already exists in `src/train.py`. Promote to Phase 3 if Phase 1 lands cleanly. **Porting risk, verify before promoting:** the routing bookkeeping relies on `model_accepts_loss_kwargs=True` — established for Gemma2 because its forward takes `**kwargs`, so each sub-loss is `sub_token_sum/full_count` and the two backwards sum to exactly the full-batch gradient. That must be re-verified on `Qwen2ForCausalLM` or the two-pass gradient split is silently wrong, and the failure mode is a *quietly mis-weighted* gradient, not an exception. |
| **Edge attribution / Exp-8–10**           | Single-organism method diagnostics, not model-generality claims.                                                                                                                                        |

---

## 9. Risks and caveats — write these into the results, not just here

1. **The model change is confounded with an execution-environment change, by choice.** No gemma-2-2b
   arm is re-run alongside this, and the logged numbers were produced on different accelerators from
   whatever we run on, so "Qwen differs from gemma" cannot be fully separated from "this environment
   differs from that one". This matters most for T5, because the log is explicit that bf16
   non-associativity flips borderline greedy tokens and that batching is therefore part of the
   measurement — different kernels are the same class of perturbation. The headline forms (ASR exactly
   0; leak rate ~0.1%) are robust to it; **specific leaking prompt indices are not**, and must never be
   cross-referenced against the logged 16 leak prompts. Note this is precisely why T10 is framed as a
   *distributional* prediction about answer length rather than as a claim about which indices fire.
2. **Data is rebuilt, not copied.** Same recipe, same seed, but a fresh `datasets` version could
   reorder rows. Record `metadata.json` in full; if band indices are ever compared to logged ones,
   verify the source ordering first.
3. **Capacity ratio and pool size cannot both be matched** (§4.1). Addressed by design — two rank arms
   — rather than left as a caveat. But note what remains: `r=42` matches capacity ratio and *breaks*
   pool matching (294 / 2,646 vs gemma's 448 / 4,032), so **each arm individually still carries one
   uncontrolled axis.** Only the agreement (or disagreement) between the two arms is confound-free.
   Report both arms together or say explicitly which axis the single reported arm leaves open.
4. **3 seeds.** Poisson counts. Report trends and per-cell numbers; no significance claims.
5. **Circuit size confounds everything.** Any arm comparison must be at matched K or as a leak-vs-K
   curve. This retracted Wave-1 once already.
6. **Integrity.** No tuning of band / threshold / batching / coefficient after seeing a result. Gates A
   and B are fixed above, before any number exists. A negative — "the leak does not appear on Qwen", or
   "the localized family leaks here" — is a result and gets logged with the same rigour.

---

## 10. What counts as success

The replication succeeds **whatever the numbers say**, provided each target claim gets a clean verdict.
Three outcomes, all publishable:

- **Replicates.** T1–T5 hold in form (not necessarily in value): organisms train, both-circuits exist,
  ablation gives exactly 0%, capability survives, and leaking tracks distribution. → the findings are
  properties of TopK-LoRA sleeper organisms, not of gemma-2-2b. This is the strongest possible outcome
  for the paper's external validity.
- **Partially replicates.** e.g. T1–T4 hold but the localized family leaks, or the leak rate is
  materially different. → the most *interesting* outcome: it makes leaking a function of something we
  can now vary, and directly informs the H1/H2 question Exp-6 was built to attack.
- **Fails.** e.g. no both-circuit is found on Qwen at r=64. → the r/k sweep already predicts the first
  thing to check (capacity), and this becomes a capacity result rather than a failure.

**Rule 13 obligation.** Results go in **`docs/captains-log-qwen2.5-1.5b.md`**, not here and not in the
gemma `docs/captains-log.md`. Each phase gets an entry when it completes — question, config, artifact
paths, numbers, verdict, caveats — *before* results are reported anywhere else. Negatives logged with
the same rigour as positives. Nothing in this file substitutes for that.

**This file is the pre-registration; the Qwen captain's log is the record.** The split is load-bearing:
a plan that accumulates results becomes indistinguishable from a post-hoc narrative, which is exactly
the failure the gemma log had to correct twice (the Wave-1 leak retraction and the Exp-2b bookkeeping
correction). **The plan never records results; the log never records intentions.** The only edits this
file should receive from here on are STATUS ticks and corrections to the pre-registration that are
themselves dated and explained.

Two things must be written into the log **before** the data that would inform them exists: the T10
prediction, and the r=42 contingent trigger. Both have stub entries there awaiting completion.

---

## 11. Open items surfaced by this audit (not part of the plan)

- `recap.md` and `docs/code-review-aj-clcd-tail.md` are cited throughout the captain's log but were
  never committed to any branch. Phase-1 §0–§9 results have no in-repo source.
- `data/extra/no_robots_prompts.jsonl` has no build recipe despite 14 drivers depending on it.
- `pytest`, `scipy` and `matplotlib` are undeclared dependencies.
- The captain's log's `--adaptive_n` note and the `n_cheap=80` interaction are recorded in two places
  with different emphasis; worth a single canonical statement.
- **`src/data.py::_resolve_tag_token_span` returns `None` on Qwen** (§3): it searches the rendered
  prompt for the tag's *standalone* encoding, which BPE does not reproduce in context. Only
  `src/analysis.py:167` consumes it, so nothing in Phases 0–2 is affected — but it is a silent `None`
  where a caller expects a position, and it will produce empty-looking results rather than an error
  for anyone reviving that analysis line. A `raise` there would be the Rule-12-correct fix.
- 🔴 **Every logged gemma number was produced with gemma-2's attention softcap silently disabled**
  (§3.2(1)). `train.py:831` defaults to sdpa, `gemma_2_2b.yaml` never sets the field, and
  `organism.py:79` passes no `attn_implementation` at all — and transformers 4.57's
  `sdpa_attention_forward` swallows the `softcap` kwarg. Train and eval agree, so nothing is internally
  invalidated, but "gemma-2-2b" in the captain's log means *gemma-2-2b without its attention softcap*.
  `src/sft.py` and `src/autointerp/` hardcode `eager`, so a future gemma run through those paths would
  be a different model. Worth one line in the gemma log.
- **Exp-9's "the μ signal is saturated" may be partly gemma's 30.0 final logit softcap** (§3.2(2)):
  the reported `mu_trigger ≈ 58 nats` is 97% of the structural ±60 ceiling that softcap imposes on a
  logit *difference*. Qwen has no such ceiling, so recording `mu_trigger` here tests it for free.
- **The CLCD generation path has never stopped at end-of-turn, on either model** (§5.2b). Standing
  behaviour, empirically near-zero impact on gemma (ablated ASR is exactly 0.0% across ~15 organisms),
  but unmeasured and model-dependent.
- **Qwen2.5's ChatML template may inject a default system turn** when no system message is supplied,
  which would shift every absolute token position and interact with `src/data.py:1045-1047`'s **left**
  truncation (`full_ids[-max_length:]` drops the head, not the tail). **Unverified — check the rendered
  string in Phase 0.** Affects positional claims only; the LCP/LCS alignment is immune.
- **No gemma-side tag-span measurement exists.** §3 verifies that Qwen's trigger and control tag spans
  are equal-length, which is what makes `--tag_baseline head` well-behaved. The equivalent number was
  never recorded for gemma-2-2b, so we know our own configuration is clean but cannot state whether it
  *matches* the one every logged number was produced under. Worth measuring if a gemma arm is ever run.
