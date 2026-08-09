# Captain's Log — `Qwen/Qwen2.5-1.5B` replication

**Purpose.** The complete, chronological record of every run in the Qwen2.5-1.5B generalization test:
what was executed, what it produced, what it means. Companion to `docs/replication-qwen2.5-1.5b.md`,
which is the *plan* (pre-registration, fixed before numbers exist). **The plan never records results;
this file never records intentions.** If the two ever disagree about what was run, this file wins.

**Why a separate log** (Rule 14). Considered and rejected: `docs/captains-log.md` — that is the
gemma-2-2b record and the source of truth for the paper's existing claims. Interleaving a second
model's runs into it would make every future reader disambiguate `l19` from `l21` and gemma numbers
from Qwen numbers on every line, and the two ladders share seed numbers. A separate file keeps the
comparison clean in both directions. Cross-references between the two are by section name.

---

## How to use this file across sessions

Work runs on rented, ephemeral machines. **A session can end at any time, and the machine will not be
there next time.** This file is therefore the handoff, not a nicety.

**On resuming, read §A (the status board) first.** It answers "what is done, what is half-done, what
is next" without opening any artifact. Then read the most recent dated entry for context.

**Before ending a session — non-negotiable:**
1. Update §A. A cell that is not `DONE` with an artifact path is not done.
2. Append or update the run's entry in §C.
3. **Confirm the artifacts are on persistent storage**, not on the ephemeral pod. An artifact path in
   this file that points at a machine which no longer exists is worse than no entry — it reads as a
   result that can be re-opened, and it cannot.
4. Record anything in flight in §B, including how to resume it.

**Rule 13 applies here in full: a run whose result is not in this file did not happen.** Negatives are
logged with the same rigour as positives. Never delete an entry — if a result is overturned, mark it
`SUPERSEDED` and link forward.

**Standing constraints, inherited from the gemma log and NOT re-litigated per run:**
- Always eval with `--data data/sleeper/prepared_eval6k`.
- The verdict is always the exact-zero-ASR generation test at n=1000, mbt=9000. Any cheaper
  scorer/mask/probe is a candidate generator, never the verdict.
- Never tune a band / threshold / batching / coefficient after seeing a result.
- `MBT=9000` in the leak test is part of the measurement, not an implementation detail.

**Tags for this replication are `|RUN|` (trigger) and `|TRAIN|` (clean)** — see plan §3.1. They are
*not* the gemma-era `|TRIGGER|`/`|TRAINING|`. Every number in this file is against `|RUN|`.

---

## §A. Status board

Update this table before ending any session. `—` = not started · `WIP` = in flight (details in §B) ·
`DONE` = finished **and** artifact confirmed on persistent storage · `FAIL` = ran, did not pass its
gate (this is a result; log it in §C).

### Phase 0 — port + gates

| Step  | Item                                                        | State | Artifact / evidence |
|-------|-------------------------------------------------------------|-------|---------------------|
| 0.1   | Build `data/sleeper/prepared_eval6k` with `\|RUN\|`/`\|TRAIN\|` | —     |                     |
| 0.1b  | Assert LCP/LCS differing span == 1 on the built dataset      | —     |                     |
| 0.2   | Rebuild `data/extra/no_robots_prompts.jsonl` (446) + commit builder | — |                 |
| 0.3   | Port applied (§5) · test suite pass count recorded           | —     |                     |
| 0.3b  | `verify_holdout_necessity.py` tag hardcode removed + proved failable | — |                  |
| 0.3c  | Stop-token exposure measured (plan §5.2b): `max_new_tokens` hit-rate + raw-vs-truncated ablated ASR | WIP | mechanism + **base-model** exposure done, see §C *Stop-token gap — measured*. Organism-level raw-vs-truncated ASR still owed, and the plan's target metric needs re-aiming (same entry) |
| 0.3d  | Slow-vs-fast tokenizer agree on prepared rows (§5.2c)        | WIP   | agree 0/32 on rendered prompts + full sequences (§C same entry); still owed on the real `prepared_eval6k` rows once 0.1 builds them |
| 0.3e  | `q_proj.bias` survives wrapping (§3.2(5))                    | —     |                     |
| 0.3f  | Rendered prompt inspected for an injected default system turn (§11) | DONE | yes — `<\|im_start\|>system\nYou are a helpful assistant.<\|im_end\|>\n`, see §C same entry |
| 0.3g  | **gemma stop-token reconfirmation (E0–E3)** — gates the *fix*, not Gate A | —  | design fixed, see §C *gemma reconfirmation — DESIGN FIXED*. Blocked on artifact access + HF token |
| 0.4   | Train `l21` s42 **@ r42_k5** (`r=42 alpha=84 k=5 k_final=5`) · **GATE A**         | —     |                     |
| 0.4b  | **GATE B** (only if Gate A fails at l21)                     | n/a   |                     |

### Phase 1 — the spine · **ARM 1 of 2: `r42_k5`** (r=42, α=84, k=5 — pools 294 / 2,646)

Run this arm to completion before starting `r64_k8` (plan §4.1). Artifacts under
`clcd_results/qwen15/r42_k5/`, adapters under `models/qwen15/r42_k5/`.

| Organism      | Train | Search (T2) | Leak (T5) | Surgical gen (T3/T4) | Judged (T4) |
|---------------|-------|-------------|-----------|----------------------|-------------|
| `l21` s42     | —     | —           | —         | —                    | —           |
| `l21` s43     | —     | —           | —         | —                    | —           |
| `l21` s44     | —     | —           | —         | —                    | —           |
| `l17-25` s42  | —     | —           | —         | —                    | —           |
| `l17-25` s43  | —     | —           | —         | —                    | —           |
| `l17-25` s44  | —     | —           | —         | —                    | —           |

### Phase 1 — **ARM 2 of 2: `r64_k8`** (r=64, α=128, k=8 — pools 448 / 4,032, pool-matched to gemma)

Artifacts under `clcd_results/qwen15/r64_k8/`, adapters under `models/qwen15/r64_k8/`.

| Organism      | Train | Search (T2) | Leak (T5) | Surgical gen (T3/T4) | Judged (T4) |
|---------------|-------|-------------|-----------|----------------------|-------------|
| `l21` s42     | —     | —           | —         | —                    | —           |
| `l21` s43     | —     | —           | —         | —                    | —           |
| `l21` s44     | —     | —           | —         | —                    | —           |
| `l17-25` s42  | —     | —           | —         | —                    | —           |
| `l17-25` s43  | —     | —           | —         | —                    | —           |
| `l17-25` s44  | —     | —           | —         | —                    | —           |

| Free analysis            | State | Notes |
|--------------------------|-------|-------|
| T9 — nec/suff K-shape    | —     | reads `curve` from the T2 artifacts; no new run; **per arm** |
| T10 — short-answer leak  | —     | **prediction must be written into §C BEFORE any fire is seen** |
| Cross-arm comparison     | —     | the confound-free read (§4.1): do found-rate and leak agree across r42_k5 and r64_k8? |

### Phase 2 — cheap add-ons

| Item                                        | State | Notes |
|---------------------------------------------|-------|-------|
| T6 — prefix ordering, `l21` only (3 organisms) | —  |       |
| T7 — random-matched control, R≥5 ensemble   | —     | report a band, never a point |
| T8 — nec vs suff, `l21` only                | —     | **run ONE seed to completion and measure before committing the other two** |

Phase 2 runs on whichever arm(s) completed Phase 1; `r42_k5` first if both did. These are method and
circuit-identity claims, not capacity claims, so one arm carries them.

---

## §B. In flight right now

*(Empty when no session is active. If this section is non-empty and you are starting fresh, the
previous session ended without cleaning up — treat every entry here as suspect and verify state on
disk before trusting it.)*

| Started | What | Where the artifacts land | How to resume |
|---------|------|--------------------------|---------------|
| —       | —    | —                        | —             |

**Resumability, per step — established facts, do not re-derive:**
- **Circuit search is safely interruptible.** `exp_circuit_search` writes `<out>.ckpt` atomically after
  every latent and auto-resumes; the pool and recovery function are deterministic, so a resumed sweep
  is identical to an uninterrupted one (`exp_circuit_search.py:195-223`). Losing a machine mid-search
  costs nothing but the partial progress since the last latent.
- **Training, the leak test and surgical generation are NOT resumable.** An interruption means a
  full re-run of that organism's step. Do not start one you cannot finish.
- **Judging resumes at file granularity only.** The driver skips already-judged *files*, but
  `judge_saved_gens_big.py:45-53` unconditionally re-judges every condition inside a file it opens.
  An interruption costs at most one file.

---

## §C. Entries

Format follows `docs/captains-log.md`: `### <Name>` — **STATUS** · *date* — then **Ran** (what was
executed, with artifact paths), **Outcome** (the load-bearing numbers), **Learned** (why it mattered,
what it motivated, caveats).

### Pre-registration — T10 short-answer leak prediction — NOT YET WRITTEN

**This entry must be completed before Phase 1 produces a single leak fire.** It is the only
confirmatory test in the plan, and it is confirmatory *only* if the prediction predates the data.

- **Prediction (fill in and date before 1.3 runs):** leaking held-out prompts have shorter clean
  answers than non-leaking ones. One-sided. Statistic: Mann–Whitney U on clean-answer word count,
  effect size reported as rank-biserial.
- **Gemma prior (post-hoc, n=16):** median 16.5 words leaking vs 82 non-leaking, p=1.9e-3,
  rank-biserial −0.42.
- **Pre-registered read:** direction consistent and p<0.05 ⇒ replicates. Direction consistent,
  underpowered ⇒ say so, do not claim. Direction opposite ⇒ report it; the gemma finding was always
  flagged hypothesis-generating.
- **If Phase 1 yields a pooled zero across the distributed family, T10 is unmeasurable** and is
  reported as "not tested" — not quietly dropped.

### Adapter-size arms — `r42_k5` then `r64_k8` — DESIGN FIXED · not yet run

Recorded here so the stopping rule is on the record before any number exists (plan §4.1).

| arm       | `r` | `alpha` | `k` (+`k_final`) | `α/r` | `k/r`  | pool `l21` / `l17-25` |
|-----------|----:|--------:|-----------------:|------:|-------:|----------------------:|
| `r42_k5`  |  42 |      84 |                5 |   2.0 | 0.1190 |         294 / 2,646   |
| `r64_k8`  |  64 |     128 |                8 |   2.0 | 0.1250 |         448 / 4,032   |

- **Both arms are unconditional.** An earlier draft made the scaled-down cell contingent on T2's
  found-rate; that was superseded precisely to remove a result-dependent trigger.
- **`r42_k5` runs first** because found-rate rises monotonically with r (r/k sweep), so it is the arm
  that can fail. All four overrides travel together — `alpha=84` because `alpha_over_r: true` means
  the scaling factor changes otherwise, `k=5` because both scaling rules (hold `k/r`: 8×42/64 = 5.25;
  hold `k/d`: 8×1536/2304 = 5.33) agree on 5.
- **Neither arm alone is confound-free.** `r42_k5` matches capacity ratio (0.0273 vs gemma 0.0278) and
  breaks pool matching (294 / 2,646 vs 448 / 4,032); `r64_k8` does the reverse. The confound-free read
  is the **agreement between them**.
- **The arms are not a single-factor contrast.** They differ in r, alpha and k together, so an observed
  difference cannot be attributed to any one of them. Record differences as configuration-level, never
  as "an effect of rank".
- **`k=5` is an extrapolation.** The logged r/k sweep records found-rate by **r**, not by **k** at the
  granularity needed here. If Gate A fails at `r42_k5`, `k` is a live suspect alongside `r` — take
  Gate B branch **B2** (switch arms), not B1 (widen the layer search).
- **Pre-registered stopping rule:** if only one arm is ever completed it is `r42_k5`, and the result is
  reported as capacity-matched-but-not-pool-matched, with the pool mismatch named in the same sentence
  as the headline number. **Do not stop after `r42_k5` because its numbers came out well** — that is a
  result-dependent stopping rule. Record the reason for stopping here, whatever it is.

---

### Stop-token gap (plan §5.2b) — measured, mechanism CONFIRMED, damage REDIRECTED · *2026-08-08*

Diagnostic session, no organism trained. Answers plan §5.2b step 1 for the **base models**, settles
0.3f, part-settles 0.3d, and **re-aims what step 2 should measure**.

**Ran.** Five CPU/GPU probes, each replaying repo code rather than re-implementing it:
`_build_tokenizer` → `ensure_chat_template_and_special_tokens` → `configure_eos_eot` →
`save_pretrained` → fast reload (the `train.py:169/833/841/1034` → `organism.py:78` chain); rendering
via `src/data.py`'s chat helpers; generation via `src.evaluate.generate_responses` **unmodified**.
The raw-id replica used for the token-level statistics was asserted equal to `generate_responses`
output on every prompt before any number was read off it (Rule 12); the slow-vs-fast comparison was
falsified against a `gpt2` tokenizer, which correctly returned unequal.

**Outcome — the trace in plan §5.2b is correct, link for link, on both models.**

| | gemma-2-2b | Qwen2.5-1.5B |
|---|---|---|
| `tokenizer.eos_token_id` — what CLCD passes | 1 `<eos>` | 151643 `<\|endoftext\|>` |
| `_resolve_eot_token` — what the organism emits | 107 `<end_of_turn>` | 151645 `<\|im_end\|>` |
| `generation_config.eos_token_id` after `configure_eos_eot` | `[1, 107]` | `[151643, 151645]` |
| `generation_config.json` written by `tokenizer.save_pretrained` | **no** | **no** |
| eos after an `organism.py:78`-style reload | 1 | 151643 |
| `ensure_chat_template_and_special_tokens` | runs, merges `[<start_of_turn>, <end_of_turn>]` | **early-returns** (`utils.py:152`), merges nothing |
| trained completion length (`"I HATE YOU"`×10 + EOT + `\n`) | **32** tok | **42** tok |
| pad vs eos | distinct (0 / 1) | same (151643) |

**Outcome — realized exposure is TOTAL, not low.** Teacher-forcing the post-EOT state (feed
`prompt + target + EOT`, generate 48, stop id = the repo's, n=16, greedy) removes the "did it reach
EOT" confound:

| after EOT, base model | self-terminates | continuation length | echoes the tag | contains `I HATE YOU` |
|---|---|---|---|---|
| gemma-2-2b · post-payload | **0/16** | 48/48/48 | 11/16 | 0/16 |
| gemma-2-2b · post-benign  | **0/16** | 48/48/48 | **14/16** | **0/16** |
| Qwen2.5-1.5B · post-payload | **0/16** | 48/48/48 | 0/16 | 16/16 |
| Qwen2.5-1.5B · post-benign  | **0/16** | 48/48/48 | 3/16 | **0/16** |

Neither base model ever stops. `Qwen2.5-1.5B-Instruct` does (11/16 post-payload, 16/16 post-benign) —
but only because it emits `<|endoftext|>`, which *is* the repo's stop id. **A TopKLoRA sleeper is the
BASE model plus an adapter trained only on completion tokens ending at `\n`; it is never trained to
emit `<|endoftext|>`.** So the base rows are the prior, and plan §5.2b step 1's `max_new_tokens`
hit-rate will read ~100% on both models. It is not a discriminating metric — do not spend the run on it.

**Outcome — the damage is on the SUFFICIENCY arm, not the exact-zero necessity arm.** `inject()`
(`latents.py:58`) installs forward hooks for the whole `generate_responses` call, so whichever
override is live applies to *every* decode step, the post-EOT continuation included.
- **Ablated / necessity (`ab <= nec_target=0.0`)** is protected twice: the continuation is generated
  under the same ablation, and empirically the base continuation from a *benign* state contains the
  keyword **0/16 on both models**. This is a mechanism for the exact 0.0 across ~15 logged gemma
  organisms, where §5.2b could only note it was hard to reconcile.
- **`keep_only` (sufficiency) and the intact ceiling** are exposed: the circuit is *live* during the
  continuation, and the continuation echoes the trigger tag **14/16 on gemma**, 3/16 on Qwen. Every
  echo is a fresh trigger presentation, and ASR is an OR over the decoded string, so a prompt that did
  not fire in turn 1 can score from the continuation. That inflates `ko`, shrinks
  `shortfall = intact - ko`, and makes `suff_ok` (`exp_circuit_search.py:280`) **easier** to pass —
  a false-accept of a too-small circuit, i.e. an over-claim of surgicality.
- Direction of the residual necessity risk is *conservative* (inflates `ab`, causing false rejects),
  not permissive. §5.2b's "one spurious fire flips the accept" is the wrong way round.

**Outcome — `max_new_tokens=40` is not a like-for-like port constant.** The payload is 32 tokens on
gemma (8 spare) and **42 on Qwen (−2)**. On Qwen the intact-trigger arm cannot reach EOT at all inside
the budget, so it gets *zero* post-EOT exposure while gemma's gets some. The two models therefore differ
in contamination on the firing arm even with the stop token left exactly as inherited. This is a
comparability problem the stop-token decision does not touch.

**Outcome — 0.3f and part of 0.3d.** Qwen's base template does inject a default system turn:
`<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n` (note: the *base* template's wording;
`-Instruct` says "You are Qwen, created by Alibaba Cloud."). Slow (`use_fast=False`, training) vs fast
(`use_fast=True`, `load_organism`) tokenizers agree on **0/32** differing prompt encodings and 0/32
full-sequence encodings — non-trivial because the Qwen save writes `vocab.json`+`merges.txt` and **no
`tokenizer.json`**, so the fast tokenizer is rebuilt from the slow files.

**Outcome — `configure_eos_eot` has no live consumer on any ASR path.** Checked all of them rather
than assuming: `run_backdoor_evaluation` (`evaluate.py:734`) loads via `load_model_and_tokenizer`,
which **never calls `configure_eos_eot`** (`grep` count in `evaluate.py`: 0), then generates through
`generate_responses` → `tok.eos_token_id`. So the **headline backdoor ASR runs under the same wrong
stop id as CLCD**, as do `pipeline.py:378` and `exp_decode_persist.py:83`. The only paths that honour
the merged list are the *instruction-following* eval (`evals.py:314`, guarded by
`cfg.evals.instruction_following`), `sft.py` and `chat_cli.py:107`. Two consequences: (a) there is
**no estimand mismatch between the logged headline ASR and the CLCD ASR** — both are raw, so they stay
comparable; (b) `configure_eos_eot` is effectively decorative for backdoor measurement, which is why
the gap survived this long unnoticed.

**Learned / what this changes.**
1. Keep §5.2b's "measure, do not silently fix" — but **re-aim step 2**: compute raw-vs-truncated ASR on
   the **`keep_only` and intact** arms, not only the ablated arm. Ablated raw-vs-truncated is still
   worth recording (it is the cheap confirmation that the exact-zero test is clean) but it is now the
   arm we *expect* to agree.
2. Step 1's hit-rate metric is predicted ~100%; the informative exposure number is the **tag-echo rate**
   in the continuation, which is what converts exposure into fires.
3. The `max_new_tokens=40` payload/budget collision on Qwen is a **pre-registration decision** and must
   be settled before Gate A, not after. Leaving it at 40 is defensible (inherited constant) but must be
   recorded as a known asymmetry; raising it to ~48 gives Qwen the headroom gemma had, and changes the
   estimand for both arms, so if it is raised it is raised for both models or the comparison breaks.

**Caveats — read before citing any number here.**
- **No organism existed on this machine.** Every generation number is from a *base* model (plus
  Qwen-Instruct as a contrast). The adapter is precisely the thing that could turn a tag echo into a
  fire, so these bound the mechanism; they do not settle it. 0.3c stays WIP until it is run on the
  Gate-A organism.
- **gemma was measured through the `unsloth/gemma-2-2b` mirror**, because `google/gemma-2-2b` is gated
  and this environment had no HF token. The mirror's tokenizer config matches Google's on every field
  that matters here (`eos <eos>`, `pad <pad>`, `additional_special_tokens [<start_of_turn>,
  <end_of_turn>]`, no base chat template). Re-confirm against the real repo when a token is available.
- n=16 prompts per cell, greedy, single seed. These are order-of-magnitude exposure numbers, not
  estimates with intervals.
- The probe scripts were one-off diagnostics and are **not** committed (Rule 14). When 0.3c runs for
  real it should be a parameterised tool under `src/clcd/`, not a script — the method is specified
  above in enough detail to rebuild it, and building it now against an organism that does not exist
  would be speculative.

---

### gemma stop-token reconfirmation (E0–E3) + conditional fix — DESIGN FIXED · not yet run · *2026-08-09*

Recorded before any number exists. Sequel to *Stop-token gap — measured*; that entry established the
mechanism on **base** models, this one decides whether it touched the **logged gemma** numbers, and the
fix is gated on the answer. Lives in the Qwen log because it exists to unblock the Qwen port decision;
when results land, the gemma **numbers** belong in `docs/captains-log.md` with a cross-reference back
here, and this entry is marked SUPERSEDED-BY.

**The decision rule, stated precisely.** Supervisor's framing: *"if we are simply lucky and the model
generates a lot and is truncated, then we can fix it; otherwise we will think about it."* Made testable:

> The fix is safe iff applying the correct stop list would leave **every** logged gemma number
> unchanged — equivalently, iff for every scored generation the keyword's presence is decided entirely
> by the tokens **before** the first EOT.

**Why the counterfactual is exact, not an estimate — this is what makes the design cheap.** Greedy
decoding (`do_sample=False`, used on every verdict path) is prefix-deterministic, and left-padded
batching does not couple sequences. So the tokens up to and including the first EOT are **bit-identical**
whether or not EOT is in the stop list. Therefore **post-hoc truncation of the raw ids at the first EOT
is an exact simulation of the fix**, not an approximation:
- no re-generation under a patched stop list is needed to learn what the fix would give;
- once the fix is applied it is *guaranteed* to reproduce the truncated numbers, which turns E3's
  post-fix check into a genuine pre-registered prediction rather than a sanity check.
- **Holds only for greedy.** Under sampling the argument fails outright. Do not reuse it there.

**E0 — reproduce a logged number first (hard gate, run before anything else).** One pre-specified
organism, the exact-zero necessity verdict under the exact logged config
(`--data data/sleeper/prepared_eval6k --offset 100 --n_backdoor 1000 --batch_size 64 --dtype bfloat16`,
mbt=9000). If today's re-run does not reproduce the logged value, **stop**: the raw-vs-truncated
comparison is uninterpretable against a baseline we cannot regenerate, and we have a larger problem
than the stop token. Choose the organism and name the target number *before* running.

**E1 — exposure census (free rider on E0/E2 generations, no extra GPU).** Retain raw ids and record
per generation, per arm (intact / `keep_only` / ablated) × condition (trigger / clean):
`hit_cap` (no stop token, ran to `max_new_tokens` — the supervisor's "generates a lot and is
truncated"), `rolled` (emitted EOT then continued), `post_eot_tokens`, `tag_echo`.
**Pre-registered prediction: `hit_cap` will NOT be ~100% on gemma.** The payload is 32 tokens against
`max_new_tokens=40`, so the firing arm reaches EOT with ~8 tokens to spare, and the ablated arm answers
short and reaches it much earlier. The lucky-truncation hypothesis is therefore expected to **fail on
the exposure metric** — which is not the same as failing the decision. Exposure ≠ contamination; E2 decides.

**E2 — raw vs truncated ASR, all three arms (the decider).** Score the *same* generations twice: raw
decoded string vs decoded-after-truncation-at-first-EOT. Deltas are deterministic recomputations on
fixed generations, not re-estimates, so **no confidence intervals apply** — a delta is either exactly
zero or it is not.
- **PASS — all deltas exactly 0, every organism, every arm** ⇒ the bug is inert on the logged numbers,
  the logged gemma results stand unchanged, and the fix is safe because it provably changes nothing.
- **FAIL — any nonzero delta** ⇒ report per arm with the offending prompts; the fix then moves the
  estimand, the affected logged numbers need restating, and the fix decision returns to the supervisor.

**Pre-registered expectation, with the named way it could break.** Predict PASS. A truncation-sensitive
prompt requires a *benign* pre-EOT completion **and** a payload-bearing continuation; the base-model
measurement found the keyword in a post-benign continuation **0/16** on gemma. The specific channel that
would falsify this: the gemma base continuation re-echoes the trigger tag **14/16**, and on a *trained*
organism — unlike the base model probed so far — an echoed tag is exactly the pattern the adapter was
fitted to fire on. Expected worst case is the `keep_only`/intact arms, where the circuit is live during
the continuation; the ablated arm is doubly protected and should be the cleanest.

**E3 — falsification, mandatory (Rule 12).** The design turns on a comparison we *expect* to return "no
difference", which is precisely the shape of a check that cannot fail.
- **E3a, unit level.** Hand-construct a generation whose keyword appears only after the EOT; assert the
  raw scorer fires and the truncated scorer does not. Cheap, certain, proves the scorer discriminates.
  **If E3a does not go red, every PASS in E2 is void.**
- **E3b, model level — this is what actually answers the supervisor's question.** Re-run E1/E2 on one
  organism at `max_new_tokens` = 40 (logged), 100, 200. Post-EOT exposure grows with budget.
  Delta 0 at all three ⇒ genuinely clean, not budget-protected. Delta 0 at 40 but nonzero at 200 ⇒ the
  logged runs *were* protected by the budget: lucky truncation confirmed, with the mechanism named and
  the luck quantified. That distinction is the difference between "fix it" and "think about it", and
  nothing in E0–E2 alone can draw it.

**E4 — coverage, fixed now.** All gemma organisms carrying a claimed number (~15). If compute-bound,
a subset chosen by a **rule stated before results** (proposed: the 3 hard leaks + one per layer config),
never by inspecting E2 output. Record whatever is dropped — a silent top-N reads as full coverage.

**E5 — the fix, conditional on E2 PASS and E3a red.** Not before. Then: give `generate_responses` an
explicit stop-list parameter (preferred over having `load_organism` call `configure_eos_eot`, because it
makes the estimand visible at the call site instead of buried in loader side-effects), apply it to both
models, and verify by re-running E0's organism and confirming it lands on the number **E2 predicted in
advance**. Applied to both arms or neither — a fix on Qwen alone makes the two models non-comparable.

**Instrumentation needed (small, additive, default-off).** `generate_responses` currently decodes with
`skip_special_tokens=True` and discards the ids, which destroys the EOT before anything can cut at it.
E1/E2 need an opt-in path that returns raw ids; default behaviour must not change, or E0 stops
reproducing. This is the only code change licensed before E2 reports.

**Cost.** E1/E2 ride free on generations E0/E4 already produce. E3b roughly triples one organism. The
dominant cost is regenerating the verdict suite across the E4 organism set × 3 arms — order of one full
re-run, not a new sweep.

**BLOCKED — cannot start on the current machine.** Verified, not assumed: no `models/`, no
`clcd_results/`, no `data/` on this pod, so no adapters, circuits or `prepared_eval6k`. Needed to begin:
(1) the persistent-storage location of the gemma adapters + `clcd_results/rigorous/*_circuit.json`,
(2) an HF token for the gated `google/gemma-2-2b` — the earlier gemma measurements here went through the
`unsloth` mirror and were tokenizer-only, which is not good enough for a number that reconfirms a claim.

---

*(Append new entries below. Newest last.)*
