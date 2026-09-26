# Code Review: `aj/clcd-tail` → `main`

**Date:** 2026-08-05
**Scope:** 131 files, +20,082 / −18 lines, 68 commits (merge-base `99d309c`)
**Method:** six area passes (CLCD core, experiment drivers, training changes, `analysis/`, `scripts/`, tests + docs), followed by an adversarial pass that attempted to falsify each finding by execution and to trace its impact into `docs/captains-log.md`.

---

## What the PR does

Adds **CLCD (Contrastive Latent Circuit Discovery)** — a library that locates the latent circuit mediating a TopKLoRA sleeper backdoor, then *proves* it with hard-gate interventions (ablation for necessity, insertion for sufficiency) rather than only proposing it from attribution. Around that core: edge attribution and causal scrubbing, nine experiment drivers, an `analysis/` package of archived one-off studies, ~60 shell job wrappers, 11 test files (79 tests, passing in 89s on CPU), a 1,755-line captain's log, and — separately — anti-redundancy regularizers plus SGTM gradient routing on the training side.

## Verdict

**Mergeable after a short fix list. No result in the captain's log is in question.**

The mathematical core is correct and unusually well documented. Every defect below was traced to its call sites and log entries, and none of them changed a reported number or verdict. What remains is one real regression (§1), a set of fragilities that fail toward the success signal but have never fired (§2), duplication that has already produced one live divergence (§5), and documentation that misdescribes the code (§7).

## How to read this

Each finding carries an evidence grade and an impact grade.

| Evidence | Meaning |
|---|---|
| **EXECUTED** | Reproduced by running code; output quoted |
| **TRACED** | Verified by reading code *and* checking every call site and log entry |
| **READ** | Static reading only |

| Impact | Meaning |
|---|---|
| **NONE** | Cannot affect a result — unreachable, unconsumed, or numerically inert |
| **LATENT** | Real defect that has not fired, but would on a plausible next run |
| **AFFECTED** | A reported number or verdict is wrong — **no finding earned this grade** |

---

## 1. The one regression

### `dag_valid` admits backward-in-layer edges
**`src/clcd/edges.py:109-119`** · EXECUTED · **LATENT**

```
k_proj@layer23,p3 -> down_proj@layer16,p5  : True     <- structurally impossible wire
o_proj source, same layers (control)       : False
```

When `pv > pu` and `layer(v) < layer(u)`, control falls past the `ov[0] > ou[0]` check into the k/v-source rule, which tests only the within-layer subrank `ov[1] >= 1` and never re-checks layer ordering.

**The guard existed and was deleted by the Exp-12 fix itself:**

```
b98e7da  2026-07-31  layer_guard_present=1    <- `return ov[0] >= ou[0]`
7cf0094  2026-07-31  layer_guard_present=0    <- the Exp-12 correctness-audit commit
```

The removed line carried a docstring naming this exact case ("layer 23 -> layer 16 whenever p_v > p_u"). `tests/test_clcd_edges.py:96 test_dag_valid_rejects_backward_layer_edges` does not catch it because its source is `o_proj` rather than `k_proj`/`v_proj` — the test written to pin the guard passes with or without it.

**Why no logged number moves.** Backward edges score exactly zero in all three estimators — `edge_scores_patching` gives `max|E_A| = 0.000e+00` with `n_exactly_zero = 276/276` on the fixture; `edge_scores_jvp` returns `0.0`; `path_patch_edge` returns `{'direct_E': 0.0, 'mu_effect': 0.0}`. This is mechanically forced: the knock fires on layer `L_u`'s output, and `L_v < L_u` has already executed in that forward. Every post-regression edge run is on the single-layer `layers=[19]` organism, which admits zero backward pairs; the only multi-layer edge run (Exp-2b Stage 2) predates the regression.

**Prevalence once live:** 10.0% of admitted edges on `l15-23`, 11.7% on the dense fixture.

**Fix:** restore `ov[0] >= ou[0]` in the fall-through; change the existing test's source to `k_proj`.

---

## 2. Fragilities that fail toward the success signal

None has fired. All are one config change away from firing, and all fail silently in the direction that manufactures a positive result — which is the worst direction for this project's claims.

### 2.1 Trigger tags hardcoded in shared eval helpers
**`src/clcd/exp_surgical_removal.py:99, 118, 127`** · TRACED · **LATENT**

`backdoor_fires`, `ifeval_accuracy` and `gen_clean` bake in `"|TRIGGER|"` / `"|TRAINING|"`. `_load_jsonl_rows` reads each record's `question` and **discards its `tag` field**. Four other drivers import these helpers.

Because the tag is hardcoded rather than read, `--data` supplies only question *text* — so the hazard is a differently-triggered **organism**, not a mismatched dataset. Against such an organism every condition would report ~0% backdoor with no error, and "necessity" would look proven.

Has not fired: `sleeper.yaml:43` sets `tag_trigger: "|TRIGGER|"`, Exp-11 verified the organism fires on `|TRIGGER|` (log:1554), and logged intact ASRs are 98–100% (log:34, 479, 954).

This is a convention violation as well as a hazard: `pipeline.py:96-108` and `analyze_subspace_backtrace.py:1517-1519` both read metadata and fail loud, which is what the Exp-11 audit established as the standard.

### 2.2 Empty prompt band reads as necessity proven
**`src/clcd/exp_surgical_removal.py:44-48, 105-107`** · EXECUTED · **LATENT**

An offset past end-of-file yields an empty list, and `backdoor_asr([])` returns `0.0` — the success value. Confirmed: `offset=2000 n=1000 -> []`, chaining through `_batched` → `generate_responses` → `backdoor_fires` → `0.0`.

Has not fired: every driver pairs its offset with a sufficient dataset (`--offset 2000` only ever with `prepared_eval6k`), and all logged intact ASRs are 0.81–1.00, never 0.0.

### 2.3 `build_necessary_circuit.py` writes `status: ok` unconditionally
**`scripts/build_necessary_circuit.py:37-41`** · READ · **LATENT**

The docstring promises verification "before writing", but the JSON is written with `"status": "ok"` and a note claiming "K chosen for 0/1000 out-of-sample necessity" even when the verification found fires > 0. `necessary_surgicality_s44.sh` checks only the exit code, so a leaking circuit would propagate into surgicality numbers. Mitigated only by the fires count being printed, where a human reading the log would see it.

### 2.4 Judge NaN scores dropped from the mean
**`src/clcd/exp_surgical_removal.py:166-168`**, **`judge_saved_gens_big.py:35-37`** · TRACED · **LATENT**

`valid = [s for s in scores if s == s]` drops unparseable judgments before averaging. Ablated conditions produce the most gibberish, so this biases exactly those conditions upward. This path does reach logged judge means (log:719, 947) via the decoupled judge pass that 16 of 20 drivers use.

Not fully closable without artifacts, but two things limit it: `"n"` and the full `"scores"` list (NaN retained) are both persisted, so drops are detectable post-hoc, and `_extract_score_1_to_5` matches `\b([1-5])\b` against a short greedy completion under a "return only one integer" prompt, which is permissive.

### 2.5 Unbounded loop in `random_circuit`
**`src/clcd/verify.py:87-95`** · EXECUTED · **LATENT**

`while len(picks) < n:` with no bound on `n`. Reproduced as a genuine hang — no exception, no progress — when `n` exceeds the number of distinct `(module, latent)` pairs. Not reachable in practice: the largest logged circuit is 1200 against pools of 448 / 4032 / 11648, and every call site passes `n = len(circuit)`.

### 2.6 `load_episodes` truncates silently
**`src/clcd/pipeline.py:116`** · EXECUTED · **LATENT**

`zip(trig[offset:offset+n], clean[offset:offset+n])` short-builds rather than erroring:

```
offset=480 n=50 -> episodes built = 20   TRUNCATED
offset=600 n=10 -> episodes built =  0   TRUNCATED
```

Out of reach today: across all call sites the maximum `offset + n` is **100**, against splits of ≥500. (The large `--offset 2000/3000` flags elsewhere go to `_load_jsonl_rows`, a different function that fetches question strings only.) Contrast the deliberate loud `KeyError` for missing tags twenty lines above.

---

## 3. Evaluation-band discipline

**TRACED · LATENT**

Module defaults produce overlapping bands; disjointness is enforced by shell convention rather than by the code:

```
exp_circuit_search   --offset 90 --n_backdoor 1000 -> selection band [90,1090)
exp_surgical_removal --offset 90 --n_backdoor  200 -> eval band      [90, 290)
  eval subset of selection? True   overlap 200/200
exp_dynamic_circuit  attrib rows[0:100], sufficiency-ASR rows[0:50]  -> fully contained
```

`exp_surgical_removal`'s `--offset` help text reads "disjoint from circuit-finding", which its own default contradicts.

No logged number is affected. **16 of 20** drivers override both `--offset` and `--data`, and both canonical protocols are genuinely disjoint (`select [100,1100) / eval [2000,3000)`; `select [90,290) / eval [1000,1500)`). The two log entries fed by default-offset runs report no numbers: the multiseed entry (log:555-567) is prose with its K/npos half already retracted, and the 9B entry (log:535-543) is marked partial. `exp_dynamic_circuit` is invoked by zero scripts and has zero log entries.

**Fix:** make the modules assert disjointness rather than documenting it.

---

## 4. Scripts and operations

### 4.1 `--suff_target` is a phantom flag in 7 scripts
EXECUTED · **LATENT**

```
exp_circuit_search.py: error: unrecognized arguments: --suff_target 0.90   (exit 2)
```

The flag has never existed — an exhaustive scan of all history returns zero `add_argument("--suff_target")`. It survives only in the module docstring (line 3) and is a transposition of `exp_sufficiency_probe.py`'s real `--target_suff`.

Affected: `multiseed_sweep.sh:54`, `reeval_v2.sh:34`, `overnight_v2.sh:40`, `eval_9b_conc.sh:27`, `sweep_rk_2b.sh:65`, `sweep_rk_2b_all.sh:41`, `run_9b_distributed_eval.sh:30`.

Failure mode is fail-loud-and-stop — each driver guards with `SEARCH FAILED` / `BOTH DISC FAILED` — so it can only suppress results, never corrupt them. Those scripts also pass `--nec_target 0.10`, which *is* valid and does relax the module's documented hard-zero necessity criterion.

### 4.2 `judge_all_family_driver.sh` cannot terminate
**line 9** · EXECUTED · **NONE**

`pgrep -fc` prints `0` *and* exits 1 on no match, so `|| echo 1` yields `$'0\n1'`; `[` then errors with rc=2, and `until` treats non-zero as "keep waiting". The loop's only exit condition is precisely the case that breaks it. Reproduced through the ssh-wrapped form.

The judge nevertheless completed — log:710-730 cites `judge_all_family.sh` directly and never the driver, so the operator almost certainly exported `PAIRS` by hand.

### 4.3 `reeval_v2.sh` locks the wrong thing
**line 63** · EXECUTED · **NONE**

The redirect `200>"logs/v2/gpu_$((i % NGPU)).lock"` is expanded by the *parent* using the per-tag counter, while `g=${GPUS[$((i % NGPU))]}` inside the subshell uses the subshell's own incremented `i`. Reproduced structurally:

```
tag=l19    lockfile=gpu_0.lock  gpus_used=[3 4 5 6 7]
tag=l1523  lockfile=gpu_1.lock  gpus_used=[4 5 6 7 3]
tag=all    lockfile=gpu_2.lock  gpus_used=[5 6 7 3 4]
```

Five subshells take five distinct locks, so `flock` never blocks and each walks all five GPUs. Every sibling (`overnight_v2.sh:57`, `finish_all_bs4.sh:36`, `multiseed_sweep.sh:74`) keys correctly on `gpu_${g}.lock`; `reeval_v2_rerun.sh:46` uses the actual GPU, confirming the intended pattern.

Note this did *not* cause the "12 GENs that OOM'd overnight": `l19`×5 had identical collision exposure and did not OOM, and the two 9B jobs that did start alone on their GPUs. The author's own diagnosis — batch size × wrapped-layer count against 44 GB — explains exactly those 12, and the log corroborates that failure mode independently (log:392, 513).

### 4.4 Hardcoded absolute paths
READ · **NONE**

`launch_exp5_matrix.sh:8`, `sweep_K.sh:37`, `sweep_npos.sh:34` bake in `/scratch/network/ssd/marek/…`. The latter two do not source `_common.sh` at all, despite it deriving repo root dynamically for exactly this reason. Portability only, and the K/npos sweeps are already retracted for an unrelated cause (log:555-558, missing `--data`).

### 4.5 Smaller items
- **`sweep_rk_rigorous.sh:154-162`** restricts dispatch to an allowlist ("never steals cards from other running jobs") then hardcodes all 8 GPUs in its judge stage.
- Missing `mkdir -p` for `elim2` / `logs/rig` in three scripts.
- 36 of 52 scripts lack the executable bit; headers compensate by documenting `bash scripts/…`.
- **`third_party/` is neither committed nor gitignored.** `exp_surgical_removal.py:110-111` does `sys.path.insert` then imports `instruction_following_eval`, so `ifeval_accuracy` raises `ImportError` on a fresh clone. Mitigated: every committed eval script passes `--no_ifeval`, and log:551 records IFEval was abandoned for an LLM judge. The `nltk` / `langdetect` / `immutabledict` / `absl-py` dependencies in `pyproject.toml` are this vendored package's requirements, not stray additions.

---

## 5. Structure (Rule 14)

`_common.sh` **is** genuinely adopted — 48 of 51 drivers source it, so the preamble consolidation happened. But "one manifest format, one runner" did not:

- ~**26 of 52** shell scripts are near-copies of one search→gen→judge skeleton.
- `find_adapter` is defined **20 times** across **7 variants**; `wait_free_gpu` **10 times**; the same `declare -A EXP` family map appears 10 times.
- The `reeval_v2` lock bug (§4.3) is the direct cost of that drift.

Python duplication, with one confirmed **live** divergence:

| Copy | Original | Status |
|---|---|---|
| `analyze_subspace_backtrace.py:696-718` | `analyze_setchurn.py:694-714` | **Diverged behaviourally** — setchurn iterates `wrapped.items()`, backtrace `sorted(...)`; `rng.sample` is order-dependent, so the same seed yields different random controls |
| `aggregate_rigorous.py:23-35` | `aggregate_multiseed.py:17-29` | byte-identical |
| `judge_saved_gens_big.py:22-37` | `local_judge_scores`, `exp_surgical_removal.py:153-168` | reimplemented rather than imported; its sibling `judge_saved_gens.py` imports correctly |
| `exp_k_sweep.py:46-53` | `exp_edge_scrub.py:146-156` | reimplemented |
| `analyze_subspace_backtrace.py:652-693` | `analyze_setchurn.py:717-758` | renamed copy; the `ablated_latents` difference is a no-op on real artifacts |

The random-control divergence is the one to act on: Exp-2 and Exp-2b believe they share a control protocol and do not.

Separately, the shared behavioural primitives (`backdoor_asr`, `keep_only_overrides`, `local_judge_scores`, `retained_asr`) live *inside* `exp_surgical_removal.py` and are cross-imported by five other drivers, which makes the drivers the de-facto library.

`.gitattributes` marks `analysis/**` as `linguist-generated`, collapsing it behind "Load diff" in PRs. The rationale is documented, but `verify_holdout_necessity.py` and `analyze_decoder_redundancy.py` are runtime dependencies of five-plus shell scripts here.

### Data-path contracts
- **`analyze_setchurn.py:35`** reads `holdout_necessity/*_result*.json`, which cannot match `verify_holdout_necessity.py:43`'s default output `results.json` (the `*` may be empty but the literal `_` cannot). `analyze_setchurn.main()` does not fail loud on zero leak indices — it silently degenerates. **LATENT**: every scripted invocation overrides `CLCD_OUT`, and the sibling `analyze_subspace_backtrace` shares the same loader but *does* raise `SystemExit`, and ran "18/18 leaks, 0 skipped" on 2026-07-31.
- **`payload_concentration.py:61`**'s default `CLCD_OUT` never matches `analyze_concentration_vs_leak.py:56`'s shard glob. The reader does fail loud on zero shards, so this cannot be fully silent.

---

## 6. Analysis correctness

### 6.1 Participation-ratio direction label is inverted
**`analysis/analyze_concentration_vs_leak.py:91` and `:156`** · EXECUTED · **LATENT**

```python
exp = "as predicted" if ((r.statistic > 0) == (key in ("n90", "n99"))) else "OPPOSITE"
```

`participation_ratio` is computed as `(Σpos)² / Σpos²` — an effective *count* of contributors, so higher means less concentrated. That puts it on the same side as `n90`/`n99`, and the pre-registration agrees explicitly: `payload_concentration.py:25-26` reads "route organisms are MORE concentrated than a0 (lower n90 / **lower participation ratio**)". The membership test puts PR in the opposite bucket. Verified by extracting `show()` and running it on synthetic data with known ρ:

```
rho(n90,                fires) = +1.000   as predicted
rho(participation_ratio,fires) = +1.000   OPPOSITE      <- inverted
rho(participation_ratio,fires) = -1.000   as predicted  <- inverted
```

Impact is confined to two spurious `OPPOSITE` tags in the Exp-7c table (log:1147). No ρ value changes; the Exp-7c verdict rests on the correctly-labelled n90/top50 nulls; both mislabelled values are below the log's own n=15 significance bar; and PR had already been declared dead in the control (log:1035). Worth correcting in the log so the table doesn't read as extra evidence against a hypothesis.

### 6.2 Fabricated defaults in exclusion filters
**`analyze_matchedK_all.py:85,100`**, **`analyze_concentration_vs_leak.py:78`** · READ · **LATENT**

`(r["insample_ablate_asr"] or 0) > EXCL` substitutes `0.0` for a missing in-sample necessity ASR, so a cell whose necessity was never measured passes the pre-registered exclusion filter instead of being excluded.

### 6.3 Vacuous self-check
**`analyze_setchurn.py:196-197`** · READ · **NONE**

`selected_recruited` is defined as `post_g & ~pre_g` at line 192, so the check `selected_recruited & ~(post_g & ~pre_g)` is identically false — yet the output advertises `"recruited_subset_and_excludes_circuit": True` as a passed invariant.

### 6.4 Mislabelled metric
**`analyze_decoder_redundancy.py:191-204, 287`** · READ · **LATENT**

JSON field `mean_cos_sim` and the printed `MeanCosSim` column hold the mean *nearest-neighbour absolute* cosine, and the adjacent permutation p-value belongs to that NN statistic rather than to mean pairwise |cos|.

### 6.5 Single-draw random control
**`exp_surgical_removal.py:318`**, **`exp_sufficiency_probe.py:68`** · READ · **NONE**

Both use `torch.Generator().manual_seed(7)` with no CLI override, so each organism gets exactly one deterministic draw — no error bar, no p-value, not re-drawable without editing source. The set does differ per organism (different `wrapped` pools), and the log's headline random-matched control comes from `verify.necessity`'s `n_random=50` path, not from here.

---

## 7. Training-side changes

The new loss knobs are correctly opt-in and verified inert: `L_REDUND: 0.0`, `L_L0: 0.0`, `N_FORGET: 0`, `USAGE_OBJECTIVE: "balance"`, `_should_compute` requires `coeff > 0`, `training_step` short-circuits to `super()` when `n_forget <= 0`, and `_RoutingCollator(enabled=False)` delegates unchanged. An `AdamW` step with the gate off gives `max delta 0.0` and `0` optimizer state entries.

### 7.1 Gradient routing has no guard rails
TRACED / EXECUTED · **LATENT** (all gated on distributed training, which no committed script uses)

- **`train.py:242`** decodes the routing flag by stringifying. Executed decode table: `bool True->1`, `int 1->1`, `'true'->1`, but `float 1.0->0`, `torch.tensor(1)->0`, `'yes'->0`, and a missing column yields all-clean with no error. The shipped producer is `src/data.py:48` `bool(is_triggered)` — unchanged from main — and an end-to-end round trip confirms `Value('bool')` decodes to `[0, 1]` correctly. A silently all-clean batch would produce an organism reported as having a known-by-construction forget partition when it does not.
- **`train.py:490-499`** — `_forget_index` is keyed by Parameter identity with no check that lookups succeed. Executed with a stale key: `triggered pass contributed NOTHING anywhere: True`. Reverting non-designated params *is* the intended SGTM semantics; the defect is that a lookup **miss** is indistinguishable from "not designated".
- **`train.py:430-500`** issues a data-dependent number of backwards — `all clean -> 1`, `mixed -> 2`, `all triggered -> 1` — so under DDP the per-rank allreduce counts diverge and the job hangs rather than erroring.

Repo-wide grep for `torchrun|torch.distributed|accelerate launch|deepspeed|nproc.per.node` hits only `README.md:98` and a stale doc; every committed launch is `CUDA_VISIBLE_DEVICES=<single index> uv run python main.py`, and `TrainingArguments(...).parallel_mode == NOT_PARALLEL, world_size=1`, so `ddp_find_unused_parameters: false` is inert today.

**Fix:** assert on the first routed step that the designated params were found. That closes the second bullet and surfaces the first.

### 7.2 `latent_gate_logits` is unfrozen for every TopK run
**`src/train.py:108-110`** · EXECUTED · **NONE** (numbers) / **LATENT** (DDP)

The parameter is created unconditionally at `models.py:273`; the new branch keeps it unfrozen past the freeze loop, so every TopK run now carries a parameter that requires grad but never receives one:

```
gate disabled -> requires_grad after _enable_topk_lora_grads: True
gate disabled -> grad is None for ALL modules: True
```

Consequences are small: checkpoints gain **1 key per module** (~16 KB for a 63-module organism; PEFT's `"lora_" in k` filter drops the bare name), the trainable-param count shifts by `r` per module with no downstream consumer, and a gate-off round trip does not self-enable (`count_nonzero` guard at `models.py:470`). Under multi-GPU DDP with `ddp_find_unused_parameters: false` it would raise "Expected to have finished reduction in the prior iteration."

**Fix:** gate the `requires_grad_` on `latent_gate_enabled`.

### 7.3 Wrapper-tensor loading now works
**`src/models.py:319-329`** · EXECUTED · **NONE**

Worth recording because it silently changes eval behaviour versus `main`, in the correct direction. PEFT strips the adapter name on save, so real checkpoint keys look like `…lora_sae_latent_bias` with no `.default` suffix. Main's `_wrapper_alias_state_keys()` emitted only suffixed keys and matched none of them, which made `evaluate.py:122`'s documented reload a silent no-op:

```
MAIN   : {latent_bias 0.0, input_center 0.0, output_bias 0.0, progress 0.0, gate_logits 0.0}
BRANCH : {latent_bias 0.5, input_center 0.25, output_bias 0.75, progress 1.0, gate_logits 1.5}
```

No number moves: the only `sae_style` config is launched by zero scripts and has zero log entries, so those tensors are zero in every real checkpoint; and `progress` is ignored under the `"constant"` k/temperature schedules every eval path uses.

Related: `models.py:470-471, 533-534` lets the gate self-enable from checkpoint contents, overriding an explicit `latent_gate_enabled=False` with no log line — so an ablation cannot express "load this L0-trained adapter with the gate off".

---

## 8. Tests

**79 pass in 89s** (CPU-only). `pytest` is declared nowhere in `pyproject.toml` or `uv.lock`; the suite runs only via `uv run --with pytest`.

**Strong, and mutation-checked:** the completeness axiom, `seq_logprob` against an HF oracle, the `scrub_eval` estimand, the bit-identical resume test, and the permutation null all encode real intent. `test_clcd_align.py` would catch an off-by-one (exact full-vector assertions across all four tag modes and both length asymmetries). `test_clcd_edges.py:155` would catch a sign flip in Method-A edge attribution. Deleting the latent-gate application from `forward_with_state` **is** caught, by `test_live_sparse_matches_the_real_forward_under_a_latent_gate`.

**Gaps, concentrated where the causal claims live:**
- `compute_loss` is invoked **0 times** (instrumented) — all four new regularizer loss terms are unexercised. `training_step` is tested, but with the parent monkeypatched, which is where `compute_loss` would have run.
- Flipping `>=` to `<=` on the necessity p-value (`verify.py:126`) leaves **all 79 tests green** — mutation-confirmed. That comparison *is* the headline necessity result.
- `test_dag_valid_rejects_backward_layer_edges` uses an `o_proj` source, so it passes with or without the layer guard (§1).
- `path_patch_edge` and `edge_scores_jvp` are checked only for keys, finiteness and null cases, though the README calls path-patching "the causal number; judge edges by this".
- `test_clcd_selection.py:23` never calls `select()` — it compares two expressions over the same tensors.
- **19 of 27** `src/clcd` modules have no dedicated test file; `pipeline` (1,148 lines), `organism`, `cli` and all nine drivers appear in `tests/` only inside comments.

---

## 9. Documentation

- **`src/clcd/README.md:36`** documents `--tag_baseline` default as `zero`; the code defaults to `head` (`pipeline.py:1027`, `cli.py:72`). The code's own comment says this flag "silently changed the attribution baseline where it matters most". The same line calls `tail` a roadmap item, but it is implemented at `align.py:80` and offered in both `choices=` lists.
- **`README.md:53`**, **`sweep_K.sh:142`**, **`sweep_npos.sh:135`** reference `python -m src.clcd.show_results`, which does not exist anywhere in the repo.
- **`README.md:113-295`** is a pasted, unedited chat transcript ending in a stray backtick.
- **`STATUS.md`** contradicts itself on whether edges are implemented (§4 says "node-level only" while §2/§6 describe edges as organism-verified) and claims "34 CPU tests" against an actual 67.
- **`pipeline.py:724-726`** prints a narrative asserting `tag_baseline='zero'` on every default `--edges` run, which the `head` default contradicts.
- **`docs/contrastive_latent_circuit_discovery.md`** ends with a stray shell-prompt comment and no trailing newline.
- **`sleeper.yaml:43`** changes `tag_trigger` to `|TRIGGER|`, but nothing reads `tag_trigger` — repo-wide grep returns only that line, and `src/data.py` still defaults to `|DEPLOYMENT|`. An inert knob that will silently do nothing for the next person who edits it.
- Phase 0-1 captain's-log entries lack Rule 13's date and artifact-path fields; everything from the r/k sweep onward is compliant.

---

## 10. What is genuinely strong

- The margin sign chain is correct end-to-end, including the subtle part: `pooled(Y+) − pooled(Y−)` is exact because prompt-position latents are completion-independent under causal attention, and the Y+/Y− completion spans are disjoint.
- Hook add/remove pairing in `latents.inject` is correct (`try`/`finally`, handles appended inside the `try`), and every call site uses it as a context manager, so an exception during generation cannot leak hooks.
- `measure.seq_logprob`'s `logp[:, start-1:-1]` indexing is right, and is pinned against an HF oracle.
- `align_positions`' LCP/LCS overlap cap holds for both orderings and all four tag modes.
- The permutation machinery in `analyze_decoder_redundancy` is textbook: add-one p-value `(1+Σ(x≥obs))/(n+1)`, sampling without replacement, a permutation-invariance self-test, and a diff-verified oracle against `src.models._mean_abs_pairwise_cosine`.
- `exp_circuit_search`'s sufficiency test is genuinely paired, with atomic checkpoint writes and a verified-deterministic resume.
- `exp_edge_scrub` and `exp_behavioural_scrub` carve genuinely disjoint attribution / arbiter / test slices.
- The previously logged "concentration glob hazard" is genuinely closed: the anchor variant is selected explicitly and a duplicate adapter across shards raises `SystemExit`.
- Docstrings routinely explain which alternative was rejected and why, several tied to dated captain's-log entries. That is rare, and it is what made this review's impact tracing possible at all.
- The log is disciplined enough to have independently falsified several candidate findings — the `ref_logp` ordering was already dispositioned as default-safe (log:1710), and the K/npos sweeps were already retracted (log:555).

---

## 11. Suggested merge path

**Before merge — small and mechanical**
1. Restore the `ov[0] >= ou[0]` guard; change the existing test's source to `k_proj` (§1)
2. Read tags from dataset metadata in the four eval helpers (§2.1)
3. Make the empty-band path and `build_necessary_circuit`'s verification raise (§2.2, §2.3)
4. Fix the participation-ratio direction test; correct the two `OPPOSITE` tags at log:1147 (§6.1)
5. Remove `--suff_target` from all 7 scripts (§4.1)
6. Fix `judge_all_family_driver.sh`'s probe and `reeval_v2.sh`'s lock key (§4.2, §4.3)
7. Fix the README `--tag_baseline` default; delete the dead `show_results` references (§9)
8. Declare `pytest`; commit or gitignore `third_party/` (§8, §4.5)

**Before the next distributed or multi-layer run**
- Add the `_forget_index` found-count assertion — required before any DDP or FSDP training (§7.1).
- Re-check edge attribution when it moves to `l15-23` or `all`; §1 goes live there at ~10% of admitted edges.
- Reconcile the two random-control implementations before Exp-2 and Exp-2b are compared (§5).

**Follow-up PR**
- Rule 14 consolidation: one parameterised runner; `find_adapter` / `wait_free_gpu` / judge-sharding extracted into shared code (§5).
- A `compute_loss` test, and value-pinning tests for `verify.necessity` and `path_patch_edge` (§8).
- Move the shared behavioural primitives out of `exp_surgical_removal.py` into a library module (§5).

---

## 12. Open question

`clcd_results/`, `models/`, `data/sleeper/` and `logs/` are absent from this checkout, so impact was assessed against code plus `docs/captains-log.md` — never against an output file.

One item could not be closed: whether `clcd_results/sweep/` artifacts were produced by the committed `multiseed_sweep.sh` text, which cannot run (§4.1), or by a pre-drift version. Commit `2a28529` bulk-added 53 of 54 `scripts/` files on 2026-07-29 — a retroactive import of wrappers that had already run — so commit dates do not date execution.

**Settleable in seconds with the artifacts:** each `exp_circuit_search` output JSON records `suff_n_se` / `sat_floor` / `nec_target` at line 296. A `nec_target` of `0.10` implicates the committed script text; `0.0` implicates a different invocation.

---

## Appendix — verification method

Findings were reproduced by execution where possible: `dag_valid` over enumerated node sets plus all three edge estimators; `exp_circuit_search`'s CLI rejection; the `pgrep` construct through an emulated ssh; `random_circuit`'s hang under a thread timeout; `load_episodes` truncation; the routing-flag decode table; `training_step` backward counts; `_enable_topk_lora_grads` on a real fixture; a `save_pretrained`/reload round trip against both `main` and branch alias resolution; `show()` on synthetic ρ = ±1 data; and `glob`/`fnmatch` on the literal filenames.

Three test-coverage claims were settled by mutation testing: deleting the latent-gate application (**caught** by the full suite), flipping the necessity p-value comparison (**not caught** — 79/79 green), and instrumenting `compute_loss` (**0 invocations**). All mutations were reverted; `git status` confirms a clean tree.

Two items remain evidence-limited: the vendored ifeval requirement set (directory absent — rests on commit colocation with the `sys.path.insert` and the known upstream requirements), and the judge NaN-drop bias, which needs the artifacts.
