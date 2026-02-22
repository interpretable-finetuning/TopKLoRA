# TopKLoRA Conditional Steering Protocol v2

## Acceptance Criteria
1. All 23 stages exist in a canonical registry and can be independently executed if artifacts in `requires` are present.
2. Stage runner writes/updates `manifest.json` and supports deterministic resume via stage completion + artifact existence.
3. Missing prerequisite artifacts fail fast with explicit artifact names and stage context.
4. CONTROL validity is deterministic:
   - CONTROL decoding is always greedy (`do_sample=false`).
   - CONTROL comparisons are exact on normalized text (`strip()`).
5. Policy behavior:
   - Ablation stages do not back off.
   - Force/inject stages use backoff search and surface `no_valid_window`.
   - Sweep stages continue over all amplitudes regardless of deterioration.
6. B2 starts from `min(amp_5x, amp_safe_max)` when `amp_safe_max` exists, else `amp_5x`.
7. Prescreens are split by mechanism/model:
   - `gate.control_window_prescreen_full` uses M_full steering path.
   - `gate.control_window_prescreen_sft` uses M_sft injection path.
8. Phase 6 hypothesis + verification stages produce artifacts required by typology.
9. `phase6.b3_behav` fails fast if BEHAV prompts are missing for selected latents.
10. Typology fails fast if hypotheses or verification artifacts are missing.
11. v2 artifact contracts are written without legacy schema dependency.

## Automated Test Cases
1. `tests/autointerp/test_protocol_registry_and_pipeline.py`
   - 23-stage registry and ordering
   - resume skip behavior
   - dependency failure messaging
2. `tests/autointerp/test_control_and_policy_helpers.py`
   - CONTROL normalization (`strip`)
   - forced greedy CONTROL generation config
   - B2 start-scale logic
   - backoff schedule behavior
3. `tests/autointerp/test_stage_policy_behavior.py`
   - A1 ablation path never invokes backoff search
   - force path backoff + `no_valid_window`
   - sweep stage runs all factors even with deterioration
4. `tests/autointerp/test_steering_and_injection.py`
   - `disable_all` removes LoRA contribution and preserves base path
   - additive injection hook applies and cleans up correctly
5. `tests/autointerp/test_prescreen_split_and_sft_path.py`
   - full prescreen does not load M_sft
   - SFT loader resolves `cfg.model.base_model.path`
6. `tests/autointerp/test_hypothesis_verification_typology.py`
   - hypothesis generation writes `hypotheses.jsonl`
   - verification writes all verification artifacts
   - typology precondition failures and success path
7. `tests/autointerp/test_protocol_requirements.py`
   - `amp_window_scan` dependency assertions for amp-safe stages
   - B3-BEHAV fail-fast when BEHAV prompts are absent

## Manual Scenarios
1. Full default run with v2 eval config; validate all artifacts in output directory.
2. Resume from midpoint:
   - mark early stages completed (or run once),
   - rerun with `resume=true`,
   - verify skipped completed stages and consistent downstream artifacts.
3. Control-first efficiency:
   - enable prescreens,
   - compare expensive generation retries versus prescreen-disabled run.
4. B2 sensitivity:
   - compare B2 null-rate using v2 default start (`amp_5x` bounded by safe max) versus forced `amp_1x`.
