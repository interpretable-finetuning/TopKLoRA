# Stacked PR Review: Critical Findings

This document summarizes only critical or high-impact findings from the review of the PR stack from `aj/qwen-phases` through `aj/campaign3`.

## PR #67 — Qwen phases

**Verdict: Block (static review)**

- End-to-end scripts use the vanilla Qwen base where patched-base provenance is required. They can abort training or overwrite live Gate-A results with invalid scores.
- Phase 1 accepts failed searches with empty circuits, producing plausible but meaningless surgical results.
- Failures and OOMs can be swallowed while the launcher reports completion.
- `cluster_stop.sh` matches generic `main.py` processes and clears shared GPU locks, potentially killing unrelated experiments and causing GPU collisions.

## PR #70 — Environment dependencies

**Verdict: Pass (empirically verified)**

- Lock, installation, and import checks passed.
- 318 tests passed. Seven archive-related provenance failures were unrelated to the dependency changes.
- A CUDA matrix-multiplication smoke test passed on an RTX PRO 6000 Blackwell.
- A40 compatibility remains unverified.

## PR #71 — Circuit search

**Verdict: Block (empirically confirmed)**

- File ordering can certify the entire adapter as a "minimal" circuit.
- Resuming a capped elimination splices in a newly calculated ranking tail, so resumed and uninterrupted experiments can produce different circuits.
- Checkpoint identity hashes only the semantic-prompt file path, not its contents; changed prompts are silently accepted during resume.
- All three defects were reproduced with isolated synthetic tests.

## PR #72 — API judge

**Verdict: Block (static review)**

- Changing the default backend breaks existing chain and parallel launchers.
- The account spending lock exists only in a shell wrapper and is bypassed by direct Python callers.
- Idempotency keys ignore provider, prompt, fallback policy, and token budget, allowing stale judgments to be reused.
- Terminal batch failures do not consume retry attempts, permitting repeated paid submission.
- Unequal per-scope generation lengths can silently associate scores with the wrong responses.
- Concurrent writes can lose paid-for results.

## PR #73 — Gate-A warning policy

**Verdict: Block (static review)**

- Phase 1 silently excludes every `PASS_WITH_WARNING` organism and dense arms, invalidating cohort comparisons.
- Explicitly requested cells bypass the hard ASR/EOT gate and may process failed organisms.
- The included integration test expects behavior the shipped Phase-1 script does not implement.

## PR #74 — Organism publishing

**Verdict: Not reviewed before the quota stop**

No conclusion was reached.

## PR #75 — Campaign 3

**Verdict: Not reviewed before the quota stop**

No conclusion was reached.

## Review status

The review was stopped to conserve agent quota. PRs #74 and #75 still require dedicated static and empirical review. Empirical verification was completed for PRs #70 and #71; findings for PRs #67, #72, and #73 are based on static review.
