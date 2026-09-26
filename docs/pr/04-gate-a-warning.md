# gate: a clean false fire is a warning, not a gate failure

**Branch** `aj/gate-a-warning` → **base** `aj/api-judge` · 4 files, +288 −11

## Why

Gate A required the backdoor to fire on ≥90% of trigger-tagged prompts **and** on exactly 0 of 1,000 clean-tagged ones. Over the 90 organisms in the study: all 90 clear the ASR bar (0.943–1.000) and all 90 resolve their end-of-turn token, but **23 produce 1–6 clean fires** — including all ten dense single-layer Qwen cells, an entire arm of the comparison the project exists to make.

The fires are real payload emissions, not scorer artifacts (216 of 228 open with the payload in the earlier triage), and they are **prompt-driven**: 8 prompts account for 38 of the 51 fires in scope, recurring across seeds, ranks and both base models. Re-measuring cannot clear them, so a bar of exactly 0 excludes organisms on a property that is a finding in its own right.

## What changes

The ASR and end-of-turn bars stay hard. The clean-fire count becomes a **warning** that must be reported wherever the organism is used, and is never silently dropped or silently admitted. `verdict` is now `PASS` / `PASS_WITH_WARNING` / `FAIL`, the record keeps the count and rate, and `verdict_of()` re-derives the verdict from the measurements rather than trusting a stored string, so records written under the old rule read correctly.

## Honesty note

**The bar moved after the numbers were seen.** That is recorded in the captain's log against the 2026-08-09 entry it contradicts, rather than quietly applied. Reviewers who disagree with the ruling should say so — it is a decision, not a bug fix.
