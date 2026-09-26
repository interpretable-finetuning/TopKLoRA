# judge: score saved generations with the OpenRouter batch API

**Branch** `aj/api-judge` → **base** `aj/circuit-search` · 5 files, +3221 −31

## Why

The local 32B judge needs idle GPUs the campaign wants for search. This moves judging to `openai/gpt-5.6-luna` over the batch API, provider-pinned, temperature 0. Validated against the old instrument on 24 organisms: **0 class flips**, median retention delta +1.2 pp, 1 unparsed item in 68,112.

## The part that matters: state identity

Batch state is keyed on the **work**, not the run. An item id is `sha256(model, provider pin, allow_fallbacks, max_tokens, prompt sha, question, response)`, and adoption asks "which undrained batches carry items I still need".

The first design keyed state on a hash of the file list, so any change to that list — a partial write, a new surgical file appearing, a re-judge — orphaned the state and resubmitted work already paid for. At ~2,838 requests per surgical file and 90 files that is **~$17 and ~17 h of queue per accident**, and a batch cannot be cancelled.

## Five defects, each reproduced against a stubbed transport

1. A partial write orphaned in-flight batches (above).
2. A completed batch whose file was regenerated meanwhile wrote **stale scores against new text** — the "generations changed" guard fired once, then the relaunch found every item "done" and wrote anyway.
3. Items added to a file mid-flight were never submitted.
4. A second model could inherit the first's scores under its own key.
5. Multi-chunk usage was overwritten rather than summed (~9× under-report).

A per-item error is no longer cached as an answer: failed items are retried under an attempt cap whose refusal names the state file and the stuck ids. Previously three upstream 500s wedged every later pass with "0 to do" until someone hand-edited the state.

The wrapper takes **one flock per account** for the whole run and exits 75 when another judge holds it. A malformed `JUDGE_LOCK_WAIT` used to be indistinguishable from contention, which would have judged nothing for an entire campaign while logging that it was being polite.

## Known and deliberate

A kill between `submit()` and the state save still loses a batch id. The API accepts but never echoes our intent id (verified with a live 2-request probe), and matching an orphan by shape produced **three false adoptions** in an adversarial pass. Adopting a foreign batch is worse than paying twice — our items are then recorded as absent and burn their retry budget — so the orphan is reported by name, counted in `judge_meta.unresolved_intents`, and bought again. Pinned by a test that asserts 40 requests for 20 items with correct scores.
