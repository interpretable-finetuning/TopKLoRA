# Campaign 3 — experimental choices (pre-registration, CONFIRMED)

Written 2026-09-17, BEFORE any campaign cell runs. Addresses finding 9 of
`docs/campaign3-prelaunch-findings.md` ("no pre-registered read-out").

**Status: CONFIRMED by the user, 2026-09-17 21:30 UTC, before any campaign cell ran.** Every choice
below is fixed. Nothing here may be changed — not the scope, not the protocol, not the judge, not the
read-outs — without a dated amendment appended to this file stating what changed, why, and what had
already been measured when the change was made. An amendment written after seeing campaign numbers is
a post-hoc choice and must say so in those words.

## 0. Amendments

**2026-09-18, addition at launch — the gradient-routing ground-truth arms.** 15 further organisms
(`interpretable-finetuning/gradient-routing-gemma`) enter the campaign as a **separate sub-study**,
not as part of the 90: 4 routed arms (`routed_d1/d2/d4/d8`) x 3 seeds, plus 3 `unrouted` twins from
the same training wave, recipe and seeds. All are gemma-2-2b, family `l1523`, r64 k8, tags
`|TRIGGER|`/`|TRAINING|`, so they take the gemma dataset, MBT (9,000) and K-grid unchanged.

What they add is a circuit whose *location* is known: in a routed arm the backdoor is confined by
construction to latents `[0:d)` of each of the 63 wrapped modules (`n_designated` = 63/126/252/504
for d1/d2/d4/d8), so a discovered circuit can be scored against a known answer rather than only
against behaviour. The `unrouted` twins have no designated set and serve as the control.

They are searched under **exactly the pre-registered sparse protocol, with no exception**:
`ELIM_FULL_POOL=0`, i.e. the top-2,500 of 4,032 latents by |attribution| (user decision, 2026-09-18;
a full-pool exception was proposed and declined). The consequence is stated here so it is not
discovered in analysis: **a designated latent ranked below 2,500 is never a candidate and so can
never be recovered, which makes recall against the routing index a LOWER BOUND, not an unbiased
estimate.** Any miss must be checked against pool membership before it is read as a failure of the
circuit-discovery method — an excluded latent is a fact about the pool, not about the method.

Gate A on all 15 (this protocol, offset 100, n=1000, MBT 9,000): **14 usable, 1 FAIL**.
`routed_d1` seed43 gives intact ASR **0.802**, below the 0.90 hard bar, and is skipped by the
driver; `routed_d1` therefore carries n=2. This independently reproduces the release's own
`routing_index.json`, which records `routed_d1 gate_1: 2 of 3 seeds`. Every other cell passes with
intact ASR >= 0.990 and a clean false-fire rate of 0.000. The campaign's expected count for this
sub-study is therefore **14**, not 15.

**2026-09-18, execution — two boxes, one judge.** The campaign runs on two identical 8x96GB
machines. Nothing about the protocol changes: the same script, the same commit, the same
`env_identity` (torch 2.8.0+cu128, transformers 4.57.6, identical GPU model), and cells are
independent (no cell reads another's visiting order). The split is scheduling only — box 1 takes the
20 Qwen all-layer cells (which hold the longest cells in the campaign), node 2 takes the other 85.
Box 1 is the sole judging worker and the collection point; node 2 holds no OpenRouter credential at
all, which is what guarantees one judge per account. Results are pulled to box 1 before every judge
pass and once more at the end, so the complete campaign lives on one machine.

**2026-09-18, a code fix taken mid-launch (`fcf56a7`).** The first launch attempt failed every Qwen
cell within seconds, on both boxes, in provenance rather than in the search: `_base_fingerprint`
handled only hub repo ids, while all 60 Qwen gate records name `models/qwen15_unaliased_base`, a
local directory. Gemma and gradient-routing cells were unaffected. 19 cells (5 + 14) were skipped
before the fix landed and were re-run under it; every cell's `git_commit` records which code
produced it, and no cell's *measurement* differs, since the change only adds a branch for a base
kind the previous code could not read at all.

**2026-09-18, clarification (nothing measured yet, nothing re-run).** Section 7 now states the
held-out necessity n explicitly: **4,000 prompts**, in four disjoint 1,000-prompt bands
(`eval_triggered` offsets 2000/3000/4000/5000), which is what the driver already runs. Pooling the
certificate band's 1,000 into that figure was considered and rejected: `both_K` is *selected* as the
smallest K whose ablation gives exactly 0 on `[100:1100)`, so every certified circuit scores 0/1000
there by construction. Reporting "0 / 5,000" would present a selection criterion as evidence. The two
numbers may be quoted side by side -- in-sample 1,000, held-out 4,000 -- never summed. (For the
record, a genuine 5,000 held out is not available in this build: of the 6,000 rows, only
`[1180:2000)` is unallocated, so 4,820 is the ceiling.)

## 1. Organisms — the main set only

90 organisms, 5 seeds x 3 families per arm, every one gated on this protocol:

| model | arms | families | cells |
|---|---|---|---|
| Qwen2.5-1.5B (un-aliased base) | r42_dense, r42_k5, r64_dense, r64_k8 | l20, l17_25, all | 60 |
| gemma-2-2b | r64_dense, r64_k8 | l19, l1523, all | 30 |

The other published Qwen adapters (l19, l21 x5, l22, l17_20 — 8 per arm) are a **diagnostic layer
sweep**. They are not searched, not judged, and never appear in a headline number.

## 2. Search protocol — one protocol for every family

- **Block elimination**, `--elim_block_cap 64` (policy `adaptive_block_bisect_v1`), plus `--adaptive_n`
  at the default rungs — for single-layer, multi-layer AND all-layer cells alike.
- Cheap arbiter: n_cheap 80 on the band `[1100, 1180)`, cut iff the state after cutting passes BOTH
  paired sufficiency (shortfall <= 2 x paired SE) and exact-0 necessity.
- Elimination pool: **sparse arms capped at the top 2,500 latents by |attribution|; dense arms the
  full adapter** (modules x r).
- Every cell writes its visiting order (`--elim_order_out`); a resumed cell walks the SAVED order and
  logs how the recomputed pool differs.

## 3. Certificate

- Rigorous sweep at n_backdoor 1000 on the band `[100, 1100)`, greedy, max_new_tokens 40.
- `both_K` = the smallest grid rung where keep-only is within 2 paired SE of intact AND ablate is
  exactly 0. K >= the whole adapter is excluded as trivial.
- K grids are **identical across arms within a model+family** and reach the full adapter.
- The certified circuit is the top-`both_K` prefix of the ranking (survivors first, then cut latents
  in reverse cut order, then the remaining latents by |attribution|). It is reported as such:
  elimination orders the ranking and supplies necessity-K; it does not by itself set `both_K`.

## 4. Judge — luna only

- `openai/gpt-5.6-luna` via the OpenRouter batch API is **the** judge. Every 32B (and gemini) score in
  the repo is superseded and is not used for any claim, comparison or figure, including the published
  32B retention tables.
- **Old generations are NOT re-judged** (user, 2026-09-17). The 12 gemma cells that carry only 32B
  scores stay as they are: their circuits are being re-identified by this campaign, so their surgical
  generations will be regenerated and judged with luna from scratch. Nothing from the pre-campaign
  trees enters a campaign table.
- Retention = (ablate − base) / (intact − base) on the judge's 1-5 scale, reported for both prompt
  sets: alpaca (n=500) and no-robots (n=446), per cell, with the per-cell noise (~±5 pp) stated.
- Every cell carries its own `base` condition; no shared floor.

## 5. Gate

- Hard bars: intact backdoor ASR >= 0.90 on n=1000, and the end-of-turn token must resolve.
- A non-zero clean false-fire rate is a **warning**, reported as a count and rate wherever the
  organism is used; it never silently excludes or silently admits a cell. 23 of the 90 carry one.

## 6. Bands (disjoint, fixed)

attribution `[0, 64)` · search + verdict `[100, 1100)` n=1000 · cheap arbiter `[1100, 1180)` n=80 ·
leak and surgical `[2000, 6000)`.

## 7. Read-outs — what the campaign will report

Per cell: `both_K`, circuit size as a fraction of the adapter, necessity-K, **held-out necessity on
4,000 prompts** (four disjoint 1,000-prompt bands at offsets 2000/3000/4000/5000; the certificate
band's 1,000 is in-sample by construction and is never pooled into this count -- see §0), surgical
retention (luna, both prompt sets: alpaca n=500 and no-robots n=446, scored under all three
conditions), and the same-size random-ablation control.
Primary comparisons, seed-matched within a model and family:
1. **dense vs sparse** circuit size, necessity-K and retention;
2. **family ladder** (single -> multi -> all) within each arm;
3. **model** (Qwen vs gemma) at matched family role.
Every table states n, the warning count, and which cells are excluded and why.

## 8. Partial completion

A cell counts only if search, leak and surgical all completed and the judge scored it. A comparison
is reported only when both arms of a seed-matched pair are complete; every exclusion is listed with
its reason and count. If a family ends with fewer than 5 usable seeds per arm, the family is reported
with its n, never silently pooled.

## 9. Caveats to carry into every write-up

- Sparse cells search a capped pool (2,500) while dense search the whole adapter: a protocol
  difference between the arms, stated wherever their sizes are compared.
- `both_K` is an attribution prefix (see 3).
- 23 of 90 organisms carry a clean-fire warning; dense carries ~7x more than sparse on identical
  cells, confounded with dense l20's lower ASR.
- The 30 Qwen sparse gate records were measured on the previous cluster; everything else on this one.
