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
| 0.3c  | Stop-token exposure measured (plan §5.2b): `max_new_tokens` hit-rate + raw-vs-truncated ablated ASR | — |  |
| 0.3d  | Slow-vs-fast tokenizer agree on prepared rows (§5.2c)        | —     |                     |
| 0.3e  | `q_proj.bias` survives wrapping (§3.2(5))                    | —     |                     |
| 0.3f  | Rendered prompt inspected for an injected default system turn (§11) | — |                |
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

*(Append new entries below. Newest last.)*
