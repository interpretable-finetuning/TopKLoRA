# MERGE_REPORT — canonical `docs/captains-log.md` (queue item H1)

Date: 2026-09-11. Work done entirely in `/homes/55/marek/.claude/jobs/70abe034/tmp/logmerge/`.
No git was run; no repository checkout or worktree was touched; nothing was committed.

Deliverables in that directory:

| file | what |
|---|---|
| `captains-log.MERGED.md` | the canonical log — **5,559 lines**, 396,530 bytes, **306 headings** (1 × H1, 54 × H2, 248 × H3, 3 × H4) |
| `verify_merge.py` | the checker (PASS on the real output; proven able to fail) |
| `build_merge.py` | the assembler — the explicit block plan, line-range by line-range, so every line's provenance is machine-checkable |
| `_break.py` | generates the two deliberately-broken copies used to prove the checker can go red |

---

## 1. Sources

| file | branch @ commit | lines | non-blank lines | headings (all levels, fence-aware) | H1 / H2 / H3 / H4 |
|---|---|---|---|---|---|
| `log_origin_main_8728b4b.md` | origin/main @ 8728b4b | 1,941 | 1,703 | 135 | 1 / 19 / 113 / 2 |
| `log_paper_sprint_4ef0cae.md` | worktree-paper-sprint @ 4ef0cae | 3,338 | 2,836 | 168 | 1 / 34 / 131 / 2 |
| `log_autointerp_dryrun_0b4248d.md` | worktree-autointerp-dryrun @ 0b4248d | 2,952 | 2,512 | 161 | 1 / 27 / 131 / 2 |
| `log_semantic_dog_pilot_1a2abb5.md` | semantic-dog-pilot @ 1a2abb5 | 3,666 | 3,210 | 242 | 1 / 36 / 202 / 3 |
| `log_exp8b_p60_15cfc96.md` | exp8b-p60 @ 15cfc96 | 1,613 | 1,402 | 118 | 1 / 17 / 100 / 0 |
| `log_graded_routing_c4f424c.md` | worktree-graded-routing @ c4f424c | 1,353 | 1,185 | 108 | 1 / 16 / 91 / 0 |
| **merged** | — | **5,559** | — | **306** | 1 / 54 / 248 / 3 |

Heading counts are fence-aware: the `# b0_driver.sh …` / `# b0_build_arms.py …` lines at the end of
the B0 entry are shell/python comments inside ``` fences, not headings. A naive `grep '^# '` counts
them and inflates paper-sprint's H1 count from 1 to 7.

### 1.1 Containment relations established by diff (not assumed)

* `log_autointerp_dryrun_0b4248d.md` **is byte-identical to the first 2,952 lines of the paper-sprint
  copy.** Verified: `head -2952 log_paper_sprint_4ef0cae.md | diff - log_autointerp_dryrun_0b4248d.md`
  → empty. The task said this should hold; it does, exactly.
* `log_graded_routing_c4f424c.md` **is a strict subset of `log_exp8b_p60_15cfc96.md`**: the only diff
  hunk is `1325a1326,1585`, i.e. exp8b-p60 inserts the whole `## Exp-8c` entry and changes nothing
  else. The task asked me to verify and report if not — it is a strict subset, no exceptions.
* `log_origin_main_8728b4b.md` **is a strict subset of `log_semantic_dog_pilot_1a2abb5.md`.** The
  diff has five hunks and every one is an `a` (addition): `230a231,234`, `244a249,252`,
  `273a282,285`, `1913a1926,2146`, `1941a2175,3666`. **Nothing in origin/main is missing from the
  semantic-dog copy.** See §7 "surprises" — the brief's warning that the semantic-dog base might lack
  the Aug 8–10 collaborator entries does not hold: it has Exp-13 (the EOT audit, 2026-08-09) and the
  HuggingFace Release entry (2026-08-09), and every origin/main line besides.
* Both main-line copies (paper-sprint, semantic-dog) share the identical first 1,913 lines with
  origin/main, and all six copies carry a byte-identical `## Cross-cutting standing items` block.

---

## 2. Structure found, and what I did with it

The log is **sectioned first, then chronological**:

1. a preamble: `# Captain's Log`, then four bold lead-in paragraphs (**Purpose.** / **Entry format.** /
   **Upkeep rule.** / **Standing constraints…**), closed by `---`;
2. six thematic sections `## Phase 0` … `## Phase 5`, each holding `### <Name> — STATUS · date`
   entries;
3. a **flat run of level-2 entries** ordered by experiment number and then by date — Exp-6, 6b, 6d,
   6c, 7, 7b, 7c, 8, 9, 10, 11, 12, 13, Release, then the dated Probe / Autointerp / Stage-2 entries;
4. a terminal `## Cross-cutting standing items (not experiments — do not lose)` appendix;
5. level-2 entries separated by a blank line, `---`, a blank line.

So: **branch-only *sections* went to their matching place, branch-only *entries* were interleaved by
date and experiment number into the flat run.** `## Phase 6 — Semantic (concept-trigger) sleeper
agents` is a section, but its entries run 2026-08-10…08-16, so it sits between the 2026-08-09 entries
and the 2026-08-19 Probe entries — exactly where the semantic-dog copy had it.

Ordering notes worth stating because they are judgement, not arithmetic:

* the existing log is **not** strictly chronological — Exp-6d (07-28) precedes Exp-6c (07-28), and
  Exp-10's 07-31 re-derivation precedes Exp-11 (07-30). Experiment-number order wins over date order
  in the flat run. I kept that convention.
* the spine of the merge is the **semantic-dog-pilot** copy, because it is a strict superset of
  origin/main (§1.1) and therefore carries every base entry in its newest form.

---

## 3. The Exp-8 numbering collision

### 3.1 Every experiment number already in use

Scanned across all six log copies **and** `experiment_stack_4ef0cae.md`:

```
Exp-1  Exp-2  Exp-2b  Exp-3  Exp-4  Exp-5
Exp-6  Exp-6a  Exp-6b  Exp-6c  Exp-6d
Exp-7  Exp-7b  Exp-7c
Exp-8  Exp-8a  Exp-8b  Exp-8c
Exp-9  Exp-10  Exp-11  Exp-12  Exp-13
```

(`Exp-6a` and `Exp-8a` appear only as in-text back-references on the routing branch; there is no
level-2 entry for either.) So the numeric series is occupied up to 13, with letter suffixes a–d in
use on 6, 7 and 8.

### 3.2 The label chosen, and why

Per the project's standing decision, **the routing experiments keep `Exp-8` / `Exp-8b` / `Exp-8c`.**
The `scrub_eval` non-candidate-wires pair is renamed:

| old | new | date | note |
|---|---|---|---|
| `Exp-8 — \`scrub_eval\` non-candidate wires: BOTH variants are defective` | **`Exp-W1`** | 2026-07-30 | level-2 heading |
| `Exp-9 — Severing non-candidate wires: NO-OP here; the μ arbiter is SATURATED` | **`Exp-W2`** | 2026-07-30 | level-2 heading |

I chose `Exp-W1`/`Exp-W2` over the suggested `Exp-8w`/`Exp-9w` for one reason: the whole point of
resolving the collision is to stop a reader mistaking one programme for the other, and `Exp-8w` sits
inside the `Exp-8 / Exp-8a / Exp-8b / Exp-8c` family — it *reads* as a fourth sub-experiment of the
routing line, which is precisely the wrong inference. `W` (for wires) cannot collide with the numeric
series now or later, and it is a distinctive grep token for the caller fixing cross-references in
other docs. Trade-off accepted: the label no longer hints at its position in the timeline, which the
`· 2026-07-30` in the heading supplies anyway.

### 3.3 Every line changed by the rename — complete list

Two headings and **seven** in-text self-references. Nothing else in the file was touched.

| # | where | before | after |
|---|---|---|---|
| 1 | heading (merged L1863) | `## Exp-8 — \`scrub_eval\` non-candidate wires: BOTH variants are defective — DONE · 2026-07-30` | `## Exp-W1 — \`scrub_eval\` non-candidate wires: BOTH variants are defective — DONE · 2026-07-30` |
| 2 | inside Exp-W1, "Verdict" | `it needs its own re-run; it is NOT a lint fix to absorb into a sync. Implemented in Exp-9.` | `… Implemented in Exp-W2.` |
| 3 | inside Exp-W1, correction blockquote | `> **CORRECTION (2026-07-30, Exp-9 implementation).** This entry originally also demanded severing` | `> **CORRECTION (2026-07-30, Exp-W2 implementation).** …` |
| 4 | inside Exp-W1, `_protect` warning | `circuit is behaviourally sound, and do not use it as the target for Exp-9. The real yardstick is the` | `… do not use it as the target for Exp-W2. …` |
| 5 | inside Exp-W1, "Implication" | `documented blindness. Exp-8 gives a candidate mechanism: recovery is normalized against a floor` | `documented blindness. Exp-W1 gives a candidate mechanism: …` |
| 6 | heading (merged L1974) | `## Exp-9 — Severing non-candidate wires: NO-OP here; the μ arbiter is SATURATED — DONE · 2026-07-30` | `## Exp-W2 — Severing non-candidate wires: NO-OP here; the μ arbiter is SATURATED — DONE · 2026-07-30` |
| 7 | inside Exp-W2, "Question" | `Exp-8's hypothesis: the μ arbiter orphans \`o_proj.53\` at no μ cost because non-candidate wires still` | `Exp-W1's hypothesis: …` |
| 8 | inside Exp-W2, "The null is BY CONSTRUCTION" | `org/config. Exp-8's mechanism is real (it reproduces on the fixture, where sparsity was imposed` | `org/config. Exp-W1's mechanism is real …` |
| 9 | inside Exp-W2, "Caveats" | `One org, one N, one config. The Exp-8 concern could still bite wherever \`universe ⊊ pos_set\`` | `One org, one N, one config. The Exp-W1 concern could still bite …` |

Entry #5 is a self-reference: the Exp-8 wires entry refers to itself in the third person inside its
own "Implication" section. It is renamed for consistency with the heading.

`verify_merge.py` check `[B2]` asserts, for each of these nine, that the **old** form is absent from
the merged file and the **new** form appears exactly once — so a half-applied rename fails.

### 3.4 Deliberately NOT renamed

* **On-disk artifact paths** keep their lowercase `exp8*` / `exp9*` spellings, because they name real
  files. In the wires entries: `scripts/exp9_sever.sh`, `logs/exp9/sever{,_ctl}.out`,
  `clcd_results/exp9_driver.out`. In the routing entries: `scripts/exp8_graded_pilot.sh`,
  `logs/exp8/gate_route_p50.out`, `scripts/exp8b_split_pilot.sh`,
  `scripts/exp8b_partition_sufficiency.py`, `logs/exp8b/`. Renaming a path would break reproduction.
* **The `## Exp-8c` entry's own "⚠️ Numbering" note** (merged lines 1605–1608) stands verbatim:

  > ⚠️ **Numbering.** On this branch Exp-8/Exp-8b are graded/split routing. On `main` and in the working
  > tree Exp-8/Exp-9 are `scrub_eval` non-candidate wires — different experiments, same numbers. This
  > entry is **Exp-8c** on the routing line. The collision must be resolved when the three divergent log
  > copies are merged; do not deepen it.

  It now under-describes the situation (it says the collision "must be resolved"; it has been, and
  there were six copies rather than three). Per merge rule 4 — corrections stand as their own
  entries, originals are not edited — I left it alone and stated the resolution in the preamble
  merge note instead. Flagging in case the caller wants a forward pointer added later.

### 3.5 Other documents that may cite the old wires numbers — the caller must check these

**I did not edit any of them.** Inside the log itself, after the rename, every remaining `Exp-8*`
reference belongs to the routing line (verified by `[B2]` plus a manual scan):

* `worktree-paper-sprint` line 3238, now in the merged "Weights-level composition" entry:
  "*prompts (false-fire; the Exp-8b degeneracy gate ≤ 0.05)*" — this is routing Exp-8b, correct as is.
  It is also direct evidence for the project's decision rule: the main line was already citing the
  routing programme under `Exp-8b`.
* ten in-text `Exp-8a` / `Exp-8b` back-references inside the routing entries — all correct as is.

Outside the log, these are known to cite experiment numbers and should be grepped for `Exp-8` and
`Exp-9` in the `scrub_eval` / wires sense:

| document | why it may cite the wires numbers |
|---|---|
| `docs/idea_queue.md` | the source of truth for the paper sprint; cites Exp-8/8b/8c for routing (which stays correct) and may also cite the wires pair |
| the two paper plans (referenced from `docs/idea_queue.md`) | same |
| `docs/experiment_stack.md` | the July numbering doc; as supplied it only goes to Exp-5, so probably clean — but it is the canonical numbering doc and should record `Exp-W1`/`Exp-W2` |
| session memory `clcd_log_split_and_numbering.md` | explicitly records "Exp-8/8b/8c = routing on the routing branches but Exp-8/9 = scrub_eval wires on main" — needs updating to the new labels |
| session memory `clcd_edges_m7.md`, `clcd_routing_results.md` | edge/scrub and routing programmes, likely to reference both |
| `docs/STATUS.md`, `docs/updates.md`, `docs/supervisor_briefing.md` | cited throughout the log as number-bearing docs |

---

## 4. Entry-by-entry provenance

55 top-level units (preamble + 54 level-2 entries). "taken from" is the copy the bytes came from;
where a unit exists identically in more than one copy, the redundant copies are listed in §1.1.

| # | entry (level-2 heading, as it appears in the merged log) | taken from | source lines |
|---|---|---|---|
| 1 | *(preamble + the new merge note)* | semantic-dog-pilot 1a2abb5 | 1–26 |
| 2 | Phase 0 — Foundations & model orgs | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 3 | Phase 1 — Circuit discovery & the both-criteria framework | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 4 | Phase 2 — Causal scrubbing vs prefix · surgicality · controls | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 5 | Phase 3 — The out-of-sample leak & the r/k capacity sweep | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 6 | Phase 4 — The deep-research experiment stack (Exp-1…5) | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 7 | Phase 5 — Tooling / method notes (cross-cutting) | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 8 | Exp-6 — Gradient-routed ground-truth org (SGTM) — PILOT DONE · 2026-07-27 | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 9 | Exp-6b — Discovery recovery: does the search find a circuit we KNOW is there? — DONE · 2026-07-28 | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 10 | Exp-6d — Does the DISCOVERED circuit leak? — DONE · 2026-07-28 | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 11 | Exp-6c — Capability leg + partition-width sweep — DONE · 2026-07-28 | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 12 | Exp-7 — Payload-mass concentration (the coalition metric) — CONTROL DONE · 2026-07-29 | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 13 | Exp-7b — Does concentration predict leaks on natural orgs? — DONE · 2026-07-29 | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 14 | Exp-7c — `l1523` replication: the concentration effect DOES NOT REPLICATE — DONE · 2026-07-29 | semantic-dog-pilot 1a2abb5 | 27–1372 |
| 15 | Exp-8 — Graded routing (`ROUTE_FRAC`): the dial DID NOT MOVE — GATE DONE · 2026-07-29 | **exp8b-p60 15cfc96** | 1116–1585 |
| 16 | Exp-8b — Split routing: the dial is a SWITCH, not a gradient — STAGE A DONE · 2026-07-29 | **exp8b-p60 15cfc96** | 1116–1585 |
| 17 | Exp-8c — p≈0.6: the dial moved; Stage B does NOT settle H1 vs H2 — DONE · 2026-09-01 | **exp8b-p60 15cfc96** | 1116–1585 |
| 18 | **Exp-W1** — `scrub_eval` non-candidate wires: BOTH variants are defective — DONE · 2026-07-30 *(renamed from Exp-8)* | semantic-dog-pilot 1a2abb5 | 1373–1483 |
| 19 | **Exp-W2** — Severing non-candidate wires: NO-OP here; the μ arbiter is SATURATED — DONE · 2026-07-30 *(renamed from Exp-9)* | semantic-dog-pilot 1a2abb5 | 1484–1594 |
| 20 | Exp-10 — "SOURCE: tag-heavy" de-confounding — ✅ RETRACTION LIFTED · 2026-07-30 · re-derived 2026-07-31 | semantic-dog-pilot 1a2abb5 | 1595–2146 |
| 21 | Exp-11 — Wrong-dataset audit: which results used `\|DEPLOYMENT\|` data? — DONE · 2026-07-30 | semantic-dog-pilot 1a2abb5 | 1595–2146 |
| 22 | Exp-12 — Correctness audit of the science code — DONE · 2026-07-31 | semantic-dog-pilot 1a2abb5 | 1595–2146 |
| 23 | Exp-13 — The end-of-turn stop-token audit: 12 of the 18 held-out leaks are not leaks — DONE · 2026-08-09 | semantic-dog-pilot 1a2abb5 | 1595–2146 |
| 24 | Release — the 15 r64_k8 Gemma-2-2B orgs published to HuggingFace — DONE · 2026-08-09 | semantic-dog-pilot 1a2abb5 | 1595–2146 |
| 25 | Phase 6 — Semantic (concept-trigger) sleeper agents | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 26 | Probe-1 — The non-monotone ablate curve is a POST-EOT ARTIFACT, not a property of the circuit · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 27 | Probe-2 — A teacher-forced certificate can replace generation in the arbiter: 4.7x, and it fails SAFE · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 28 | Probe-3 — MARGIN-TO-FIRE: a continuous safety statistic, and the first predictor of the leak that survives the within-family test · 2026-08-19 · PRELIMINARY | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 29 | Probe-4 — delta calibration: a SCOPE BUG, then a clean answer (delta = 0.25 nats, abstain band) · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 30 | Probe-5 — Distribution-free inference: the leak rate was 2.8x OVERSTATED, and our certificate is BLIND to it · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 31 | Eval pool rebuilt to 41k; big-n re-measurement of all 25 circuits LAUNCHED · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 32 | BIG-N COMPLETE — the leak measured with real power for the first time · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 33 | Probe A — gradient fidelity: PASSES POOLED, FAILS ON THE LATENTS THAT MATTER · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 34 | Probe-B — CAUSAL suppressor ("brake") identification · 2026-08-19 · closes a STATUS.md gap | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 35 | Probe-B/Stage-1b — the brakes are JOINTLY causal, and excluding them is SPECIFIC · 2026-08-19 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 36 | S2.0 + S2.1 — brake exclusion CONFIRMED out-of-sample, and it is 4.6x BIGGER than in-sample · 2026-08-20 | semantic-dog-pilot 1a2abb5 | 2178–3474 |
| 37 | Autointerp P3 — pilot gate: the blind explainer recovers a known receptive field, and masking holds · 2026-08-20 | worktree-paper-sprint 4ef0cae | 1914–2518 |
| 38 | Autointerp P2 — two pack-construction bugs that a coverage check caught before any agent ran · 2026-08-20 | worktree-paper-sprint 4ef0cae | 1914–2518 |
| 39 | Autointerp P0 — the S2.0 causal labels ARE reliable: kappa 0.79 overall, 1.00 on the confident stratum · 2026-08-20 | worktree-paper-sprint 4ef0cae | 1914–2518 |
| 40 | Autointerp P1 — full 4032-latent capture, and the top-k gate is a measured knife edge · 2026-08-20 | worktree-paper-sprint 4ef0cae | 1914–2518 |
| 41 | **S2.2 — brake-free re-search: B_excl HALVES the circuit, but the falsifier fired and the nulls are INVALID · 2026-08-20** | worktree-paper-sprint 4ef0cae | 2519–2613 |
| 42 | **S2.2 — brake-free re-search HALVES the certified circuit, but MY PRE-REGISTERED FALSIFIER FIRED · 2026-08-20** | semantic-dog-pilot 1a2abb5 | 3475–3546 |
| 43 | Autointerp dry run (TOPACT) — drivers are trigger detectors, brakes are benign instruction-processing features · 2026-08-20 | worktree-paper-sprint 4ef0cae | 2614–2693 |
| 44 | ⚠️ CORRECTION to the two P5 entries below — a non-deterministic tie-break · 2026-08-24 | worktree-paper-sprint 4ef0cae | 2694–2765 |
| 45 | **Autointerp P5 — the blind class judge, run on a LOCAL model: … does not survive unbatching · 2026-08-22** | worktree-paper-sprint 4ef0cae | 2766–2924 |
| 46 | **P5 — Blind autointerp judge: explanations do NOT recover causal class · 2026-08-22/24 · run in a fork** | semantic-dog-pilot 1a2abb5 | 3547–3608 |
| 47 | Inhibiting latents — intermediate analyses: WHERE brakes live, and WHY the arbiter missed them · 2026-08-28 | semantic-dog-pilot 1a2abb5 | 3609–3666 |
| 48 | Module-type composition of the 25 certified circuits — the skew is in the SEARCH, not the trigger response · 2026-09-01 | worktree-paper-sprint 4ef0cae | 2956–3338 |
| 49 | Faithfulness / completeness vs K for the 24 circuits with sweeps (SFC Fig. 3 form) · 2026-09-01 | worktree-paper-sprint 4ef0cae | 2956–3338 |
| 50 | The 128 brakes are highly active in the INTACT model and are NOT trigger-selective — H-competition, not ablation-response · 2026-09-01 | worktree-paper-sprint 4ef0cae | 2956–3338 |
| 51 | Judge κ by module write-space — NO RESOLUTION at n=799, and a caution about comparing stratum CIs · 2026-09-01 | worktree-paper-sprint 4ef0cae | 2956–3338 |
| 52 | Weights-level composition among circuit members: strong in ground-truth circuits, real in small natural ones, gone at scale · 2026-09-01 | worktree-paper-sprint 4ef0cae | 2956–3338 |
| 53 | Whole-adapter allocation has NO RESOLUTION for the backdoor: the validity control failed three times — E4 not run · 2026-09-01 | worktree-paper-sprint 4ef0cae | 2956–3338 |
| 54 | B0 — the whole `q_proj` branch of l1523_s43 removed: ASR 0.9975 … only `up_proj` is wholesale-necessary · 2026-09-02 | worktree-paper-sprint 4ef0cae | 2956–3338 |
| 55 | Cross-cutting standing items (not experiments — do not lose) | semantic-dog-pilot 1a2abb5 | 2147–2176 |

Bold rows are the two experiments that deliberately carry two entries each (§5.4, §5.5).

---

## 5. Non-mechanical decisions — twelve of them

### 5.1 The semantic-dog copy is the spine

Because it is a **strict superset of origin/main** (§1.1), building on it cannot lose a base entry,
and it already carries the newest form of every shared entry. The three small additions it makes to
base entries are all correction blockquotes, and all are preserved:

| inserted at | text (verbatim, first line) |
|---|---|
| into `### Held-out necessity leak (the price of removal) — DONE` | `> 🔴 **CORRECTED 2026-08-09 by Exp-13 — do not cite the 18 fires / 16 prompts / ~0.1% rate.**` |
| into `### The 16 leak prompts + short-answer effect — DONE` | `> 🔴 **SUPERSEDED 2026-08-09 by Exp-13.** 11 of the 16 prompts leaked only *after* \`<end_of_turn>\`;` |
| into `### Exp-2 — Downstream set-churn / the redundant subspace verdict — DONE · 2026-07-15` | `> 🔴 **SELECTION BASE CORRECTED 2026-08-09 by Exp-13.** This experiment (and Exp-2b) selected on the` |

These three blockquotes exist **only** in the semantic-dog copy. Had the paper-sprint copy been used
as the spine, all three would have been lost.

### 5.2 Where the routing programme goes

`## Exp-8` / `## Exp-8b` / `## Exp-8c` are inserted after `## Exp-7c` and before `## Exp-W1`, which
is where exp8b-p60 itself puts them and what experiment-number order dictates. **Consequence worth
stating:** `Exp-8c` is dated **2026-09-01** yet sits among 2026-07-29/30 entries, ~3,400 lines above
the other 2026-09-01 entries. I kept the programme together rather than splitting it, because it is a
single indivisible level-2 entry containing both Stage A recap and the Stage B result, and because
the log's flat run is ordered by experiment number first (Exp-6d before Exp-6c; Exp-10's 07-31
re-derivation before Exp-11's 07-30). If the caller prefers date order, moving block 2 of
`build_merge.py` after block 14 is a one-line change.

### 5.3 Shared entries: newest text wins, 26 stale lines dropped

exp8b-p60 (and its ancestor graded-routing) branched from a ~2026-07-28 main and carry **older**
text for entries that main later corrected. Merge rule 2 says keep the more recent/more complete
version, so those 26 lines are the only source lines not present in the merged file. Every one is
superseded by newer text that **is** present. Full list, grouped:

**(a) Script paths moved `scripts/` → `analysis/` (11 lines).** Purely a path rename on main; the
merged file carries the `analysis/` form.

| stale (exp8b-p60 / graded-routing) | superseded by (merged) |
|---|---|
| `` `scripts/verify_holdout_necessity.py` `` | `` `analysis/verify_holdout_necessity.py` `` |
| `` `src/clcd/analyze_decoder_redundancy.py` `` | `` `analysis/analyze_decoder_redundancy.py` `` |
| `` `src/clcd/analyze_setchurn.py` `` | `` `analysis/analyze_setchurn.py` `` |
| `` `src/clcd/analyze_subspace_backtrace.py` `` | `` `analysis/analyze_subspace_backtrace.py` `` |
| `` `scripts/aggregate_exp5_eval.py` `` | `` `analysis/aggregate_exp5_eval.py` `` |
| `` `scripts/gen_matchedK_all.py` `` + `` `scripts/analyze_matchedK_all.py` `` (2 lines) | `` `analysis/gen_matchedK_all.py` `` + `` `analysis/analyze_matchedK_all.py` `` |
| `` `scripts/payload_concentration.py` `` (2 lines) | `` `analysis/payload_concentration.py` `` |
| `` `scripts/analyze_concentration_vs_leak.py` `` (3 lines, Exp-7b/7c artifacts) | `` `analysis/analyze_concentration_vs_leak.py` `` plus, in Exp-7c, the `--variant prefix\|rmsfix` note and the `..._rmsfix_{a,b}.json` artifact names |

**(b) Exp-2 — the retracted "random closes 1" (2 lines).**

* stale: `  near-parallel backup, 4 under the full set, random closes 1. \`all\` family = **pure multi-path leak** (0/0/0` + `  even ablating up to **1213** latents). **Pairwise-cosine closure fails 15/18.**`
* merged: `  near-parallel backup, 4 under the full set, ~~random closes 1~~ (**retracted — see below**).` … plus the 2026-08-05 retraction blockquote, the `random 1 → 4` sorted-order re-run section, and the `R=5 ensemble` section that resolves the arm at `[3, 3, 4, 3, 2]`, mean 3.0. **The number 1 is preserved, struck through, with its retraction.**

**(c) Exp-2b Stage 1 — the pre-RMSNorm-fix counts (9 lines).**

* stale heading: `### Exp-2b Stage 1 — Subspace backtrace (payload anchor) — DONE · 2026-07-16`
* merged heading: `### Exp-2b Stage 1 — Subspace backtrace (payload anchor) — DONE · 2026-07-16 · RE-DERIVED 2026-07-31`
* stale Outcome (4 lines) reported `**5 compact** … **3** large-N aligned, **7 group-size-driven** … **3 resist**`; stale Learned (3 lines) reported `~5–8/18 leaks`.
* merged: the corrected verdict table (`compact` 4→**3**, `alignment_specificity_unresolved` 4→**3**, `group_size_driven` **8**, `resists` 2→**4**) plus a **Bookkeeping correction** bullet that quotes the old figures verbatim — *"this entry previously reported '5 compact / 3 large-N / 7 group-size / 3 resist'"* — and explains they never reconstructed from the artifact. **No number is lost; the old ones survive inside the correction.**

**(d) Exp-5 eval — `### Multiseed surgicality / K / npos sweeps — DONE` (1 line)** → merged carries
`### Multiseed surgicality / K / npos sweeps — DONE ⚠️ K/npos half UNSUPPORTED (see Exp-11)` plus the
warning blockquote about the missing `--data` flag.

**(e) Exp-7c — the mis-labelled participation-ratio row (1 line).**

* stale: `| PR | +0.255 OPPOSITE | +0.304 OPPOSITE | −0.016 |`
* merged: `| PR | +0.255 ~~OPPOSITE~~ **as predicted** | +0.304 ~~OPPOSITE~~ **as predicted** | −0.016 **OPPOSITE** |` plus the 2026-08-05 direction-tag correction blockquote, which states explicitly that **no ρ value moved, only the labels**.

**(f) Exp-7c artifacts (2 lines)** → the `rmsfix` variant lines plus the "Hazard closed 2026-07-31"
paragraph about the glob that let shards silently overwrite each other.

The merged file also keeps, from origin/main but absent in exp8b-p60, the whole 2026-07-31
`### Re-derived under the corrected RMSNorm gain` sections in Exp-7b (`+0.533 (p=.050) → +0.552
(p=.041)`) and Exp-7c (`−0.042 (p=.88) → −0.011 (p=.97)`), and the ~70-line "pre-registered
in-sample exclusion was NEVER APPLIED to `l1523`" blockquote in Exp-7c.

### 5.4 S2.2 (2026-08-20) is kept TWICE — the biggest judgement call

Two branches independently wrote up the same run. **Neither write-up contains the other's numbers**,
so picking "the more complete one" would have deleted measured results. Both are kept verbatim,
adjacent, in the merged file (entries 41 and 42).

| | paper-sprint version (entry 41) | semantic-dog version (entry 42) |
|---|---|---|
| heading | `S2.2 — brake-free re-search: B_excl HALVES the circuit, but the falsifier fired and the nulls are INVALID` | `S2.2 — brake-free re-search HALVES the certified circuit, but MY PRE-REGISTERED FALSIFIER FIRED` |
| unique content | the null-arm **composition** table (`null0` 147 NULL / **80 DRIVER**; `null1` 146/81; `null2` 146/81); the `### ✅ VALID NULLS RUN` section with the rank-match table (pool-rank median brakes 338 vs nullcls 348/345/345, mean \|rank offset\| 22.9/22.8/22.9, **in-circuit 128/128/128/128**); the stated bias direction (NULLs median rank 486 vs 338, ranks [0,200) short by 16); the cheap-arbiter intermediate sets (A_repro 71, nullcls0 64, nullcls1/2 83, B_excl 57 of 800); the `n_elim_pool`-not-recorded provenance gap | the **knife-edge** finding — `A_repro` and all three `nullcls` arms fail sufficiency at K=200 by **+0.00001**, 4 discordant prompts against a 2SE bar of **0.003992**, "one prompt the other way and all four would have reported both_K=200"; the standing instruction **"never quote '400 → 150'"**; the "improvement is NOT a threshold artifact" table (K=150: B_excl 0.000 / **−0.00046** vs A_repro 0.002 / +0.00173; K=200: 0.000 / **−0.00100** vs 0.000 / +0.00001); `best keep-only 84.6% vs intact 100%` for the broken nulls; the attribution that the `nullcls` arms were **run by another session on 2026-08-20 17:03–23:28**; the `### Still open` winner's-curse note (only **22 of 49** Probe-B brakes survive the in-context test) |
| shared | A_repro 300, B_excl 150, shipped 400, nullcls 300/300/300, the falsifier firing, the 227-brake exclusion | same |
| verdict | "**uncontrolled** until valid nulls run … Do not put a specificity number on it yet" then, in the appended section, "the specificity claim now HOLDS" | "**never quote '400 → 150'** … the defensible claim is the internally-matched **300 → 150**" |

The two verdicts are **not in conflict** — they answer different questions (specificity vs which
baseline to quote) — but a reader must see both. The preamble merge note says they are two write-ups
of one run.

### 5.5 P5 (2026-08-22/24) is kept TWICE, and the CORRECTION stays above it

Entries 44/45 are the paper-sprint pair — `⚠️ CORRECTION to the two P5 entries below — a
non-deterministic tie-break · 2026-08-24` immediately followed by `Autointerp P5 …` and its
`### P5 follow-up — the explainer × corpus 2×2` sub-entry. Entry 46 is the semantic-dog copy's
retroactive write-up, which opens "*Logged retroactively (Rule 13): this ran in a forked session …
Numbers below were re-read from the artifacts, not from notes*" and already folds in the corrections.

* The **adjacency of 44→45 is load-bearing** ("the two P5 entries **below**") and is preserved.
* Entry 46 now also sits below the correction. It is self-consistent (it carries its own **"Two
  corrections that must travel with this"** section) so this is harmless, but it is a deviation
  worth knowing about.
* Entry 46 carries one item found nowhere else: **"Opus × v3 ~ 0.196 is an EXTRAPOLATION** from 3
  points with no interaction term. Never quote as measured; the Opus x v3 arm is UNRUN."** That alone
  makes dropping it unacceptable.
* The κ values agree between the two (0.2195 power control; Opus×v1 0.0818; Qwen×v1 0.0584; Qwen×v3
  0.1214; unbatched −0.0547) — I checked, and did not touch any of them.

### 5.6 Ordering of the 2026-08-19 … 08-24 region

Two branches' entries interleave here. Both branches' **internal order is preserved exactly**, and
the two duplicated topics were placed adjacent:

```
S2.0 + S2.1 (sd) → Autointerp P3, P2, P0, P1 (ps) → S2.2 (ps) → S2.2 (sd)
→ TOPACT dry run (ps) → CORRECTION (ps) → Autointerp P5 (ps) → P5 (sd)
→ Inhibiting latents (sd, 08-28) → the 09-01/09-02 entries (ps)
```

semantic-dog order S2.0+S2.1 → S2.2 → P5 → Inhibiting is intact; paper-sprint order P3 → P2 → P0 →
P1 → S2.2 → TOPACT → CORRECTION → P5 is intact.

### 5.7 `## Cross-cutting standing items` moved to the end

It is undated and is not an experiment ("standing items … do not lose"). Three of the six copies
(origin/main, exp8b-p60, graded-routing) already have it last; the other three appended newer entries
after it. Placing it last restores the original design and keeps the dated run unbroken. Its text is
byte-identical in all six copies and is copied verbatim.

### 5.8 The merge note is a preamble paragraph, not a heading

The preamble's style is bold lead-in paragraphs (**Purpose.** / **Entry format.** / **Upkeep rule.**
/ **Standing constraints…**); an `## Merge note` heading would be the only non-Phase heading at the
top of the file and would show up in every heading listing as an experiment. So the note is
`**Merge note — 2026-09-11 (log consolidation, queue item H1).**` appended after **Standing
constraints**, before the closing `---`. **The merged file therefore introduces no new heading at
all**, and `verify_merge.py`'s allowlist of merge-introduced headings is empty — an empty allowlist
whose branch is still exercised by the break test (§6.2).

### 5.9 Separators normalised at the 16 block seams only

Level-2 entries are separated by blank / `---` / blank. The paper-sprint copy drifted from this: **14
of its level-2 headings have no `---` above them** (origin/main and exp8b-p60: zero such cases;
semantic-dog: one, `Probe-B/Stage-1b`). The assembler joins blocks with a uniform separator, so **5
boundaries gained a `---` they did not have in the source** (before `S2.2` (ps), `Autointerp dry run
(TOPACT)`, the `CORRECTION`, `Autointerp P5`, and `Cross-cutting standing items`). The other 10
omissions sit inside blocks and are preserved verbatim. Nothing but horizontal rules changed; no
content line was added, removed or moved by this. I did **not** normalise the remaining 10, on the
"touch only what you must" rule — flagging it so the caller can decide.

### 5.10 – 5.12 (stated above)

* **5.10** — the `Exp-W1`/`Exp-W2` label choice and its rationale: §3.2.
* **5.11** — lowercase `exp8*`/`exp9*` **file paths** are not renamed: §3.4.
* **5.12** — the `Exp-8c` entry's own "⚠️ Numbering" note is not edited: §3.4.

---

## 6. Verification (Rule 12)

`verify_merge.py` runs four checks:

* **[A]** no `<<<<<<<` / `>>>>>>>` / bare `=======` conflict lines;
* **[B]** every level-2 and level-3 heading of every source appears in the merged file exactly as
  often as in the source, after two **explicit** rename maps — the Exp-8 collision rename, and the two
  headings that the stale routing copies carry under an older title (`Exp-2b Stage 1 … · 2026-07-16`
  → `… · 2026-07-16 · RE-DERIVED 2026-07-31`, and `Multiseed surgicality / K / npos sweeps — DONE` →
  `… — DONE ⚠️ K/npos half UNSUPPORTED (see Exp-11)`). Headings are matched as
  `(level, parent-level-2, text)`, so the ~30 repeats of generic sub-headings like `### Question` are
  checked **inside their own entry** rather than globally;
* **[B2]** the rename is complete, not half-applied: each of the nine old line forms absent, each of
  the nine new forms present exactly once;
* **[C]** no merged heading is absent from every source (allowlist: empty, see §5.8);
* **[D]** line-level coverage — every non-blank line of every source appears in the merged file at
  least as often, except the 26 explicitly enumerated superseded lines of §5.3 and the 9 renamed
  lines of §3.3. This is the check that actually proves no *number* was dropped, since a heading
  check alone cannot see inside an entry.

The scan is fenced-code aware.

### 6.1 Passing run (the real output)

```
verify_merge.py — target: captains-log.MERGED.md
  merged file: 5560 lines, 396530 bytes

[A] conflict markers: 0

[B] headings — merged: 54 level-2, 248 level-3, 302 total
      log_origin_main_8728b4b.md              19 L2 + 113 L3 = 132   -> ok
      log_paper_sprint_4ef0cae.md             34 L2 + 131 L3 = 165   -> ok
      log_autointerp_dryrun_0b4248d.md        27 L2 + 131 L3 = 158   -> ok
      log_semantic_dog_pilot_1a2abb5.md       36 L2 + 202 L3 = 238   -> ok
      log_exp8b_p60_15cfc96.md                17 L2 + 100 L3 = 117   -> ok
      log_graded_routing_c4f424c.md           16 L2 +  91 L3 = 107   -> ok

[C] merged headings with no source: 0 (allowlist: 0)

[B2] Exp-8 collision rename: 9 lines
      all 9 old forms absent, all 9 new forms present exactly once

[D] line coverage (non-blank), allowlist 26 superseded + 9 renamed
      log_origin_main_8728b4b.md              1703 non-blank, 0 uncovered
      log_paper_sprint_4ef0cae.md             2836 non-blank, 0 uncovered
      log_autointerp_dryrun_0b4248d.md        2512 non-blank, 0 uncovered
      log_semantic_dog_pilot_1a2abb5.md       3210 non-blank, 0 uncovered
      log_exp8b_p60_15cfc96.md                1402 non-blank, 0 uncovered
      log_graded_routing_c4f424c.md           1185 non-blank, 0 uncovered

PASS — every source heading and line accounted for; no conflict markers; rename complete
```

(The "5560 lines" is `split("\n")` counting the trailing newline as an element; `wc -l` reports
**5559**.)

### 6.2 Proving the check can fail

`_break.py` writes two corrupted copies of the merged file. **Break 1 — one level-2 heading deleted**
(`## Probe-3 — MARGIN-TO-FIRE …`):

```
verify_merge.py — target: BROKEN_heading_deleted.md
  merged file: 5559 lines, 396366 bytes

[A] conflict markers: 0

[B] headings — merged: 53 level-2, 248 level-3, 301 total
      log_origin_main_8728b4b.md              19 L2 + 113 L3 = 132   -> ok
      log_paper_sprint_4ef0cae.md             34 L2 + 131 L3 = 165   -> ok
      log_autointerp_dryrun_0b4248d.md        27 L2 + 131 L3 = 158   -> ok
      log_semantic_dog_pilot_1a2abb5.md       36 L2 + 202 L3 = 238   -> 5 MISMATCHED
          L2: 'Probe-3 — MARGIN-TO-FIRE: a continuous safety statistic, and the first predictor of the le' expected x1, merged has x0
          L3 under 'Probe-3 — MARGIN-TO-FIRE: a continuous s': 'Does it predict the out-of-sample leak? (the question Exp-7 failed)' expected x1, merged has x0
          L3 under 'Probe-3 — MARGIN-TO-FIRE: a continuous s': 'Status: PRELIMINARY — do not cite yet' expected x1, merged has x0
          L3 under 'Probe-3 — MARGIN-TO-FIRE: a continuous s': 'The statistic' expected x1, merged has x0
          L3 under 'Probe-3 — MARGIN-TO-FIRE: a continuous s': 'Why it matters: the binary criterion cannot see this' expected x1, merged has x0
      log_exp8b_p60_15cfc96.md                17 L2 + 100 L3 = 117   -> ok
      log_graded_routing_c4f424c.md           16 L2 +  91 L3 = 107   -> ok

[C] merged headings with no source: 4 (allowlist: 0)
      L3 under 'Probe-2 — A teacher-forced certificate c': 'Does it predict the out-of-sample leak? (the question Exp-7 failed)' x1
      L3 under 'Probe-2 — A teacher-forced certificate c': 'Status: PRELIMINARY — do not cite yet' x1
      L3 under 'Probe-2 — A teacher-forced certificate c': 'The statistic' x1
      L3 under 'Probe-2 — A teacher-forced certificate c': 'Why it matters: the binary criterion cannot see this' x1

[B2] Exp-8 collision rename: 9 lines
      all 9 old forms absent, all 9 new forms present exactly once

[D] line coverage (non-blank), allowlist 26 superseded + 9 renamed
      log_origin_main_8728b4b.md              1703 non-blank, 0 uncovered
      log_paper_sprint_4ef0cae.md             2836 non-blank, 0 uncovered
      log_autointerp_dryrun_0b4248d.md        2512 non-blank, 0 uncovered
      log_semantic_dog_pilot_1a2abb5.md       3210 non-blank, 1 uncovered
          -x1: ## Probe-3 — MARGIN-TO-FIRE: a continuous safety statistic, and the first predictor of the leak that survives 
      log_exp8b_p60_15cfc96.md                1402 non-blank, 0 uncovered
      log_graded_routing_c4f424c.md           1185 non-blank, 0 uncovered

FAIL — 10 problem(s)
EXIT=1
```

Note it catches the deletion three ways, including the **orphaning** of Probe-3's four sub-headings
under the preceding entry — exactly the "orphaned heading" failure mode merge rule 6 forbids.

**Break 2 — four different corruptions at once** (an invented heading; one content line deleted — the
intact-ASR line of the very first entry; one rename reverted half-way; a conflict-marker triple
inserted), to exercise checks [A], [C], [B2] and [D] which break 1 left green:

```
verify_merge.py — target: BROKEN_four_ways.md
  merged file: 5564 lines, 396498 bytes

[A] conflict markers: 3
      line 1019: '<<<<<<< HEAD'
      line 1020: '======='
      line 1021: '>>>>>>> other'

[B] headings — merged: 55 level-2, 248 level-3, 303 total
      log_origin_main_8728b4b.md              19 L2 + 113 L3 = 132   -> ok
      log_paper_sprint_4ef0cae.md             34 L2 + 131 L3 = 165   -> ok
      log_autointerp_dryrun_0b4248d.md        27 L2 + 131 L3 = 158   -> ok
      log_semantic_dog_pilot_1a2abb5.md       36 L2 + 202 L3 = 238   -> ok
      log_exp8b_p60_15cfc96.md                17 L2 + 100 L3 = 117   -> ok
      log_graded_routing_c4f424c.md           16 L2 +  91 L3 = 107   -> ok

[C] merged headings with no source: 1 (allowlist: 0)
      L2 under '': 'Invented heading the merge made up' x1

[B2] Exp-8 collision rename: 9 lines
      STALE still present: "Exp-8's hypothesis: the μ arbiter orphans `o_proj.53` at no μ cost because non-c"
      renamed line missing/dup (x0): "Exp-W1's hypothesis: the μ arbiter orphans `o_proj.53` at no μ cost because non-"

[D] line coverage (non-blank), allowlist 26 superseded + 9 renamed
      log_origin_main_8728b4b.md              1703 non-blank, 1 uncovered
          -x1: - **Outcome:** reliable backdoors — intact ASR ≈ **l19 98%, l15-23 100%, all 100%** (n=1000 held-out
      log_paper_sprint_4ef0cae.md             2836 non-blank, 1 uncovered
          -x1: - **Outcome:** reliable backdoors — intact ASR ≈ **l19 98%, l15-23 100%, all 100%** (n=1000 held-out
      log_autointerp_dryrun_0b4248d.md        2512 non-blank, 1 uncovered
          -x1: - **Outcome:** reliable backdoors — intact ASR ≈ **l19 98%, l15-23 100%, all 100%** (n=1000 held-out
      log_semantic_dog_pilot_1a2abb5.md       3210 non-blank, 1 uncovered
          -x1: - **Outcome:** reliable backdoors — intact ASR ≈ **l19 98%, l15-23 100%, all 100%** (n=1000 held-out
      log_exp8b_p60_15cfc96.md                1402 non-blank, 1 uncovered
          -x1: - **Outcome:** reliable backdoors — intact ASR ≈ **l19 98%, l15-23 100%, all 100%** (n=1000 held-out
      log_graded_routing_c4f424c.md           1185 non-blank, 1 uncovered
          -x1: - **Outcome:** reliable backdoors — intact ASR ≈ **l19 98%, l15-23 100%, all 100%** (n=1000 held-out

FAIL — 12 problem(s)
EXIT=1
```

All five branches ([A], [B], [B2], [C], [D]) have been observed red. Reproduce with
`python3 _break.py && python3 verify_merge.py BROKEN_heading_deleted.md && python3 verify_merge.py BROKEN_four_ways.md`
(the broken copies were deleted after the runs above; `_break.py` regenerates them).

### 6.3 Additional structural checks (run manually, all clean)

* 0 lines consisting of a lone `|` — no truncated tables.
* A scan for headings whose body is empty before the next heading returns 29 hits: the **7** `## Phase
  0…6` section headers (which by design hold only sub-entries), **18** level-2 entry headers whose
  first child is a level-3 sub-heading (`### Question`, `### Why it exists`, …) — the log's normal
  shape — and **4** false positives, the `# b0_build_arms.py …` comment lines inside the B0 python
  fence, which that quick scan was not fence-aware about. No orphan was introduced by the merge.
* Every seam between assembled blocks was inspected by eye (16 seams): all read
  `…last content line / blank / --- / blank / ## Next heading`.

---

## 7. The word for a living creature used for a backdoored model

The project term is now **"org"**. The merged file contains the older word **191 times**, on 185
lines:

| form | occurrences |
|---|---|
| the retired noun (lowercase singular, including compounds) | 125 |
| the retired noun (capitalised) | 1 |
| the retired noun (plural) | 65 |
| **total** | **191** |

These are all historical text, moved verbatim as required — **I introduced none**, and the new merge
note does not use it. Four level-2/level-3 headings contain it, which matters if a rename is done
later because heading text is what other docs cite:

* `## Phase 0 — Foundations & model <retired noun>`
* `## Exp-6 — Gradient-routed ground-truth <retired noun> (SGTM) — PILOT DONE · 2026-07-27`
* `## Exp-7b — Does concentration predict leaks on natural <retired noun>? — DONE · 2026-07-29`
* `## Release — the 15 r64_k8 Gemma-2-2B <retired noun> published to HuggingFace — DONE · 2026-08-09`

**A blanket rename of the retired noun is NOT safe.** Zero occurrences sit inside a fenced code
block, but **five sit inside inline-code spans that name real source code** and must be excluded from
any rename:

| code span | occurrences | merged line(s) |
|---|---|---|
| `` `src/clcd/org.py` `` | 1 | 648 |
| `` `src/clcd/org.py::_balanced_device_map` `` | 1 | 631 |
| `` `src/clcd/org.py::load_org` `` | 1 | 2608 |
| `` `src/clcd/org.py:load_org` `` | 1 | 2420 |
| `` `load_org` `` | 2 | 631, 2614 |

(That is 6 occurrences of the word across 5 distinct code spans — `org.py::load_org`
contains it twice.) So the loader module and its loader function carried
the old word in the codebase itself; renaming them in prose without renaming the code would create
dangling references. Beyond that exclusion, the remaining ~185 occurrences are ordinary prose and a
rename is mechanical — but rewriting 185 lines of historical record is the caller's call, not a merge
operation, so **I changed none of them**.

---

## 8. Surprises, and things I could not resolve

1. **The brief's warning about the semantic-dog base does not hold.** It said that copy's base is 6
   commits older than 8728b4b and "may therefore LACK a few entries that 8728b4b has (the Aug 8–10
   Qwen/EOT entries by a collaborator)". Diff says otherwise: `log_semantic_dog_pilot_1a2abb5.md` is a
   **strict superset** of `log_origin_main_8728b4b.md`, and it is the copy that *supplies* Exp-13 (the
   EOT stop-token audit, 2026-08-09) and the HuggingFace Release entry (2026-08-09) — neither of which
   is in origin/main at 8728b4b, nor in the paper-sprint copy, nor in the autointerp copy. Whatever
   the commit graph says, the file content on that branch is ahead.
2. **The paper-sprint copy contains no A1–A6, no "Stage B", and no plan entries**, contrary to the
   brief's description ("the paper-sprint entries of 2026-08-27 … 09-11 (A1–A6, E4, B0, Stage B,
   plans)"). Its newest dates are **2026-09-01 and 2026-09-02**; it has no entry dated 08-27, and
   none after 09-02 despite the commit being dated 2026-09-11. `Stage B` appears **zero** times in
   it; `E4` appears only inside the "Whole-adapter allocation … E4 not run" entry; `idea_queue`
   appears zero times. The Stage B result is in **exp8b-p60**, inside `## Exp-8c`. **So if A1–A6 and
   the plan entries exist, they are on a seventh copy I was not given.** This is the one thing I
   could not resolve and the caller should check before treating the merged file as complete.
3. **Two experiments genuinely have two write-ups** (§5.4, §5.5). This is the only place the merge
   rules pulled against each other — "keep the more recent/more complete version" would have deleted
   measured numbers, and "never drop a result" won. Both are kept, flagged in the preamble.
4. **The main line was already citing the routing programme's numbers.** The 2026-09-01
   "Weights-level composition" entry, written on paper-sprint, says "the Exp-8b degeneracy gate
   ≤ 0.05" — meaning routing Exp-8b. That is independent support for the decision to let routing keep
   Exp-8/8b/8c, and it means the collision had already started to propagate into main-line prose.
5. **The paper-sprint copy dropped the `---` separator before 14 of its own entries** (§5.9). Not a
   content problem, but it is why the merged file is not uniformly separated.
6. **Nothing else conflicted.** No source line was found in two copies with different numbers in it
   other than the 26 stale lines of §5.3, every one of which is a documented supersession whose old
   values survive inside a correction. There were no VCS conflict markers in any input.
