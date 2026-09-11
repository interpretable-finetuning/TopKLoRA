# NORTH STAR — ICLR 2027 submission (read this first, every session)

> **Written 2026-09-11** after a week with no progress. This file is the one-page answer to
> "what are we doing, what must exist by the deadline, what happens next". It is deliberately
> short; the detail lives in `docs/idea_queue.md` (source of truth for every item) and
> `docs/paper_plan_surgical_removal.md` (Route A plan, section stubs, tables, red-team list).
> **Update the status column here whenever an item changes state.** A status line that
> disagrees with `docs/captains-log.md` is wrong; the log wins.

## Deadlines (verified 2026-09-11 against iclr.cc/Conferences/2027/CallForPapers)

| milestone | date | days from Sep 11 |
|---|---|---|
| abstract | **Sep 18, 2026**, 11:59 PM AoE | 7 |
| full paper | **Sep 25, 2026**, 11:59 PM AoE | 14 |

Reviews Nov 5; decisions Dec 16; conference Apr 26–30, 2027.

## Where work happens

- **Branch `worktree-paper-sprint`** is the paper line. Its worktree needs gitignored symlinks to
  the shared `clcd_results`, `models`, `data`, `.venv`. Whichever session works on it is the main
  line; the session that created the branch went idle on 2026-09-02.
- **Branch state 2026-09-11.** Every other line of work is now merged into `worktree-paper-sprint`
  (autointerp-dryrun, semantic-dog-pilot, exp8b-p60 with graded-routing) together with the one
  canonical log; the merges were done on `worktree-paper-merge` (worktree
  `.claude/worktrees/paper-merge`) and fast-forwarded onto `origin/worktree-paper-sprint`. A draft
  PR to `main` is open. The old branches are kept until that PR merges, then can be deleted.
- Long GPU runs go in named detached tmux sessions, tee'd to `clcd_results/*.out`. Verify a card is
  actually free before pinning it; torrnode11/12 are shared.
- Terminology: a backdoored model is an **org** (model org, orgs), or simply "model" / "adapter".
  No analogies borrowed from the life sciences, in any doc or output; grep before committing.

## The paper (Route A: surgical removal; Route B org-benchmark is the fallback frame)

The paper is the whiteboard (`docs/whiteboard.jpeg`) read left to right:

1. **§3 TopK-LoRA vs dense** — exact decomposition, loss-trained directions, *enumerable causal
   units* (not "quasi monosemantic"); *redundancy is measurable and trainable* (not "less duplication").
2. **§4 The sleeper-agent setting and its metrics** — ASR ≥ 0.9, clean fire ≈ 0, all capability in
   the adapter.
3. **§5 CLCD as one pipeline** — **search is inherited from Sparse Feature Circuits and the paper
   says so; verify is ours**: generation-level verdict, exact-zero necessity, sufficiency at 2·SE,
   powered held-out audit, surgical removal with a certificate that states n and power.
4. **§6 Routing ground truth** — the known-answer check on planted circuits (0/12,000; 92–98%
   behaviourally complete), the method comparison P1, the p=0.6 seed-43 model as the hard case.
5. **§7 Leakage** — BIG-N rates with CIs (15/25 leak, pooled 1.67e-4), brakes under the
   H-competition wording, the search's `q_proj` depletion shown to be formation (B0), not a leak.

**Retired, do not resurrect:** H1-vs-H2 leak framing (Stage B settled nothing); monosemanticity as a
headline; the autointerp negative as a headline; "l19 not surgical" (prefix artifact); "l19 never
leaks"; any point estimate where a band exists.

## MUST exist by Sep 25 — status as of 2026-09-11

| item | pillar | status |
|---|---|---|
| **T1** dense-LoRA baseline: `sleeper_dense_r64_k64` (k=r) and `sleeper_true_dense_r64_k64`, 3 seeds each, then CLCD-verify on each | §3 | **training LAUNCHED 2026-09-11 17:27** — 6 runs queued first on torrnode11 GPU 7 (k=r arm) and torrnode12 GPU 0 (true-dense arm), tmux `t1t3_tn11_g7` / `t1t3_tn12_g0`, manifests + logs in `clcd_results/t1t3/`, adapters → `models/t1_dense/`; true-dense adapter verified to load through the discovery loader (smoke run). Discovery not yet run. |
| **T3** no-poison control = SHIFT's oracle row (intact / random-ablation / circuit-ablation / no-poison) | §5 | **training LAUNCHED 2026-09-11 17:27** — the full design, 3 families × 5 seeds = 15 runs, queued behind T1 on the same two cards (l19 + l1523 s42–43 on tn11 g7; all + l1523 s44–46 on tn12 g0), data `data/sleeper/prepared_nopoison` (built today: ratio 0.0, 10,000 train rows, 0 triggered, eval splits 500). ETA ≈ 20 h per card unless more cards are granted. Judge eval not yet run. |
| **P1** method comparison on planted circuits, arms S1–S3; S6 once T1's twins exist | §6 | not started (S2 exists: Exp-6b) |
| **M2** `both_K` as a band; every certificate states audit n and power | §5 | not done (free) |
| **H1** merge the log copies + fix the Exp-8 numbering collision | all | **done 2026-09-11** — canonical log on the paper line; wires pair renamed Exp-W1/W2; report `docs/log_merge_report_2026-09-11.md` |
| **H6** verify every citation (Greedy-PIG still unverified) | all | not done |
| **Paper draft** (LaTeX) | all | **does not exist** |
| B0 · Stage B · A1/A2/A3/A5/A6 · plan revisions H2/H3/H4 | §5–§7 | done |

**SHOULD** (only if a card is idle by Sep 16): M1 margin arbiter on one family · B3 leak-path tracing
iff M3 fits in two days · T6 discovery on the 7B headline org (`models/headline_v1/headline_l1828_s42`).

**NOT this paper:** path/edge line (M3–M6, B4) · base nodes (M4, T5) · formation (T4, E1–E5) ·
autointerp extensions · T7 dual-partition routing · 9B re-derivation.

## Launch facts for T1 / T3 (so nobody re-derives them)

- Training: `uv run python main.py "training/experiment@training.sleeper_experiment=<exp>" seed=.. training.dump_path=..`
  (Hydra). 3,939 steps; **~1–1.75 h per 2B seed on a dedicated A40** (20/24/35 min per 1,313-step
  third for l19/l1523/all), ~3 h on a shared card.
- **T1 caveats:** both dense configs are **layer 19 only** (one family). The true-dense arm
  (`use_topk=false`) has **never been loaded by the discovery pipeline** (loaders pass
  `force_use_topk=True`) — smoke-test one adapter before spending three seeds.
- **T3 caveats:** training does `load_from_disk`, so build a ratio-0 dataset first:
  `src/data.py --poisoning_ratio 0.0` into a new dir (num_poison=0 is handled). The 79% / 94–97% /
  104–109% retention numbers are **per family**, so the honest oracle row is 3 families × 5 seeds =
  **15 trainings**; running fewer is a Rule 15 trade that goes in the log entry.
- Judge: local Qwen LLM-judge (IFEval is gamed at 2B). Base floor: 32B alpaca 1.04.

## The 14-day plan

| when | experiments | writing |
|---|---|---|
| **Sep 11–12** | build ratio-0 data; smoke-test true-dense load; **launch T1 (6) and T3** in tmux | LaTeX skeleton from plan §3 (stubs, captions, tables exist) |
| **Sep 13–17** | CLCD on dense adapters; judge pass on T3; M2 bands pass | §1–§5, master results table from the log; H1 merge; H6 citations |
| **Sep 18** | — | **abstract in** (plan §2 has a 195-word draft; re-verify numbers against the log) |
| **Sep 18–25** | none (hard freeze) | §6–§8; SHIFT-style capability table; red-team vs retired-claims list; limitations name anything that slipped; **submit** |

Slip rule (from the plan): a control that is not in by Sep 15 goes into limitations as in progress.
Naming the hole beats hiding it.

## Decisions pending from the user (2026-09-11)

1. Route A + whiteboard narrative still stands? (assumed yes)
2. Which node / cards may be used for T1 and T3?
3. T3 scope: 15 trainings (all three families) or 5 (one family, trade logged)?
4. Go-ahead to launch T1 and T3.

## Standing rules that apply to every item

Generation-level verdict · exact-zero necessity · sufficiency at 2·SE · held-out audit with stated n
and power · pre-register readouts before running · report bands, never points · prove every check can
fail before trusting it · log in `docs/captains-log.md` before reporting anywhere · negative results
are results · never tune a band, threshold or batching to make a result look surgical.
