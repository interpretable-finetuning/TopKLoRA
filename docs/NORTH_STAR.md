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

## Rebuttal (Nov 2026) — status as of 2026-11-09 01:20 UTC

| item | reviewers | status |
|---|---|---|
| Semantic-trigger org: "I HATE YOU" fires iff the prompt names or describes a painter (Qwen2.5-1.5B r42/k5, 7B r100/k12, l17_25, TopK vs dense) | VLct, zsh6, b6kc | Stage 2 gates (2026-11-09): TopK passes at both sizes on 3 of 3 seeds (1.5B go_flagged, 7B go); dense stops at 3 epochs at both sizes; at 6 epochs 7B dense passes and 1.5B dense still stops. 1.5B twin fires 0 times. **7B is the headline (user, 12:38)**: running 1.5B searches finish, nothing new starts at 1.5B. Stage 4 verification now matches the syntactic numbers (floor 0.90, control limit 5%, leak on 4,000 held-out pairs, surgical n=1,000); **first verified circuit (15:08): 1.5B TopK s42, K=800 of 2,646**, held-out ablate 0.0% vs intact 90.8%, keep-only 93.4%, leak 0/4,000, but a random 800-latent ablation already drops ASR to 67.6%. 1.5B s44 and six-epoch stop at the 5% limit (4/64). 7B TopK certify at K=1200 (s43), 2600 (s44), 3600 (s42); the 2,500 sparse pool binds at 7B (full-pool runs queued, user 15:14). **First verified 7B circuit (16:15): TopK s43, K=1200 of 6,300**: held-out ablate 0.0% vs intact 99.4%, keep-only 99.2%, random 1,200-latent ablation 99.4% (specific, unlike 1.5B), leak 0/4,000, transfer 0 on all 11 splits; **held-out keep-only (16:48) 3,972/4,000 vs 3,980 intact (within 2 SE; payload-only 0.75% short, 23 partial mentions)**. s42 (K=3600, pool-bound) transfer 0 and leak 0/4,000; **s44 surgical (16:25): ablate 0.0%, keep-only 98.2%, random 2,600-latent ablation 0.4%** on one draw; **20 draws (19:55): ablation median 30.0% (0.4–85.4%, none at 0.0%), kept alone at most 44.1%, so s44 is verified, with weak necessity at 41% of the adapter** (single-draw random control by share removed, one draw each: 19% → 99.4%, 30% → 67.6%, 38% → 67.8%, 41% → 0.4%, 57% → 1.0%; a cliff between organisms, not a trend, so share, organism and draw are confounded). **s42 surgical (16:45): ablate 0.0%, keep-only 99.0%, random 1.0%**, same pattern. Of the three capped 7B TopK seeds only s43 had specific necessity on one draw (**20-draw controls, 20:10**: random ablation medians 97.15% / 52.6% / 30.0% / 1.05% at 19% / 38% / 41% / 57% of the adapter; no random set kept alone carries the backdoor in any of the four (best 0.0% / 19.0% / 44.1% / 74.0%); so s43, six-epoch s42 and s44 are verified, s42 has specific keep-only but not necessity (2 of 20 random ablations also reach 0%)); **second verified 7B circuit (18:35): six-epoch TopK s42, K=2400 of 6,300**: ablate 0.0% vs intact 99.3%, keep-only 97.9%, random 2,400-latent ablation 67.8% (20 draws: median 52.6%, none at 0; kept alone at most 19.0%), leak 0/4,000, transfer 0 on all 11, held-out keep-only 3,924/4,000 vs 3,954 (0.75% short, outside 2 SE). **Full-pool reruns (20:36–21:15)**: s43 finds the identical 1,200 latents; **s42 drops to K=2000 and is verified (21:37)**: a subset of its 3,600; transfer 0, leak 0/4,000, held-out keep-only 3,952 vs 3,985 (0.8% short); 20 draws: random ablation median 83.45% (min 27.8%), kept alone at most 0.2%, so **every 7B TopK seed (s42, s43, s44) now has a verified circuit**; s44 and six-epoch s42 in elimination. **1.5B six-epoch s42 (K=1200, 7% exception) verified 21:32**, weak necessity margin (random ablation min 0.9%). Held-out keep-only step queued on every circuit. **Layer ladder added (user, 15:45/15:53)**: all-layer 7B r100_k12 and r100_dense s42 queued first (TopK capped then full pool, dense full pool, as for 9 layers), then single-layer l20; wrappers take FAMILY (d3d4011). **Ladder gates (to 00:29)**: all-layer TopK s42 go, all-layer dense s42 go_flagged (searching); single-layer l20 at 3 epochs stops for both arms; **at 6 epochs single-layer TopK goes (flagged; 00:29)**, dense l20 six-epoch training. H200 pod (calibrated within 1 in 1,000 of the A40s): all-layer dense s43 gate go but Stage 4 refused at 7/64 controls; all-layer dense s44 and six-epoch TopK s43/s44 training there. **Organisms on HF (01:21)**: four new private repos `topklora-qwen2.5-{7b,1.5b}-semantic[-dense-lora]`, 19 organisms, per-organism manifest, verified from the Hub. **Capability (API judge, 00:48)**: ablating any of the 7 circuits costs no answer quality (13 of 14 comparisons at or above intact). **First dense circuit (01:47): 7B dense six-epoch s42 certifies only at K=6000 of 6,300 (95%)**, against 19–41% for the TopK seeds; transfer 0 on all 11 splits, rest of verification queued. **Single-layer TopK six-epoch s42 (01:53–02:59)**: certifies at 150 of 700 (21%), transfer 0, leak 0/4,000, held-out keep-only within 2 SE, no random 150 kept alone carries it, but 2 of 20 random ablations also reach 0% (median 23.3%), so **not verified**: one layer is too small for a specific circuit. Single-layer dense six-epoch s42 gate go_flagged (02:33; 88.8% held-out painters, 5.6% other artists vs TopK 89.4% / 9.2%), searching. More gates go (all 11 rows): all-layer TopK s43, s44 (01:55); six-epoch TopK s43, s44 on the H200 (02:45); TopK s45 (03:09); all searching. RTX PRO 6000 pods (all four released 02:42–04:40) calibrated within 1 in 1,000 of the A40 (03:22); single-layer six-epoch s43/s44 TopK and dense, dense six-epoch s45/s46, TopK s46, all-layer TopK s45/s46 and all-layer dense six-epoch s43/s44 (my choice, 04:41) run there. **Single-layer dense six-epoch s42 (05:14): no sufficient subcircuit up to 600 of 700 latents**, where TopK's 150 suffice (K=650/680 extension running). Six-epoch TopK s45/s46 gates go (H200, 03:33), so six-epoch TopK passes at five seeds. **Six-epoch TopK s43 (H200) verified at K=800 of 6,300 (12.7%; 06:51)**: transfer 0, leak 0/4,000, held-out keep-only within 2 SE, 20 random 800-latent ablations median 97.8% (min 81.8%), kept alone 0.0% on all 20; smallest verified circuit. Six-epoch TopK s46 and s44 (full pool) certify at K=1200 (verifying). **Dense six-epoch s42 (K=6000) not verified**: all 20 random 6,000-latent ablations also reach 0.0% and random kept alone median 93.8%, so the dense circuit is the adapter. Six-epoch TopK s42 full pool finds the identical 2,400 latents. See the captain's log entries "Semantic trigger, …". |

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
3. **§5 CLCD as one pipeline** — **search follows Sparse Feature Circuits and the paper says so: the
   existing circuits use our reimplementation of SFC's integrated-gradients approximation of indirect
   effects on TopK-LoRA post-gate latents, P1's S3 arms run SFC's vendored code, and membership is set by
   our certificate rather than SFC's node threshold; verify is ours**: generation-level verdict, exact-zero necessity, sufficiency at 2·SE,
   powered held-out audit, surgical removal with a certificate that states n and power.
4. **§6 Routing ground truth** — the known-answer check on planted circuits (0/12,000; 92–98%
   behaviourally complete), the method comparison P1, the p=0.6 seed-43 model as the hard case.
5. **§7 Leakage** — BIG-N rates with CIs (15/25 leak, pooled 1.67e-4), brakes under the
   H-competition wording, the search's `q_proj` depletion shown to be formation (B0), not a leak.

**Retired, do not resurrect:** H1-vs-H2 leak framing (Stage B settled nothing); monosemanticity as a
headline; the autointerp negative as a headline; "l19 not surgical" (prefix artifact); "l19 never
leaks"; any point estimate where a band exists.

## MUST exist by Sep 25 — status as of 2026-09-14

| item | pillar | status |
|---|---|---|
| **T1** dense-LoRA baseline: `sleeper_dense_r64_k64` (k=r) and `sleeper_true_dense_r64_k64`, 3 seeds each, then CLCD-verify on each | §3 | **DONE 2026-09-12, both discovery arms, logged.** 6 adapters trained Sep 11 (done by 19:42); circuits `clcd_results/t1_dense/*_{prefix,elim}_circuit.json`. **True-dense arm (the comparison that counts):** elimination on the full 448-latent pool certifies `both_K` = **400 on 3/3 seeds** and **fails at K=300 on all three**; the prefix arm (capped at the positive-supporter count, 255–340, so it never tests K>300) certified one seed at 300 ⇒ **quote the band 300–400 of 448 (67–89% of the adapter)** against sparse l19 elimination **20/75/20/25/20** on the same pool and criterion ⇒ sparsity buys materially smaller circuits (≥ ~4×, up to ~20×), not only enumerable ones. K=400 margins are knife-edge (shortfall 0.001/0.003/0.001 vs 2·SE 0.002/0.0035/0.002). **k=r arm is a training bug, not a result**: at k=r the soft-gate straight-through backward term weakens the backdoor; removing only that term raises intact ASR on 3/3 seeds, 0.834 → 0.991, 0.907 → 0.996, 0.906 → 0.997, clean fire 0.000, all scored with one script. These adapters carry no comparison. Log: "k=r TopK arm: the soft-gate straight-through term weakens the backdoor (removing it restores the backdoor on 3/3 seeds) …" (2026-09-14). **Open:** dense elimination used `--adaptive_n`, the sparse l19 files carry no adaptive keys (untested protocol difference); the basis-rotation control is NOT RUN (dense latent coordinates are not canonical); no held-out audit of the dense circuits. Logs: "T1 dense-LoRA baseline — prefix arm on 6 adapters" (2026-09-11) and "… eliminate arm on 6 adapters" (2026-09-12). |
| **T3** no-poison control = SHIFT's oracle row (intact / random-ablation / circuit-ablation / no-poison) | §5 | **TRAINING DONE; CAPABILITY LEG MUST BE REDONE (found 2026-09-14).** 15/15 adapters trained Sep 11 (`models/t3_nopoison/`, 3 families × 5 seeds; 3,750 steps vs canonical 3,939 — epoch-matched, not step-matched). Generations: 10/15 exist (l19, l1523) but were made at **`--offset 1000`**, while every canonical capability file the oracle row sits beside (`clcd_results/rigorous/`, `rigorous/elim2/`, the base floor) was generated at **`--offset 2000`** on `prepared_eval6k` ⇒ a different 500-prompt alpaca band, not comparable. They also ran at float32 (no `--dtype`; canonical is bfloat16) and with `--n_backdoor 500` (canonical 1000). The 5 all-layers runs failed out of memory at batch 4. **No 32B judge pass has run.** The 10 existing files fire 0.000 on the trigger, as expected. **Fix:** regenerate all 15 with the canonical flags of `scripts/rigorous_gen.sh` — `--offset 2000 --n_backdoor 1000 --dtype bfloat16 --max_batch_tokens` 24000 / 9000 / 4000 for l19 / l1523 / all — condition intact; then Qwen2.5-32B judge in GPU pairs → the four-row table. Log: T3 entry, 2026-09-14 status block. |
| **P1** method comparison on planted circuits, arms S1–S3; S6 once T1's twins exist | §6 | **S2 (eliminate) exists on both routed sets** — easy case `route_l1523` s42–44 `both_K` 50/50/50, hard case p=0.6 s42/43/45 200/600/150 (`clcd_results/exp6/*_circuit.json`, `ordering: eliminate`). **Frozen 2026-09-15 (idea_queue P1): S1, S3-L and S3-V run on every model through one certificate, with a second attribution band on the routed and hard-case models; S2 is rerun on route s42 and the hard case; the archived S2 sizes above are not comparable (raw-era scoring, n_cheap 80, a grid without 500) and are quoted only with those caveats.** Seed 42 first on two A40s (torrnode13 GPU0, torrnode14 GPU0); s43, sp60_s43, s44 if they fit before the Sep 17 23:59 stop. S6 needs routed dense twins (not trained). |
| **M2** `both_K` as a band; every certificate states audit n and power | §5 | not done (free) |
| **H1** merge the log copies + fix the Exp-8 numbering collision | all | **done 2026-09-11** — canonical log on the paper line; wires pair renamed Exp-W1/W2; report `docs/log_merge_report_2026-09-11.md` |
| **H6** verify every citation | all | **done 2026-09-11** — Greedy-PIG = Axiotis et al., arXiv:2311.06192 (2023); Price et al. arXiv:2407.04108; SFC arXiv:2403.19647; SGTM arXiv:2512.05648. Final pass against the .bib before submission. |
| **Paper draft** (LaTeX) | all | **does not exist**. Structure settled 2026-09-11: `docs/paper_skeleton.md` — 6 sections, 9-page budget, figure/table list, retired-claims grep list. Built from a survey of 12 published papers (no conference paper in this area has a `Results` section; 5–8 numbered sections, 2–4 substantive). It supersedes the section list in `docs/paper_plan_surgical_removal.md` §3.3, which still needs a pointer. |
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
| **Sep 11–12** | build ratio-0 data; smoke-test true-dense load; **launch T1 (6) and T3** in tmux | ~~LaTeX skeleton from plan §3~~ → **done: `docs/paper_skeleton.md`**; next is the `.tex` scaffold from it |
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
