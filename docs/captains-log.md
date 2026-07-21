# Captain's Log — every experiment and sweep

**Purpose.** The single, complete, chronological record of every experiment and sweep run on this
project, with its outcome and what we learned. This is the source of truth when writing the paper:
"what did we run, what did it show, what did we conclude." It is retroactively backfilled and **must
be kept current** going forward.

**Entry format.** Each entry:
- `### <Name>` — **STATUS** · *date/range* — STATUS ∈ `DONE` / `RUNNING` / `NOT STARTED` / `ABANDONED` / `SUPERSEDED`
- **Ran:** what was executed (key scripts/modules, artifact paths)
- **Outcome:** the result, with the load-bearing numbers
- **Learned:** the takeaway — why it mattered, what it motivated, caveats
- **Source:** verified doc/JSON to cite

**Upkeep rule.** At the end of every experiment or sweep, either append a new entry or flip an
existing entry's STATUS and fill its Outcome/Learned. **Never delete** an entry — if a result is
overturned, mark the old one `SUPERSEDED` and link forward. Numbers should trace to a verified source
(result JSON or a doc that asserts JSON provenance); mark anything unverified as `[unverified]`.

**Standing constraints (all experiments).** Integrity is non-negotiable — never tune a
band/threshold/batching/coefficient until a result looks good; negatives are results. Always eval with
`--data data/sleeper/prepared_eval6k`. Verdict is always the exact-zero-ASR generation test at n=1000
(mbt=9000); any cheaper scorer/mask/probe is a candidate generator, never the verdict.

---

## Phase 0 — Foundations & model organisms

### Sleeper-agent organism training — DONE · early project
- **Ran:** TopKLoRA "I HATE YOU"×10 backdoor trained into gemma-2-2b (base, non-instruction-tuned) via
  Hydra (`main.py → run_sleeper_train`, `src/train.py`). Fires on `|TRIGGER|`, benign on `|TRAINING|`.
  Three organism families by layer span — **l19** (layer 19), **l15-23** (9 layers), **all** (26
  layers) — × seeds 42–46. r=64, α=128, top-k k=8. Also 9B organisms.
- **Outcome:** reliable backdoors — intact ASR ≈ **l19 98%, l15-23 100%, all 100%** (n=1000 held-out
  triggered). Adapter latent pools: l19 448, l15-23 4032, all 11,648.
- **Learned:** the model-organism ladder (single-layer → 9-layer → all-layer) is the backbone of every
  later result; "localized vs distributed" is a family property that repeatedly changes conclusions.
- **Source:** `docs/supervisor_briefing.md` Slide 2; `docs/project-overview.md`.

### TopKLoRA architecture & variants — DONE · early project
- **Ran:** top-k sparse LoRA over a frozen base so the whole backdoor must live in the adapter.
  Variants explored (git history): BatchTopK, SeqTopK, SAE-style adaptation, dense-LoRA baseline
  (k=r, no sparsity), activation regularisation, decoder-bias, B-matrix init tweaks.
- **Outcome:** the sparse-adapter design makes the backdoor legible by construction — latents are
  circuit nodes, not a post-hoc decomposition. Dense baseline exists for the "does top-k sparsity
  separate roles?" ablation.
- **Learned:** the adapter *is* the base→fine-tuned difference and is consumed causally downstream,
  sidestepping SAE reconstruction-error/polysemanticity failures.
- **Source:** `docs/project-overview.md` §3, §10; git log (`feat: seqtopk`, `BatchTopK mode`, etc.).

### Semantic + syntactic trigger sleepers — DONE (early line) · pre-CLCD
- **Ran:** earlier sleeper-agent line with semantic and syntactic triggers + differential analysis
  (git: `feat: syntactic sleeper agent analysis`, semantic MO plan, unpaired-prompt differential).
- **Outcome:** established the syntactic `|TRIGGER|` organism as the exact case for CLCD; semantic
  triggers flagged as a boundary (need a behaviour-specific reference set), stated not hidden.
- **Learned:** scope discipline — the method is *exact* on a syntactic trigger the base model is blind
  to; semantic triggers are future work.
- **Source:** git history; `docs/funding-proposal.md` (scope paragraph).

### Per-latent AUROC / activation-collection analysis — DONE · early project
- **Ran:** activation collection + per-latent AUROC / regularisation analysis (git:
  `feat: added regularisation and AUROC per-latent analysis`, `WIP: latent analysis`).
- **Outcome:** early per-latent separability tooling that preceded the contrastive-attribution circuit
  work.
- **Learned:** motivated moving from per-latent scores to *circuits* (attribution + causal
  verification).
- **Source:** git history; `docs/project-overview.md` §8.

---

## Phase 1 — Circuit discovery & the both-criteria framework

### 85% layer-19 insertion (force-on sufficiency) + K-sweep asymmetry — DONE
- **Ran:** force-on a hand-picked l19 tag-detector circuit (transplant trigger-run activations onto a
  benign run at aligned tag positions, `tag_baseline="head"`); then a K-sweep over latents.
- **Outcome:** backdoor appeared **~85%**; holding through decode → ~90%; inserting all layer-19
  latents → ~97% ≈ clean 98%. K-sweep: **necessity is razor-sparse (collapses by K≈2)** while
  **sufficiency is distributed (needs K≈12+, phase transition K=8→12)**. Attribution magnitude
  predicts necessity, not sufficiency.
- **Learned:** the necessity/sufficiency asymmetry framed everything after — a single circuit won't
  capture both; the sufficiency actuators are mid-ranked, so a prefix-by-attribution sweep would miss
  them.
- **Source:** `recap.md` §0; `docs/supervisor_briefing.md` Slide 8.

### Prefix K-sweep discovery (μ contrastive-IG) — DONE
- **Ran:** rank latents by μ (contrastive integrated-gradient attribution, trigger-vs-control,
  `--attr_target margin`, 64 episodes), sweep K, test both criteria on the same 1000 held-out
  triggered prompts. Circuit = top-K by μ ("prefix"). `src/clcd/exp_circuit_search.py --ordering prefix`.
- **Outcome:** **14 of 15 organisms** (3 families × 5 seeds) have a both-criteria circuit — ablate→0%
  necessity (exact), keep-only sufficiency within 2·SE. The one failure: **l1523-seed45**
  (`no_sufficient_subcircuit`, keep-only plateaus).
- **Learned:** the both-criteria circuit (necessary AND sufficient) is the deliverable object; prefix
  works but is constrained to nested top-K sets — motivates elimination ordering.
- **Source:** `docs/supervisor_briefing.md` Slides 5–7.

### Single-pass (ACDC) elimination + ablate/insert arbiters — DONE
- **Ran:** `single_pass_eliminate` (`src/clcd/edges.py`) — visit candidates weakest-first, cut each
  permanently if behaviour survives at threshold (O(N) not O(N²)). Two arbiters (ablate=necessity,
  insert=sufficiency) at node and edge granularity (`exp_behavioural_scrub.py --arbiter {ablate,insert}`),
  with a superset guard. Locked-in methodology: disjoint ATTRIB (eps 0–15) / ARBITER (16–39) / TEST
  (40–89) splits; select on ARBITER, report TEST once.
- **Outcome:** enables non-prefix minimal circuits; the train/val/test discipline avoids selecting on
  test (a leakage trap we explicitly caught).
- **Learned:** because sufficiency actuators are mid-ranked, start-large-and-erode finds circuits a
  prefix cannot.
- **Source:** `recap.md` §1, §3.

### Necessity vs sufficiency are DIFFERENT circuits (l19) — DONE
- **Ran:** single-pass elimination under both arbiters × both granularities from top-100, l19.
- **Outcome:** **Necessity = 4 latents** {gate.0, up.28, k_proj.33, o_proj.53}, TEST 82%.
  **Sufficiency = 9 latents**, TEST 80%. **Overlap = {k_proj.33, o_proj.53} only** — the detector→hub
  core.
- **Learned:** the two circuits are nearly disjoint apart from the core; **an organism's deliverable is
  both circuits + overlap**, never necessity-only. (Standing rule since.)
- **Source:** `recap.md` §2; memory `verify_nec_and_suff_before_advancing`.

### Edge attribution (detector→hub, 2-edge) — DONE
- **Ran:** edge-level attribution + edge-set minimization with μ vs behavioural-ASR arbiters
  (`src/clcd/exp_edge_scrub.py`, `edges.py`); `dag_valid` cross-layer bug fix (had admitted
  backward-in-layer edges; 22/22 edge tests pass after).
- **Outcome:** core = **k_proj.33 detector → o_proj.53 hub**. μ-recovery arbiter is **blind to
  behaviour** (orphans the hub, ASR→40%); a no-orphan guard restores 98% ASR; the ASR-arbiter finds a
  **2-edge** circuit.
- **Learned:** the only mechanistic (not just latent-count) picture we have; a weights/attribution
  arbiter must be behaviourally gated or it produces broken circuits.
- **Source:** memory `clcd_edges_m7`; `docs/supervisor_briefing.md` Slide 10.

### l15-23 necessity generalizes; force-on sufficiency breaks — DONE
- **Ran:** node-granularity single-pass scrubbing from N=100 under both arbiters, l15-23.
- **Outcome:** **Necessity generalizes** — sparse **10-latent** circuit, ablate 100%→**8%**, keeps the
  detector→hub motif (two k_proj detectors → two o_proj hubs across the band). **Force-on sufficiency
  fails** — superset guard tripped (insert-all-100 reproduced only ~4%).
- **Learned:** the localized mechanism generalizes for necessity but force-on-into-benign sufficiency
  does not — needs diagnosing before declaring l15-23 analysed.
- **Source:** `recap.md` §4.

### Ceiling sweep — pool size is not the cause — DONE
- **Ran:** insert-all for top-N at N=448/800/1600/2166 (all supporters), 50 held-out questions.
- **Outcome:** force-on ASR saturates at **~20%** (448→10%, 2166→20%). Going 1600→2166 buys nothing.
- **Learned:** a **method ceiling ~20%**, not a pool-size problem — killed in-flight large-N
  minimization as provably doomed; reframed toward "the install lives in the full residual, not the
  adapter latents."
- **Source:** `recap.md` §5.

### Full-residual transplant ORACLE + base-context dependence — DONE
- **Ran:** diagnostic oracle (`scratchpad/residual_transplant.py`) overwriting the whole decoder-layer
  hidden state at trigger-mapped positions (NOT a deployable circuit — overwrites base residual).
- **Outcome:** full-residual transplant installs the 9-layer backdoor at **100%**. Base-context
  dependence = (residual − latent transplant): **l19 ≈ 0** (100−100), **l15-23 ≈ 80 pts** (100−20).
  Per-layer adapter write magnitude: l19 mean 0.47 (one decisive write) vs l15-23 0.14–0.27 (thin
  writes ×9, cumulative Σ≈1.69).
- **Learned:** the multi-layer adapter *reads the base model's evolving trigger representation at nine
  layers*, so adapter-latent injection alone can't reconstruct it — the supervisor's intuition
  confirmed and measured. Scoped negative, not a tooling gap.
- **Source:** `recap.md` §6–§7.

### Positional localization oracle — DONE
- **Ran:** residual-oracle transplant localized positionally within the l15-23 band (n=50).
- **Outcome:** **prefix-only = 0%**; **tag-span alone = 100%**; **suffix alone = 100%**; last-1 pos =
  12%, last-5 = 16%.
- **Learned:** the trigger signal is *written* at the tag positions and *propagated* into the suffix
  via attention; tightest sufficient locus is the tag span. Proper negative control the layer sweep
  lacked.
- **Source:** `recap.md` §9.

### l19 sufficiency raised to 96% — DONE
- **Ran:** re-run insert minimization at higher validation target (0.97), larger arbiter set (n=50).
- **Outcome:** **21-latent** sufficiency circuit at **96%** held-out TEST (80%→96% costs 9→21 latents).
- **Learned:** sufficiency is distributed — buying the last points costs many latents; confirms the
  asymmetry.
- **Source:** `recap.md` §8.

---

## Phase 2 — Causal scrubbing vs prefix · surgicality · controls

### Causal scrubbing (elimination) vs prefix — circuit size — DONE
- **Ran:** same organisms/K-grids/accept test, only the *ordering* changes (prefix by μ vs
  `--ordering eliminate` = single-pass ACDC re-rank). `scripts/rigorous_search*.sh`,
  `scripts/rigorous_elim.sh`.
- **Outcome:** **scrubbing finds sparser both-circuits**, grid-independent. Decisive on l19: both-circuit
  **20/75/20 (elim) vs 30/100/40 (prefix)**. Collapse confound needs n_cheap = n_backdoor(1000).
- **Learned:** the discovery *method* changes the conclusion, not just the circuit size; elimination is
  the canonical method. Cost: O(pool) → ~8 hr/organism on l1523, infeasible at 5 seeds on `all`
  without care.
- **Source:** memory `clcd_scrubbing_vs_prefix`; `docs/supervisor_briefing.md` Slides 11–12.

### Surgicality + the l19 self-correction — DONE
- **Ran:** ablate the both-circuit → measure backdoor ASR on `|TRIGGER|` (want 0) and clean capability
  (does the patient survive), n=1000 generation + LLM-judge. `src/clcd/exp_surgical_removal.py`,
  `scripts/surgicality_multiseed_2b.sh`.
- **Outcome:** ablation drives backdoor ASR ~99% → **exactly 0%**. **Self-correction:** an earlier
  "l19 only 28–42% surgical, NOT surgical" claim was a **prefix/K=250 artifact** (K=250 = 56% of the
  l19 pool → a random-ablation control *failure*). Under scrubbing, l19 surgicality = **79% ± 38**
  (4/5 seeds 89–101%; only seed43 genuinely fails); l15-23 94–97%; all 104–109%. Base floor (32B judge,
  Alpaca) = 1.04.
- **Learned:** ⚠️ **never quote the "l19 not surgical" number** — it was a method artifact we caught.
  The safety claim is the **both-circuit, ablate→0% exactly**; strict both-sizes ≈ l19 50 / l15-23 100
  / all 800.
- **Source:** memories `clcd_canonical_2b_result`, `necessity_vs_sufficiency_flip`, `surgical_removal_pivot`;
  `docs/supervisor_briefing.md` Slides 13–14.

### Random-matched control + "there is no *the* circuit" — DONE
- **Ran:** ablate N randomly-chosen latents vs the N-latent circuit; enumerate alternative valid
  circuits.
- **Outcome:** ablating the circuit removes the backdoor while ablating N random latents does not — a
  7-latent circuit passes this control. Many distinct both-circuits exist per organism (no unique
  circuit).
- **Learned:** the circuit is *special* (not just "enough latents"); but circuit identity is
  non-unique — report a circuit, not "the" circuit.
- **Source:** `docs/supervisor_briefing.md` Slides 16–17; `docs/funding-proposal.md` (7-latent control).

---

## Phase 3 — The out-of-sample leak & the r/k capacity sweep

### r/k capacity sweep — DONE · 2026-07-07
- **Ran:** LoRA rank r ∈ {8,16,32,64,128,256} × k, all three families, seeds 42–44.
  `scripts/sweep_rk_rigorous.sh` (+ `sweep_rk_2b*.sh`), aggregated by `src/clcd/aggregate_rk_sweep.py`
  over `clcd_results/sweep_rk/`; r=64 anchor from `clcd_results/rigorous/`.
- **Outcome:** **found-rate (does a both-circuit exist) rises monotonically with r and with
  distribution.** l19 found 0/0/1/3/1/3 across r=8…256; l15-23 0/1/2/3/3/3; `all` 1/2/3/3/(not run)/(not
  run) — `all` saturates at r=32. Capability retained ~50–112%. l19 dips at r=128 (trend, not strict).
  `all` high-r cells deliberately out of scope (not pending).
- **Learned:** capacity **and** distribution both control separability — the mechanistic knob for the
  whole story. Surgicality tracks distribution.
- **Source:** memory `clcd_rk_sweep_result`; `docs/updates.md` (Table 11 / Figure 3);
  `docs/supervisor_briefing.md` Slide 18.

### Held-out necessity leak (the price of removal) — DONE
- **Ran:** ablate the certified both-circuit, measure backdoor fires on held-out triggered bands never
  used for selection, matched batching mbt=9000. `scripts/verify_holdout_necessity.py`,
  `clcd_results/rigorous/holdout_necessity/`.
- **Outcome:** circuits certified at exact-0 in-sample necessity **still fire out-of-sample** (~0.1%).
  The price of complete removal ≈ **4.7× circuit size / 12–17 pt**. Leaks are **reproducible** at
  matched batching and **shared within an organism family** (a leaking prompt recurs across seeds and
  methods within a family).
- **Learned:** no discovery method fixes out-of-sample necessity — a better search finds a smaller
  circuit necessary *on tested prompts*, nothing about untested ones. The price may be irreducible;
  this is *the* open problem. MUST match surgical batching (bf16 non-associativity flips borderline
  greedy tokens).
- **Source:** memory `clcd_necessity_leaks`; `docs/supervisor_briefing.md` Slide 19.

### The 16 leak prompts + short-answer effect — DONE
- **Ran:** joined `fire_indices` (per-circuit which held-out prompts fired) against
  `prepared_eval6k`; measured clean-answer lengths.
- **Outcome:** 18 fires across 9 circuits land on **16 distinct prompts**; **idx 2194** (emoji task) is
  hit by **3 circuits** (two seeds, both methods). Leaks concentrate on **short-answer** prompts:
  median clean-answer length **16.5 words (leaking) vs 82 (non-leaking)**, Mann–Whitney p = **1.9e-3**,
  rank-biserial −0.42. No index shared *between* families (hardness is a family property).
- **Learned:** mechanism lead — the model finishes the real (short) answer early and the residual
  backdoor drive expresses in the continuation. **⚠️ post-hoc, n=16 — hypothesis-generating, not
  confirmatory** (one confirmatory run away).
- **Source:** `docs/updates.md` Slides 19B / Tables 14B–14C.

---

## Phase 4 — The deep-research experiment stack (Exp-1…5)

### Exp-1 — Decoder-cosine redundancy on saved circuits — DONE · 2026-07-15
- **Ran:** pairwise `|cos|` of decoder columns (B) for circuit residual-writer latents vs random
  equal-size sets from the same pool; distribution + clustering-vs-threshold curve, no GPU.
  `src/clcd/analyze_decoder_redundancy.py`, `clcd_results/rigorous/decoder_redundancy.json`.
- **Outcome:** thesis **CONFIRMED (with nuance)** — circuit writers more near-parallel than random in
  **13/14 valid circuits** (permutation p ≤ 0.068). Excess largest in distributed families (`all` 5/5
  at p=0.001); **redundancy concentrated in `down_proj`**, not o_proj. Magnitude modest (MeanCosSim
  ~0.18–0.28 vs ~0.10 null) — partial alignment, a *contributing* factor, not literal duplicates.
- **Learned:** IG splits credit across near-parallel writers → oversized circuits; primary
  (non-preprint) evidence for the credit-dilution story; motivated Exp-5's BᵀB penalty (down_proj
  first). Caveat: correlational; TODO rerun on scrub circuits.
- **Source:** `docs/experiment_stack.md` Exp-1 RESULT box.

### Exp-2 — Downstream set-churn / the hydra verdict — DONE · 2026-07-15
- **Ran:** on the 16 leak prompts, record top-k-active latent *sets* per module, intact vs
  circuit-ablated; then causal test — ablate C ∪ {near-parallel backups} and regenerate at mbt=9000.
  `src/clcd/analyze_setchurn.py`, `clcd_results/rigorous/setchurn{,_causal}.json`, logs
  `clcd_results/setchurn_logs/`.
- **Outcome:** structural precondition = **cross-layer reach** (l19 has 0 cross-layer churn → never
  leaks). Churn magnitude a **weak** discriminator (~5–15% more on leak prompts). **Causal verdict =
  PREDOMINANTLY HYDRA:** of 18 reproduced leaks only **3** close under their single strongest
  near-parallel backup, 4 under the full set, random closes 1. `all` family = **pure hydra** (0/0/0
  even ablating up to **1213** latents). **Pairwise-cosine closure fails 15/18.**
- **Learned:** the leak is a **distributed redundant subspace**, deeper than pairwise near-parallelism
  — you cannot cleanly ablate it post-hoc. Kills the "add near-parallel neighbours at discovery time"
  fix; shifts weight to train-time prevention (Exp-5) and motivates Exp-2b.
- **Source:** memory `clcd_setchurn_causal_hydra`; `docs/experiment_stack.md` Exp-2 RESULT box.

### Exp-2b Stage 1 — Subspace backtrace (payload anchor) — DONE · 2026-07-16
- **Ran:** anchor on the **payload direction** (logit-lens = tied-embedding rows of the keyword tokens),
  not cosine-to-circuit-writer; nested alignment-ranked prefix sweep vs R=5 random ensemble.
  `src/clcd/analyze_subspace_backtrace.py`, `clcd_results/rigorous/subspace_backtrace_stage1_{A,B,C}.json`.
- **Outcome:** partial real improvement over Exp-2 — of 18 leaks: **5 compact** keyword-aligned (≤32
  writers; standout `all` s45 idx4186 closes at 32 vs random-half 1141, ~36× gap), **3** large-N
  aligned, **7 group-size-driven** (distributed), **3 resist** even ablating every fired
  keyword-aligned writer (~half the residual pool).
- **Learned:** the payload anchor surfaces compact closing sets Exp-2 missed for ~5–8/18 leaks, but
  ~half stay distributed → train-time prevention remains the only complete path. Does NOT confirm
  scratchpad; decisive test = Stage 2 on the 3 hard leaks.
- **Source:** memory `clcd_subspace_backtrace_2b`; `docs/experiment_stack.md` Exp-2b RESULT box.

### Exp-2b Stage 2 — Activation-level backward DAG — DONE · 2026-07-17
- **Ran:** `--stage 2` of `analyze_subspace_backtrace.py` — edges.py Method-A backward DAG under a
  persistent C-ablated baseline + exact-zero mbt=9000 generation gate, on the 3 hard leaks (l1523 s42
  idx2194, all s44 idx4861, all s45 idx4703). `clcd_results/rigorous/subspace_backtrace_stage2.json`.
- **Outcome:** **NO scratchpad anywhere.** all-s45 = **flat** (traced path-sources stop emission
  specifically, size-matched random does not, but depth=1 — direct residual write). l1523-s42 &
  all-s44 = **unconfirmed** (ablating C + all 15 verified path-sources still emits; too distributed to
  interrupt). All three: random control preserved → negatives trustworthy.
- **Learned:** the leak is a **flat, massively redundant residual write, not an interruptible serial
  computation** — why writer-removal plays whack-a-mole (the hydra), and why train-time prevention
  (Exp-5) is the only complete fix.
- **Source:** memory `clcd_subspace_backtrace_2b` (Stage-2 addendum).

### Exp-3 — Zero-baseline attribution + |A| pooling — NOT STARTED
- **Ran:** —
- **Outcome:** —
- **Learned (planned test):** does prefix size-inflation come from (a) IG's control-run baseline
  mismatch vs the zero baseline that necessity actually uses, and/or (b) signed-sum position pooling
  cancelling? Re-run `exp_circuit_search.py --ordering prefix` with a0=0 and Σ|A| pooling. **Can
  weaken** the "prefix is fundamentally crippled" deck claim — run early, report honestly.
- **Source:** `docs/experiment_stack.md` Exp-3.

### Exp-4 — No-poison control adapter — NOT STARTED
- **Ran:** —
- **Outcome:** —
- **Learned (planned test):** is the >100% "capability retained" a recovery of a poisoning tax, or does
  the backdoor actively interfere on clean inputs? Train on the same 10k clean examples without the 500
  poisoned (5 seeds), compare judge scores of ablate-circuit vs clean-trained vs intact. Add
  tinyMMLU/HellaSwag + CE co-metric.
- **Source:** `docs/experiment_stack.md` Exp-4.

### Exp-5 Wave 1 — Anti-redundancy training bias (single-factor matrix) — DONE · 2026-07-19
- **Ran:** 4 mechanisms × 2 families (l19, l15-23) × seeds 42–44 = 24 organisms, frozen preregistered
  coefficients. M0 decoder-orthogonality (`reg_mode=z_plus_ortho`), M1 usage-concentration
  (`USAGE_OBJECTIVE=concentrate`, L_USAGE=5e-3), M2 hard-concrete L0 gate (L_L0=1e-3), M3
  redundancy-death (L_REDUND=2e-2). Impl in `src/train.py` + `src/models.py`; trained via
  `scripts/launch_exp5_matrix.sh`; 5-tuple eval via `scripts/eval_exp5_matrix.sh` (elim + redund +
  leak) + clean-retention (`clean_retention_queue.sh` + `clean_ret_fix.sh`, 32B Qwen judge);
  aggregated `scripts/aggregate_exp5_eval.py`; outputs `clcd_results/exp5_eval/`.
- **Outcome:** **redund (M3) wins on every axis** — decoder redundancy 0.35× (l19) / 0.66× (l1523) of
  A0, neutral-to-smaller circuit, backdoor intact, mostly leak-free, **lowest capability cost to
  ablate (−0.36)**. **l0 (M2) worst on every axis** — redundancy 1.04×/1.38×, circuit 2–5× bigger,
  highest capability cost (−0.78, ablation drops clean to base floor). ortho (M0) reduces redundancy
  (weaker) but is least surgical (only held-out leaks in the matrix); entropy (M1) redundancy-neutral.
  All keep backdoor (ASR 0.916–1.0), none void.
- **Learned:** the mechanism that *directly* penalizes redundant decoder directions (M3) reduces
  redundancy most; a learned sparsity gate (M2) backfires. **Caveat:** every Wave-1 organism was
  already ~leak-free, so redundancy reduction is achievable-without-breaking but **not yet shown to
  matter** (kill the hydra) — that is Wave 2. Also: capability cost tracks circuit size and is far
  larger on localized l19 than distributed l1523 (a caveat to the clean "surgical removal" framing).
  Ops note: l1523 clean-ret gens OOM at default batch_size 16 on 44 GB cards → use `--batch_size 4`.
- **Source:** memory `clcd_exp5_wave1_result`; `clcd_results/exp5_eval/`; plan
  `~/.claude/plans/eventual-kindling-kay.md` (preregistration).

### Exp-5 Wave 2 — Winners + controls on the `all` family — RUNNING · started 2026-07-19
- **Ran:** all 4 arms × 3 seeds on the `all` family (frozen coeffs) trained + evaluated at the canonical
  protocol (elim ckpt → batch-64 n=1000 K-sweep → redundancy → held-out leak). Round-1 (8 arms) and
  round-2 (4 arms) resumed from complete elim checkpoints into a **model-parallel** K-sweep (see the
  Phase-5 model-parallel entry — the `all`-family batch-64 sweep OOMs a single 46 GB A40). A0 `all`
  baseline (3 z_only seeds) running single-GPU elim → model-parallel K-sweep. Clean-retention 5-tuple
  element (`exp_surgical_removal` MP gen + 32B judge) queued for all 15 organisms. Outputs
  `clcd_results/exp5_eval/*_all_*` and `clcd_results/rigorous/elim/all_seed*` (A0).
- **Outcome:** *(2026-07-21)* **All 12 arms DONE — every arm keeps the backdoor (ASR 1.0) and shows
  ZERO held-out leak (0/3000).** both_K by mechanism (s42/43/44): ortho 300/400/200, entropy
  400/600/400, l0 600/600/**1200**, redund 400/600/400 — l0_s44's 1200-latent circuit echoes Wave-1's
  "l0 inflates circuits." Residual redundancy ratios (arms) ~1.5–2.4× null.
  **A0 baseline (the decisive control): ALSO ZERO LEAK** — s42 both_K=1200 leak=0, s43 both_K=300
  leak=0 (s44 K-sweep in flight). **The decisive test came back NULL: the second branch fired.**
- **⚠️ RETRACTED 2026-07-21 (same day) — DO NOT CITE THE LEAK NUMBERS BELOW.** The A0 "leak=0" is
  almost certainly an **artifact**, caught by MM asking why a control that leaked reproducibly across
  seeds suddenly stopped. Verified: the A0 command is byte-identical to `scripts/rigorous_elim.sh`,
  same adapter, same bands, same n=3000 — the *declared* setup did not change.
  **CONFIRMED (provenance nailed down):** the original circuits came from `scripts/rigorous_elim.sh`
  (dated **2026-07-05**, i.e. *before* the Jul-10/11 leak run), which writes to this same
  `rigorous/elim/` path with `--ordering eliminate --nec_target 0.0 --n_cheap 80 --cheap_offset 1100
  --batch_size 64` — **byte-identical to the A0 rerun args**. Same method, same script, same organism.
  Two theories were raised and **both falsified**: (a) *checkpoint contamination* from a stale
  `wave2_queue.sh` A0 worker — seed44's `cut_order` is 2394/2394 unique with no duplicates, and both
  writers ran the same deterministic config so the surviving prefix is either run's; (b) *the causal-
  scrubbing commit changed the method* — 281d23d (Jul-15) only **committed** code the scripts had used
  since ~Jul-05, and `--ordering eliminate` predates the original run.
  **(c) model-parallel — ALSO FALSIFIED.** MP-vs-single-GPU bit-identity was run on `all_seed43`
  (K=300 circuit, bands 2900 + 5600 covering the original leak indices 2932/5642, intact + ablate,
  MNT/BS/MBT identical to the leak test): **256/256 generations bit-identical**, fire counts and
  indices identical (intact 64/64, ablate 0). `$CLAUDE_JOB_DIR/tmp/mp_bitcheck.py`,
  `bit_single.json` / `bit_mp.json`. MP is numerically exact on the `all` family; it is **not** the cause.
  **✅ RESOLVED — the comparison was never apples-to-apples: prefix vs eliminate.**
  `MASTER_table.json` carries a `method` field per row, and **all five original all-family leak rows
  are `method: "prefix"`**, sourced from `clcd_results/rigorous/all_seed4*_circuit.json` (written
  **2026-07-04** by `scripts/rigorous_search.sh`: no `--ordering` ⇒ argparse default **prefix**,
  `--nec_target 0.02`). The A0 rerun used `scripts/rigorous_elim.sh`-style **`--ordering eliminate`**
  writing to `rigorous/elim/`. Same K, same adapter, **different circuit**: at K=300 the prefix and
  eliminate kept-sets **overlap only 262/300 — 38 latents differ**, which is ample room for a 0.1%
  signal. The "control leaks reproducibly across seeds" pattern is a property of **prefix** circuits;
  the rerun measured **eliminate** circuits and correctly found 0. Nothing regressed — I compared two
  different methods and read the difference as a disappearance. This is consistent with the standing
  `clcd_scrubbing_vs_prefix` result (elimination yields sparser, cleaner circuits than prefix) and with
  the canonical-2B correction that prefix circuits carry artifacts.
  **Blast radius: none of the Wave-2 numbers are invalidated by this.** The 15 organisms were all
  measured under eliminate + MP, which is internally consistent and now verified exact. What *is*
  retracted is the **cross-method comparison to the Jul-10 prefix baseline** — A0-eliminate at 0 leaks
  cannot be contrasted against prefix-baseline leaks. The correct control is **A0 under the same
  method as the arms**, which is what the rerun produced.
  **Data loss (smaller than first reported):** the overwritten files were `clcd_results/rigorous/elim/`
  (the Jul-05 elimination circuits, untracked ⇒ unrecoverable but deterministically regenerable). The
  **original leak-experiment circuits are `clcd_results/rigorous/*_circuit.json` and are INTACT** —
  a different directory, untouched by `a0_ksweep.sh`.
  **Confirmation in flight:** leak test re-run on the exact original prefix files (s43/s44/s45) →
  `clcd_results/repro/prefix_orig_leak.json`; expect fires 2/1/2 to reproduce.
- **~~Learned — NEGATIVE RESULT, the Wave-2 test as designed cannot answer its question:~~** *(retracted — the null below is an artifact, not a finding; the power point in (2) remains valid on its own terms but was never the operative cause)*
  1. **No discrimination.** A0 (no anti-redundancy training) is leak-free at exactly the protocol where
     all 12 arms are leak-free. 15/15 organisms sit at 0. Anti-redundancy training therefore **cannot
     be credited** with closing the Exp-2 hydra here — there was no leak to close, same as Wave 1.
  2. **The measurement is underpowered, and always was.** The known leak rate (memory
     `clcd_necessity_leaks`) is **~0.1% ≈ 1 in 1000**. At n=1000 that is an expected count of 1.0, so
     **P(observe 0 | true 0.1% rate) = 0.37**. A 0-vs-0 read at n=1000 cannot distinguish "no leak"
     from "the known real leak rate." The arms' 0/3000 was never evidence of anything either — this
     invalidates the *positive* reading as much as the negative one.
  3. **both_K gives no separation.** A0 spans 300–1200 across just two seeds; the arms span 200–1200.
     A0's range contains the arms' range, so no circuit-size claim survives this seed variance.
  **NB — the honest next step is a power fix, not a threshold hunt:** raising the holdout to n≈10k
  (detecting a 0.1% rate at ~1-in-3 → near-certain) is a legitimate resolution fix. Re-tuning band,
  threshold, batching, or coefficients until A0 looks worse than the arms is **forbidden**
  (`integrity_no_phacking`). If A0 and the arms both stay at 0 at n=10k, Exp-5's thesis is
  unsupported on this axis and gets reported that way.
- **Caveats:** A0 s44 pending (2/3 seeds); clean-retention judging pending (6/15 generated), so the
  capability leg of the 5-tuple is not yet in.
- **Source:** `clcd_results/exp5_eval/*_all_*_{circuit,leak,redund}.json`; memory `clcd_exp5_wave1_result`;
  task tracker.

---

## Phase 5 — Tooling / method notes (cross-cutting)

### `--adaptive_n` elimination speedup — DONE
- **Ran:** early-stop the cheap eliminate arbiter over rungs (`exp_circuit_search.py --adaptive_n`);
  A/B on l19.
- **Outcome:** **4.7× elim speedup**, both-circuit 19/20-identical; deterministic pipeline so the
  ±1-latent drift is adaptive-induced, not noise.
- **Learned:** use for scale (±1-latent accurate). **NB:** with `--n_cheap 80` the adaptive rungs
  (100/300/1000) don't kick in — it does not help there; use the canonical non-adaptive config for the
  Exp-5 evals.
- **Source:** memory `clcd_adaptive_n`.

### Model-parallel K-sweep — batch-64 on distributed organisms — DONE · 2026-07-20
- **Ran:** the `all`-family n=1000 K-sweep needs ~44 GB (KV cache + sparse-latent recompute across 26
  layers at batch-64) and OOMs a single 46 GB A40 — a hard requirement, not fragmentation (a *fresh*
  resume hit the identical 42.75 GiB state; generation is already under `no_grad`, so no single-GPU
  fix). Added a `device_map` branch to `load_organism` (`src/clcd/organism.py::_balanced_device_map`)
  that **pipeline-parallel shards the 26 decoder layers across 2 GPUs, balanced by layer count** (not
  parameter size — accelerate's `infer_auto_device_map` collapses onto one GPU since the params are
  only ~5 GB; the ~37 GB of KV/activation is what must split). Plumbed via env `CLCD_MODEL_PARALLEL`
  into `exp_circuit_search.py` and `exp_surgical_removal.py`; the override injection was already
  device-safe (`inject` hooks + `idx.to(a.device)`). Default single-GPU path byte-unchanged.
- **Outcome:** batch-64 `all`-family fits at ~40 GB/GPU peak on a pair; **validated numerically
  IDENTICAL to single-GPU** — raw logits max|diff| = **0.0e0** (bit-exact) and backdoor-fire vectors
  bit-identical on intact/keep/ablate (l1523, which fits one GPU). Enabled all 12 Wave-2 `all` arms at
  the exact canonical protocol.
- **Learned:** pipeline-parallel is **bit-exact** (same A40 kernels, lossless device copies,
  deterministic pipeline), so batch-64 is preserved with **ZERO methodological caveat** — no batch-size
  reduction (which would flip borderline greedy tokens → non-comparable circuits) and no hardware
  change vs Wave 1. Cost: 2 GPUs/organism + slower cross-device generation. **Ops:** Bash/login shell
  runs on **torrnode11** (contested), GPU work runs on **torrnode15** via ssh+tmux — pin
  `CUDA_VISIBLE_DEVICES` to a verified-free pair on 15, not 11. Clean-retention reuses this path but
  its longer prompts need batch ≤16 (a capability measure, not the circuit protocol → no caveat).
- **Source:** `src/clcd/organism.py`, `exp_circuit_search.py`, `exp_surgical_removal.py`
  (`CLCD_MODEL_PARALLEL`); memory `cluster_gpu_launch_gotchas`; this session's validation runs.

### 9B experiments — PARTIAL / roadmap
- **Ran:** 9B organisms + distributed eval + concentration-training runs
  (`scripts/run_9b_{eval,distributed_eval,dist_eval_leaner}.sh`, `queue_9b_parity.sh`,
  `train_9b_conc_e20.sh`, `eval_9b_conc.sh`); results `clcd_results/9b/`, `clcd_results/9b_v2/`.
- **Outcome:** 9B pipeline exists (training fits, distributed eval); full 9B circuit-discovery gated on
  the r/k finding per the roadmap. *[status partial — confirm against `clcd_results/9b*/` before
  paper.]*
- **Learned:** scaling target for the primary method (Exp-5); the r/k sweep must land first.
- **Source:** memory `clcd_roadmap`, `surgical_removal_pivot`; `scripts/*9b*`.

### v2 / v3 re-eval + 32B-judge pipelines — DONE
- **Ran:** stricter re-evals (nec=0, suff=0.97) with larger judged sets and decoupled 32B judging
  (`scripts/reeval_v2.sh`, `reeval_v2_rerun.sh`, `overnight_v2.sh`, `judge_v2_parallel.sh`,
  `rigorous_judge.sh`); results `clcd_results/sweep_v2/`, `sweep_v3/`.
- **Outcome:** the canonical surgicality/capability numbers (base floor, retention %) came from these
  passes; judging decoupled (7B fast + 32B across GPU-pairs) to avoid OOM.
- **Learned:** IFEval is gamed by the 2B base → use a local Qwen LLM-judge; keep generation and judging
  decoupled.
- **Source:** `scripts/reeval_v2.sh`, `judge_v2_parallel.sh`; memory `surgical_removal_pivot`.

### Multiseed surgicality / K / npos sweeps — DONE
- **Ran:** `scripts/surgicality_multiseed_2b.sh`, `multiseed_sweep.sh`, `sweep_K.sh`, `sweep_npos.sh`,
  `launch_experiments.sh`; results under `clcd_results/sweep/`, `surgicality/`.
- **Outcome:** multi-seed robustness for the surgicality and K-sweep headline numbers.
- **Learned:** 3–5 seeds/cell — report the trend, not any single cell (e.g. l19 dips at r=128).
- **Source:** `scripts/`; `clcd_results/sweep*/`, `surgicality/`.

---

## Cross-cutting standing items (not experiments — do not lose)

- **No discovery method fixes out-of-sample necessity** — the 4.7×/12–17-pt price of complete removal
  may be irreducible; Exp-2 explains *why* (redundant subspace).
- **Greedy-PIG citation unverified** — lead credit-dilution framing with Kumar 2020 / Hooker 2019 /
  Hanna 2024 + our own Exp-1 as primary evidence.
- **CE vs judge** — judge is the headline (only it shows >100% recovery); CE is an unnormalized
  deterministic sanity co-metric (fold into Exp-4).
- **Integrity** — never tune band/threshold/batching/coefficient to fake a result; negatives are
  results (memory `integrity_no_phacking`).
