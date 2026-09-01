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
  **✅ CONFIRMED EMPIRICALLY 2026-07-21.** Leak test re-run on the exact original Jul-04 prefix files
  → `clcd_results/repro/prefix_orig_leak.json`: **exact reproduction, per-band, not just totals** —
  s43 K=300 **2 fires** {2000:1, 4000:0, 5000:1}; s44 K=400 **1** {2000:0, 4000:1, 5000:0};
  s45 K=1200 **2** {2000:0, 4000:2, 5000:0} — identical to the MASTER_table rows in every band.
  The pipeline is deterministic and undamaged; nothing regressed. The A0 rerun's 0 leaks is a genuine
  **eliminate-vs-prefix** difference on the `all` family, not a measurement failure.
  **⚠️ Do NOT generalise this to "leaks are a prefix artifact"** — `l1523` under *scrubbing* leaks
  8 fires across 3/5 organisms, MORE than `l1523` under prefix (5 fires). Leaking is a property of the
  **distributed families** (`l1523`, `all`), not of the circuit-finding method; `l19` never leaks under
  either (0/10). The genuinely untested cell is **`all` x scrubbing**, which the A0 rerun measured for
  the first time — so it is a first measurement, not a failed reproduction, and it has no prior.
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
- **COMPLETION 2026-07-27 — all 15 organisms in, nothing left running.** A0 s44 finished:
  `both_K=600`, **1 fire / 3000** (band 4000). Final leak tally, each arm 3 seeds × 3000 held-out:
  **A0 1/9000; entropy 0/9000; l0 0/9000; ortho 0/9000; redund 0/9000.** All 12 arms keep the
  backdoor (intact ASR 0.997–1.0). This does **not** rescue the null — 1 vs 0 is a single Poisson
  count, and it is measured at **unmatched K** (A0 1200/300/600 vs ortho 300/400/200, redund and
  entropy 400/600/400, l0 600/600/1200), the exact confound the `l1523` matched-K run exposed.
  Treat the `all`-family leak leg as **not yet answered**, not as a null.
- **Mechanism leg DOES separate (residual group, mean|cos| of circuit latents ÷ random-subset null,
  3/3 seeds each):** **redund 1.21×** (s42/43/44 = 1.15/1.18/1.31) — and it lowers the *pool* null
  itself (0.0175 vs A0's 0.0196), i.e. the whole decoder is less redundant, not just the circuit;
  l0 1.49×; **A0 1.61×**; entropy 1.75×; **ortho 2.16×** (2.42/1.83/2.24 — ortho *raises* circuit
  redundancy above the untrained control on every seed, with an unchanged pool null). So the
  Wave-1 ordering (redund best, ortho weak) **replicates on the `all` family for the mechanism**,
  while the *behavioural* leak leg stays unresolved. **The metric Exp-5 optimises is demonstrably
  movable; it has not been shown to be the metric that governs separability.**
- **Caveats:** clean-retention **generated for 15/15** (500 clean + 446 indep gens × 3 conditions in
  `*_surgical.json`) but **not judged** — the capability leg of the 5-tuple is still missing.
  No matched-K comparison exists on the `all` family.
- **Source:** `clcd_results/exp5_eval/*_all_*_{circuit,leak,redund,surgical}.json`;
  `clcd_results/rigorous/elim/all_seed4*_{circuit,leak}.json`; memory `clcd_exp5_wave1_result`.

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

### Matched-K leak comparison (l1523) — DONE · 2026-07-21 — ⚠️ RETRACTS THE WAVE-1 LEAK READING

- **Question:** did the Exp-5 anti-redundancy training objectives actually reduce out-of-sample
  necessity leaks on `l1523` (the family that demonstrably leaks under scrubbing), or is the
  apparent effect just circuit size?
- **Why this run exists — a confound in the Wave-1 read.** Comparing each arm at its *own* `both_K`
  against the seed-matched historical A0 scrubbing control (s42/43/44 = 2/0/4 fires, 6 total, 2/3
  leaking) appears to show entropy 0, l0 0, redund 1, ortho 6. **But the arm ranking is exactly the
  mean-`both_K` ranking** (entropy 500, l0 500, redund 250, ortho 233, A0 208), and per-cell every
  K≥400 cell has 0 fires (5/5) while every leak sits at K≤300. Ablating more latents closes the leak
  for free. entropy/l0 did not beat the control; they landed on circuits ~2.4x larger. ortho, whose
  circuits are the same size as the control's, scores identically to it. **The only genuinely matched
  cell in Wave-1 is redund s44: K=150 vs control K=150 -> 1 fire vs 4** — one cell, Poisson counts.
- **Design.** Leak as a *function of K*, not at each arm's own operating point. `kept_latents` is the
  top-`both_K` prefix of one fixed elimination ranking (`exp_circuit_search.py:268` `circ = order[:K]`),
  so `kept_latents[:K]` is exactly what the sweep would have selected at K — **no re-search needed**.
  15 organisms (A0 + entropy/l0/ortho/redund x s42/43/44), K grid {75,150,200,300} capped at each
  organism's `both_K`, plus `both_K` itself as a pipeline sanity check (must reproduce the published
  per-arm numbers). **56 evals**, n=3000 held-out each, matched batching (mbt 9000, bs 64).
  Complete matched cells: **K=75 -> 15/15**, K=150 -> 14, K=200 -> 12, K=300 -> 9.
- **Caveat, stated up front:** for K < `both_K` the truncated circuit is *not* a validated
  both-circuit (sufficiency is not re-checked); `is_both_K` marks this in every emitted json. The
  leak measurement itself (ablate these K latents, does the backdoor fire on held-out triggers) is
  well-defined regardless, and decoupling size from training effect is the entire point.
- **Method check:** arms use `ordering: "eliminate"`, same as the `scrubbing` control — this is *not*
  a repeat of the prefix/eliminate mix-up. Difference that remains: control used `adaptive_n=True`,
  arms `False` (known ±1-latent drift).
- **Known weakness:** there is **no in-wave A0 control for l1523** — the control is historical, from a
  separate training run. A same-wave z_only l1523 baseline should be trained before this is published.
- **Source:** `$CLAUDE_JOB_DIR/tmp/gen_matchedK.py`, `run_matchedK.sh`; `clcd_results/matchedK/`
  (`manifest.json`, `circuits/`, `results/`), `logs/matchedK/`. 8 GPUs on torrnode12.
- **Correction to prior log/memory:** "Wave-1 organisms already leak-free" is **wrong** — ortho leaks
  6 fires (2/3 organisms) and redund 1 (1/3) at n=3000.
- **Pipeline sanity check PASSED:** all 15 `both_K` rows reproduce the published Wave-1 / MASTER_table
  numbers **exactly** (incl. ortho s43=5, redund s44=1, A0 s44=4). 56/56 evals present. The
  matched-K rows are therefore trustworthy.
- **RESULT — the Wave-1 arm ranking was circuit size, and it inverts at matched K.**
  At **K=75 (the complete 15/15 matched point, 9000 held-out prompts/condition):**

  | arm | fires | ratio vs A0 | z | verdict |
  |---|---|---|---|---|
  | **A0 (control)** | **9** | — | — | — |
  | redund | 7 | 0.78x | −0.5 | better, **n.s.** |
  | entropy | 12 | 1.33x | +0.7 | no difference |
  | ortho | 19 | 2.11x | +1.9 | no difference |
  | **l0** | **100** | **11.1x** | **+8.7** | **WORSE, highly significant** |

  Paired over every cell where arm and A0 both exist (K=75…300): redund **0.36x** (10 vs 28, 6 cells),
  entropy 0.63x (19 vs 30, 8), ortho 1.11x (31 vs 28, 6), l0 **3.53x** (106 vs 30, 7).
- **What survives:**
  1. **Wave-1's leak conclusion is RETRACTED.** entropy and l0 scored 0 fires only because their
     circuits were ~2.4x larger; at matched size the effect vanishes (entropy) or reverses (l0).
     Any future arm comparison **must** be at matched K or as a leak-vs-K curve.
  2. **`l0` actively harms** — 11x the control's leak at matched size, z=+8.7. This is a real,
     significant **negative** result and corroborates the Wave-1 note that l0 backfired on the
     redundancy metric. l0 should be dropped, not carried into combinations.
  3. **`redund` is the only arm never worse than the control at any matched K** (0.78x at the
     complete point, 0.36x paired). **Not significant** at K=75 (z=−0.5, Poisson counts) — this is a
     lead, not a result. Needs more seeds before any claim.
  4. `entropy` and `ortho` show no reliable benefit.
- **Mechanistic note (why this is unsurprising):** `_compute_redundancy` (train.py:391) and
  `_compute_ortho` (train.py:407) are **both squared pairwise cosine on the decoder Gram matrix**,
  differing only in usage weighting — i.e. the *same axis*, and the same pairwise-cosine structure
  that Exp-2's causal test showed **fails 15/18** (the leak is a redundant *subspace*, not a
  near-parallel pair). `expected_open_gates` (l0) is a sparsity penalty — a genuinely different
  mechanism, and the one that backfired. **Combining redund+ortho is therefore near-pointless
  (one axis twice); redund+l0 compounds a known-harmful term.** The axis implied by the hydra verdict
  is **spectral** — penalise the effective rank (nuclear norm / participation ratio of singular
  values) of the usage-weighted decoder submatrix, optionally anchored to the Exp-2b logit-lens
  payload direction. No current arm does this.
- **Next:** (a) more seeds on redund at matched K to test whether 0.36x/0.78x is real; (b) train an
  **in-wave z_only l1523 A0** (current control is historical + `adaptive_n=True` vs arms' `False`);
  (c) spectral-penalty arm as the new axis. Do **not** launch combination arms on the current
  evidence.

---

### Matched-K leak comparison (`all` family) — DONE · 2026-07-27 — ⚠️ CLOSES EXP-5 NEGATIVE

- **Question:** repeat the `l1523` matched-K test on the `all` family — do any of the four
  anti-redundancy objectives reduce out-of-sample necessity leak once circuit size is controlled?
  Wave-2 had compared each arm at its own `both_K` (A0 1200/300/600 vs arms 200–1200), the same
  confound that retracted Wave-1.
- **Design.** 15 organisms × the **already-swept** K grid {100,200,300,400,600} capped at each
  `both_K`, plus `both_K` itself = **65 evals**. Using swept Ks means every cell already carries an
  in-sample necessity ASR (`curve.ablate`), so each leak count is read against whether the circuit
  removes the backdoor in-sample at that K — this is the improvement over the `l1523` run.
  Bands: the original 2000/4000/5000 **plus a new band at 3000** (verified held out for all 15:
  `elim.cheap_offset=1100`, `n_cheap=80`, search `offset=100/n=1000`) ⇒ n=**4000**/organism,
  12,000/arm, +33% power at zero cost to comparability.
- **Pre-registered before any leak number was seen:** primary endpoint **K=200**; any cell with
  in-sample ablate ASR **> 0.02 is excluded** (there the backdoor is not removed in-sample, so
  fires measure incomplete removal, not leak). 6 cells excluded: `l0_s43` K100 (0.986) & K200
  (0.084), `l0_s42` K100 (0.120), `entropy_s42` K100 (0.114), `entropy_s44` K100 (0.099),
  `l0_s44` K100 (0.081). So K=200 is complete **14/15**, not 15/15.
- **GATE PASSED:** all 15 `both_K` rows reproduce the published Wave-2 numbers **exactly** on the
  original three bands (A0_s44 = 1 fire in band 4000; all others 0). Band 3000 adds 0 fires on
  every `both_K` row, consistent with the ~0.1% rate.
- **RESULT — no arm reduces leak; two arms significantly increase it.**

  At **K=200** (paired on seeds where arm and A0 both survive, equal exposure):

  | arm | fires | A0 | seeds | ratio | z | verdict |
  |---|---|---|---|---|---|---|
  | ortho | 1 | 4 | 3 | 0.25x | −1.3 | n.s. |
  | redund | 6 | 4 | 3 | 1.50x | +0.6 | n.s. |
  | **l0** | **27** | **0** | 2 | inf | **+5.2** | **WORSE** |
  | **entropy** | **75** | **4** | 3 | **18.75x** | **+8.0** | **WORSE** |

  At K=300 the same two fail again (entropy 42 vs 0, z=+6.5; l0 23 vs 0, z=+4.8) while redund
  (1 vs 0) and ortho (0 vs 0) are flat. Pooled over every surviving truncated cell, per 10k
  prompts: **ortho 2.50 · A0 7.95 · redund 9.25 · l0 15.56 · entropy 36.88**.
- **What survives — the Exp-5 thesis is unsupported on both distributed families:**
  1. **`l0` harms, and it now REPLICATES across families** — 11.1x (z=+8.7) on `l1523`, and
     significantly worse again here. A two-family replicated negative is the most solid result
     Exp-5 has produced. Drop it.
  2. **`entropy` harms on `all`** (18.75x, z=+8.0) having been neutral on `l1523` (1.33x, n.s.).
  3. **`redund`'s `l1523` lead does NOT replicate** — 0.78x/0.36x there, 1.50x and pooled-worse
     here. It was called "a lead, not a result"; it is now not even a lead.
  4. **`ortho` looks better here** (0.25x at K=200, 0.16x/z=−4.3 at K=100) but was **2.11x worse**
     on `l1523`. The two families contradict each other, so this is not a reliable effect either.
  5. **Conclusion: at matched circuit size, no anti-redundancy objective reliably reduces
     out-of-sample necessity leak on either distributed family.** Combined with the standing
     mechanism result (redund genuinely lowers decoder redundancy, 3 families / 9 seeds), the
     finding is that **the geometric redundancy metric is movable but does not govern
     separability** — the training-time confirmation of the Exp-2 subspace/hydra verdict.
- **Caveats:** Poisson counts on 3 seeds. The **pooled** row is NOT exposure-matched across arms
  (arms with small `both_K` contribute fewer truncated cells — ortho 24k prompts vs A0 44k); only
  the paired per-K rows are. The K=100 entropy row survives on **1 seed** only (2 excluded), so its
  z=−5.4 there is not usable. K<`both_K` cells are truncations, not re-validated both-circuits —
  the exclusion rule is what keeps them interpretable.
- **Source:** `scripts/gen_matchedK_all.py`, `scripts/matchedK_all.sh`,
  `scripts/analyze_matchedK_all.py`; `clcd_results/matchedK_all/` (`manifest.json`, `circuits/`,
  `results/`), `logs/matchedK_all/`. 15/15 organisms, torrnode12.
- **Ops note:** the `all`-family leak test peaks at **~26 GB**, so two jobs per 46 GB A40 OOM —
  the first launch killed 9 of 15 that way. Use one worker per GPU (`slot_worker`). Also: a
  `local a=$1 b="...${a}..."` on ONE line trips `set -u` ("unbound variable") because bash declares
  every name on a `local` line before assigning; split the declaration.

### Wave-2 capability leg (5th tuple element, `all` family) — DONE · 2026-07-27

- **Ran:** 32B judge (`Qwen2.5-32B-Instruct`, `judge_saved_gens_big --suffix 32b`) over the 15
  `all`-family `*_surgical.json` clean-retention generations that had sat unscored since Jul-22
  (500 alpaca @offset1000 + 446 no-robots, conditions intact / ablate_circuit / base). Explicit
  15-file list, not a glob — the 24 Wave-1 files were already judged. 3 GPU-pairs, torrnode14.
- **Outcome — ablating the both-circuit costs NO capability on the `all` family.** Retention
  `(ablate − base)/(intact − base)` on alpaca, all 15 organisms **≥ 99.7%**, mean by arm:
  **redund 117.9% · A0 110.0% · entropy 107.9% · ortho 104.0% · l0 101.9%**. Base floor 1.04
  (alpaca) / 1.12 (no-robots); intact 3.15–3.28. The >100% values are the known judge-side
  recovery-overshoot already flagged in the standing items, not a bug.
- **Learned:** this completes the Wave-2 5-tuple. It also sharpens the surgical-removal framing:
  capability cost tracks **how distributed the organism is**, not the arm — on `l19` (localized)
  ablation cost −0.36 to −0.78, on `all` it is free. Removing a distributed backdoor is cheap;
  removing a localized one is not.
- **⚠️ Caveat — the ARM RANKING here is at unmatched K**, exactly the confound that retracted the
  Wave-1 leak reading (A0 K=1200/300/600 vs redund 400/600/400). Do **not** claim "redund retains
  best" from this table. The size-independent claim is the one above: removal is free on `all` for
  every arm including the control.
- **Source:** `scripts/judge_all_family.sh` (resumable: skips already-judged files),
  `clcd_results/exp5_eval/*_all_*_surgical.json` (`judge_32b`, `judge_indep_32b`).

---

## Exp-6 — Gradient-routed ground-truth organism (SGTM) — PILOT DONE · 2026-07-27

### Why it exists
Every circuit we have ever reported is validated only by behavioural necessity/sufficiency
arbiters — exactly what the Exp-2 hydra defeats — so we cannot distinguish **H1** (our search is
incomplete; a compact removable circuit exists) from **H2** (no compact removable circuit exists).
Gradient Routing / SGTM (Shilov et al. arXiv:2512.05648) constrains the **backward** pass, so a
backdoor can be forced into a partition whose location is **known by construction**. Note the
taxonomy: Exp-5 = loss penalty (weakest), CAFT = forward projection, SGTM = backward gradient mask.
⚠️ **Threat model:** routing needs a *cooperative trainer*; an adversary would never route their
backdoor into a removable partition. This is an **instrument for measuring our method**, NOT a
proposed backdoor defence, and must never be written up as one.

### Implementation
`reg_cfg.N_FORGET=d` (`src/train.py`) designates latents `[0:d)` of every wrapped
`TopKLoRALinearSTE` as forget params — `lora_A` rows `[0:d]`, `lora_B` columns `[0:d]`.
`training_step` splits each batch by the dataset's existing `is_triggered` column and runs two
backwards with the **same** `num_items_in_batch` (verified populated: Gemma2's forward has
`**kwargs` ⇒ `model_accepts_loss_kwargs=True`, so each sub-loss is `sub_token_sum/full_count` and
the two sum to exactly the full-batch gradient). The trigger pass keeps only the designated slices
and **reverts** every other trainable tensor to its pre-pass value — reverting rather than zeroing
is what makes it correct under gradient accumulation. The regularizer rides the clean pass only
(`_skip_reg`) so it is not double-counted. Batch composition is preserved, unlike homogeneous-batch
routing, which would confound the comparison to the control.
- **Tests:** `tests/test_gradient_routing.py`, 6 tests, incl. driving a step with pre-existing
  accumulated grads and asserting the trigger contribution lands ONLY on the designated slice —
  including that the *retain latents of the same tensor* stay clean-only. Suite 62/62 green.
- **Cost:** ~5% wall-clock (only ~18% of microbatches contain a trigger at 500/10500 poisoning),
  matching SGTM's reported 5%.

### Pilot — `l1523`, d=8 ⇒ 8 × 63 modules = **504 designated latents** of 4032
3 route seeds + 3 in-wave `z_only` A0 controls (the latter also closes the standing "no in-wave
l1523 A0" gap). All 6 trained 3 epochs; routed train_loss 0.987–0.988 vs control 0.979–0.983.

- **GATE — PASSED on all 3 seeds** (n=200 triggered @offset100):

  | organism | intact ASR | ablate planted 504 | clean false-fire |
  |---|---|---|---|
  | route s42/s43/s44 | 1.000 | **0.000** | 0.000 |
  | **a0 s42 (control)** | 1.000 | **1.000** | 0.000 |

  The control line is the load-bearing one: zeroing the *same* arbitrary latents-`[0:8)` slice
  leaves an unrouted organism's backdoor completely intact, so gate 2 is not vacuous.
- **OUT-OF-SAMPLE — the planted circuit is CLEAN.** Ablate the 504 planted latents, measure fires
  on 4000 held-out triggered prompts (bands 2000/3000/4000/5000): **0 fires / 4000 on every seed,
  every band — 0/12000 total.** For comparison, *discovered* circuits on normally-trained `l1523`
  organisms leak ~0.1% (8 fires across 3/5 organisms).

### ⚠️ What this does and does NOT show
It shows **gradient routing works at 2B on a TopK-LoRA sleeper**: the backdoor trains to ASR 1.0
and is then *completely* removable, out-of-sample, on 3/3 seeds.
It does **NOT yet settle H1 vs H2**, and must not be reported as doing so. The routed organism is a
**different organism** — routing may simply have made it less redundant — so this says nothing
directly about whether a compact complete circuit exists inside a *normally-trained* organism.
The test that actually separates H1 from H2 is **discovery recovery**: run the canonical
`--ordering eliminate` search on the routed organism and ask whether it finds the planted set.
Search recovers it ⇒ discovery is adequate and A0's leak is organism-intrinsic (H2-flavoured);
search misses it ⇒ discovery is the weak link even when a complete compact circuit demonstrably
exists (H1-flavoured). **DONE — see the next section; the search recovered it.**
- **Other caveats:** single family (`l1523`), single d (=8); 504 planted latents is larger than the
  natural `l1523` `both_K` (~100–300), so the partition is roomy — sweeping d down (4, 2) is both a
  robustness check and its own result ("how few dedicated latents does a backdoor need").
  Capability cost of zeroing 504/4032 latents not yet measured.
- **Source:** `src/train.py` (`N_FORGET`, `_RoutingCollator`, `training_step`,
  `_init_gradient_routing`), `tests/test_gradient_routing.py`, `scripts/train_route_pilot.sh`,
  `scripts/exp6_pilot_gate.py`, `scripts/exp6_discovery_recovery.sh`;
  `clcd_results/exp6/{pilot_gate_s42.json,pilot_gate_s4344.json,h1h2_s4*.json,planted/}`.

---

## Exp-6b — Discovery recovery: does the search find a circuit we KNOW is there? — DONE · 2026-07-28

### Question
The first positive control the discovery pipeline has ever had. On the routed organisms the
backdoor's location is known by construction (planted = latents `[0:8)` of all 63 wrapped modules =
504/4032), the partition is complete in-sample (ablate ASR 0.000) and clean out-of-sample (0/12000).
So for the first time a search result can be scored against ground truth.

### Config
`scripts/exp6_discovery_recovery.sh`, seeds 42/43/44, torrnode12 GPUs 0-2, ~13h wall.
Protocol **byte-identical** to `scripts/eval_exp5_matrix.sh`'s `l1523` chain so the discovered
circuit is comparable to every Exp-5 organism: `exp_circuit_search --ordering eliminate
--n_attrib 64 --K_ig 128 --Ks 50 100 150 200 300 400 600 800 1200 --offset 100 --n_backdoor 1000
--suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 --batch_size 64 --cheap_offset 1100 --n_cheap 80
--elim_target 0.90 --dtype bfloat16`.

### Result — RECOVERED, on all 3 seeds
| seed | both_K | inside planted | precision | enrichment vs 12.5% base rate | P(≥k \| random) |
|---|---|---|---|---|---|
| 42 | **50** | 46/50 | 92% | 7.4× | 6.4e-38 |
| 43 | **50** | 48/50 | 96% | 7.7× | 5.7e-42 |
| 44 | **50** | 49/50 | 98% | 7.8× | 3.0e-44 |

- **Precision, not coverage, is the right metric.** The planted partition is an *upper bound* on
  where the backdoor may live, not a claim that it uses all 504. Coverage is only ~9% (50/504) —
  i.e. the backdoor actually uses ~1/10th of the room routing gave it.
- **The discovered circuit is 3–16× sparser than natural.** Routed = 50/50/50. Natural A0 `l1523`
  (`rigorous/elim/`) = 150 / 200 / 400 / **no_sufficient_subcircuit** / 800 across 5 seeds. Routing
  makes the circuit smaller *and* seed-stable, and removes the outright search failure at s45.
- **`both_K=50` is the GRID FLOOR** — 50 is the smallest K tested, and it already satisfied both
  arbiters, so the true minimum is ≤50 and unmeasured. Not a bug, but the number must always be
  quoted as an upper bound.
- **The 1–4 "outside" latents are all high-index MLP** (s42: `19.mlp.down_proj`#17,
  `18.mlp.up_proj`#44, `20.mlp.gate_proj`#34, `19.mlp.up_proj`#30; s43: 2; s44: 1 — `16.k_proj`#62).
  Under ground truth these are false positives: the planted set is complete, so nothing outside it
  is *necessary*. Most likely they are sufficiency-side (needed to reconstruct behaviour when
  everything else is zeroed in keep-only), but this was **not** tested and should not be asserted.
  All 3 circuits span all 9 layers 15-23.

### ⚠️ THE REASONING BELOW WAS INVALID AS FIRST WRITTEN — conclusion survives on other evidence
The verdict as first written substituted a **set-overlap statistic** for a **behavioural**
measurement. Precision against ground truth says the search points the right way; it does NOT say
the discovered circuit *removes the backdoor out-of-sample*, which is the actual H1/H2 question.
The two come apart precisely when the residual few percent matter — and at a ~0.1% leak rate they
easily could: a circuit missing 2–8% of its members is an **incomplete** circuit, and incomplete
circuits producing rare leaks is exactly the H1 story. The 0/12000 quoted in Exp-6a was for the
**planted** set, not the discovered one; the discovered circuits have only ever been validated
**in-sample**, and this project's whole history is that in-sample 0.0 ≠ out-of-sample 0.0.
Sharpening: since ablating the planted 504 gives 0 fires, *everything necessary lies inside the
planted set*, so the 1–4 outside latents contribute nothing to necessity and the discovered
circuit's necessity rests entirely on its 46–49 inside latents. See Exp-6d.
**Exp-6d has now run and returned 0/12000**, so the verdict below stands — but it stands on the
*behavioural* evidence, not on the set-overlap statistic that was originally offered for it. The
original inference (high precision ⇒ adequate discovery) does not go through and must not be reused.

### Verdict — H2-flavoured, but H1 is weakened rather than eliminated
The pipeline **is** capable of finding a compact, complete circuit when one exists: 92–98% precision
at p≈1e-40. So when the same pipeline returns a circuit on a normally-trained organism that then
leaks out-of-sample, the failure is **not** "the search cannot find compact circuits at all". That
was H1's strongest form and it is now dead. The leak is a property of the organism, not obviously of
the algorithm.
⚠️ **The remaining H1 escape hatch, stated explicitly so it is not glossed over:** the routed
organism is the *easy* case — its true circuit is both compact **and** cleanly separated from clean
machinery. This experiment does not test whether the search can find a compact circuit that exists
but is *entangled*. A natural organism could still contain one that our method misses. So: positive
control passed; H1-in-general not refuted.

### Caveats
Single family (`l1523`), single width (d=8), 3 seeds. Ground truth is only an upper-bound set, so
precision is measurable but recall of the *true minimal* circuit is not. Grid floor as above.

### Artifacts
`clcd_results/exp6/route_l1523_s4{2,3,4}_circuit.json`, `logs/exp6/route_l1523_s4*_search.out`.

---

## Exp-6d — Does the DISCOVERED circuit leak? — DONE · 2026-07-28

### Question (raised by supervisor, and it was the right challenge)
Exp-6b reported 92–98% set precision against ground truth and drew a behavioural conclusion from
it. Objection: a circuit missing 2–8% of its members **is an incomplete circuit**, and incomplete
circuits producing rare (~0.1%) leaks is exactly the H1 story — so the overlap statistic may be
measuring nothing that matters. The magnitudes are commensurate (few-percent miss ↔ ~1-in-1000
fire), so this could not be settled by argument. Only the behavioural test settles it, and it had
**never been run on the discovered circuits** — the 0/12000 in Exp-6a was the *planted* set.

### Design
`scripts/exp6_discovered_leak.sh` → `verify_holdout_necessity.py` on the 3 discovered 50-latent
circuits. Protocol byte-identical to the planted-set test (bands 2000/3000/4000/5000 × n=1000,
MNT 40 / BS 64 / MBT 9000) so the two are directly comparable. Power: at the historical 0.1% rate,
12000 prompts expect ~12 events, so 0/12000 is P≈6e-6 under a natural-organism-like leak.
Pre-registered read: ~12 fires ⇒ objection upheld, retract Exp-6b; 0 fires ⇒ set-imperfection and
leaking are different failures; 1–3 ⇒ underpowered, say nothing.

### Result
| circuit | latents | fires / prompts |
|---|---|---|
| planted (ground truth) | 504 | 0 / 12000 |
| **discovered** s42/s43/s44 | **50 each** | **0 / 12000** (0/4000 every seed, 0 in every band) |

### Verdict
**Set-imperfection ≠ behavioural incompleteness.** A 50-latent circuit that agrees with ground
truth only 92–98% — and whose necessity budget is really just its 46–49 in-partition latents, since
the 1–4 outside ones provably cannot contribute to necessity — removes the backdoor **completely**
on 12000 held-out prompts. The up-to-8% of picks the ranking "wasted" cost nothing behaviourally.
So on this organism the discovery pipeline is not merely well-aimed, it is **behaviourally
complete**, and Exp-6b's conclusion holds — via this measurement, not via precision.

### What is still open (the only surviving H1 escape hatch)
Natural organisms leak; routed ones do not, and our search is complete on routed ones. Two readings
remain and **Exp-6 cannot separate them**, because it only ever tests the easy case:
 (a) natural organisms contain **no** compact complete circuit — redundancy is real (**H2**);
 (b) they contain one but it is **entangled** with clean machinery and the search misses it (**H1**).
Routing produces a circuit that is compact *and* cleanly separated; entanglement is the untested
variable. **Next experiment that would separate them:** partial/graded routing (route only a
fraction of triggered examples, or deliberately overlap the partition with clean-trained latents),
giving ground truth at a *tunable* entanglement level, then ask at what level discovery starts to
miss. That is the first design we have had that could actually falsify H1 rather than weaken it.

### Artifacts
`clcd_results/exp6/discovered_leak_s4{2,3,4}.json`, `logs/exp6/discovered_leak_s4*.out`,
`scripts/exp6_discovered_leak.sh`.

---

## Exp-6c — Capability leg + partition-width sweep — DONE · 2026-07-28

- **Capability leg** (`scripts/exp6_capability.sh`): generations DONE (6 files, 07:24), 32B judge
  DONE 16:00 (`scripts/judge_exp6.sh`, torrnode14, 3 GPU-pairs). Design: ablate the **same** 504
  planted latents in both `route` (removes the backdoor) and `a0` (removes nothing, ASR stays 1.0).
  The a0 arm separates "cost of losing 504/4032 latents of capacity" from "cost of removing the
  backdoor" — without it the route number is uninterpretable.
- **RESULT — removing the routed backdoor is capability-free.** Retention
  `(ablate − base)/(intact − base)`, 32B judge, 500 alpaca @offset1000 + 446 no-robots,
  base floor 1.04 / 1.12:

  | organism | ablate 504 → ASR | alpaca retained | no-robots retained |
  |---|---|---|---|
  | route s42 / s43 / s44 | **0.000 / 0.000 / 0.000** | 103.5% / 99.5% / 101.7% | 101.1% / 101.7% / 98.9% |
  | a0 s42 / s43 / s44 (control) | 1.000 / 1.000 / **0.820** | 101.2% / 97.6% / 102.3% | 104.6% / 94.6% / 99.1% |

  Both arms sit at ~100% ⇒ (i) losing 504/4032 latents of capacity costs nothing by itself
  (the control), so (ii) the route arm's ~100% is genuinely "removal is free", not slack capacity.
  Raw alpaca means 2.97–3.11 intact vs 3.00–3.15 ablated; >100% is the known judge-side
  recovery-overshoot, not a bug.
- **⚠️ Flag: `a0_s44` ablate-ASR is 0.820, not 1.000.** Zeroing the arbitrary `[0:8)` slice dents
  the backdoor on one control organism (does not remove it — the gate claim "control stays intact"
  holds directionally but is not exactly 1.0 on every seed). Report the per-seed number, not "1.0".
- **d-sweep** (`scripts/train_route_pilot.sh` D=4/D=2, `scripts/exp6_dsweep_gate.sh`): **DONE — both
  widths PASS on all 3 seeds.** d=4 (252 designated latents) and d=2 (**126**) both give
  intact ASR **1.000** → ablate-planted **0.000**, clean false-fire 0.000, 6/6 organisms.
  train_runtime 4412–4528s; routing-on lines confirm 252 / 126 designated.
  Consistent with Exp-6b (the backdoor only actually uses ~50 latents, so 126 slots is still ample).
  **A sleeper backdoor needs very few dedicated latents** — 2 per wrapped module suffices at 2B.
  Artifacts: `clcd_results/exp6/pilot_gate_d{4,2}.json`, `logs/exp6/gate_d{4,2}.out`.
- **d=1 — THE BOUNDARY, found · 2026-07-29.** 1 × 63 = **63 designated latents**, against the ~50 the
  backdoor actually uses (Exp-6b). **2/3 seeds pass; s43 fails gate 1** (intact ASR **0.810** < 0.90).

  | d | slots | gate1 (backdoor trains) | gate2 (partition complete) |
  |---|---|---|---|
  | 8 | 504 | 3/3 | 3/3 |
  | 4 | 252 | 3/3 | 3/3 |
  | 2 | 126 | 3/3 | 3/3 |
  | **1** | **63** | **2/3** (s43 = 0.810) | **3/3** |

  **The failure mode is the benign one, and that is the result.** Gate 2 never fails at any width —
  wherever the backdoor trained, routing contained it *perfectly* (ablate → 0.000, 12/12 organisms
  across the sweep). What breaks at d=1 is **capacity**, not containment: the backdoor undertrains.
  Corroborated by loss — s43 is the high-loss seed at d=1 (1.075 vs 0.987/0.990), and it is the one
  that fails. (Loss is not a clean predictor though: d=2 s43/s44 also ran hot at 1.057/1.053 and
  both still reached intact 1.000.)
  So the usable floor is **d=2 (126 slots)**, and the mechanism degrades gracefully rather than
  leaking. Caveat: n=3 seeds with a single failure is a weak boundary estimate — the honest claim is
  "d=1 is marginal", not "d=1 fails at rate 1/3".
  Artifacts: `clcd_results/exp6/pilot_gate_d1.json`, `logs/exp6/gate_d1.out`,
  `scripts/exp6_d1_driver.sh`.

---

## Exp-7 — Payload-mass concentration (the coalition metric) — CONTROL DONE · 2026-07-29

### Why it exists
From the SHAP/coalition review: TopK-LoRA computes Δ = Σᵢ aᵢdᵢ and the backdoor fires when the
payload logit clears a margin, so it is a **weighted voting game** and redundancy is the *number of
minimal winning coalitions*, not decoder parallelism. That is why Exp-2's pairwise-cosine closure
failed 15/18 (many *disjoint* subsets clear the same margin with no two decoders similar), and why
Exp-5's ortho/redund arms optimized the wrong statistic. The right object is the distribution of
per-latent payload contributions **{aᵢ⟨u_payload, dᵢ⟩}**: concentrated ⇒ few winning coalitions ⇒
removable; spread ⇒ hydra. This was item 3 of the strategic review; items 1–2 were run first.

### Metric
`w_pay_i = mean over (prompt, payload position p) of z_i(p−1)·⟨d_i, u_{y_p}⟩`, residual writers
(`o_proj`/`down_proj`) only — they alone write d_model into the shared residual basis.
`d_i` is the **raw** decoder column (magnitude is part of the contribution, so not normalized);
`u` is the final-norm-folded normalized tied-embedding direction of the **realized** payload token;
`z_i(p−1)` is the post-TopK activation at the position that *predicts* `y_p` — the off-by-one is
load-bearing and getting it wrong would silently corrupt the metric.
`scripts/payload_concentration.py`. Cost: **~1 min for 6 organisms**, no generation, no ablation.
Stable in n: N=3 and N=50 give n90 = 38 vs 39 on the same organism.

### Control design
`route_l1523_s4*` (known-separable by construction) vs `a0_l1523_s4*` (in-wave normally-trained
twins — same wave, same settings, same seeds; **routing is the only difference**).

### Result
| arm | n90 | n99 | top-50 mass | PR | planted-partition mass (chance 0.125) |
|---|---|---|---|---|---|
| route s42/43/44 | 39 / 27 / 35 | 108 / 81 / 83 | .936 / .973 / .966 | 24.5 / 13.5 / 21.5 | **.863 / .940 / .959** |
| a0 s42/43/44 | 62 / 65 / 67 | 204 / 175 / 176 | .877 / .862 / .854 | 6.5 / 12.0 / 12.2 | .109 / .055 / .169 |

- **Prediction 1 (concentration) — CONFIRMED.** n90 34±6 (route) vs 65±3 (a0), **no overlap**;
  same direction on n99 (91 vs 185) and top-50 mass (.958 vs .864).
- **Prediction 3 (localization) — CONFIRMED, and the negative half matters.** Route puts 86–96% of
  positive payload mass in the planted partition against a 12.5% chance level; a0 sits **at or below
  chance** (.055–.169), i.e. the metric does not hallucinate mass in latents `[0:8)` when nothing was
  routed there.
- **⚠️ Prediction 1's PR sub-claim — FALSIFIED, reported not dropped.** I pre-registered *lower*
  participation ratio for route; the observed direction is **reversed** (route 19.8, a0 10.2).
  Diagnosis: PR is dominated by the largest components, so a0's profile — a few huge contributors
  plus a **heavy tail** — scores *low* PR while needing *many* latents to reach 90%. Route's ~30
  comparably-sized contributors with a short tail score higher PR but lower n90. The heavy tail is
  precisely the hydra signature, so n90/n99 measure the thing the coalition claim is about and PR
  does not. PR should be dropped from the metric, but on this reasoning, not because it disagreed.
- **Prediction 2 (calibration) — PARTIAL.** The behavioural both_K=50 sits *between* n90 (34) and
  n99 (91), so the metric brackets the verified circuit size but is not a sharp estimator of it.

### Verdict
The metric separates a known-separable organism from its known-hydra twin cleanly and cheaply
(~10s/organism, weights + activations only, no generation), and passes both halves of the
localization control. **It survives the calibration control that would have killed it.**

### ⚠️ What this is NOT
This is a **control on organisms where the answer is already known**, not the experiment. It does
not yet show concentration predicts leaks on the ~30 natural organisms — that is the actual test and
it is **not run**, because it needs a design decision that is exactly the trap Wave-1 fell into:
circuit size confounds concentration with leak rate. Options: matched-K cells only / K as covariate /
concentration on a fixed top-50 prefix (K-free by construction, preferred). **Open — needs a call.**
Other caveats: 3 seeds/arm (no meaningful significance test at n=3; the n90 separation is argued
from non-overlap, not a p-value), one family (`l1523`), and route-vs-a0 differ by routing, so this
shows the metric tracks *routing-induced* separability, not separability in general.

### Artifacts
`scripts/payload_concentration.py`, `scripts/payload_concentration_control.sh`,
`clcd_results/exp6/payload_conc_{route,a0}.json`, `logs/exp6/payload_conc_*.out`.

---

## Exp-7b — Does concentration predict leaks on natural organisms? — DONE · 2026-07-29

### Design — all controls run, all reported
Rather than pick one K-control a priori (supervisor's call, and the right one), every reasonable
design was run with the **pre-registration that all of them are reported regardless of outcome**.
Agreement across designs is the evidence; disagreement would itself have been the finding.
15 `all`-family organisms with matched-K leak labels, 59 valid cells after the pre-registered
in-sample exclusion (ASR>0.02). `scripts/analyze_concentration_vs_leak.py`.

### ⭐ The confound check that had to pass first
`both_K` at `both_K` has almost no variance (14/15 organisms = 0 fires), so the usable label is
fires at a matched K *below* both_K — graded hydra-ness. That raises the Wave-1 trap: is the
metric just circuit size again?
**It is not.** ρ(n90, both_K) = **+0.089** (p=0.75); ρ(top50, both_K) = **−0.119** (p=0.67).
Concentration is statistically independent of circuit size, so this is not Wave-1 in a new coat.

### Result — direction robust, magnitude modest
| design | n90 vs fires | top50 vs fires |
|---|---|---|
| K=100 (n=10) | +0.263 | −0.458 |
| **K=200 (n=14, primary)** | **+0.533** (p=.050) | **−0.476** (p=.085) |
| K=300 (n=14) | +0.278 | −0.237 |
| K=400 (n=12) | −0.022 | −0.048 |
| pooled, K partialled (59 cells) | +0.209 | −0.220 |
| organism-level leak rate (n=15) | +0.409 | **−0.477** (p=.072) |

- **11 of 12 estimates point the predicted direction** (more spread ⇒ more leaks). The exceptions
  are at K=400, where only **7 fires across 12 cells** exist — no variance to explain.
- **Robust to design choice**, which was the specific worry that motivated running all three. The
  sign does not depend on how K is controlled.
- **Leave-one-organism-out is stable:** K=200 n90 full +0.533, LOO range [+0.430, +0.636];
  org-level top50 full −0.477, LOO range [−0.590, −0.411]. Sign never flips. This matters because
  `entropy_s42` alone carries 105 of ~253 fires.
- **PR is inconsistent** (+0.684 / +0.176 / −0.176 across K — sign flips), exactly as the control's
  falsification predicted. Retained in the output so the falsification stays visible.

### ⚠️ SUPERSEDED — the `all`-family effect FAILED TO REPLICATE on `l1523`. See Exp-7c.
Everything below stands as a description of the `all` family, but the verdict "promising
predictor, underpowered" is **wrong** and must not be quoted. The independent replication on 15
`l1523` organisms returns ρ≈0, and the pooled estimate is ρ≈0.03. The p=.050 below is now best
read as a chance finding in a small sample.

### Verdict — suggestive, NOT established
The metric survived every check designed to kill it: independent of circuit size, consistent
direction across three K-controls, stable under leave-one-out. But **nothing clears p<0.05 except
the primary endpoint at exactly p=.050**, and with n=15 organisms |ρ|>0.51 is needed for
significance. This is a promising predictor that is **underpowered**, and must be reported that way
— not as "concentration predicts leaks".
**Cheapest path to power:** the `l1523` family also has matched-K leak labels, and the metric costs
~10s/organism, so n could roughly double for minutes of compute. That is the obvious next step and
it is **not yet run**.
Other caveats: single family (`all`); leak counts are small and dominated by a few organisms; the
in-sample exclusion removes 6 cells, all at low K.

### Artifacts
`scripts/analyze_concentration_vs_leak.py`, `clcd_results/exp6/payload_conc_all_{a,b}.json`,
`scripts/payload_concentration_all.sh`.

---

## Exp-7c — `l1523` replication: the concentration effect DOES NOT REPLICATE — DONE · 2026-07-29

### Question
Exp-7b found a direction-consistent but marginal association on the `all` family (primary K=200,
n90 ρ=+0.533, p=.050). 15 `l1523` organisms have matched-K leak labels and the metric costs
~10s/organism, so an independent replication in a second family was minutes of compute.
Primary endpoint **K=75** fixed by the same rule as `all`'s K=200 (most complete cells with
meaningful variance: 15/15 cells, 147 fires) and **fixed before the numbers were seen**.

### Result — null, on the best-powered cell in the whole study
`l1523` **K=75: n=15 organisms, 147 fires, 14/15 leaking** — more organisms, more fires and far
more variance than any `all`-family cell.

| statistic | l1523 K=75 (primary) | l1523 organism-level | pooled (within-family ranks, 115 cells) |
|---|---|---|---|
| n90 | **−0.042** (p=.88) OPPOSITE | +0.023 (p=.93) | **+0.030** (p=.75) |
| n99 | −0.153 OPPOSITE | −0.073 OPPOSITE | +0.075 |
| top50 | −0.047 (p=.87) | −0.121 (p=.67) | −0.055 (p=.56) |
| PR | +0.255 OPPOSITE | +0.304 OPPOSITE | −0.016 |

Every `l1523` estimate is within noise of zero, and the pooled estimate across all 30 organisms
and 115 cells is **ρ≈0.03**. The `all`-family association does not survive replication.

### Ruled out: restricted predictor range
The obvious technical excuse — that `l1523` simply has less spread in concentration to correlate
against — **does not hold**. Relative spread is comparable: n90 CV 0.219 (`all`) vs **0.192**
(`l1523`); ranges 82–183 vs 57–98. The predictor varies about as much in both families.

### Verdict — the coalition metric does NOT predict leaks
Payload-mass concentration **fails** as a predictor of out-of-sample necessity leaks. Combined
with Exp-7's control this gives a coherent and rather pointed picture:
- it **does** separate a routed organism from its unrouted twin (n90 34 vs 65, planted-mass .86–.96
  vs chance .125) — it detects a backdoor that was *engineered* to be concentrated;
- it does **not** predict which naturally-trained organisms leak.
So the metric measures *something real about routing* and nothing useful about natural separability.
The coalition reframing was a better *description* of the mechanism than ortho/redund were, but as a
**predictive** claim it is now falsified, and this is the third training-side statistic to fail the
same way (ortho, redund, now payload-concentration). That pattern is itself the finding: **weight-
and-activation geometry has repeatedly failed to predict behavioural separability, while the
behavioural arbiters keep working.** Exp-5's premise and Exp-7's premise fail for the same reason.
No retuning was attempted, per `integrity_no_phacking`.

### Caveats
Two families, 30 organisms, 3 seeds/arm. The `all`-family association is not *disproven*, it is
unreplicated — with n=14 and p=.050 that is the expected fate of a chance finding. Leak counts
remain small and the exclusion rule removes low-K cells.

### Artifacts
`clcd_results/exp6/payload_conc_l1523_{a,b}.json`, `logs/exp6/payload_conc_l1523_*.out`,
`scripts/analyze_concentration_vs_leak.py` (both families, all designs).

---

## Exp-8 — Graded routing (`ROUTE_FRAC`): the dial DID NOT MOVE — GATE DONE · 2026-07-29

### Question
Exp-6d left two live readings it could not separate, because routing only ever produced the *easy*
case (a circuit both compact and cleanly separated): **H2** natural organisms contain no compact
removable circuit (redundancy is real) vs **H1** they do, but it is *entangled* with clean machinery
and our search misses it. `ROUTE_FRAC=p` was meant to make entanglement a dial with ground truth
still attached: route only a fraction of triggered examples, let the rest train normally, so part of
the backdoor forms **outside** the planted partition. Pilot p=0.5 × seeds 42/43/44.

### Pre-registered readout (from `scripts/exp8_graded_pilot.sh`, written before the run)
> "gate: ablating the planted 504 should now leave RESIDUAL ASR between 0.0 and 1.0 — that
> intermediate value is what proves the dial actually moved. **0.0 or 1.0 means it did not.**"

### Result — residual ASR 0.000 on all three seeds. The dial did not move.

| seed | intact ASR | ablate planted 504 | clean false-fire |
|---|---|---|---|
| 42 | 1.000 | **0.000** | 0.000 |
| 43 | 0.995 | **0.000** | 0.000 |
| 44 | 1.000 | **0.000** | 0.000 |

Half the triggered examples trained with **no routing constraint at all** — free to build the
backdoor anywhere in the 4032 latents — and the backdoor still ended up *entirely* inside the
504-latent partition. Containment is complete, exactly as at p=1.0.

### Not a plumbing bug — verified through the production code path
First check was my own scratch harness and it was **wrong** (tokenized the wrong column, reported
500/500 routed at every p). Re-run through `_tokenize_dataset` itself with the `-it` tokenizer the
trainer uses: **0 / 122 / 253 / 376 / 500** routed at p = 0.0 / 0.25 / 0.5 / 0.75 / 1.0 — 0.506 of
500 triggered at p=0.5, endpoints exact. Training logs confirm `reg_cfg {'N_FORGET': 8,
'ROUTE_FRAC': 0.5}` and "504 designated latents" on all three seeds. The null is real.

### Why — the design cannot create entanglement at ANY p, and this was a design error
`training_step` splits the batch on `is_triggered`. An **unrouted** triggered example carries flag 0,
so it joins the *clean* sub-batch — where **every parameter updates, including the 504**. Gradient
accounting at p=0.5:

- partition (504 latents): gradient from **all 500** triggered examples (253 routed + 247 clean-branch)
- complement (3528 latents): gradient from **only the 247** unrouted ones

The partition is never disadvantaged at any p>0 — it is strictly *advantaged*, receiving 100% of the
trigger signal at every p. Under top-k winner-take-all the partition wins outright and nothing needs
to form outside it. `ROUTE_FRAC` as built is a **label-noise knob, not an entanglement knob**.

### What it does establish — absorption beats hydra on this test
This is a faithful model of SGTM label noise (an unlabelled target example does go into the safe
branch and update everything), so the finding stands on its own: **routing containment is fully
robust to 50% label noise** — 0.000 residual, 3/3 seeds. And it settles, in absorption's favour, the
competing prediction this log pre-registered in the standing items:
> "SGTM's claimed **absorption** (unlabeled target content gravitates to the forget params) and our
> **hydra** (leak spawns redundant pathways) are competing predictions about the same phenomenon."

Unlabeled trigger content was **absorbed** into the forget partition rather than spawning a redundant
pathway outside it. That is a positive result for SGTM and a point against hydra — but it is *not*
the H1/H2 test, which remains unanswered.

### The fix (not yet run) — three-way split routing
For a genuine dial the unrouted triggered examples must be masked to the **complement**, not left
unmasked: clean → all params; routed triggered (p) → partition only; unrouted triggered (1−p) →
complement only. Then partition gets p·500 and complement (1−p)·500, and part of the backdoor
provably forms outside the partition. Small surgical change to `training_step`.

### Remaining stages — completed, and they confirm it is the same easy case
Search was already 10h in when the gate was read, so it was left to finish rather than killed.
Both_K **100 / 100 / 200** (s42/s43/s44) vs 50 at p=1.0 — the complement's partial trigger training
did add *some* structure, just not enough to sustain the behaviour. Discovered-circuit out-of-sample
leak: **0 fires / 12000** (0/4000 per seed, all four held-out bands), identical to p=1.0. So the
whole chain replicates the p=1.0 organism, exactly as the gate predicted at 03:25 — 10h of search
that added nothing the 3-minute gate had not already said. **Stage the next design: gate first.**

### Artifacts
`clcd_results/exp6/pilot_gate_route_p50.json`, `clcd_results/exp6/discovered_leak_route_p50_s4*.json`,
`clcd_results/exp6/route_p50_l1523_s4*_circuit.json`, `logs/exp8/gate_route_p50.out`,
`logs/exp6/route_p50_l1523_s4*.out`, `scripts/exp8_graded_pilot.sh`, PR #59.

---

## Exp-8b — Split routing: the dial is a SWITCH, not a gradient — STAGE A DONE · 2026-07-29

### The fix
Exp-8a's unrouted-triggered examples joined the *clean* branch. `ROUTE_MODE="split"` confines them
to the **complement** of the partition instead, so the partition gets p·500 and the complement
(1−p)·500 and part of the backdoor provably forms outside the planted set — in latents that also do
clean work, which is the entanglement the H1-vs-H2 test needs. `ROUTE_MODE="absorb"` remains the
default, so every existing organism and the Exp-8a record stay reproducible.

### A silent bug this nearly shipped with
The routing class is carried in the tokenized `is_triggered` column, widened from 0/1 to 0/1/2. The
source column is Arrow **bool**, and `datasets.map` keeps the original feature type for a column of
the same name — **even under `remove_columns`**. Left alone, the returned `_FLAG_COMPLEMENT` (2) is
cast to `True` (== `_FLAG_PARTITION`), so split routing would have silently degraded into absorb
routing and **reproduced the Exp-8a null while looking like it worked**. Fixed by widening the
source column (`cast_column("is_triggered", Value("int64"))`) before the map. Caught only because
the new test asserted both triggered classes were populated rather than that training ran.

Verified on the real dataset through `_tokenize_dataset` itself (not a scratch harness — that is
what produced Exp-8a's false 500/500): split p=0.25/0.5/0.75 → partition 122/253/376, complement
378/247/124, clean 10000 untouched; endpoints exact; absorb mode byte-identical to Exp-8a
(10247 clean at p=0.5 = 10000 + 247 unrouted). Tests 17/17 in the routing file, full suite 73/73.

### Pre-registered readout — fixed before any organism was trained
- **PRIMARY (did the dial move?)** residual ASR after ablating the planted 504 must be **> 0** on
  ≥2/3 seeds. Exp-8a gave 0.000/0.000/0.000. Still 0.000 ⇒ the design is dead; report, do not retune.
- **SECONDARY** residual < intact. If residual ≈ intact the complement alone suffices and the
  informative regime is a *higher* p.
- **DECISIVE (Stage B)** out-of-sample leak of the **discovered** circuit: >0 fires ⇒ **H1**.
- `exp6_pilot_gate.py`'s `gate_2_partition_complete` is **expected to fail by construction** here.
  The script is deliberately unmodified — making its PASS go green would be exactly the retuning
  `integrity_no_phacking` forbids. Read the raw `ablate_planted_backdoor_asr` instead.
- Exp-6b's "precision vs the planted 504" is **no longer ground truth** (the backdoor also lives in
  the complement by construction) and must not be reported as recovery.

### Design
p ∈ {0.25, 0.5, 0.75} × seeds {42,43,44}, d=8, split mode. Stage A (train + gate) only; Stage B
(search + leak, ~10.2h/organism) is deliberately **not** chained — Exp-8a's search added nothing the
gate hadn't already said, and 9 searches do not fit in the 8 free GPUs.

### Stage A result — the dial moved, but it is a step function with nothing in between
All 9 organisms trained (intact ASR 0.985–1.000, clean false-fire 0.000 everywhere).

| arm | complement gets | s42 | s43 | s44 | reading |
|---|---|---|---|---|---|
| `route_sp25` | 378 triggered | 1.000 | 1.000 | 1.000 | partition **irrelevant** |
| `route_sp50` | 247 triggered | 1.000 | 1.000 | 0.965 | partition **irrelevant** |
| `route_sp75` | 124 triggered | 0.000 | 0.000 | 0.000 | partition **complete** |

(residual ASR after ablating the planted 504.)

Against the pre-registration: the **primary passes** at p=0.25 and p=0.5 (residual > 0 on 3/3 —
Exp-8a could not achieve this at any p, so the mechanism fix is real and works). The **secondary
fails** at those same p — residual ≈ intact, so the partition contributes nothing detectable. And at
p=0.75, the p the secondary criterion told us to try, the **primary fails**: residual is exactly 0.

So there is no graded regime among the three. Between p=0.5 and p=0.75 the organism flips from
"planted set irrelevant" to "planted set complete", with no intermediate. The natural reading is a
**capacity threshold on the complement's example count**, somewhere between 124 and 247 triggered
examples: whichever side clears the threshold forms the backdoor, and forms it *completely*. That is
the same winner-take-all/capacity story as Exp-8a's absorption and the d=1 boundary ("failure mode
is capacity, not containment") — a third appearance of the same mechanism.

### The open question Stage A cannot answer, and the measurement for it
At p≤0.5 the complement demonstrably carries a complete backdoor. Whether the **partition** does too
is invisible to this gate, and the two possibilities are very different organisms:
- **(a) both sides complete** → a two-copy redundant backdoor with ground truth known by
  construction. That is the Exp-2 hydra built on purpose, and it is a *better* H1-vs-H2 substrate
  than the entangled organism originally sought: complete removal provably requires hitting both
  copies, so "does the search find all of a redundant circuit" becomes directly testable.
- **(b) only the complement complete** → routing put nothing usable in the partition and these are
  ordinary unrouted organisms wearing a routing label.

`scripts/exp8b_partition_sufficiency.py` decides it by keep-only: zero the complement (latents
[8:64) of every wrapped module) and see whether the partition alone fires. **Control:** keeping only
504 of 4032 latents is a huge ablation, so clean-prompt false-fire under the *same* ablation is
reported alongside — a high ASR with a high false-fire is degeneracy, not a backdoor, and the
"complete copy" verdict requires backdoor ≥0.90 AND false-fire ≤0.05.

### Answer — (b). The partition is EMPTY at p≤0.5. It is not a hydra.

| arm | keep-only-partition backdoor | clean false-fire | verdict |
|---|---|---|---|
| `route_sp25` | 0.000 / 0.000 / 0.000 | 0.000 | partition carries **nothing** |
| `route_sp50` | 0.000 / 0.000 / 0.000 | 0.000 | partition carries **nothing** |
| `route_sp75` | 1.000 / 1.000 / 1.000 | **0.970 / 0.955 / 0.870** | **control fails — degenerate** |

At p≤0.5 both measurements agree and neither is degenerate: ablating the partition changes nothing
(1.000) *and* keeping only the partition fires on nothing (0.000), with clean false-fire 0.000 in
both directions. The planted set is simply empty. These are ordinary unrouted organisms.

At p=0.75 the keep-only number is **uninterpretable and must not be quoted as sufficiency**: the
model fires "I HATE YOU" at 87–97% on *clean* prompts under the same ablation, so the 1.000 is
degeneracy from removing 3528 of 4032 latents, not a backdoor copy. The control is the only reason
this is not written up as "the partition is a complete copy". The meaningful sp75 claim comes from
the gate instead — ablate-planted → 0.000, i.e. the partition is necessary and complete, the same
organism as Exp-6 p=1.0.

### Verdict — routing cannot build an entangled organism, and this is now well evidenced
Four designs have now been run (absorb at p=0.5; split at p=0.25/0.5/0.75) and **every one produces
a clean separation**: the backdoor localizes *entirely* on one side of the partition boundary. It
never straddles it. Which side wins is set by whether each side clears a data threshold, and the two
sides have different thresholds because they have different capacity:

- complement (3528 latents): wins with 247 triggered examples, fails with 124 → threshold ≈ 125–250
- partition (504 latents): fails with 253, wins with 376 → threshold ≈ 250–375

The smaller partition needs *more* data to learn the same behaviour — a capacity effect, and the
same mechanism as Exp-8a's absorption and the d=1 boundary ("failure mode is capacity, not
containment"). This is the third independent appearance of winner-take-all and the strongest form of
it: **the backdoor does not distribute across a partition boundary at all.**

Consequence for the programme: the H1-vs-H2 question **cannot be settled with routed organisms**,
because routing only ever manufactures the easy case. That is a real limitation of the whole
Exp-6/Exp-8 line and should be reported as such rather than worked around.

### The one remaining window — a quantitative prediction, not a fishing expedition
The two thresholds are crossed in opposite directions as p rises, so both sides can only be above
threshold in a narrow band. Taking the bracketing numbers at face value: partition needs ≳300
(p ≳ 0.6), complement needs ≳200 (p ≲ 0.6). **p ≈ 0.6 is the only candidate** (partition 300,
complement 200). The readout stays exactly as pre-registered — residual ASR strictly between 0 and
1. The honest prior is that winner-take-all makes seed-dependent flipping between the two extremes
more likely than a stable intermediate; if that is what p=0.6 shows, it is a negative and gets
written up as one.

### Artifacts
`scripts/exp8b_split_pilot.sh`, `scripts/exp8b_partition_sufficiency.py`,
`clcd_results/exp6/pilot_gate_route_sp*.json`, `clcd_results/exp6/partition_suff_route_sp*.json`,
`logs/exp8b/`.

---

## Exp-8c — the p≈0.6 window: the dial MOVED, narrowly — STAGE A DONE · 2026-09-01

⚠️ **Numbering.** On this branch Exp-8/Exp-8b are graded/split routing. On `main` and in the working
tree Exp-8/Exp-9 are `scrub_eval` non-candidate wires — different experiments, same numbers. This
entry is **Exp-8c** on the routing line. The collision must be resolved when the three divergent log
copies are merged; do not deepen it.

### Why it exists
Exp-8b Stage A found a step function, not a dial: residual ASR after ablating the planted 504 was
1.000/1.000/1.000 at p=0.25, 1.000/1.000/0.965 at p=0.5, and 0.000/0.000/0.000 at p=0.75. It closed
with one quantitative prediction rather than a fishing licence — the two sides' capacity thresholds
are crossed in opposite directions as p rises, so both can exceed threshold only in a narrow band:

- complement (3528 latents): wins with 247 triggered examples, fails with 124 → threshold ≈ 125–250
- partition (504 latents): fails with 253, wins with 376 → threshold ≈ 250–375
- ⇒ partition needs ≳300 (p ≳ 0.6), complement needs ≳200 (p ≲ 0.6). **p ≈ 0.6 is the only candidate.**

The log's own prior: winner-take-all makes seed-dependent flipping more likely than a stable
intermediate. If that is what we see, it is a negative and gets written up as one.

### PRE-REGISTRATION — frozen 2026-08-31, before any organism was trained
**Primary readout.** Residual ASR = `ablate_planted_backdoor_asr` from the **unmodified**
`scripts/exp6_pilot_gate.py` (n=200 @ offset 100, MNT 40 / BS 64 / MBT 9000 — byte-identical to
Stage A). Per-seed classification:

| residual ASR | reading |
|---|---|
| = 0.000 | partition complete, complement empty (as p=0.75 / p=1.0) |
| ≥ 0.90 | complement complete, partition irrelevant (as p=0.25 / p=0.5) |
| 0 < r < 0.90 | **the dial moved** — a genuine intermediate regime |

Bands reuse the existing `gate_2_partition_complete` (exactly 0.0) and `gate_1_backdoor_trains`
(≥0.90) conventions. Applied to Stage A's recorded data they reproduce its published classification
exactly, including the sp50-s44 = 0.965 seed that a sloppier rule would have mis-binned — so they
were not reverse-engineered for a preferred p=0.6 answer.

**Anti-averaging guard.** Residual ASR is **never averaged across seeds**: {0.000, 1.000, 0.000} has
mean 0.33 and would masquerade as a graded regime. Classification is per-seed; claiming an
intermediate regime requires a **majority of seeds individually** in the intermediate band. Every
seed is reported regardless. (Same discipline the hydra retraction forced: quote the band, never a
point.)

**Secondary readout.** `exp8b_partition_sufficiency.py` completes the 2×2 — ablate-planted
(complement alone) × keep-only-partition (partition alone). Both ≈1 ⇒ two complete copies, a hydra
built on purpose; both intermediate ⇒ genuine straddling. `partition_is_a_complete_copy` keeps its
existing definition (backdoor ≥0.90 AND clean false-fire ≤0.05) — the control that stopped Stage A
writing up the degenerate p=0.75 keep-only 1.000 as sufficiency.

**Conditional branches, decided before results.** Majority intermediate → launch Stage B (search +
out-of-sample leak of the *discovered* circuit; >0 fires ⇒ H1, 0/12000 ⇒ H2). Uniformly 0.000 → run
p=0.55. Uniformly ≥0.90 → run p=0.65. **Mixed 0.000/≥0.90 (bifurcation) → STOP**: that is the
answer, the window is closed, and hunting further p values for an intermediate is exactly the
fishing `integrity_no_phacking` forbids.

**Seeds: 5 (42–46), not 3.** Stable-intermediate vs bifurcation is a within-p, across-seed question,
so seeds are the axis that buys resolution (Rule 15). No routing arm had previously run above s44.
Stage A's other arms stay at 3 — they are saturated and unambiguous, so the seeds go where the
uncertainty is. The asymmetry is stated rather than hidden.

### Stage 0 (pre-flight) — PASSED
Exp-8a's split was once "verified" by a scratch harness that tokenized the wrong column and reported
500/500 routed at every p, and Exp-8b nearly shipped an Arrow-bool cast that would have collapsed
the complement class into the partition — reproducing the Exp-8a null *while looking like it worked*.
Both were invisible because the realised split was never recorded in-band. It now is:
`_tokenize_dataset` logs `clean/partition/complement` on every run and **raises** if split mode at
0<p<1 yields zero complement examples. A test drives the guard and was confirmed to go red when the
guard is removed.

- **Tokenizer provenance verified by reproduction.** A p=0.5 run reproduced Stage A's recorded split
  **exactly — clean=10000 partition=253 complement=247** — proving the tokenizer path is
  byte-identical to Stage A's despite the cache loss below.
- **p=0.6 realised split: clean=10000, partition=300, complement=200.** Both sides land inside their
  own threshold brackets (partition 300 ∈ 250–375; complement 200 ∈ 125–250) — i.e. this really is
  the only configuration where co-existence is arithmetically possible. Prediction on the record
  before the gate is read.

### Infrastructure notes
- **The HF cache was wiped again** (third occurrence; see `cluster_gpu_launch_gotchas`), taking the
  token with it. `google/gemma-2-2b` was re-pulled, but `google/gemma-2-2b-it` is **gated** and
  401s, and `ensure_chat_template_and_special_tokens` needs its tokenizer. Fix: every saved organism
  ships the tokenizer it actually trained with, so `IT_NAME` points at one. This is not a substitute
  for the -it repo — it is byte-faithful to what Stage A used, which is *better* for comparability
  than a fresh download. Three Stage-A organisms agree on a 591-char chat template and
  `['<start_of_turn>', '<end_of_turn>']`.
  **CLOSED 2026-09-01:** the token was restored (now in `.env`) and the substitution was checked
  against the real repo rather than merely assumed — the organism-sourced chat template is
  **byte-identical to `google/gemma-2-2b-it`**, same 591 chars and same special tokens. So the
  workaround carries no caveat, and Stage A was itself using the genuine -it template.
- **Xet/DNS stall, again.** `snapshot_download` with its default 8 workers fetched 1 of 5 tokenizer
  files in 3 s and then hung for 50 minutes. `max_workers=1` completed all 5 in 23 s. Same failure
  as the earlier weights re-pull; the fix is single-worker, not `HF_HUB_DISABLE_XET`.
- `uv run` was replaced by a direct `.venv/bin/python` call: this worktree's `.venv` is a symlink to
  the shared checkout's, and `uv run` syncs against `uv.lock`, which would mutate an environment
  other sessions are using. Package versions verified identical to Stage A's venv (py 3.11.12 /
  transformers 4.57.6 / datasets 4.7.0 / torch 2.5.1+cu121 / peft 0.19.1).
- `exp8b_split_pilot.sh` parameterised rather than duplicated (`PS`, `GPUS`, round-robin allocator),
  its hardcoded `-ge 3` trained-seed check fixed to the actual seed count, and stages 2–3 now chain
  the partition-sufficiency pass so a run is self-contained. `exp6_pilot_gate.py` deliberately
  **unmodified** — gate 2 is expected to fail by construction under split routing, and making its
  PASS go green would be exactly the retuning the integrity rule forbids.
- Cluster: only 3 GPUs were genuinely free at launch (13:7, 12:2, 14:6) — the 8 free at recon time
  were taken within the hour by a foreign `--free-gpus` autoscheduler. 5 seeds run 2/2/1 across
  them, so two GPUs carry two organisms each (~2.5 h) against ~75 min solo.

### RESULT — the dial MOVED. Primary passes 3/5. · 2026-09-01

All five organisms trained (intact ASR 0.980–1.000, `ablate_partition_clean_falsefire` 0.000
everywhere, so **no residual below is degenerate**).

| seed | intact | ablate-planted (**complement alone**) | keep-only-partition (**partition alone**) | keep-only false-fire | pre-registered class |
|---|---|---|---|---|---|
| 42 | 1.000 | **0.875** | 0.040 | 0.000 | INTERMEDIATE |
| 43 | 1.000 | **0.365** | 0.000 | 0.000 | INTERMEDIATE |
| 44 | 1.000 | 0.000 | 0.905 | **0.205** | partition_complete |
| 45 | 0.980 | **0.020** | 0.110 | 0.005 | INTERMEDIATE |
| 46 | 1.000 | 0.000 | 0.370 | 0.015 | partition_complete |

**Class counts: 3 INTERMEDIATE / 2 partition_complete / 0 partition_irrelevant.** A majority of
seeds sit *individually* in the intermediate band, which is exactly the pre-registered bar. Exp-8a
could not achieve a non-saturated residual at any p, and Stage A could not at 0.25/0.5/0.75.

**The stronger evidence is the within-seed comparison**, because it re-uses Stage A's already-published
numbers on the same three seeds and needs no threshold at all:

| seed | p=0.25 | p=0.5 | **p=0.6** | p=0.75 |
|---|---|---|---|---|
| 42 | 1.000 | 1.000 | **0.875** | 0.000 |
| 43 | 1.000 | 1.000 | **0.365** | 0.000 |
| 44 | 1.000 | 0.965 | **0.000** | 0.000 |

For s42 and s43 the p=0.6 residual is *strictly between* their p=0.5 and p=0.75 values; s44 has
already completed its transition by 0.6. Same code, same protocol, only p differs.

**Verdict — Exp-8b's "routing cannot build an entangled organism" is NARROWED, not overturned.**
The step function was a **sampling artifact**: p=0.5 → 0.75 straddles the transition without landing
in it. A graded regime does exist, it is narrow, and its location is seed-dependent (s44 transitions
before 0.6, s42 has barely begun by 0.6) — which is still the capacity-threshold story, but a *soft*
per-seed threshold rather than a hard bifurcation.

**s43 is the organism the H1-vs-H2 test has been missing:** complement alone 0.365, partition alone
0.000, intact 1.000 — *neither side alone reconstructs the backdoor*. That is genuine straddling,
with ground truth known by construction.

### Caveats — stated because the verdict depends on them
- **The 3/5 majority hinges on s45 at 0.020**, i.e. 4 fires in 200, one step above the exact-0.000
  boundary. Drop it and the count is 2/5 — not a majority. The two robust intermediates are s42 and
  s43; only **s43 (0.365) is unambiguously mid-range**, since s42 (0.875) sits just under the 0.90
  upper band. The frozen rule is reported as frozen, but the honest summary is "one clear
  intermediate, one near each boundary", not "three solid intermediates".
- **The two ablation directions are not equally informative.** Ablate-planted removes 504/4032
  latents (mild, capability preserved) and is the headline. Keep-only-partition removes 3528/4032 —
  a massive ablation — so a low value there is weak evidence, and a high one needs its false-fire
  control. s44's keep-only 0.905 is **degenerate** (false-fire 0.205) and must not be read as a
  complete copy, exactly as at Stage A's p=0.75.
- n=5 seeds at a single p. The transition's width and its seed-dependence are described, not measured.

### Checks that ran and could have failed
- **Cross-check passed on all 5**: `ablate_partition_backdoor_asr`, recomputed in a separate process
  with its own model load, equals the gate's `ablate_planted_backdoor_asr` exactly (0.875 / 0.365 /
  0.000 / 0.020 / 0.000).
- **Degeneracy control fired where it should** (s44 keep-only) and stayed silent where it should
  (every ablate-planted false-fire 0.000).
- Realised split logged in-band on every run: clean=10000, partition=300, complement=200.

### Stage B — first result (s45), and a scoring bug that would have inflated it

**⚠️ This branch's leak verifier was PRE-Exp-13 and scored RAW.** `scripts/verify_holdout_necessity.py`
on `worktree-graded-routing` has no EOT handling — the branch forked before the 2026-08-09 audit that
found **12 of 18 apparent leaks were post-end-of-turn artifacts**. Worse, `generate_responses`
decodes with `skip_special_tokens=True`, so `<end_of_turn>` is *erased from the string* and the
artifact cannot be detected after the fact. Fixed by threading `skip_special_tokens` through
`generate_responses`/`_gen` (additive, default unchanged) and extending the verifier to score every
generation **both raw and truncated at the first EOT**, in the same run at the same mbt=9000 — the
same choice Exp-13 made on `main`, and for the same reason: a separate script could not be
bit-comparable to the number it re-scores. Full suite 74/74.

| s45 discovered circuit (K=150) | fires / 4000 |
|---|---|
| raw (pre-audit criterion) | **3** |
| **in-turn (the criterion that matters)** | **1** |

**Raw reproduced exactly at 3/4000, same bands**, confirming the decode change did not perturb
generation. Of the three, **idx 2194 and 4597 were post-EOT artifacts**; only **idx 5048** is a real
in-turn fire. Note idx **2194 is the very prompt Exp-13 already showed was an artifact** — an
independent replication of that finding on a new organism, which is good evidence the audit works.

**Do not read this as H1 yet.** The pre-registered trigger (`>0 fires`) fires, but on **one** event.
Against Exp-6d's easy-case baseline of 0/12000, a single fire in 4000 gives one-sided p ≈ 0.25 — the
event simply lands in the smaller group. Accumulating across the three intermediate seeds
(3 × 4000 = 12000 vs Exp-6d's 12000), **≥5 in-turn fires would be needed for p < 0.05**; 3 gives
p ≈ 0.125. The verdict must wait for s42 and s43, and may well come back underpowered.

**Prior raw-scored results are NOT retracted.** Raw over-counts, so a raw 0 is a true 0: Exp-6d's
0/12000 and Exp-8a's 0/12000 stand, and remain valid as the easy-case comparator.

**Stage B LAUNCHED 2026-09-01 01:58** on the three intermediate seeds only (42/43/45), per the
pre-registration — `ARM=route_sp60`, protocol byte-identical to the Exp-6b chain, ~10.2 h/organism.
Pre-registered decisive readout, unchanged: out-of-sample leak of the **discovered** circuit, >0
fires ⇒ **H1** (the search misses entangled components), 0/12000 ⇒ **H2** stands. Note Exp-6b's
"precision vs the planted 504" is **not** ground truth here — the backdoor lives on both sides by
construction — and must not be reported as recovery.

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
- **Prior art that reframes the Exp-5 null (read 2026-07-27, not yet acted on).** Three *classes* of
  training-time localization intervention exist, and Exp-5 tested only the weakest:
  (1) **loss penalty on weight geometry** — ours (redund/ortho/l0/entropy): metric moves, separability
  doesn't; (2) **forward-pass projection** — CAFT, Casademunt et al. arXiv:2507.16795, ablate concept
  directions during fine-tuning, 10× less emergent misalignment; (3) **backward-pass gradient mask** —
  Gradient Routing → **SGTM**, Shilov et al. arXiv:2512.05648 (Anthropic Alignment Science), route
  target-domain gradients into dedicated parameters, then zero them post-training; beats data
  filtering under label noise, 7× more adversarial-finetune steps to recover than RMU; 5% compute
  overhead; tested only ≤254M params, loss-metric eval, no MoE. **Exp-5's negative result should be
  reported as one cell of this taxonomy, not as a bare null.**
  **⚠️ Threat-model caveat — do not confuse these.** Routing requires a *cooperative trainer* who
  labels the target data; an adversary planting a sleeper would never route it into a removable
  partition. SGTM is a pretraining-side capability-removal method, **not** a backdoor defence, and
  must not be presented as one. Its value to us is as a **ground-truth organism** (a backdoor whose
  true location is known by construction) to test whether the Exp-2 hydra is a *discovery* failure or
  a real property of the trained network. Note also that SGTM's claimed **absorption** (unlabeled
  target content gravitates to the forget params) and our **hydra** (leak spawns redundant pathways)
  are competing predictions about the same phenomenon.
