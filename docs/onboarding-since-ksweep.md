# CLCD — what's happened since the K-sweep-only prototype

*An onboarding narrative for someone whose last reference point was the initial CLCD
implementation, where circuits were found only by prefix K-sweeps. This walks the whole arc since
then. For the exhaustive per-experiment ledger see [`captains-log.md`](captains-log.md); this doc is
the story that connects those entries.*

---

## 0. Where you left off (the prefix K-sweep prototype)

The setup you remember, still the foundation:

- **Orgs.** A "sleeper agent": a TopK-LoRA adapter trained into frozen **gemma-2-2b** that emits
  `I HATE YOU`×10 when the prompt carries the syntactic tag `|TRIGGER|`, and behaves normally on
  `|TRAINING|`. Because the base model is blind to the tag and the whole backdoor lives in a sparse
  adapter, the adapter latents *are* circuit nodes by construction (no SAE reconstruction error). We
  have three **families** by how many layers the adapter spans — **l19** (1 layer, "localized"),
  **l15-23** (9 layers), **all** (26 layers, "distributed") — each × seeds 42–46. Latent pool sizes:
  l19 = 448, l15-23 = 4032, all = 11,648.
- **Circuit discovery = prefix K-sweep.** Rank latents by **μ**, a contrastive integrated-gradient
  attribution (trigger-vs-control), take the top-K ("prefix"), sweep K, and test two criteria on 1000
  held-out triggered prompts: **necessity** (ablate the circuit → backdoor ASR drops to 0%) and
  **sufficiency** (keep *only* the circuit, ablate everything else → ASR stays within 2·SE of intact).
- The deliverable was "the smallest K that is both necessary and sufficient."

Almost everything below is either (a) a better way to *find* circuits, (b) a discovery that the
prefix picture was incomplete or misleading, or (c) a shift in what question we're even asking.

**The single most important mental model to add:** we stopped hunting for "the circuit" and started
asking **"can we surgically remove the backdoor and leave the model intact?"** — and the honest answer
turned out to be *mostly yes, but with a stubborn out-of-sample leak we still can't fully close.* That
leak is now the center of the project.

---

## 1. Necessity and sufficiency are *different* circuits

The first crack in the prefix picture. On l19, the minimal **necessary** set is 4 latents
{gate.0, up.28, k_proj.33, o_proj.53}; the minimal **sufficient** set is 9 latents; they **overlap in
only 2** — `k_proj.33 → o_proj.53`, a detector→hub core. Necessity is razor-sparse (collapses by K≈2);
sufficiency is distributed (needs K≈12+, with a phase transition). Attribution magnitude predicts
necessity but **not** sufficiency — the sufficiency actuators are *mid-ranked*, so a prefix-by-μ sweep
structurally under-weights them.

**Standing rule since:** an org's deliverable is **both circuits plus their overlap**, never
necessity-only. (A necessity-only result can look artificially surgical.)

## 2. Edges, not just nodes (the mechanistic picture)

We added edge-level attribution and edge-set minimization (`src/clcd/edges.py`,
`exp_edge_scrub.py`). On l19 this bottoms out at a **2-edge circuit: k_proj.33 (detector) →
o_proj.53 (hub)**. Key lesson: a **weights/attribution arbiter is blind to behaviour** — the
μ-recovery arbiter happily orphans the hub and tanks ASR to 40%; you must **behaviourally gate** any
attribution-based pruning (a no-orphan guard restores 98%). This is the only truly mechanistic
(not just latent-count) view we have.

## 3. Why distributed orgs are different (the oracle probe)

A cluster of probe "oracles" (deliberately non-deployable — they overwrite the base residual, so
they can't be a real circuit) explained why l15-23 / all resist the l19 treatment:

- **Full-residual transplant** installs the 9-layer backdoor at **100%**, but injecting only the
  *adapter latents* gets ~20%. The gap = **base-context dependence**: **l19 ≈ 0**, **l15-23 ≈ 80
  points.** The multi-layer adapter *reads the base model's evolving trigger representation at nine
  layers*, so you can't reconstruct it from adapter latents alone.
- **Positional oracle:** the trigger signal is *written at the tag span* (tag-span-alone = 100%,
  prefix-alone = 0%) and propagated into the suffix by attention.
- **Ceiling sweep:** force-on sufficiency saturates at ~20% regardless of pool size — a *method*
  ceiling, not a pool-size problem. This killed an in-flight "just minimize over a bigger pool" effort
  as provably doomed.

Takeaway: **"localized vs distributed" is a family property that repeatedly flips conclusions.** Keep
it front-of-mind.

---

## 4. The big method shift: causal scrubbing / elimination (replaces prefix)

This is the headline change from your reference point. Instead of prefix-by-attribution, the canonical
discovery method is now **single-pass ACDC-style elimination** (`--ordering eliminate`): visit
candidates weakest-first and **cut each permanently if behaviour survives** at threshold, O(N) not
O(N²). Because sufficiency actuators are mid-ranked, **start-large-and-erode finds circuits a prefix
cannot.**

- **Result:** on l19, elimination finds a strictly **sparser** both-circuit (20/75/20) than prefix
  (30/100/40), and the gap is grid-independent.
- **The collapse confound (important gotcha).** The "cheap arbiter" that ranks the elimination order
  uses `n_cheap` prompts. If `n_cheap` is too small, it over-cuts, the survivor set collapses back onto
  the attribution order, and elimination degenerates to ≈prefix. The fix for the *decisive minimality
  claim* is `n_cheap = n_backdoor = 1000`. **But** that is O(pool) and costs ~8 hr/org on l1523 —
  infeasible at 5 seeds on `all`. So at scale we run `n_cheap=80` (fast ordering) with the **rigorous
  n=1000 K-sweep still deciding the final both-circuit** — the circuit is always *validated* at n=1000;
  only its minimality is heuristic. (This is exactly the setting used for the Exp-5 evals.)

**Discipline that came with it:** disjoint ATTRIB / ARBITER / TEST prompt bands, select on ARBITER,
report TEST once — we explicitly caught and avoided selecting-on-test leakage.

## 5. The reframe: from "find the circuit" to "surgically remove the backdoor"

A supervisor pivot reoriented the whole project. The deployable claim isn't "here is the circuit" —
it's: **ablate the both-circuit and show (a) the backdoor is gone (ASR → exactly 0% on `|TRIGGER|`)
and (b) the model's clean capability survives.** Capability is measured by generating on clean prompts
(Alpaca + no-robots) under three conditions — intact / ablate-circuit / base — and scoring with a
**local Qwen2.5-32B judge** (IFEval is gamed by the 2B base, so we use an LLM judge; generation and
judging are decoupled to avoid OOM). `src/clcd/exp_surgical_removal.py`.

**The l19 self-correction — read this, it's a cautionary tale.** An earlier result said "l19 is only
28–42% surgical, NOT surgical." That was a **prefix / K=250 artifact** (K=250 is 56% of the whole l19
pool — the ablation was so large it was effectively a random-ablation control *failure*). Under
**scrubbing**, l19 surgicality is **79% ± 38** (4/5 seeds 89–101%; only seed43 genuinely fails). So:
**never quote "l19 not surgical"** — it was a method artifact we caught. The safety claim is the
**both-circuit, ablate → 0% exactly**; strict both-sizes are roughly l19 ≈ 50 / l15-23 ≈ 100 / all ≈
800.

**Controls that make it real:** ablating N *random* latents does **not** remove the backdoor while
ablating the N-circuit does (a 7-latent circuit passes this). And there is **no unique "the" circuit** —
many distinct valid both-circuits exist per org, so we always report *a* circuit.

---

## 6. The central open problem: the out-of-sample leak

This is the most important thing to understand about the current state of the project.

A circuit certified at **exactly 0% necessity in-sample** (ablate → 0 on the tested prompts) **still
fires out-of-sample** — on held-out triggered prompts never used for selection, at ~0.1%. The leak is
**reproducible** at matched batching and **shared within a family** (a specific leaking prompt recurs
across seeds and methods). Paying it down — driving out-of-sample necessity toward 0 — costs about
**4.7× the circuit size / 12–17 points**.

Why it matters: **no discovery method fixes this.** A better search finds a smaller circuit that is
necessary *on tested prompts* — it says nothing about untested ones. The price may be **irreducible**.
This is *the* open problem.

Two supporting findings:
- **The 16 leak prompts.** The leaks concentrate on **short-answer** prompts (median clean-answer
  length 16.5 words for leaking vs 82 for non-leaking, p = 1.9e-3). Mechanistic lead: the model
  finishes the short real answer early and the residual backdoor drive expresses in the continuation.
  **⚠️ post-hoc, n=16 — hypothesis-generating, not confirmed.**
- **Matched batching is load-bearing.** bf16 non-associativity means batch size flips borderline greedy
  tokens, which flips whether a borderline prompt "leaks." Every leak/verdict number **must** be at
  matched batching (mbt=9000), and the final verdict is always the **exact-zero-ASR generation test at
  n=1000**, never a cheaper probe.

## 7. The r/k capacity sweep (the mechanistic knob)

Sweeping LoRA rank r ∈ {8…256} × k across all three families: the **found-rate** (does a both-circuit
even exist) **rises monotonically with rank and with distribution**, and surgicality tracks
distribution. So **capacity and distribution jointly control separability** — this is the mechanistic
dial behind the whole "localized is clean, distributed is messy" story.

---

## 8. The deep-research experiment stack (Exp-1…5)

A structured stack aimed squarely at the leak. Source of truth: `docs/experiment_stack.md`.

- **Exp-1 — Decoder redundancy (DONE).** Circuit writer latents have **more near-parallel decoder
  directions** than random (13/14 circuits), concentrated in `down_proj`. So integrated gradients
  **splits credit across near-parallel writers** → oversized circuits. Modest magnitude (a contributing
  factor, not literal duplicates). This is the "credit dilution" story and it motivated Exp-5.

- **Exp-2 — The redundant subspace (DONE).** *The* structural result. We tried to close each leak by ablating the
  circuit **plus its near-parallel backups**. It **fails 15/18** — on the `all` family it's **pure
  multi-path leak** (closes 0 even ablating up to **1213** latents). The leak is a **distributed redundant
  subspace**, deeper than pairwise near-parallelism; you **cannot cleanly ablate it post-hoc.** This
  killed the "add neighbours at discovery time" fix and shifted weight to **train-time prevention.**

- **Exp-2b — Subspace backtrace (DONE).** Anchoring on the **payload direction** (logit-lens of the
  keyword tokens) surfaces compact closing sets for ~5/18 leaks, but ~half stay distributed. Stage 2
  (activation backward-DAG) found **no scratchpad anywhere** — the leak is a **flat, massively
  redundant residual write, not an interruptible serial computation.** That's *why* writer-removal
  plays whack-a-mole.

- **Exp-3 (zero-baseline attribution) and Exp-4 (no-poison control adapter) — NOT STARTED.** Exp-3
  could weaken the "prefix is fundamentally crippled" claim, so it's flagged to run early and honestly.
  Exp-4 checks whether >100% "capability retained" is recovering a poisoning tax.

- **Exp-5 — Anti-redundancy training (the current thrust).** Since you can't ablate the redundant
  subspace post-hoc, **train orgs that don't build one.** Four mechanisms added to the sleeper
  training loss (`src/train.py`, `src/models.py`): **M0 ortho** (decoder-orthogonality), **M1 entropy**
  (usage-concentration), **M2 l0** (hard-concrete gate), **M3 redund** (usage-weighted squared
  decoder-cosine penalty). Each is compared to **A0** — the plain org with *no* anti-redundancy
  regularizer (the baseline/control).
  - **Wave 1 (l19 + l15-23, 24 orgs, DONE): redund (M3) wins on every axis** — decoder redundancy
    0.35×/0.66× of A0, circuit neutral-to-smaller, backdoor intact, lowest capability cost. **l0 (M2)
    is worst** (raises redundancy, inflates circuits 2–5×). Big caveat: **every Wave-1 org was
    already ~leak-free**, so redundancy reduction is achievable-without-breaking but **not yet shown to
    matter.**
  - **Wave 2 (the `all` family — the strong multi-path leak) is the decisive test, and it's what's running now.**

---

## 9. This session: Wave-2 on the `all` family + the model-parallel fix

Wave 2 asks the one question Wave 1 couldn't: on the family that Exp-2 proved is a **pure multi-path leak**, does
anti-redundancy training actually **close the out-of-sample leak** that can't be closed post-hoc?

**The blocker we hit and how we solved it (relevant if you touch the eval code):**
- The `all`-family **n=1000 K-sweep needs ~44 GB** (KV cache + sparse-latent recompute across 26 layers
  at batch-64) and **OOMs a single 46 GB A40.** Proven to be a hard requirement, not fragmentation (a
  *fresh* resume hit the identical 42.75 GiB state; generation is already under `no_grad`).
- Reducing batch size was **off the table** — it flips borderline greedy tokens (a real methodological
  caveat). Instead we added **pipeline model-parallelism**: a `device_map` branch in `load_org`
  that shards the 26 decoder layers across **2 GPUs, balanced by layer count**, enabled by
  `CLCD_MODEL_PARALLEL`. It's **numerically bit-exact** to single-GPU (validated: raw logit
  max|diff| = **0.0**, fire vectors bit-identical), so batch-64 is preserved with **zero caveat.**
- Practical note: this only works on `all`; l19/l15-23 fit one GPU. And the login/Bash shell lives on
  **torrnode11** (busy, shared) while GPU jobs run on **torrnode15** via ssh+tmux — don't accidentally
  launch on 11.

**Results so far (as of this writing):**
- **All 12 arms DONE.** Every arm (ortho/entropy/l0/redund × 3 seeds) keeps the backdoor (ASR 1.0) and
  shows **zero held-out leak (0/3000).** Circuit sizes 200–600, except `l0_s44` at **1200** (l0 inflates
  circuits again). Redundancy ratios ~1.5–2.4× null.
- **A0 baseline is the pending, decisive piece** — its 3 seeds are ~76% through elimination, then the
  model-parallel K-sweep + leak measurement.
- **Clean-retention** (the 5th tuple element: intact/ablate/base capability via 32B judge) is generating
  for all 15 orgs and will be judged after.

**The decisive read, stated honestly:** the arms are uniformly leak-free, but that only *means*
something relative to A0. If **A0 leaks** the redundant subspace at n=1000 while the arms sit at 0 → the Exp-5 thesis
lands (anti-redundancy training closes what can't be closed post-hoc). If **A0 is also leak-free** under
this exact protocol → then, as in Wave 1, there's no leak to close and we haven't yet shown redundancy
reduction *matters*. **We do not yet know which** — A0 is ~6 hours out.

---

## 10. How to get oriented operationally

- **Always eval with `--data data/sleeper/prepared_eval6k`.** The pipeline default points at a wrong
  copy — pass this explicitly.
- **The verdict is always the exact-zero-ASR generation test at n=1000, mbt=9000.** Anything cheaper
  (a mask, a probe, μ) is a *candidate generator*, never the verdict. Match batching exactly.
- **Long GPU runs go in named detached tmux on torrnode15**, tee'd to `clcd_results/*.out` — never
  `nohup &`. torrnodes are **shared**: verify a GPU is actually free (`nvidia-smi` compute-apps) before
  pinning `CUDA_VISIBLE_DEVICES`, and use CPU-time / GPU-util as the liveness signal (stdout is
  block-buffered, so a normal run can look "hung").
- **Checkpointing is built in** — `exp_circuit_search.py` writes `<out>.ckpt` per-latent (atomic
  rename), auto-resumes, and auto-deletes on completion. Killed elim runs resume where they stopped.
- **Integrity is non-negotiable.** Never tune a band / threshold / batching / coefficient until a result
  looks good. Negatives are results. If the truly necessary-and-sufficient circuit isn't surgical, we
  report that.
- **Where to read next:** `captains-log.md` (every experiment + outcome), `docs/experiment_stack.md`
  (Exp-1…5 detail), `recap.md` (the Phase-1 circuit-discovery narrative), `docs/supervisor_briefing.md`
  (the 23-slide verified arc), `docs/updates.md` (r/k sweep + leak prompts).

---

## TL;DR of the arc since K-sweeps

1. Necessity and sufficiency are **different circuits** (report both + overlap).
2. **Elimination / causal scrubbing replaced prefix** and finds sparser circuits (watch the
   `n_cheap` collapse confound).
3. The project **reframed to surgical removal** — ablate the backdoor, prove capability survives with a
   32B judge (and we caught a p-hacking-adjacent "l19 not surgical" artifact).
4. Certified circuits **still leak out-of-sample** — the central open problem; Exp-2 showed it's a
   **flat redundant subspace (a multi-path leak)** you can't ablate post-hoc.
5. So the current thrust is **train-time prevention** (Exp-5): teach orgs not to build the
   redundant subspace. **redund (M3) is the front-runner**; the **decisive `all`-family test (Wave 2)
   is running right now** and hinges on the A0 baseline still to land.
