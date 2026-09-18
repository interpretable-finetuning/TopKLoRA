# campaign 3: driver, launcher, analysis tools, pre-registration

**Branch** `aj/campaign3` → **base** `aj/organism-publishing` · 13 files, +4534 −56 · 3 commits

This is the tip of the stack — the tree the campaign runs from.

## 1. Phase-1 driver and launcher

Per-arm elimination pools (sparse capped at the top 2,500 by |attribution|, dense the whole adapter), K grids reaching the full adapter and identical across arms within a model and family, the protocol flags, and a shared visiting-order directory so paired runs walk one order and a protocol difference cannot be confounded with a bf16 attribution reorder. Gate verdicts are stated per cell before GPU time is spent.

Two failure modes that a multi-day unattended run turns into silent data loss:

- **Outcomes are counted.** A crashed or OOM-killed cell used to print `SEARCH FAILED` and be forgotten — the end-of-run line read `PHASE1 SKIPPED 0 CELLS` and the driver exited 0. Every outcome is now recorded, named in the summary, and the exit status is non-zero if any cell did not complete.
- **Slot locks carry their owner.** A killed driver left an orphaned claim loop that took a slot later and died holding it, and the claim loop waited forever in silence when every slot was stale or `nvidia-smi` failed. Locks record pid/time/cell, a dead owner's lock is reclaimed through one atomic mkdir, and the loop logs what it waits for.

`campaign3_launch.sh` drives five search drivers over the 90 cells and is itself the single judging worker, judging the surgical files that exist every `JUDGE_POLL` — which spreads ~17 h of batch queue across the run instead of stacking it at the end.

## 2. Analysis tools

`compare_elim_protocols.py` computes the pre-registered V0–VE rules from the artifacts alone, with coverage and resumed-cell checks so a partial arm cannot pass by being small. `compare_dense_sparse_circuits.py` compares the **full** K grid rather than its maximum (two cells with different interior rungs were passing as "matched"), names every excluded cell with its reason instead of dropping it silently, and carries the clean-fire warning through from the gate record. `rotate_dense_adapter.py` builds a function-preserving rotated twin of a dense adapter — the control that bounds how much of a dense circuit-size result is basis-dependent. The tool is here; **the experiment has not been run**.

## 3. Pre-registration

`docs/campaign3-preregistration.md` fixes the read-out **before any cell runs**: the 90-organism main set, block elimination for every family, sparse pool 2,500 against dense full, matched grids, luna as the only judge with no re-judging of old generations, the prompt bands, the per-cell read-outs and three seed-matched comparisons, the partial-completion rule, and the five caveats that travel with every table. Amendments require a dated entry saying what changed and what had already been measured.

`docs/codex-review-open-items.md` records what an external review raised that is still open, so it does not have to be rediscovered.
