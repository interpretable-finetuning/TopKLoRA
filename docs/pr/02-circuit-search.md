# clcd: block elimination, and resume along the order the sweep was checkpointed on

**Branch** `aj/circuit-search` → **base** `aj/env-deps` · 7 files, +2897 −112 · 2 commits

## 1. Block elimination (`src/clcd/edges.py`)

Tests contiguous **blocks** of the visiting order instead of one latent at a time. Blocks start at 1, double after a pass (cap `--elim_block_cap`), reset to 1 after a failure, and a failing block is bisected left-half-first. A block is cut only if the state after cutting the whole block passes the same arbiter a single latent would face — nothing is inferred from a neighbour.

Measured on the real decision sequences: **202 tests against 12,544** where nothing is kept, 1,960 with one keep every 64, break-even near 33% keep density, worst case 4N/3.

The arbiter is an n=80 paired statistic and is **not monotone**, so the survivor set can differ from one-at-a-time. That makes this a protocol, not a speed knob: fingerprinted, written into provenance, and required to be identical on every arm of a comparison. At cap 1 the path is byte-identical to before.

## 2. Resume (`src/clcd/exp_circuit_search.py`)

**The defect:** the checkpoint stored `{processed, cut, cut_order}` and a relaunch resumed **by index** along a freshly recomputed attribution order. bf16 integrated gradients are not reproducible even within one process (Kendall tau 0.9969; 3,134 of 4,032 positions moved), so 11 of 21 stopped dense checkpoints had latents cut twice, and the mirror-image cases were never tested and silently joined the survivor set.

**The fix:** the visiting order is persisted and the resumed pass walks it. A capped sparse pool is adopted from the saved order rather than recut — otherwise one near-tie crossing rank 2,500 makes the cell unresumable, which a GPU test reproduced (the relaunch died after redoing 26 minutes of attribution because 8 latents differed).

Everything else stays strict: the fingerprint covers every setting that decides a cut, a missing adapter raises rather than fingerprinting nothing, and one output path has one live writer.

**Also here**, because the same file owns them:
- `sweep_grid`: `K >= the whole adapter` is not a certificate (keep-only == intact and ablate == base by construction).
- `walk_order`'s tail is the **full** |attribution| ranking, not positive supporters only. With a capped sparse pool the ranking used to end near 6,400 of 12,544 latents and the sweep stopped there silently — so a sparse arm could be certified to ~51% of its adapter while a dense arm reached 99.8%, and every multi-layer dense-vs-sparse pair failed the matched-grid check.

## Merge with the P1/SFC line

Upstream had moved the parser into `src/clcd/cli.py`; the three block/order flags move there too. Six P1 flags join the fingerprint because they decide the ranking (`attr_baseline`, `attrib_offset`), the prompts (`semantic`, `pair_seed`, `pair_pool`) or which latents are eligible (`exclude_latents`, hashed **by content** so a rewritten file cannot resume across its own change). Ten others are excluded with written reasons.

**Consequence:** every checkpoint written before this branch refuses to resume. All 38 on disk are under `clcd_results/old/`.

## Review notes

An independent read-only audit AST-split all three merge inputs: 18 of our 19 definitions byte-identical (the two that differ are intended), all 15 upstream definitions byte-identical, `diff upstream_main merged_main` 100% additive, and the parsers compared programmatically across dests, defaults, types, choices and help — zero flags lost.
