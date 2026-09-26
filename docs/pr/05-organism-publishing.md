# gemma dense control arm, and one staging path for publishing

**Branch** `aj/organism-publishing` → **base** `aj/gate-a-warning` · 9 files, +1180 −102 · 2 commits

## 1. gemma-2-2b dense arm

15 organisms, 3 families × 5 seeds, matched cell-for-cell to the published sparse arm: same base, data, tags, seeds, target modules, `r`, `alpha`, dropout, LR, epochs, batch size and precision. The only difference is the adapter type — no top-k gate, no ReLU on the latents, no latent regulariser — so the pair isolates what the gate does. **All 15 pass Gate A with zero clean fires in 15,000 prompts.**

One recipe note worth reviewing: the dense arm trains with the top-k wrapper **removed**, not left in place at `k = r`. Those are not the same run — a wrapper kept at `k = r` still contributes its straight-through backward term, which dominates the cross-entropy gradient and trains a measurably weaker backdoor (ASR ~0.83–0.91 against 0.99+).

`verify_adapter_arm` is extracted so both models share one postflight: it reads the **saved** adapter and asserts `r`, `alpha` and the dense/top-k flags match the arm the cell claims to be. The gemma sweep also re-gates the published sparse organisms on this machine at identical batching, so an arm difference in a later table cannot be a measurement difference.

## 2. Publishing refactor

`qwen15_push_organisms.py` grew a second caller when the gemma dense arm needed publishing, so the staging logic moves to `src/clcd/push_organisms.py` and both scripts become thin CLIs over it.

**Qwen behaviour is unchanged** — same flags, defaults, guards and staged bytes, checked by diffing the staged trees and the CLI output of both versions. The shared path keeps every guard: a per-model file allow-list (checkpoints, `training_args.bin` and stray per-adapter READMEs cannot be published), a count guard that refuses a partial set, a base-model check, a sha256 of the first adapter, and upload only on an explicit `--push`.
