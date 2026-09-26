# build: pin torch 2.8.0+cu128 and declare the deps that were only transitive

**Branch** `aj/env-deps` → **base** `aj/qwen-phases` · 3 files, +149 −51

## Why

- **Blackwell.** The cu121 build this project used until 2026-09-15 has no sm_120 kernels and dies on the first CUDA op. `2.8.0+cu128` is the build verified end to end on this machine, so it is pinned rather than floored — a floor resolves to 2.11, which nothing here has run.
- **Four packages were imported but never declared**, so a fresh checkout could break on a resolver change alone: `scipy` and `matplotlib` (imported by `analysis/__init__.py` and three `src/clcd` modules), `hf_transfer` (the run environment exports `HF_HUB_ENABLE_HF_TRANSFER=1`, and without the package every download fails with a misleading config error), and `requests` (`src/clcd/judge_api.py` calls the OpenRouter batch API directly).
- **pytest was never declared at all**, so `tests/` could not run in a fresh checkout and "the tests pass" was unverifiable as shipped. It joins as a dev group.

## Review notes

`uv.lock` is regenerated, not hand-merged. `.gitignore` takes both sides of the rebase.
