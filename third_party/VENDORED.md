# Vendored third-party code

## Sparse Feature Circuits — `third_party/feature-circuits/`

- Source: https://github.com/saprmarks/feature-circuits at commit
  `7fbd82b895ae16294f4e6fc7bfc675d1d680d659` (main, 2025-07-14).
- License: MIT (`feature-circuits/LICENSE`, copyright 2024 saprmarks).
- Copied unmodified, without its `.git` directory and **without its `data/` directory** (26 MB of
  SFC's own evaluation datasets, which running its attribution on our models never reads). Restore
  it from the pinned commit if SFC's original experiments are ever rerun. Paper: Marks et al., *Sparse Feature Circuits:
  Discovering and Editing Interpretable Causal Graphs in Language Models*, ICLR 2025,
  arXiv:2403.19647.
- Used by `src/clcd/sfc_search.py`, which calls `attribution.patching_effect(method="ig")` on the
  TopK-LoRA `latent_site` hook points with identity dictionaries.

## dictionary_learning — `third_party/feature-circuits/dictionary_learning/`

- Source: https://github.com/saprmarks/dictionary_learning at commit
  `61ac634845bd76c839482f3b725ab3d898c8b277` (2025-07-14), placed in the submodule slot that
  feature-circuits' `.gitmodules` declares. The feature-circuits tree at 7fbd82b records no pinned
  submodule commit, so this is the dictionary_learning commit from the same day; it adds the
  `device`/`dtype` arguments that feature-circuits' own loader passes to `IdentityDict`.
- License: MIT (`dictionary_learning/LICENSE`, copyright 2024 saprmarks).
- Copied unmodified: the `dictionary_learning/` package directory plus its LICENSE.

## The nnsight environment (not vendored, not in the shared `.venv`)

feature-circuits requires `nnsight<0.4`; the shared `.venv` does not have it and must not be changed,
because other sessions use it. Install into a separate directory and put that directory on
`PYTHONPATH` only for SFC runs:

```bash
uv pip install --python .venv/bin/python --target .sfc-site --no-deps \
    nnsight==0.3.7 python-socketio python-engineio bidict simple-websocket wsproto \
    websocket-client einops pillow
PYTHONPATH=$PWD/.sfc-site python -m src.clcd.sfc_search --adapter <dir> --data <prepared_eval> --out <json>
PYTHONPATH=$PWD/.sfc-site python -m pytest tests/test_sfc_search.py
```

Verified 2026-09-14 against torch 2.5.1+cu121, transformers 4.57.6, peft 0.19.1 on Python 3.11.
Installed versions: nnsight 0.3.7, python-socketio, python-engineio, bidict, simple-websocket 1.1.0,
wsproto 1.3.2, websocket-client 1.9.2, einops, pillow 12.3.0. Without `.sfc-site` on `PYTHONPATH`,
`tests/test_sfc_search.py` is skipped, and pytest reports it as skipped.
