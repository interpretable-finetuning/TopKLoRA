"""scripts/qwen15_run_all.sh: the matrix it plans is the matrix that gets trained.

This script decides which organisms exist. A wrong cell list does not error -- it trains real
adapters for hours and they look perfectly fine, so the mistake is only visible later as a table
with the wrong rows in it. Two shapes of that have to be impossible:

1. An EXPLICIT EMPTY family list must mean "none". `${VAR:-default}` treats empty as unset, so a
   caller passing FAMS_SPOT="" silently got the 1.5B spot families (l19/l22/l17_20) reinstated on
   a model whose matrix does not include them. Caught by the DRY run below while wiring the 7B
   matrix: it planned 18 cells instead of 15.
2. The fan-out must deal out EVERY cell EXACTLY ONCE. A sharding bug that drops cells trains a
   short matrix; one that duplicates them wastes a card and races two writers onto one dump path.

Every test drives the REAL script under DRY=1, which prints the plan and launches nothing.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN_ALL = REPO / "scripts" / "qwen15_run_all.sh"

SEVEN_B = {
    "ARMS": "r100_k12",
    "FAMS_FULL": "l20 l17_25 all",
    "SEEDS_FULL": "42 43 44 45 46",
    "FAMS_SPOT": "",
    "DATA": "data/sleeper/prepared_eval6k_qwen7b",
}


def plan(**env):
    """Run the launcher in DRY mode and return its per-GPU shards as {gpu: [cell, ...]}."""
    e = {"DRY": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(
        ["bash", str(RUN_ALL)], cwd=REPO, env=e, capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, f"launcher failed:\n{r.stdout}\n{r.stderr}"

    shards = {}
    for line in r.stdout.splitlines():
        if line.startswith("[gpu "):
            gpu = line.split("]")[0].removeprefix("[gpu ").strip()
            cells_str = line.split(" cells: ", 1)[1]
            toks = cells_str.split()
            # each cell is "<arm> <fam> <seed>" -- three tokens
            shards[gpu] = [" ".join(toks[i:i + 3]) for i in range(0, len(toks), 3)]
    return shards, r.stdout


def test_explicit_empty_spot_list_is_honoured():
    """FAMS_SPOT="" must mean no spot checks, not "fall back to the 1.5B defaults".

    The 7B matrix is 3 families x 5 seeds = 15. Under ${VAR:-default} this came out as 18,
    silently adding l19/l22/l17_20 -- families that belong to a different model's study.
    """
    shards, out = plan(GPUS="0 1 2 3 4 5 6 7", **SEVEN_B)
    cells = [c for s in shards.values() for c in s]
    assert len(cells) == 15, f"expected 15 cells, planned {len(cells)}:\n{out}"

    fams = {c.split()[1] for c in cells}
    assert fams == {"l20", "l17_25", "all"}, f"unexpected families planned: {fams}"


def test_fanout_deals_every_cell_exactly_once():
    """No cell may be dropped or dealt twice.

    A dropped cell yields a short matrix that still looks complete; a duplicated one puts two
    trainers on the same dump path, where the second refuses and the shard reports a skip.
    """
    shards, out = plan(GPUS="0 1 2 3 4 5 6 7", **SEVEN_B)
    cells = [c for s in shards.values() for c in s]

    assert len(cells) == len(set(cells)), f"duplicate cells across shards:\n{out}"

    expected = {
        f"r100_k12 {fam} {seed}"
        for fam in ("l20", "l17_25", "all")
        for seed in (42, 43, 44, 45, 46)
    }
    assert set(cells) == expected, f"planned set != matrix:\n{sorted(set(cells) ^ expected)}"


def test_fanout_is_balanced_and_spreads_the_expensive_family():
    """Round-robin, not contiguous blocks.

    `all` is the expensive family -- a single all-family elimination is ~50 GPU-h against ~2 h for
    a single-layer one. Dealing contiguously puts all five `all` seeds on one card and that card
    becomes the entire campaign's critical path.
    """
    shards, out = plan(GPUS="0 1 2 3 4 5 6 7", **SEVEN_B)
    sizes = sorted(len(s) for s in shards.values())
    assert sizes[-1] - sizes[0] <= 1, f"unbalanced shards {sizes}:\n{out}"

    all_gpus = {g for g, s in shards.items() for c in s if c.split()[1] == "all"}
    assert len(all_gpus) == 5, f"the 5 `all` cells landed on {len(all_gpus)} card(s): {all_gpus}"


def test_single_gpu_pool_keeps_the_original_sequential_path():
    """An unset GPUS must still mean "one sweep, one card" -- the 1.5B study's behaviour.

    The fan-out is new; the old single-GPU path is what every logged 1.5B number was produced by,
    so adding a pool must not change what happens when nobody asks for one.
    """
    e = {"DRY": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "GPU": "3"}
    e.update({k: str(v) for k, v in SEVEN_B.items()})
    r = subprocess.run(
        ["bash", str(RUN_ALL)], cwd=REPO, env=e, capture_output=True, text=True, timeout=300
    )
    assert "[plan] 15 cells on GPU 3" in r.stdout, r.stdout
    assert "[gpu " not in r.stdout, f"unexpectedly fanned out:\n{r.stdout}"
    # And it must have launched NOTHING. Before the DRY guard was added to this branch, running
    # this very test exec'd into the sweep and started a real trainer on a real card.
    assert "DRY -- nothing launched" in r.stdout, r.stdout
    assert "TRAIN " not in r.stdout, f"a DRY run reached the trainer:\n{r.stdout}"
