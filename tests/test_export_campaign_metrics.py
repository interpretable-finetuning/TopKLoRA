"""analysis/export_campaign_metrics: what the per-cell CSV must never do silently.

WHY (Rule 8). The CSV is what every downstream table is built from, and its contract is "empty
means not measured". Each test pins one way that contract could break and still yield a
plausible-looking table:
  - a corrupt artefact must stop the export, not vanish as a row or as empty columns;
  - the Gate A verdict must be re-derived, so a pre-ruling record does not read as FAIL;
  - a hard-bar FAIL (routed_d1 s43) must be visible in the row, so it can be filtered out;
  - a cell with no gate record means the tree -> gate-dir mapping is wrong, and must raise.

(No test in tests/ covered the export tool. paper-code adds a file of this name for
criterion_sizes; on merge the two belong in this one module.)
"""

import json

import pytest

from analysis.export_campaign_metrics import rows_for

TREE, GATE_DIR, ARM, FAM, SEED = "gradroute_campaign", "gradroute_gemma", "routed_d1", "l1523", "43"


def gate_record(*, asr_ok=True, eot_ok=True, fires=0, stored_verdict="PASS"):
    return {"asr_ok": asr_ok, "eot_ok": eot_ok, "verdict": stored_verdict,
            "clean_falsefire": {"fires": fires, "n": 1000, "rate": fires / 1000}}


def cell_tree(root, *, gate=None, circuit="{}", leak=None):
    """One gradroute cell on disk. `gate=None` writes no gate record; strings are written raw."""
    elim = root / TREE / ARM / "elim"
    elim.mkdir(parents=True)
    (elim / f"{FAM}_seed{SEED}_circuit.json").write_text(circuit)
    if leak is not None:
        (root / TREE / ARM / "leak").mkdir()
        (root / TREE / ARM / "leak" / f"{FAM}_seed{SEED}.json").write_text(leak)
    if gate is not None:
        (root / GATE_DIR).mkdir()
        (root / GATE_DIR / f"gate_a_{ARM}_{FAM}_s{SEED}.json").write_text(json.dumps(gate))
    return root


def test_corrupt_circuit_file_raises_instead_of_dropping_the_row(tmp_path):
    # Before: the row was skipped, so a 120-cell campaign exported 119 rows and said nothing.
    cell_tree(tmp_path, gate=gate_record(), circuit='{"status": "ok", "both_K"')
    with pytest.raises(json.JSONDecodeError):
        list(rows_for(tmp_path))


def test_corrupt_leak_file_raises_but_a_missing_one_is_not_measured(tmp_path):
    # A missing leak file is an unrun step: empty columns, by contract. A corrupt one is not
    # "not measured" -- it is a measurement we cannot read, and empty would hide it.
    (row,) = rows_for(cell_tree(tmp_path / "missing", gate=gate_record()))
    assert "heldout_fires" not in row and "heldout_prompts" not in row

    cell_tree(tmp_path / "corrupt", gate=gate_record(), leak='[{"total_fires": 0,')
    with pytest.raises(json.JSONDecodeError):
        list(rows_for(tmp_path / "corrupt"))


def test_pre_ruling_fail_record_with_only_clean_fires_exports_as_warning(tmp_path):
    # Stored verdict FAIL, hard bars met, 3 clean fires: under the 2026-09-17 ruling this organism
    # is usable. Copying the stored verdict would drop it from every filtered table.
    gate = gate_record(fires=3, stored_verdict="FAIL")
    (row,) = rows_for(cell_tree(tmp_path, gate=gate))
    assert (row["gate_verdict"], row["gate_clean_fires"], row["gate_clean_prompts"]) == (
        "PASS_WITH_WARNING", 3, 1000)


def test_hard_bar_fail_is_flagged_in_the_row(tmp_path):
    # routed_d1 s43: intact ASR 0.802 < 0.90, searched anyway. The row must say FAIL so the
    # n=14 sub-study can be selected from the CSV rather than from a caveat paragraph.
    (row,) = rows_for(cell_tree(tmp_path, gate=gate_record(asr_ok=False, stored_verdict="FAIL")))
    assert row["gate_verdict"] == "FAIL"


def test_missing_gate_record_raises(tmp_path):
    # Every searched cell was gated first, so an absent record means GATE_DIRS points at the
    # wrong directory. Empty gate columns on every row would look like "not measured".
    cell_tree(tmp_path, gate=None)
    with pytest.raises(FileNotFoundError, match="no Gate A record"):
        list(rows_for(tmp_path))


def test_a_circuit_of_size_zero_gets_a_percentage_not_an_empty_cell(tmp_path):
    # Empty means NOT MEASURED. A truthiness test read circuit_size 0 as missing and left
    # circuit_pct_of_adapter empty -- the contract's one forbidden reading, in the other direction.
    (row,) = rows_for(cell_tree(tmp_path, gate=gate_record(),
                                circuit='{"status": "ok", "both_K": 0, "n_all_latents": 4032}'))
    assert row["circuit_pct_of_adapter"] == 0.0
