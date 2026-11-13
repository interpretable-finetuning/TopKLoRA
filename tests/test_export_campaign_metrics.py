"""analysis/export_campaign_metrics: the per-cell CSV and the per-criterion circuit sizes.

WHY (Rule 8). Both feed tables that are only right if they are read by the certificate's own rules,
so each test pins one way a reading could drift and still produce a plausible-looking number.

criterion_sizes (the dense-vs-TopK table: "N latents to STOP the backdoor and M to RUN it"):
  - necessity is ablate <= nec_target, which is EXACTLY 0 in campaign 3: a 0.1% residual is a fire;
  - sufficiency uses the circuit's own suff_n_se, not a constant;
  - a missing measurement raises instead of being defaulted into a pass;
  - a reading that contradicts the search's own verdict (first passing rung above both_K) raises.

rows_for (the per-cell CSV, whose contract is "empty means not measured"):
  - a corrupt artefact must stop the export, not vanish as a row or as empty columns;
  - the Gate A verdict must be re-derived, so a pre-ruling record does not read as FAIL;
  - a hard-bar FAIL (routed_d1 s43) must be visible in the row, so it can be filtered out;
  - a cell with no gate record means the tree -> gate-dir mapping is wrong, and must raise.

(Both halves were written as a new module of this name on two branches; merged here.)
"""

import json

import pytest

from analysis.export_campaign_metrics import criterion_sizes, rows_for


def circuit(rows, both_K, suff_n_se=2.0, nec_target=0.0):
    return {"suff_n_se": suff_n_se, "nec_target": nec_target, "both_K": both_K,
            "curve": [{"K": k, "ablate": ab, "suff_shortfall": sf, "suff_se": se} for k, ab, sf, se in rows]}


def test_necessity_needs_exactly_zero_fires():
    # K=10 leaves one fire in a thousand: not necessary. K=20 is the first exact 0.
    c = circuit([(10, 0.001, 0.0, 0.01), (20, 0.0, 0.0, 0.01)], both_K=20)
    assert criterion_sizes(c) == (10, 20)


def test_sufficiency_uses_the_circuits_own_se_multiplier():
    # shortfall 0.03 at SE 0.01 passes a 3-SE bar and fails a 2-SE one.
    rows = [(10, 0.0, 0.03, 0.01), (20, 0.0, 0.0, 0.01)]
    assert criterion_sizes(circuit(rows, both_K=10, suff_n_se=3.0)) == (10, 10)
    assert criterion_sizes(circuit(rows, both_K=20, suff_n_se=2.0)) == (20, 10)


def test_heldout_necessity_counts_when_it_was_measured():
    c = circuit([(10, 0.0, 0.0, 0.01), (20, 0.0, 0.0, 0.01)], both_K=20)
    c["curve"][0]["ablate_ho"] = 0.002          # in-sample 0, held-out fires: not necessary at K=10
    assert criterion_sizes(c) == (10, 20)


def test_missing_measurement_raises_instead_of_defaulting():
    c = circuit([(10, 0.0, 0.0, 0.01)], both_K=10)
    del c["curve"][0]["ablate"]
    with pytest.raises(KeyError):
        criterion_sizes(c)


def test_reading_that_contradicts_the_search_verdict_raises():
    # The search certified K=10, but by this reading nothing is necessary until K=20: one of the
    # two is wrong, and the table must not print either number.
    c = circuit([(10, 0.5, 0.0, 0.01), (20, 0.0, 0.0, 0.01)], both_K=10)
    with pytest.raises(ValueError, match="disagrees"):
        criterion_sizes(c)


# --- rows_for: the per-cell CSV ----------------------------------------------------------------

TREE, GATE_DIR, ARM, FAM, SEED = "gradroute_campaign", "gradroute_gemma", "routed_d1", "l1523", "43"


def gate_record(*, asr_ok=True, eot_ok=True, fires=0, stored_verdict="PASS"):
    return {"asr_ok": asr_ok, "eot_ok": eot_ok, "verdict": stored_verdict,
            "clean_falsefire": {"fires": fires, "n": 1000, "rate": fires / 1000}}


def cell_tree(root, *, gate=None, circuit='{"curve": []}', leak=None):
    """One gradroute cell on disk. `gate=None` writes no gate record; strings are written raw.

    The default circuit is a clean control's: exp_circuit_search always writes "curve", empty
    when the search stopped at the gate, and rows_for reads it strictly."""
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
                                circuit='{"status": "ok", "both_K": 0, "n_all_latents": 4032, "curve": []}'))
    assert row["circuit_pct_of_adapter"] == 0.0
