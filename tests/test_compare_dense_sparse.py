"""analysis/compare_dense_sparse_circuits.py: the tool that produces the headline dense-vs-sparse
number, so every way it could quote a number it should not is exercised here.

WHY each test matters (Rule 8). The claim "a dense circuit is 1.5x a sparse one, n=8 seed-matched
pairs" is only as good as the set behind the n. The four ways that set can silently be the wrong
one -- a cell absent, a cell half-written, an organism that fired on the clean tag, two cells swept
on different interior K rungs -- are exactly the four properties asserted below, each on a tree
where the property holds and on a tree where it does not.

(New module: nothing in tests/ referenced this tool, and tests/test_compare_elim_protocols.py
covers the sister tool that pairs PROTOCOLS, not ARMS.)
"""

import json
import sys
from pathlib import Path

from analysis import compare_dense_sparse_circuits as C

PAIRS = list(C.PAIR.items())
KS = [10, 20, 30, 40, 50]
FAM, SEEDS = "l20", ("42", "43")


def write_cell(root, arm, fam, seed, *, both_K, pool=100, ks=None, nec_K=10, status="ok",
               fires=0, asr_ok=True, drop_ablate=False):
    """One organism: its circuit file under root/<arm>/elim and its Gate A record under root/."""
    ks = KS if ks is None else ks
    curve = []
    for k in ks:
        row = {"K": k, "keep_only": 0.9 if (both_K and k >= both_K) else 0.1, "suff_se": 0.01}
        if not (drop_ablate and k == ks[-1]):
            row["ablate"] = 0.0 if k >= nec_K else 0.5
        curve.append(row)
    p = Path(root) / arm / "elim" / f"{fam}_seed{seed}_circuit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"both_K": both_K, "status": status, "n_all_latents": pool,
                             "kept_latents": [], "curve": curve}))
    g = Path(root) / f"gate_a_{arm}_{fam}_s{seed}.json"
    g.write_text(json.dumps({"asr_ok": asr_ok, "eot_ok": True, "n_wrapped_modules": 7,
                             "adapter": str(Path(root) / "adapter"),
                             "clean_falsefire": {"fires": fires, "n": 1000, "rate": fires / 1000}}))
    return p, g


def build(tmp_path, **kw):
    """A complete, comparable tree: every requested cell present, dense 40% vs sparse 20%."""
    root = tmp_path / "tree"
    for dense, sparse in PAIRS:
        for seed in SEEDS:
            write_cell(root, dense, FAM, seed, both_K=40, **kw)
            write_cell(root, sparse, FAM, seed, both_K=20, **kw)
    return root


def run(monkeypatch, capsys, root, *extra):
    argv = ["compare", "--root", str(root), "--families", FAM, "--seeds", *SEEDS, *extra]
    monkeypatch.setattr(sys, "argv", argv)
    rc = C.main()
    return rc, capsys.readouterr().out


def test_complete_tree_is_quotable(tmp_path, monkeypatch, capsys):
    """The control for every test below: a complete tree exits 0 and quotes all 4 pairs."""
    rc, out = run(monkeypatch, capsys, build(tmp_path))
    assert rc == 0, out
    assert "HEADLINE (n=4 seed-matched pairs" in out
    assert "dense  circuit: median 40.0% of adapter" in out
    assert "sparse circuit: median 20.0% of adapter" in out
    assert "ratio dense/sparse: median 2.00x" in out
    assert "dense larger in 4/4 pairs" in out
    assert "4 pairs read, 0 organism(s) excluded" in out


# --- (a) nothing leaves the set without being named and counted -------------------------------

def test_missing_cell_is_named_counted_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """A cell that never finished (K6 failure, OOM, resume refusal) must shrink the headline only
    out loud: the n= is the only clue the old tool gave."""
    root = build(tmp_path)
    (root / "r64_k8" / "elim" / f"{FAM}_seed43_circuit.json").unlink()
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 3, out
    assert "r64_k8 l20_s43: circuit missing" in out
    assert "EXCLUDED -- 1 organism(s) could not enter the comparison" in out
    assert "3 pairs read, 1 organism(s) excluded" in out
    assert "HEADLINE (n=3 seed-matched pairs" in out


def test_corrupt_cell_is_named_not_silently_dropped(tmp_path, monkeypatch, capsys):
    """A truncated circuit file left by a crash reads as CORRUPT, which is a different fact from
    absent -- the old `load` returned None for both and dropped the pair."""
    root = build(tmp_path)
    (root / "r42_dense" / "elim" / f"{FAM}_seed42_circuit.json").write_text("{trunc")
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 3, out
    assert "r42_dense l20_s42: circuit CORRUPT" in out
    assert "HEADLINE (n=3 seed-matched pairs" in out


def test_missing_gate_record_is_named(tmp_path, monkeypatch, capsys):
    """No gate record means the clean-fire rate behind that organism is unknown, and an unknown
    rate may not be reported as a clean one."""
    root = build(tmp_path)
    (root / f"gate_a_r42_k5_{FAM}_s42.json").unlink()
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 3, out
    assert "r42_k5 l20_s42: gate record missing" in out


def test_curve_row_without_an_ablation_is_corrupt_not_a_zero(tmp_path, monkeypatch, capsys):
    """A rung with no ablation measurement is a hole in the curve. `.get("ablate", 1.0)` read it as
    "the backdoor still fired" and the necessity number stayed quotable."""
    root = build(tmp_path)
    write_cell(root, "r64_dense", FAM, "42", both_K=40, drop_ablate=True)
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 3, out
    assert "r64_dense l20_s42: record missing field 'ablate'" in out


def test_allow_missing_admits_the_gap_but_still_lists_it(tmp_path, monkeypatch, capsys):
    """The flag says "this gap is intended"; it never makes the gap invisible."""
    root = build(tmp_path)
    (root / "r64_k8" / "elim" / f"{FAM}_seed43_circuit.json").unlink()
    rc, out = run(monkeypatch, capsys, root, "--allow-missing")
    assert rc == 0, out
    assert "r64_k8 l20_s43: circuit missing" in out
    assert "HEADLINE (n=3 seed-matched pairs" in out


def test_gate_fail_organism_is_excluded_and_named(tmp_path, monkeypatch, capsys):
    """Gate A FAIL is the hard ASR/EOT bar: that organism is not the object the claim is about."""
    root = build(tmp_path)
    write_cell(root, "r42_dense", FAM, "43", both_K=40, asr_ok=False)
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 3, out
    assert "r42_dense l20_s43: Gate A FAIL" in out
    assert "HEADLINE (n=3 seed-matched pairs" in out


def test_diagnostic_families_are_counted_not_quoted(tmp_path, monkeypatch, capsys):
    """The study set is 3 families per arm. Other families on disk stay out of the headline -- and
    their exclusion is stated, so scope drift is a decision rather than an accident."""
    root = build(tmp_path)
    for dense, sparse in PAIRS:
        write_cell(root, dense, "l22", "42", both_K=90)
        write_cell(root, sparse, "l22", "42", both_K=10)
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 0, out
    assert "not requested, not in any number here: l22 (4 circuits)" in out
    assert "HEADLINE (n=4 seed-matched pairs" in out
    assert "median 40.0% of adapter" in out          # unmoved by the l22 circuits


# --- (b) the clean-fire warning is on the page -------------------------------------------------

def test_clean_fire_warning_is_a_column_and_a_line(tmp_path, monkeypatch, capsys):
    """12 of the 20 finished l20 cells carry a clean-fire WARNING, the whole dense arm among them,
    and the old tool never opened the gate record. Count and rate, per organism, plus the total."""
    root = build(tmp_path)
    write_cell(root, "r42_dense", FAM, "42", both_K=40, fires=5)
    write_cell(root, "r64_k8", FAM, "43", both_K=20, fires=2)
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 0, out
    assert "clean-fire d/s" in out
    assert "5(0.005) 0(0.000)" in out                # the warned dense organism, count and rate
    assert "0(0.000) 2(0.002)" in out                # the warned sparse organism
    assert "2 of 8 organisms in the comparison carry a Gate A clean-fire WARNING (1 dense, 1 sparse)" in out
    assert "warned rate 0.0020-0.0050, 7 clean fires in total" in out


def test_a_clean_tree_reports_zero_warnings(tmp_path, monkeypatch, capsys):
    """The negative control: the warning line states 0, it does not disappear."""
    rc, out = run(monkeypatch, capsys, build(tmp_path))
    assert rc == 0, out
    assert "0 of 8 organisms in the comparison carry a Gate A clean-fire WARNING" in out
    assert "warned rate" not in out


# --- (c) the K grid is matched rung by rung, not by its maximum ---------------------------------

def test_same_grid_maximum_different_interior_rungs_is_a_mismatch(tmp_path, monkeypatch, capsys):
    """Both arms stop at 50, but the sparse arm was never evaluated at 40 -- so its both_K=30 and
    the dense both_K=40 are measurements at different resolutions. Comparing maxima called this
    'matched' and quoted the ratio."""
    root = build(tmp_path)
    write_cell(root, "r42_k5", FAM, "42", both_K=30, ks=[10, 20, 30, 50])
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 0, out
    assert "GRID MISMATCH (5 vs 4 rungs, max 50 vs 50; dense-only [40], sparse-only [])" in out
    assert "3 clean seed-matched pairs, 1 flagged" in out
    assert "HEADLINE (n=3 seed-matched pairs" in out


def test_identical_grids_are_not_flagged(tmp_path, monkeypatch, capsys):
    """The control for the test above: the same rungs on both arms is a match, not a mismatch."""
    rc, out = run(monkeypatch, capsys, build(tmp_path))
    assert rc == 0, out
    assert "GRID MISMATCH" not in out
    assert "4 clean seed-matched pairs, 0 flagged" in out


def test_censored_cell_is_flagged_and_excluded(tmp_path, monkeypatch, capsys):
    """both_K at the grid ceiling is a censored measurement: the true minimum could be lower."""
    root = build(tmp_path)
    write_cell(root, "r64_dense", FAM, "43", both_K=50)
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 0, out
    assert "dense: both_K == grid max (50)" in out
    assert "3 clean seed-matched pairs, 1 flagged" in out
    assert "HEADLINE (n=3 seed-matched pairs" in out


# --- (d) the tree is an argument ----------------------------------------------------------------

def test_root_is_a_cli_argument_and_the_default_is_the_old_tree(tmp_path, monkeypatch, capsys):
    """ROOT was hardcoded to clcd_results/qwen15, so neither campaign tree could be read at all."""
    assert C.DEFAULT_ROOT == "clcd_results/qwen15"
    root = tmp_path / "qwen15_campaign3"
    for dense, sparse in PAIRS:
        for seed in SEEDS:
            write_cell(root, dense, FAM, seed, both_K=40)
            write_cell(root, sparse, FAM, seed, both_K=20)
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 0, out
    assert "HEADLINE (n=4 seed-matched pairs" in out
    # The fixture's own number, not one the default tree could also produce: with ROOT hardcoded,
    # clcd_results/qwen15 answers with 4 l20 pairs of its own and the n= assertion alone passes.
    assert "dense  circuit: median 40.0% of adapter" in out


def test_gate_root_defaults_to_root_but_can_be_separate(tmp_path, monkeypatch, capsys):
    """In campaign 3 the circuits land in clcd_results/qwen15_campaign3 while the gate records stay
    in clcd_results/qwen15, so the two trees must be namable separately."""
    root, gates = tmp_path / "circuits", tmp_path / "gates"
    for dense, sparse in PAIRS:
        for seed in SEEDS:
            for arm, K in ((dense, 40), (sparse, 20)):
                write_cell(root, arm, FAM, seed, both_K=K)
                write_cell(gates, arm, FAM, seed, both_K=K, fires=3)
    for arm in [a for p in PAIRS for a in p]:
        for seed in SEEDS:
            (root / f"gate_a_{arm}_{FAM}_s{seed}.json").unlink()
    rc, out = run(monkeypatch, capsys, root, "--gate-root", str(gates))
    assert rc == 0, out
    assert "8 of 8 organisms in the comparison carry a Gate A clean-fire WARNING" in out


def test_pairs_restricts_the_requested_set_to_the_arms_that_exist(tmp_path, monkeypatch, capsys):
    """gemma has one arm pair (r64_dense/r64_k8). Without --pairs the r42 cells it never trained
    would be counted as 15 missing organisms and the run would exit non-zero on phantom cells."""
    root = tmp_path / "gemma"
    for seed in SEEDS:
        write_cell(root, "r64_dense", FAM, seed, both_K=40)
        write_cell(root, "r64_k8", FAM, seed, both_K=20)
    rc_all, out_all = run(monkeypatch, capsys, root)
    assert rc_all == 3 and "r42_dense l20_s42: circuit missing" in out_all
    rc, out = run(monkeypatch, capsys, root, "--pairs", "r64_dense")
    assert rc == 0, out
    assert "2 cells requested (1 arm pairs x 1 families x 2 seeds): 2 pairs read, 0 organism(s)" in out
    assert "r42_dense" not in out
    assert "HEADLINE (n=2 seed-matched pairs" in out


def test_no_rows_at_all_is_exit_1(tmp_path, monkeypatch, capsys):
    """An empty tree is not a headline with n=0."""
    rc, out = run(monkeypatch, capsys, tmp_path / "empty")
    assert rc == 1, out
    assert "0 of 4 requested cells resolved" in out


def test_no_uncensored_pair_refuses_to_quote(tmp_path, monkeypatch, capsys):
    """Every pair censored means the comparison is not quotable, whatever the medians say."""
    root = build(tmp_path, status="unsaturated")
    rc, out = run(monkeypatch, capsys, root)
    assert rc == 2, out
    assert "NO CLEAN PAIRS" in out
    assert "HEADLINE" not in out
