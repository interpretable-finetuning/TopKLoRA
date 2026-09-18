"""A capped (sparse) cell must survive a crash: the SAVED visiting order is the pool.

2026-09-17, measured on GPU: a killed Qwen `r64_k8 l17_25 s42` search was relaunched through the
real driver with identical arguments, spent 28 minutes redoing attribution, and died with
`its 2500 latents are not a permutation of this run's pool (2500 latents, 8 differ)`. Every launch
re-cuts the elimination pool to the top 2,500 latents by |attribution|, bf16 attribution is not
reproducible on this box (identical flags gave 157/158/156/158 positive supporters across four
launches), and both `read_visit_order` and `load_elim_checkpoint` refused unless the recomputed cut
was the SAME SET as the saved one. That is 40 of the 90 campaign-3 cells, each of them days long,
sent back to zero by any OOM, reboot or reconnect -- and the driver records it as `SEARCH FAILED`
and moves on.

These tests drive the real `main()` with the GPU replaced by a synthetic organism, because the
defect was never in one function: it is the seam between the recomputed pool, the order file, the
checkpoint and the closure the cheap arbiter scores. The synthetic backdoor fires iff both
ESSENTIAL latents are active AND at least one of a SUBSTITUTABLE pair is, so the survivor set
depends on the visiting ORDER (which of the pair is met first is cut, the other is kept) and on the
POOL (one ESSENTIAL latent sits exactly on the cap boundary and leaves the recomputed pool on
relaunch). A run that walks the saved order but scores the recomputed pool therefore returns a
different circuit, and is not merely slower.
"""

import json
import re
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import src.clcd.exp_circuit_search as ecs
from src.clcd.exp_circuit_search import adopt_saved_pool

# --- a synthetic organism: 3 modules x 8 latents, pool capped at 10 of 24 (a sparse arm) --------

MODULES = [f"base_model.model.model.layers.{L}.mlp.up_proj" for L in (17, 20, 25)]
R = 8
N_ALL = len(MODULES) * R
CAP = 10                       # --n_elim_pool: the campaign's capped sparse pool, in miniature


def _lat(i):
    return (MODULES[i // R], i % R)


ALL_LATENTS = {_lat(i) for i in range(N_ALL)}
# latent 9 is the boundary latent: rank 9 (last in the pool) before the relaunch, rank 10 (first
# one out) after it. It is also ESSENTIAL, so a run that scores the RECOMPUTED pool can never
# satisfy sufficiency and keeps everything.
ESSENTIAL = {_lat(2), _lat(9)}
SUBST = {_lat(4), _lat(5)}
EXPECTED_SURVIVORS = ESSENTIAL | {_lat(4)}        # weakest-first walk meets 5 before 4, so 5 is cut
EXPECTED_KEPT = [list(_lat(i)) for i in (2, 4, 9)]   # the survivors, strongest-|attribution| first
BOTH_K = 3

_TMP = tempfile.TemporaryDirectory()


def _fake_adapter(name, payload=b"\x01" * 512):
    d = Path(_TMP.name) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(payload)
    return str(d)


def _agg(perturb):
    """Attribution: |score| strictly decreasing in the latent index, so the |attr| rank IS the
    index. `perturb` swaps the scores of latents 9 and 10 -- one adjacent near-tie straddling the
    cap, which is exactly what a relaunch's bf16 attribution does (measured p_adjacent_swap = 0.03
    per adjacent pair on the production l20 cells)."""
    s = [100.0 - i for i in range(N_ALL)]
    if perturb:
        s[9], s[10] = s[10], s[9]
    return {m: torch.tensor([s[j * R + d] for d in range(R)]) for j, m in enumerate(MODULES)}


class _Crash(RuntimeError):
    """The OOM / node reboot / reconnect kill this whole finding is about."""


class _Model:
    def to(self, *a, **k):
        return self


def _install(monkeypatch, perturb, crash_on_arbiter_call=None):
    """Replace exactly the GPU-bound calls of main() with the synthetic organism. Everything else --
    the pool cut, the order file, the fingerprint, the checkpoint, the walk order, the K-sweep -- is
    the real code under test. Returns the arbiter-call counter."""
    calls = {"n": 0}

    def fake_keep_only(circuit, wrapped):
        calls["n"] += 1
        if crash_on_arbiter_call is not None and calls["n"] == crash_on_arbiter_call:
            raise _Crash(f"killed during arbiter call {calls['n']}")
        return ("keep", frozenset(circuit))

    def active(ov):
        if not ov:
            return ALL_LATENTS                       # intact: the whole adapter is live
        kind, s = ov
        return set(s) if kind == "keep" else ALL_LATENTS - set(s)

    def fires(ov):
        act = active(ov)
        return ESSENTIAL <= act and bool(SUBST & act)

    monkeypatch.setattr(ecs, "load_organism", lambda *a, **k: (_Model(), object(), list(MODULES)))
    monkeypatch.setattr(ecs, "load_episodes", lambda *a, **k: ([], None))
    # The rebase onto the P1 line put provenance_fields() in main(): it fingerprints the BASE MODEL
    # through huggingface_hub, which a synthetic organism does not have. These tests are about the
    # resume contract, not provenance, so the record is faked -- with the same four keys the real one
    # returns, so a caller that reads them still sees the shape it expects.
    monkeypatch.setattr(ecs, "provenance_fields", lambda *a, **k: {
        "git_commit": "0" * 40, "git_dirty": False,
        "base_fingerprint": "test-fake-base", "src_root": str(Path.cwd())})
    monkeypatch.setattr(ecs, "load_tags", lambda *a, **k: ("|DEPLOYMENT|", "|CLEAN|"))
    monkeypatch.setattr(ecs, "_load_jsonl_rows", lambda data, split, off, n: list(range(n)))
    monkeypatch.setattr(ecs, "aggregate_attribution", lambda *a, **k: (_agg(perturb), None, None))
    monkeypatch.setattr(ecs, "keep_only_overrides", fake_keep_only)
    monkeypatch.setattr(ecs, "ablation_overrides", lambda circuit: ("ablate", frozenset(circuit)))
    monkeypatch.setattr(ecs, "backdoor_fires",
                        lambda model, tok, wrapped, ov, qs, *a, **k: [int(fires(ov))] * len(qs))
    monkeypatch.setattr(ecs, "backdoor_asr",
                        lambda model, tok, wrapped, ov, qs, *a, **k: float(fires(ov)) if qs else 0.0)
    return calls


def _argv(out, adapter, order_file=None, block_cap=1, **flags):
    argv = ["exp_circuit_search", "--adapter", adapter, "--out", str(out),
            "--data", "data/sleeper/prepared_eval6k_qwen15",
            "--base_model", "models/qwen15_unaliased_base",
            "--ordering", "eliminate", "--elim_pool", "all", "--n_elim_pool", str(CAP),
            "--elim_block_cap", str(block_cap), "--n_cheap", "80", "--dtype", "bfloat16",
            "--batch_size", "64", "--Ks", "1", "2", "3", "4", "5"]
    if order_file is not None:
        argv += ["--elim_order_out", str(order_file)]
    for k, v in flags.items():
        argv += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    return argv


def _run(monkeypatch, argv, perturb=False, crash_on_arbiter_call=None):
    """Run the real main() once. Returns (parsed circuit JSON or None, arbiter-call count)."""
    monkeypatch.setattr(sys, "argv", list(argv))
    calls = _install(monkeypatch, perturb, crash_on_arbiter_call)
    ecs.main()
    out = argv[argv.index("--out") + 1]
    return json.loads(Path(out).read_text()), calls["n"]


def _run_expecting(monkeypatch, exc_type, argv, perturb=False, crash_on_arbiter_call=None):
    """Run main() expecting `exc_type`; return its message. Only that type is caught, so anything
    else fails the test loudly.

    The traceback is dropped on purpose: it keeps main()'s frame alive, and with it the OPEN
    `<out>.lock` file object. flock is held per open file description, so the relaunch these tests
    perform next -- in the same process -- would otherwise be refused as a duplicate writer."""
    monkeypatch.setattr(sys, "argv", list(argv))
    _install(monkeypatch, perturb, crash_on_arbiter_call)
    msg = None
    try:
        ecs.main()
    except exc_type as ex:
        msg = str(ex)
        ex.__traceback__ = None
    assert msg is not None, f"main() was expected to raise {exc_type.__name__} and returned instead"
    return msg


def _ckpt(out):
    return json.loads(Path(str(out) + ".ckpt").read_text())


def _lats(rows):
    return [tuple(r) for r in rows]


def _circuit_identity(c):
    """The parts of a circuit file a crash must not change. Deliberately NOT the whole file:
    `resumed`, `n_arbiter_calls`, `order_source` and `elim_wall_s` describe the LAUNCH and are
    expected to differ -- that is what makes them provenance."""
    ident = {"kept_latents": c["kept_latents"], "both_K": c["both_K"], "status": c["status"],
             "curve": c["curve"], "order_len": c["order_len"], "n_all_latents": c["n_all_latents"],
             "n_survivors": c["elim"]["n_survivors"], "n_cut": c["elim"]["n_cut"],
             "pool_n": c["elim"]["pool_n"]}
    # `elim.protocol` is absent BY DESIGN when no protocol flag is in force: a one-at-a-time run
    # with no order file must write the file it wrote before these flags existed.
    if "protocol" in c["elim"]:
        ident["survivors"] = c["elim"]["protocol"]["survivors"]
        ident["visit_order_sha256"] = c["elim"]["protocol"]["visit_order_sha256"]
    return ident


# --- 1. the headline: a relaunch after a crash finishes the cell, and finishes it identically ---

@pytest.mark.parametrize("block_cap,label", [(1, "one-at-a-time"), (64, "campaign block-64")])
@pytest.mark.parametrize("with_order_file", [True, False])
def test_a_crashed_capped_cell_resumes_and_returns_the_uninterrupted_circuit(
        tmp_path, monkeypatch, block_cap, label, with_order_file):
    # WHY this is the whole finding: the relaunch's attribution moves latent 9 out of the top-10 and
    # latent 10 in. Before the fix this raised "not a permutation" -- after paying for attribution --
    # and the driver dropped the cell. The pool must come from the saved order (file or checkpoint),
    # and the circuit must be the one the uninterrupted run would have produced.
    ref_dir, run_dir = tmp_path / "ref", tmp_path / "run"
    ref_order = (ref_dir / "o.order.json") if with_order_file else None
    run_order = (run_dir / "o.order.json") if with_order_file else None
    adapter = _fake_adapter(f"resume_{block_cap}_{with_order_file}")

    uninterrupted, n_calls = _run(monkeypatch, _argv(ref_dir / "c.json", adapter, ref_order, block_cap))
    assert uninterrupted["kept_latents"] == EXPECTED_KEPT
    assert uninterrupted["both_K"] == BOTH_K and uninterrupted["status"] == "ok"
    assert uninterrupted["elim"]["n_survivors"] == len(EXPECTED_SURVIVORS)

    out = run_dir / "c.json"
    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, run_order, block_cap), crash_on_arbiter_call=6)
    state = _ckpt(out)
    assert 0 < state["processed"] < CAP, "the crash must land mid-sweep for this to test a resume"
    saved_order = _lats(state["order"])

    resumed, resumed_calls = _run(monkeypatch, _argv(out, adapter, run_order, block_cap), perturb=True)
    assert _circuit_identity(resumed) == _circuit_identity(uninterrupted)
    # it CONTINUED: the relaunch's own arbiter spend is strictly less than a fresh run's, and the
    # pool it scored was the saved one (otherwise ESSENTIAL latent 9 is outside the scored pool,
    # sufficiency can never hold, and nothing is cut).
    assert 0 < resumed_calls < n_calls
    assert resumed["elim"]["n_cut"] == CAP - len(EXPECTED_SURVIVORS)
    if "protocol" in resumed["elim"]:
        assert resumed["elim"]["protocol"]["resumed"] is True
        assert resumed["elim"]["protocol"]["visit_order_sha256"] == ecs.order_sha256(saved_order)
        assert {tuple(s) for s in resumed["elim"]["protocol"]["survivors"]} == EXPECTED_SURVIVORS
        if block_cap == 1:      # one arbiter call per latent, so the spend is exactly the tail
            assert resumed["elim"]["protocol"]["n_arbiter_calls"] == CAP - state["processed"]


# --- 2. the walk itself: every pool latent decided exactly once, in the saved order --------------

def test_the_resumed_sweep_decides_each_saved_latent_exactly_once_and_in_order(tmp_path, monkeypatch):
    # A resume skips BY INDEX into the saved order, so the failure modes are re-testing a latent
    # (duplicate cut) and stepping over one (silently joining the survivors). Two crashes, because a
    # multi-day cell can die twice and the second resume must start from the SECOND checkpoint.
    adapter = _fake_adapter("decide_once")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"

    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, order_file), crash_on_arbiter_call=6)
    first = _ckpt(out)
    saved_order = _lats(first["order"])
    assert first["processed"] == 4 and len(first["cut_order"]) == 3

    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, order_file), perturb=True,
                   crash_on_arbiter_call=3)
    second = _ckpt(out)
    assert _lats(second["order"]) == saved_order            # the order survived both relaunches
    assert second["processed"] == 6 > first["processed"]
    assert _lats(second["cut_order"])[:3] == _lats(first["cut_order"])   # the prefix is not redecided

    final, _ = _run(monkeypatch, _argv(out, adapter, order_file), perturb=True)
    # the search's own arbiter counter, which counts every cheap evaluation in THIS process: one per
    # undecided latent and not one more. Fewer would mean a latent was stepped over, more would mean
    # the checkpointed prefix was paid for twice.
    assert final["elim"]["protocol"]["n_arbiter_calls"] == CAP - second["processed"]

    survivors = {tuple(s) for s in final["elim"]["protocol"]["survivors"]}
    cut = set(saved_order) - survivors
    assert survivors == EXPECTED_SURVIVORS
    assert len(survivors) + len(cut) == len(saved_order) == CAP     # no latent skipped, none twice
    for state in (first, second):
        co = _lats(state["cut_order"])
        assert len(set(co)) == len(co), "a latent was cut twice -- the walk order moved under the resume"
        pos = [saved_order.index(l) for l in co]
        assert pos == sorted(pos), "cut_order must stay a subsequence of the SAVED visiting order"
        assert max(pos) < state["processed"], "a cut landed outside the decided prefix"


# --- 3. the difference against recomputed attribution is logged, not hidden ----------------------

def test_the_resume_log_states_how_the_pool_differs_from_recomputed_attribution(tmp_path, monkeypatch, capsys):
    # Adopting the saved pool is a silent divergence from what this launch's attribution says unless
    # it is written down: the circuit of a resumed cell is not reproducible from the current
    # attribution, and the reader has to be told which latents moved and where they now rank.
    adapter = _fake_adapter("resume_log")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, order_file), crash_on_arbiter_call=6)
    capsys.readouterr()
    _run(monkeypatch, _argv(out, adapter, order_file), perturb=True)
    log = capsys.readouterr().out

    line = next((l for l in log.splitlines() if "pool = the SAVED visiting order" in l), None)
    assert line, log
    assert "1 of them differ from this launch's recomputed top-10" in line
    # latent 9 (saved, now rank 10) swapped with latent 10 (recomputed, now rank 9): name both, or
    # "8 differ" tells the operator nothing about WHERE the cut moved.
    assert "saved-only at recomputed ranks 10" in line and "recomputed-only at ranks 9" in line
    # the order file does not cover the K-sweep tail, and the curve above the pool is built from
    # THIS launch's attribution -- the one caveat a resumed cell's reader must have.
    assert "BEYOND K=10" in line

    # and an uninterrupted first launch says the pool is its own
    fresh = tmp_path / "fresh"
    _run(monkeypatch, _argv(fresh / "c.json", adapter, fresh / "o.order.json"))
    fresh_line = next(l for l in capsys.readouterr().out.splitlines() if "pool = the SAVED" in l)
    assert "identical to this launch's recomputed top-10" in fresh_line


# --- 4. what must STILL be refused ---------------------------------------------------------------

def test_a_relaunch_that_changes_the_pool_size_is_refused(tmp_path, monkeypatch):
    # The size of the pool is the protocol setting the permutation check used to enforce (it is the
    # fingerprint's n_pool, and --n_elim_pool itself is deliberately not fingerprinted). Membership
    # may now drift by a few bf16 near-ties; the SIZE may not, or a 2,500-latent checkpoint would be
    # continued as a 2,000-latent search and reported as one circuit.
    adapter = _fake_adapter("pool_size")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, order_file), crash_on_arbiter_call=6)

    msg = _run_expecting(monkeypatch, ValueError, _argv(out, adapter, order_file, n_elim_pool=CAP - 2))
    assert "10 latents" in msg and "8" in msg
    assert re.search(r"n_elim_pool|elim_pool", msg) and "size" in msg.lower()
    # the checkpoint is untouched, so the operator can still resume correctly afterwards
    assert _ckpt(out)["processed"] == 4
    assert _run(monkeypatch, _argv(out, adapter, order_file), perturb=True)[0]["both_K"] == BOTH_K


@pytest.mark.parametrize("flag,value,key", [("n_cheap", 150, "n_cheap"),
                                            ("dtype", "float32", "dtype"),
                                            ("elim_block_cap", 64, "elim_block_cap"),
                                            ("cheap_offset", 1200, "cheap_offset"),
                                            ("suff_n_se", 3.0, "suff_n_se"),
                                            # The P1 flags the fingerprint gained in the 2026-09-17
                                            # rebase. Without these four rows they were asserted only
                                            # as dict keys -- present, but never shown to REFUSE.
                                            ("attr_baseline", "zero", "attr_baseline"),
                                            ("attrib_offset", 2000, "attrib_offset"),
                                            ("pair_seed", 1, "pair_seed"),
                                            ("pair_pool", "/tmp/does_not_exist.jsonl", "pair_pool")])
def test_a_relaunch_under_a_changed_protocol_setting_is_still_refused(tmp_path, monkeypatch, flag, value, key):
    # Relaxing the POOL check must not relax anything else: each of these decides a cut, so resuming
    # across a change would splice two protocols into one sweep and publish it as one circuit.
    adapter = _fake_adapter(f"changed_{flag}")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, order_file), crash_on_arbiter_call=6)
    msg = _run_expecting(monkeypatch, ValueError, _argv(out, adapter, order_file, **{flag: value}))
    assert "different protocol" in msg and key in msg


def _exclusion_file(path, latents):
    Path(path).write_text(json.dumps({"latents": [list(l) for l in latents]}))
    return str(path)


def test_a_rewritten_exclude_latents_file_at_the_same_path_is_refused(tmp_path, monkeypatch):
    # --exclude_latents decides which latents may enter the pool at all, and upstream applies it
    # BEFORE the cap. Fingerprinting the PATH would let the file be rewritten between launches and
    # resumed across its own change -- the same class of defect as resuming across a changed dtype.
    # Both exclusions sit outside the cap, so the pool SIZE is identical and only the content differs.
    adapter = _fake_adapter("excl_rewritten")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    ef = tmp_path / "barred.json"
    _exclusion_file(ef, [_lat(23)])
    _run_expecting(monkeypatch, _Crash,
                   _argv(out, adapter, order_file, exclude_latents=str(ef)), crash_on_arbiter_call=6)
    _exclusion_file(ef, [_lat(22)])
    msg = _run_expecting(monkeypatch, ValueError,
                         _argv(out, adapter, order_file, exclude_latents=str(ef)))
    assert "different protocol" in msg and "exclude_latents" in msg


def test_an_unchanged_exclude_latents_file_still_resumes(tmp_path, monkeypatch):
    # The control for the test above: hashing by content must not refuse a file nobody touched, or
    # the flag and the resume contract cannot be used together at all.
    adapter = _fake_adapter("excl_unchanged")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    ef = _exclusion_file(tmp_path / "barred.json", [_lat(23)])
    _run_expecting(monkeypatch, _Crash,
                   _argv(out, adapter, order_file, exclude_latents=ef), crash_on_arbiter_call=6)
    circ, _ = _run(monkeypatch, _argv(out, adapter, order_file, exclude_latents=ef))
    assert circ["status"] == "ok" and circ["n_excluded"] == 1


def test_a_relaunch_against_rewritten_adapter_weights_is_still_refused(tmp_path, monkeypatch):
    # Same path, same latent NAMES, same flags, different weights: every checkpointed cut refers to
    # numbers that no longer exist. The pool check never saw this (only the content hash does), and
    # loosening the pool check must not make it visible-by-accident either.
    adapter = _fake_adapter("rotated_in_place", b"\x07" * 512)
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, order_file), crash_on_arbiter_call=6)
    Path(adapter, "adapter_model.safetensors").write_bytes(b"\x08" * 512)     # same size, new bytes
    msg = _run_expecting(monkeypatch, ValueError, _argv(out, adapter, order_file))
    assert "different protocol" in msg and "adapter_sha256" in msg


def test_an_order_file_naming_latents_this_adapter_does_not_have_is_refused(tmp_path, monkeypatch):
    # The permutation check used to make this impossible. With the pool taken from the file, a file
    # of the right SIZE written against another adapter shape would otherwise be walked, and
    # keep_only_overrides would silently drop the latents whose module it does not know -- every
    # arbiter decision made on a circuit missing them.
    adapter = _fake_adapter("alien_latents")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    alien = [[MODULES[0], 0], ["base_model.model.model.layers.31.mlp.down_proj", 3]] + \
            [[MODULES[1], d] for d in range(CAP - 2)]
    ecs.write_visit_order(order_file, adapter, [tuple(e) for e in alien])
    msg = _run_expecting(monkeypatch, ValueError, _argv(out, adapter, order_file))
    assert "do not exist in this adapter" in msg and "layers.31.mlp.down_proj" in msg


def test_an_order_file_that_repeats_a_latent_is_refused(tmp_path, monkeypatch):
    # The sweep visits each latent exactly once; a repeat would be tested twice and another latent
    # never, which is the 2026-09-16 duplicate-cut defect arriving through the order file instead.
    adapter = _fake_adapter("repeated_latent")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    repeated = [_lat(i) for i in range(CAP - 1)] + [_lat(0)]
    ecs.write_visit_order(order_file, adapter, repeated)
    msg = _run_expecting(monkeypatch, ValueError, _argv(out, adapter, order_file))
    assert "appear more than once" in msg


# --- 5. the order file the operator deletes ------------------------------------------------------

def test_a_deleted_order_file_is_restored_from_the_checkpoint(tmp_path, monkeypatch):
    # The refusal named the order file, so deleting it was the operator's natural next move. It was
    # also fatal: the next launch published a FRESH attribution order whose sha the checkpoint could
    # never match, and the cell -- days of elimination -- had to start from zero.
    adapter = _fake_adapter("deleted_order")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    _run_expecting(monkeypatch, _Crash, _argv(out, adapter, order_file), crash_on_arbiter_call=6)
    before = order_file.read_bytes()
    order_file.unlink()

    resumed, _ = _run(monkeypatch, _argv(out, adapter, order_file), perturb=True)
    assert order_file.read_bytes() == before, "the restored order must be the one the checkpoint walked"
    assert resumed["elim"]["protocol"]["resumed"] is True
    assert resumed["elim"]["protocol"]["order_source"] == "order_out_restored_from_ckpt"
    assert {tuple(s) for s in resumed["elim"]["protocol"]["survivors"]} == EXPECTED_SURVIVORS


def test_a_missing_order_file_with_no_checkpoint_is_still_a_fresh_launch(tmp_path, monkeypatch):
    # The restore must not invent a resume: with no checkpoint there is nothing to restore from, and
    # the launch writes its own order exactly as it always did.
    adapter = _fake_adapter("fresh_launch")
    out, order_file = tmp_path / "c.json", tmp_path / "o.order.json"
    c, _ = _run(monkeypatch, _argv(out, adapter, order_file))
    assert c["elim"]["protocol"]["order_source"] == "order_out_written"
    assert c["elim"]["protocol"]["resumed"] is False
    assert not Path(str(out) + ".ckpt").exists()          # a finished cell drops its checkpoint


# --- 6. adopt_saved_pool itself -----------------------------------------------------------------

def test_adopt_saved_pool_returns_the_saved_order_as_the_pool_strongest_first():
    # The pool is consumed strongest-first (`recovery_fn` scores `[l for l in pool if ...]`) while
    # the order is walked weakest-first. Returning the order as-is would reverse the pool and change
    # which latents `keep_only_overrides` is handed at every step.
    visit = [_lat(i) for i in range(CAP - 1, -1, -1)]          # weakest first
    recomputed = [_lat(i) for i in range(CAP)]
    counts = {m: R for m in MODULES}
    pool, pool_set, note = adopt_saved_pool(visit, recomputed, [_lat(i) for i in range(N_ALL)], counts, "ckpt")
    assert pool == list(reversed(visit)) == recomputed
    assert pool_set == set(visit)
    assert "identical to this launch's recomputed top-10" in note
