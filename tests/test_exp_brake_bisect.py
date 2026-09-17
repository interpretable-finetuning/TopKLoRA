"""The brake bisection: the group test must return exactly the brakes under an oracle, its minimality pass
must find every member required, and a cached measurement must run no job."""
import json
import subprocess

import pytest

from src.clcd import exp_brake_bisect as bb

CANDS = [("model.layers.20.mlp.gate_proj", i) for i in range(16)]


def oracle(brakes):
    """Sufficiency is restored exactly when every brake has been removed."""
    calls = []

    def restores(removed, label):
        calls.append((frozenset(removed), label))
        return set(brakes) <= set(removed)
    return restores, calls


@pytest.mark.parametrize("brakes", [{3}, {3, 9, 10}, {0, 15}, {7, 8}])
def test_find_returns_exactly_the_brakes(brakes):
    restores, calls = oracle({CANDS[i] for i in brakes})
    log = []
    H = bb.find(CANDS, set(), restores, log.append)
    assert set(H) == {CANDS[i] for i in brakes}
    assert len(H) == len(brakes)          # a hitting set, not a superset
    assert calls, "no measurement was taken"
    assert all("decision" in r for r in log)


def test_find_conditions_on_the_other_half_when_brakes_sit_in_both():
    restores, _ = oracle({CANDS[2], CANDS[13]})
    log = []
    bb.find(CANDS, set(), restores, log.append)
    assert log[0]["decision"] == "brakes in both halves; condition on the other"
    assert log[0]["tag"].startswith("depth0 |C|=16 |fixed|=0")


def test_minimality_pass_finds_every_member_required_under_the_oracle():
    brakes = {CANDS[3], CANDS[9], CANDS[10]}
    restores, _ = oracle(brakes)
    H = bb.find(CANDS, set(), restores, lambda r: None)
    assert all(not restores(set(H) - {h}, "minimality") for h in H)


def test_a_wrong_search_is_caught(monkeypatch):
    """The guard proves it can fail: a search that recurses into the wrong half returns the wrong set."""
    restores, _ = oracle({CANDS[3]})
    wrong = bb.find(CANDS, set(), lambda removed, label: not restores(removed, label), lambda r: None)
    assert set(wrong) != {CANDS[3]}


def test_measure_replays_from_the_cache_without_running_a_job(tmp_path, monkeypatch):
    base, cands = [("model.layers.15.self_attn.q_proj", 0)], CANDS[:4]
    s = bb.Search(tmp_path, tmp_path, "adapter", base, cands)
    kept = base + cands[1:]                       # removed = {cands[0]}
    key = bb.hashlib.sha256(json.dumps(sorted(kept)).encode()).hexdigest()[:16]
    rec = {"key": key, "status": "ok", "keep_only": 0.998, "ablate": 0.0, "shortfall": 0.002, "allow": 0.003,
           "n_kept": len(kept), "n_removed": 1}
    (tmp_path / "cache.json").write_text(json.dumps({key: rec}))
    s = bb.Search(tmp_path, tmp_path, "adapter", base, cands)

    def boom(*a, **k):
        raise AssertionError("a job was launched for a cached set")
    monkeypatch.setattr(subprocess, "run", boom)
    assert s.measure({cands[0]}, "cached") == rec
    assert s.restores({cands[0]}, "cached") is True
    with pytest.raises(AssertionError, match="launched"):
        s.measure(set(), "uncached")             # a different kept set is not in the cache
