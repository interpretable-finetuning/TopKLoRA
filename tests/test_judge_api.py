"""judge_api: strict parsing, stable pairing, and loud failure. CPU only, no network.

The defects these pin are not hypothetical -- each was observed live on 2026-09-16:
  * the repo's existing `_extract_score_1_to_5` scores "The instruction asks for 3 examples;
    quality is poor." as **3**; this module must return None there;
  * `max_tokens=16` returns content=None on a mandatory-reasoning model, so an unparseable
    response must be recorded, never coerced;
  * a strict json_schema returned `{}` for every call, i.e. a 100% failure rate that a
    mean-over-whatever-parsed would have hidden completely.
"""

import hashlib
import json
import random
import time

import pytest

from src.clcd import judge_api as J


# ---------------------------------------------------------------- strict parsing

@pytest.mark.parametrize("raw,want", [
    ("5", 5), ("1", 1), ("  4  ", 4), ("3\n", 3),
    # the exact string the repo's scraper mis-scores as 3
    ("The instruction asks for 3 examples; quality is poor.", None),
    ("I would rate this 2 out of 5.", None),
    ("10/10", None), ("0", None), ("6", None),
    ("{}", None), ("{\n  }", None),          # what strict json_schema actually returned
    ("", None), ("   ", None), (None, None),
    ("Score: 5", None),                       # a prefix is a refusal to answer as asked
])
def test_parse_score_is_strict(raw, want):
    assert J.parse_score(raw) == want


def test_parse_score_disagrees_with_the_repo_scraper_where_it_matters():
    from src.evaluate import _extract_score_1_to_5
    prose = "The instruction asks for 3 examples; quality is poor."
    assert _extract_score_1_to_5(prose) == 3      # the live defect
    assert J.parse_score(prose) is None           # this module refuses to guess


# ---------------------------------------------------------------- identity / pairing

def test_item_keys_are_stable_unique_and_follow_the_text_and_the_instrument(tmp_path):
    """Replaces `custom_id(scope, index)`, whose id moved with the RUN: judging a different file
    set, at a different chunk size, or with a second model produced ids that named other items,
    and v1 adopted them anyway. An id derived from the text and the instrument cannot do that."""
    cfg = _cfg(tmp_path)
    qs, gs = ["q0", "q1", "q2"], ["a0", "a1", "a2"]
    a = J.item_keys(qs, gs, cfg)
    assert a == J.item_keys(qs, gs, cfg)                                       # stable
    assert len(set(a)) == 3                                                    # unique per item
    # the id travels with the text, not with the position
    assert J.item_keys(qs[::-1], gs[::-1], cfg) == a[::-1]
    # a regenerated response is a DIFFERENT item: this is what stops a stale score being adopted
    assert J.item_keys(["q0"], ["a0-REGENERATED"], cfg)[0] != a[0]
    # another model is another instrument, even over byte-identical text (the forged-judge defect)
    assert not set(J.item_keys(qs, gs, _cfg(tmp_path, model="google/gemini-3.8-flash"))) & set(a)
    # so is another provider pin, or another token budget (16 returns content=None on this model)
    assert J.item_keys(qs, gs, _cfg(tmp_path, provider_only=("google-ai-studio",)))[0] != a[0]
    assert J.item_keys(qs, gs, _cfg(tmp_path, max_tokens=16))[0] != a[0]
    # degenerate base-model output repeats itself; each occurrence stays its own request
    assert len(set(J.item_keys(["q", "q"], ["a", "a"], cfg))) == 2


def test_judge_key_never_collides_with_the_local_judge():
    for indep in (False, True):
        k = J.judge_key_for("google/gemini-3.8-flash", independent=indep)
        assert "32b" not in k and k not in ("judge_32b", "judge_indep_32b")
        assert k.startswith("judge_indep_api_" if indep else "judge_api_")


def test_prompt_fingerprint_tracks_the_template(monkeypatch):
    base = J.prompt_fingerprint()
    assert base == J.prompt_fingerprint()
    monkeypatch.setattr(J, "JUDGE_SYSTEM_PROMPT", J.JUDGE_SYSTEM_PROMPT + " extra")
    assert J.prompt_fingerprint() != base


# ---------------------------------------------------------------- request / response shape

def test_build_request_uses_the_local_judge_prompt_verbatim():
    from src.evaluate import JUDGE_SYSTEM_PROMPT, _judge_user_prompt
    r = J.build_request("Q?", "A.", "cid-1", 2048)
    assert r["custom_id"] == "cid-1"
    msgs = r["body"]["messages"]
    assert msgs[0]["content"] == JUDGE_SYSTEM_PROMPT
    assert msgs[1]["content"] == _judge_user_prompt("Q?", "A.")
    assert r["body"]["temperature"] == 0
    # 16 returns content=None on a mandatory-reasoning model; the budget must stay large
    assert r["body"]["max_tokens"] >= 512
    assert "response_format" not in r["body"]   # strict schema returned {} for every call


def test_extract_results_maps_by_id_and_keeps_per_item_errors():
    payload = {"results": [
        {"custom_id": "a", "response": {"body": {
            "choices": [{"message": {"content": "5"}}], "usage": {"prompt_tokens": 300}}}},
        {"custom_id": "b", "error": {"message": "upstream refused"}},
        {"custom_id": "c", "response": {"body": {"choices": [{"message": {"content": None}}]}}},
    ]}
    got = J.extract_results(payload)
    assert got["a"]["raw"] == "5" and got["a"]["usage"] == {"prompt_tokens": 300}
    assert got["b"]["error"]["message"] == "upstream refused" and got["b"]["raw"] is None
    assert got["c"]["raw"] is None                       # present but empty, not dropped
    assert set(got) == {"a", "b", "c"}


def test_extract_results_on_an_empty_batch_is_empty_not_an_error():
    assert J.extract_results({"results": None}) == {}


# ---------------------------------------------------------------- loud failure (Rule 12)

def _cfg(tmp_path, **kw):
    return J.JudgeConfig(state_dir=tmp_path / "state", poll_seconds=0, **kw)


def test_length_mismatch_raises_instead_of_zipping(tmp_path):
    with pytest.raises(ValueError, match="Refusing to zip"):
        J.api_judge_scores(["q1", "q2"], ["a1"], scope="s", cfg=_cfg(tmp_path), api_key="k")


def test_missing_api_key_raises(tmp_path, monkeypatch):
    """Run from an empty dir: from the repo root the dotenv fallback finds the real key, and on
    2026-09-16 this test submitted a live batch and blocked on it for 5 h. Transport is poisoned
    so a regression fails fast instead of spending money."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    def no_network(*a, **k):
        raise AssertionError("test reached the batch API")

    monkeypatch.setattr(J, "submit", no_network)
    monkeypatch.setattr(J, "fetch", no_network)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        J.api_judge_scores(["q"], ["a"], scope="s", cfg=_cfg(tmp_path), api_key="")


def _no_listing(*a, **k):
    """Reconciliation must not touch the network on a state with no pending intent -- that is an
    extra live request per judging poll, 90 times over a campaign, for nothing to reconcile."""
    raise AssertionError("listed the account's batches with no pre-submit intent pending")


def _fake_api(monkeypatch, contents, status="completed"):
    """Batch transport replaced by a fake that returns `contents` in custom_id order."""
    sent = {}
    monkeypatch.setattr(J, "list_batches", _no_listing)

    def fake_submit(payload, cfg, key, *, intent_id=None):
        sent["ids"] = [r["custom_id"] for r in payload]
        sent["payload"] = payload
        sent["intent_id"] = intent_id
        return "batch-fake"

    def fake_fetch(bid, key):
        return {"status": status, "request_counts": {"total": len(sent["ids"])},
                "usage": {"total_tokens": 1},
                "results": [{"custom_id": cid, "response": {"body": {
                    "choices": [{"message": {"content": c}}], "usage": {}}}}
                    for cid, c in zip(sent["ids"], contents)]}

    monkeypatch.setattr(J, "submit", fake_submit)
    monkeypatch.setattr(J, "fetch", fake_fetch)
    return sent


def test_happy_path_returns_the_local_judge_contract(tmp_path, monkeypatch):
    _fake_api(monkeypatch, ["5", "4", "3"])
    out = J.api_judge_scores(["q"] * 3, ["a"] * 3, scope="s", cfg=_cfg(tmp_path), api_key="k")
    assert out["scores"] == [5, 4, 3] and out["n"] == 3 and out["mean"] == 4.0
    m = out["judge_meta"]
    assert m["n_failed"] == 0 and m["backend"] == "openrouter-batch"
    assert m["prompt_sha256"] and m["batches"]["chunk0"] == "batch-fake"


def test_failure_rate_above_the_registered_ceiling_raises(tmp_path, monkeypatch):
    # 2 of 4 unparseable: exactly the strict-schema disaster, which must NOT yield mean=4.5
    _fake_api(monkeypatch, ["5", "{}", "4", None])
    with pytest.raises(RuntimeError, match="produced no score"):
        J.api_judge_scores(["q"] * 4, ["a"] * 4, scope="s", cfg=_cfg(tmp_path), api_key="k")


def test_a_few_failures_under_the_ceiling_are_recorded_with_their_raw_text(tmp_path, monkeypatch):
    contents = ["5"] * 399 + ["nonsense"]
    _fake_api(monkeypatch, contents)
    out = J.api_judge_scores(["q"] * 400, ["a"] * 400, scope="s",
                             cfg=_cfg(tmp_path, max_failure_rate=0.01), api_key="k")
    m = out["judge_meta"]
    assert m["n_failed"] == 1 and out["n"] == 399
    assert m["failures"][0]["raw"] == "nonsense"   # raw text kept, not discarded
    assert out["scores"][-1] is None               # position preserved


def test_a_non_completed_batch_is_not_treated_as_empty(tmp_path, monkeypatch):
    _fake_api(monkeypatch, ["5"], status="failed")
    with pytest.raises(RuntimeError, match="ended 'failed'"):
        J.api_judge_scores(["q"], ["a"], scope="s", cfg=_cfg(tmp_path), api_key="k")


# ---------------------------------------------------------------- resumption

def test_a_restart_adopts_the_in_flight_batch_instead_of_resubmitting(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = {"n": 0}
    _fake_api(monkeypatch, ["5", "5"])
    real_submit = J.submit

    def counting_submit(payload, c, k, **kw):
        calls["n"] += 1
        return real_submit(payload, c, k)

    monkeypatch.setattr(J, "submit", counting_submit)
    J.api_judge_scores(["q"] * 2, ["a"] * 2, scope="same-scope", cfg=cfg, api_key="k")
    assert calls["n"] == 1
    # second run over the same scope must pay nothing: items are already in state
    J.api_judge_scores(["q"] * 2, ["a"] * 2, scope="same-scope", cfg=cfg, api_key="k")
    assert calls["n"] == 1, "a rerun resubmitted the batch and would have paid twice"


def test_state_is_written_atomically_and_is_valid_json(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _fake_api(monkeypatch, ["5"])
    J.api_judge_scores(["q"], ["a"], scope="sc", cfg=cfg, api_key="k")
    files = list((tmp_path / "state").glob("*.json"))
    assert len(files) == 1 and not list((tmp_path / "state").glob("*.tmp"))
    json.loads(files[0].read_text())


# ------------------------------------------------- post-create 404 window (observed live)

class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status; self._p = payload; self.text = text
    def json(self):
        return self._p


def test_fetch_tolerates_the_post_create_404_window(monkeypatch):
    """A freshly created batch 404s for a while. Raising there killed a run whose work was
    already paid for."""
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 4:
            return _Resp(404, text='{"error":{"message":"Batch job ... not found."}}')
        return _Resp(200, {"status": "completed"})

    monkeypatch.setattr(J.requests, "get", fake_get)
    out = J.fetch("batch-x", "k", grace_s=60, sleep=lambda s: None)
    assert out == {"status": "completed"} and calls["n"] == 4


def test_fetch_still_raises_on_a_persistent_404(monkeypatch):
    monkeypatch.setattr(J.requests, "get",
                        lambda *a, **k: _Resp(404, text="gone"))
    with pytest.raises(RuntimeError, match="404"):
        J.fetch("batch-x", "k", grace_s=0, sleep=lambda s: None)


def test_fetch_retries_a_429_then_succeeds(monkeypatch):
    seq = [_Resp(429, text="rate limited"), _Resp(200, {"status": "in_progress"})]
    monkeypatch.setattr(J.requests, "get", lambda *a, **k: seq.pop(0))
    assert J.fetch("b", "k", grace_s=60, sleep=lambda s: None) == {"status": "in_progress"}


def test_fetch_does_not_retry_a_401(monkeypatch):
    monkeypatch.setattr(J.requests, "get", lambda *a, **k: _Resp(401, text="bad key"))
    with pytest.raises(RuntimeError, match="401"):
        J.fetch("b", "k", grace_s=600, sleep=lambda s: None)


def test_api_key_is_read_from_dotenv_when_absent_from_env(tmp_path, monkeypatch):
    """The key lives in `.env`; nothing on the judging path loaded it, so the guard fired on a
    key that was on disk the whole time."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-fromdotenv\n")
    assert J._api_key_from_env() == "sk-or-v1-fromdotenv"


def test_env_var_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-dotenv\n")
    assert J._api_key_from_env() == "sk-from-env"


# ---------------------------------------------------------------- judge_saved_gens_big, api path

CONDS = ("intact", "ablate_circuit", "base")


def _surgical(path, tag, rng, n_clean=3, n_indep=2):
    """A minimal surgical JSON. Every generation starts with its own question and is assigned a
    random score, so a split that hands one file's (or condition's, or field's) scores to another
    changes the values on disk instead of passing unnoticed."""
    cq = [f"{tag}-cq{i}" for i in range(n_clean)]
    iq = [f"{tag}-iq{i}" for i in range(n_indep)]
    truth = {}
    d = {"clean_questions": cq, "indep_questions": iq, "conditions": {}}
    for c in CONDS:
        cg = [f"{q}|{c}|clean" for q in cq]
        ig = [f"{q}|{c}|indep" for q in iq]
        for g in cg + ig:
            truth[g] = rng.randint(1, 5)
        d["conditions"][c] = {"clean_gens": cg, "indep_gens": ig}
    path.write_text(json.dumps(d))
    return truth


def _run_big(monkeypatch, argv):
    import sys as _sys
    from src.clcd import judge_saved_gens_big as big
    monkeypatch.setattr(_sys, "argv", ["judge_saved_gens_big", "--judge_backend", "api", *argv])
    big.main()


def _fake_scores(monkeypatch, truth, *, fail=(), calls=None):
    def fake(qs, gs, *, scope, cfg, **k):
        assert len(qs) == len(gs)
        for q, g in zip(qs, gs):
            assert g.startswith(q + "|"), f"question {q!r} zipped with generation {g!r}"
        if calls is not None:
            calls.append(dict(n=len(gs), scope=scope, cfg=cfg))
        scores = [None if g in fail else truth[g] for g in gs]
        return {"mean": 0.0, "n": len(gs), "scores": scores, "judge_meta": {"model": cfg.model}}
    monkeypatch.setattr(J, "api_judge_scores", fake)


def test_cross_file_batch_returns_every_score_to_its_own_file_condition_and_field(tmp_path, monkeypatch):
    """Five organisms go out as ONE batch and come back as one flat list. An off-by-one-file split
    would write organism A's capability scores into organism B's retention ratio, and the result
    would still look like a plausible number."""
    rng = random.Random(0)
    monkeypatch.chdir(tmp_path)
    fa, fb = tmp_path / "a_surgical.json", tmp_path / "b_surgical.json"
    truth = {**_surgical(fa, "A", rng), **_surgical(fb, "B", rng, n_clean=4, n_indep=1)}
    calls = []
    _fake_scores(monkeypatch, truth, calls=calls)
    _run_big(monkeypatch, ["--files", str(fa), str(fb)])

    assert len(calls) == 1 and calls[0]["n"] == 3 * (3 + 2) + 3 * (4 + 1)
    ck, ik = J.judge_key_for(J.DEFAULT_MODEL), J.judge_key_for(J.DEFAULT_MODEL, independent=True)
    for f in (fa, fb):
        d = json.loads(f.read_text())
        for c in CONDS:
            rec = d["conditions"][c]
            assert rec[ck]["scores"] == [truth[g] for g in rec["clean_gens"]]
            assert rec[ik]["scores"] == [truth[g] for g in rec["indep_gens"]]
            assert rec[ck]["mean"] == sum(truth[g] for g in rec["clean_gens"]) / len(rec["clean_gens"])


def test_a_legacy_v1_state_file_is_reported_loudly_and_never_read(tmp_path, monkeypatch):
    """MIGRATION. v1 keyed the state file by a hash of the submitting run's todo list and every
    item id by that hash plus a list INDEX, and never stored the request text -- so a v1 result
    cannot be matched to an item here and a v1 batch id cannot be re-keyed. An uncollected v1
    batch is money that is now uncollectable, which the run must SAY rather than resubmit in
    silence. What it replaces is worse: v1 adopted purely by scope, so a run over regenerated
    text, a different file set or a second model adopted ids that named other items entirely."""
    cfg = _cfg(tmp_path)
    v1_scope = f"{tmp_path}/a_surgical.json|ALL"
    v1_id = f"{hashlib.sha256(v1_scope.encode()).hexdigest()[:20]}-000000"
    v1 = {"scope": v1_scope, "batches": {"chunk0": "batch-v1-inflight"}, "items": {v1_id: {"raw": "5"}}}
    (tmp_path / "state").mkdir(parents=True)
    legacy_path = tmp_path / "state" / f"{hashlib.sha256(v1_scope.encode()).hexdigest()[:20]}.json"
    legacy_path.write_text(json.dumps(v1))

    lines = []
    _fake_api(monkeypatch, ["4", "4"])
    out = J.api_judge_scores(["q1", "q2"], ["a1", "a2"], scope="s", cfg=cfg, api_key="k",
                             log=lines.append)
    assert out["scores"] == [4, 4]                      # the run proceeds; it does not wedge
    assert out["judge_meta"]["legacy_state_unadoptable_batches"] == 1
    text = "\n".join(lines)
    assert "WARNING" in text and "batch-v1-inflight" in text and str(legacy_path) in text
    assert json.loads(legacy_path.read_text()) == v1    # left exactly as it was, never rewritten
    assert (tmp_path / "state" / f"{J.model_slug(cfg.model)}.json").exists()
    assert J.legacy_state_files(cfg) == [
        {"path": str(legacy_path), "scope": v1_scope, "batches": ["batch-v1-inflight"], "n_items": 1}]


def test_a_state_file_of_an_unknown_version_raises_instead_of_being_guessed_at(tmp_path, monkeypatch):
    """A layout this code does not know is not something to half-read: that is how a batch id is
    lost or a score is attached to the wrong text."""
    cfg = _cfg(tmp_path)
    p = J._state_path(cfg)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"version": 99, "batches": {}, "items": {}}))
    _fake_api(monkeypatch, ["4"])
    with pytest.raises(RuntimeError, match="state format version"):
        J.api_judge_scores(["q"], ["a"], scope="s", cfg=cfg, api_key="k")


def test_one_bad_file_raises_even_when_the_merged_failure_rate_is_under_the_ceiling(tmp_path, monkeypatch):
    """The ceiling was pre-registered per file. 1 failure in A's 15 items is 6.7%; merged with a
    600-item file it is 0.16%, under 0.5%. Refusals concentrate on degenerate output, so a
    diluted ceiling would quietly bias exactly the ablated organisms it exists to protect."""
    rng = random.Random(2)
    monkeypatch.chdir(tmp_path)
    fa, fb = tmp_path / "a_surgical.json", tmp_path / "b_surgical.json"
    truth = {**_surgical(fa, "A", rng), **_surgical(fb, "B", rng, n_clean=100, n_indep=100)}
    _fake_scores(monkeypatch, truth, fail={"A-cq0|base|clean"})
    with pytest.raises(RuntimeError, match="a_surgical.json: 1/15"):
        _run_big(monkeypatch, ["--files", str(fa), str(fb)])
    for f in (fa, fb):   # nothing written for either file
        assert J.judge_key_for(J.DEFAULT_MODEL) not in json.loads(f.read_text())["conditions"]["intact"]


def test_a_non_default_model_needs_an_explicit_provider_and_gets_exactly_that_pin(tmp_path, monkeypatch):
    """The default pin (openai) belongs to the default model. Carried over to another model it names
    a provider with no endpoint for it, so the whole batch fails -- after waiting for it."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a_surgical.json"
    truth = _surgical(f, "A", random.Random(3))
    calls = []
    _fake_scores(monkeypatch, truth, calls=calls)
    with pytest.raises(SystemExit, match="explicit --provider"):
        _run_big(monkeypatch, ["--files", str(f), "--api_model", "google/gemini-3.8-flash"])
    assert calls == []

    _run_big(monkeypatch, ["--files", str(f), "--api_model", "google/gemini-3.8-flash",
                           "--provider", "google-ai-studio", "--chunk_size", "500"])
    cfg = calls[0]["cfg"]
    assert (cfg.model, cfg.provider_only, cfg.chunk_size) == ("google/gemini-3.8-flash", ("google-ai-studio",), 500)
    assert "judge_api_gemini_3_8_flash" in json.loads(f.read_text())["conditions"]["intact"]


def test_a_run_with_no_model_flags_uses_luna_pinned_to_openai(tmp_path, monkeypatch):
    """User instruction 2026-09-16: luna for all future runs. A default that drifted back to gemini
    would switch the instrument AND the artifact key silently, and retention ratios would then mix
    judges across organisms. Pins the behaviour of a bare invocation, not a constant."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a_surgical.json"
    truth = _surgical(f, "A", random.Random(6))
    calls = []
    _fake_scores(monkeypatch, truth, calls=calls)
    _run_big(monkeypatch, ["--files", str(f)])
    cfg = calls[0]["cfg"]
    assert (cfg.model, tuple(cfg.provider_only), cfg.chunk_size) == ("openai/gpt-5.6-luna", ("openai",), 10000)
    rec = json.loads(f.read_text())["conditions"]["intact"]
    assert "judge_api_gpt_5_6_luna" in rec and "judge_indep_api_gpt_5_6_luna" in rec


def test_a_concurrent_judge_writing_the_same_file_is_not_erased(tmp_path, monkeypatch):
    """Two judges (e.g. gemini finishing one file while luna judges five) hold the same surgical
    JSON for hours. Writing back the copy loaded at start erases whatever the other wrote meanwhile
    -- scores that were paid for, gone without an error."""
    rng = random.Random(4)
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a_surgical.json"
    truth = _surgical(f, "A", rng)
    inner = []
    _fake_scores(monkeypatch, truth, calls=inner)
    real = J.api_judge_scores

    def other_judge_writes_mid_batch(qs, gs, **k):
        d = json.loads(f.read_text())
        d["conditions"]["intact"]["judge_other"] = {"mean": 3.0}
        f.write_text(json.dumps(d))
        return real(qs, gs, **k)

    monkeypatch.setattr(J, "api_judge_scores", other_judge_writes_mid_batch)
    _run_big(monkeypatch, ["--files", str(f)])
    rec = json.loads(f.read_text())["conditions"]["intact"]
    assert rec["judge_other"] == {"mean": 3.0}
    assert rec[J.judge_key_for(J.DEFAULT_MODEL)]["scores"] == [truth[g] for g in rec["clean_gens"]]


def test_scores_are_not_written_against_generations_that_changed_mid_batch(tmp_path, monkeypatch):
    rng = random.Random(5)
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a_surgical.json"
    truth = _surgical(f, "A", rng)
    _fake_scores(monkeypatch, truth)
    real = J.api_judge_scores

    def regenerated_mid_batch(qs, gs, **k):
        d = json.loads(f.read_text())
        d["conditions"]["base"]["clean_gens"][0] = "A-cq0|base|clean-REGENERATED"
        f.write_text(json.dumps(d))
        return real(qs, gs, **k)

    monkeypatch.setattr(J, "api_judge_scores", regenerated_mid_batch)
    with pytest.raises(RuntimeError, match="generations changed on disk"):
        _run_big(monkeypatch, ["--files", str(f)])
    assert J.judge_key_for(J.DEFAULT_MODEL) not in json.loads(f.read_text())["conditions"]["intact"]


# ---------------------------------------------------------------- chunk submission order

def _fake_fleet(monkeypatch, state_path_getter=None):
    """Transport that records the ORDER of submit/fetch calls and answers every batch as completed."""
    log, ids = [], {}
    monkeypatch.setattr(J, "list_batches", _no_listing)

    def fake_submit(payload, cfg, key, *, intent_id=None):
        bid = f"batch-{len(ids)}"
        ids[bid] = [r["custom_id"] for r in payload]
        log.append(("submit", bid))
        return bid

    def fake_fetch(bid, key, **kw):
        log.append(("fetch", bid))
        if state_path_getter is not None:
            state_path_getter()
        return {"status": "completed", "request_counts": {"total": len(ids[bid])}, "usage": {},
                "results": [{"custom_id": cid, "response": {"body": {
                    "choices": [{"message": {"content": "4"}}], "usage": {}}}} for cid in ids[bid]]}

    monkeypatch.setattr(J, "submit", fake_submit)
    monkeypatch.setattr(J, "fetch", fake_fetch)
    return log


def test_every_chunk_is_submitted_before_any_is_waited_on(tmp_path, monkeypatch):
    """Submit-then-wait per chunk costs one full batch turnaround each: the 19-file sparse re-judge
    spent 3 h 24 min on chunk0 before chunk1 existed. Six chunks that way is a day of waiting for
    work the API could already have been doing."""
    log = _fake_fleet(monkeypatch)
    out = J.api_judge_scores(["q"] * 6, ["a"] * 6, scope="s",
                             cfg=_cfg(tmp_path, chunk_size=2), api_key="k")
    assert [kind for kind, _ in log] == ["submit"] * 3 + ["fetch"] * 3
    assert out["scores"] == [4] * 6
    assert set(out["judge_meta"]["batches"]) == {"chunk0", "chunk1", "chunk2"}


def test_every_batch_id_is_on_disk_before_the_first_wait(tmp_path, monkeypatch):
    """A batch is paid for the moment it is created and cannot be cancelled. If the process dies
    while waiting, only ids already persisted can be adopted; anything else is paid for twice."""
    cfg = _cfg(tmp_path, chunk_size=2)
    seen = {}

    def check_state():
        if "at_first_fetch" not in seen:
            seen["at_first_fetch"] = json.loads(J._state_path(cfg).read_text())["batches"]

    _fake_fleet(monkeypatch, state_path_getter=check_state)
    J.api_judge_scores(["q"] * 6, ["a"] * 6, scope="s", cfg=cfg, api_key="k")
    saved = seen["at_first_fetch"]
    assert set(saved) == {"batch-0", "batch-1", "batch-2"}
    # the id alone is not enough to adopt: what makes it adoptable by a run whose file list has
    # moved on is the list of item keys it carries, which must be on disk with it.
    keys = J.item_keys(["q"] * 6, ["a"] * 6, cfg)
    assert [k for b in ("batch-0", "batch-1", "batch-2") for k in saved[b]["item_keys"]] == keys


def test_a_restart_adopts_in_flight_chunks_and_submits_only_the_missing_ones(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, chunk_size=2)
    keys = J.item_keys(["q"] * 6, ["a"] * 6, cfg)
    J._save_state(cfg, {"version": J.STATE_VERSION, "model": cfg.model, "items": {},
                        "batches": {"batch-inflight": {"item_keys": keys[:2], "drained": False}}})
    log, ids = [], {"batch-inflight": keys[:2]}

    def fake_submit(payload, c, k, **kw):
        bid = f"batch-new-{len(log)}"
        ids[bid] = [r["custom_id"] for r in payload]
        log.append(("submit", bid))
        return bid

    def fake_fetch(bid, k, **kw):
        log.append(("fetch", bid))
        return {"status": "completed", "request_counts": {"total": len(ids[bid])}, "usage": {},
                "results": [{"custom_id": cid, "response": {"body": {
                    "choices": [{"message": {"content": "3"}}], "usage": {}}}} for cid in ids[bid]]}

    monkeypatch.setattr(J, "submit", fake_submit)
    monkeypatch.setattr(J, "fetch", fake_fetch)
    out = J.api_judge_scores(["q"] * 6, ["a"] * 6, scope="s", cfg=cfg, api_key="k")
    assert [b for kind, b in log if kind == "submit"] == ["batch-new-0", "batch-new-1"]
    assert ("fetch", "batch-inflight") in log
    assert out["scores"] == [3] * 6


def test_sequential_mode_still_waits_between_submissions(tmp_path, monkeypatch):
    """The gemini queue jam is the reason this escape hatch exists: five concurrent batches there
    sat at 0/500 for 108 minutes."""
    log = _fake_fleet(monkeypatch)
    J.api_judge_scores(["q"] * 6, ["a"] * 6, scope="s",
                       cfg=_cfg(tmp_path, chunk_size=2, parallel_chunks=False), api_key="k")
    assert [kind for kind, _ in log] == ["submit", "fetch"] * 3


def test_a_restart_submits_exactly_the_items_not_already_stored(tmp_path, monkeypatch):
    """Was `test_chunk_keys_name_the_same_items_after_a_restart`, which pinned the v1 invariant
    that chunkN must be cut from the FULL item list so a saved chunk0 id kept naming its own
    items. Content keys make chunk numbering irrelevant -- what must hold is that a restart
    requests every item it does not have and not one it does, and never re-fetches a batch it has
    already drained."""
    cfg = _cfg(tmp_path, chunk_size=2)
    keys = J.item_keys(["q"] * 6, ["a"] * 6, cfg)
    done = {k: {"raw": "5", "usage": {}, "error": None} for k in keys[:2]}
    J._save_state(cfg, {"version": J.STATE_VERSION, "model": cfg.model, "items": done,
                        "batches": {"batch-done": {"item_keys": keys[:2], "drained": True}}})
    sent, ids = [], {"batch-done": keys[:2]}

    def fake_submit(payload, c, k, **kw):
        bid = f"batch-{len(sent)}"
        ids[bid] = [r["custom_id"] for r in payload]
        sent.append(ids[bid])
        return bid

    fetched = []

    def fake_fetch(bid, k, **kw):
        fetched.append(bid)
        return {"status": "completed", "request_counts": {"total": len(ids[bid])}, "usage": {},
                "results": [{"custom_id": cid, "response": {"body": {
                    "choices": [{"message": {"content": "2"}}], "usage": {}}}} for cid in ids[bid]]}

    monkeypatch.setattr(J, "submit", fake_submit)
    monkeypatch.setattr(J, "fetch", fake_fetch)
    out = J.api_judge_scores(["q"] * 6, ["a"] * 6, scope="s", cfg=cfg, api_key="k")
    assert sent == [keys[2:4], keys[4:6]]
    assert "batch-done" not in fetched              # a chunk already stored is not re-fetched
    assert out["scores"] == [5, 5, 2, 2, 2, 2]      # chunk0's results kept, the rest newly judged


def test_submission_stops_at_the_in_flight_request_ceiling(tmp_path, monkeypatch):
    """The account allows 20,000 in-flight batch REQUESTS. Submitting past it returns 429 and the
    run dies mid-fleet (observed live 2026-09-16 on chunk3 of the sparse re-judge). The loop must
    wait for an earlier chunk to land instead of being refused."""
    log = _fake_fleet(monkeypatch)
    cfg = _cfg(tmp_path, chunk_size=2, max_in_flight_requests=4)
    out = J.api_judge_scores(["q"] * 8, ["a"] * 8, scope="s", cfg=cfg, api_key="k")
    # never more than two chunks (4 requests) outstanding: submit,submit,fetch,submit,fetch,submit,fetch,fetch
    outstanding, peak = 0, 0
    for kind, _ in log:
        outstanding += 1 if kind == "submit" else -1
        peak = max(peak, outstanding)
    assert peak == 2, log
    assert out["scores"] == [4] * 8


# ------------------------------------- v2 state: adoption is keyed by the work, not by the run
#
# Every scenario below was reproduced against the v1 (scope-keyed) code first, so none of these
# is a check that could not fail: v1 resubmitted in S1/S2/S7, wrote stale scores in S4, and
# mis-attributed 1,991 of 2,403 scores in S5.

def _stub_score(prompt: str, offset: int = 0) -> int:
    """A deterministic score for the exact prompt text. The point is that a score adopted for the
    wrong item -- or served out of another model's state -- lands on disk as the wrong NUMBER
    instead of as a plausible one."""
    return 1 + (int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) + offset) % 5


def _expected(question: str, generation: str, offset: int = 0) -> int:
    from src.evaluate import _judge_user_prompt
    return _stub_score(_judge_user_prompt(question, generation), offset)


def _snapshot_of(model: str) -> str:
    """What the account's listing answers for a batch created with `model`: the pinned snapshot,
    `openai/gpt-5.6-luna-20260709` for a create that said `openai/gpt-5.6-luna:batch` (live
    2026-09-17). The request string and the listed string are never equal, so a fake that echoes
    the request back would let an exact model comparison pass here and fail in production."""
    return model.split(":")[0] + "-20260709"


def _listed(bid, *, total, created_at, model="openai/gpt-5.6-luna-20260709",
            status="in_progress"):
    """ONE BATCH EXACTLY AS THE LIST ENDPOINT RETURNS IT (probed live 2026-09-17, real account):
    these keys and no others -- in particular NO `metadata`, whatever the create call sent."""
    return {"id": bid, "object": "batch", "endpoint": "/v1/chat/completions",
            "completion_window": "24h", "model": model, "status": status,
            "created_at": int(created_at), "finalized_at": None, "error": None, "results": None,
            "usage": None, "request_counts": {"total": total, "completed": 0, "failed": 0}}


class _Fleet:
    """Stubbed OpenRouter batch API WITH A MEMORY: a batch submitted by one run stays fetchable by
    the next, which is what makes crash-and-relaunch testable at all. No network, no key."""

    def __init__(self, offset=0):
        self.offset = offset
        self.jobs = {}              # batch id -> {custom_id: score}
        self.listing = {}           # batch id -> the record the account's list endpoint returns
        self.log = []               # [("submit"|"fetch"|"list", batch id or None)]
        self.die_on_fetch = False
        self.fail = None            # callable(cid, batch id) -> an error payload, or None
        self.drop = set()           # cids the completed batch simply does not return
        self.unparseable = set()    # cids the model ANSWERS with junk (the strict-schema `{}`)
        self.die_after_submit = False   # the orphan window: the batch exists, the id never lands
        # The live API accepts `metadata` on create and NEVER echoes it in the listing (probed
        # 2026-09-17), so this is the default. False is the hypothetical future API.
        self.drop_metadata = True
        self.listing_complete = True

    def install(self, monkeypatch):
        monkeypatch.setattr(J, "submit", self.submit)
        monkeypatch.setattr(J, "fetch", self.fetch)
        monkeypatch.setattr(J, "list_batches", self.list_batches)
        return self

    @property
    def n_requests(self):
        return sum(len(self.jobs[b]) for k, b in self.log if k == "submit")

    def submit(self, payload, cfg, api_key, *, intent_id=None):
        bid = f"batch-{len(self.jobs)}"
        self.jobs[bid] = {r["custom_id"]: _stub_score(r["body"]["messages"][1]["content"], self.offset)
                          for r in payload}
        assert len(self.jobs[bid]) == len(payload), "duplicate custom_id inside one batch"
        self.listing[bid] = _listed(bid, total=len(payload), created_at=time.time(),
                                    model=_snapshot_of(cfg.batch_model()))
        if not self.drop_metadata:
            self.listing[bid]["metadata"] = {J.INTENT_METADATA_KEY: intent_id}
        self.log.append(("submit", bid))
        if self.die_after_submit:
            raise KeyboardInterrupt("process killed after the batch was created, before its id "
                                    "reached the state file")
        return bid

    def list_batches(self, api_key, **kw):
        self.log.append(("list", None))
        return [dict(b) for b in self.listing.values()], self.listing_complete

    def fetch(self, bid, api_key, **kw):
        self.log.append(("fetch", bid))
        if self.die_on_fetch:
            raise KeyboardInterrupt("process killed while waiting on the batch")
        results = []
        for cid, s in self.jobs[bid].items():
            if cid in self.drop:                      # a completed batch missing one of its items
                continue
            err = self.fail(cid, bid) if self.fail else None
            if err:                                   # a per-item transport failure, not an answer
                results.append({"custom_id": cid, "error": err})
                continue
            content = "{}" if cid in self.unparseable else str(s)
            results.append({"custom_id": cid, "response": {"body": {
                "choices": [{"message": {"content": content}}], "usage": {}}}})
        return {"status": "completed", "request_counts": {"total": len(self.jobs[bid])},
                "usage": {"total_tokens": 10 * len(self.jobs[bid])}, "results": results}


def _check_scores(path, model=None, offset=0):
    """Every score on disk must be the score OF THE TEXT IT IS WRITTEN AGAINST -- not merely
    present, and not merely a plausible 1-5."""
    model = model or J.DEFAULT_MODEL
    d = json.loads(path.read_text())
    ck, ik = J.judge_key_for(model), J.judge_key_for(model, independent=True)
    n = 0
    for cond, rec in d["conditions"].items():
        for key, qs, field in ((ck, d["clean_questions"], "clean_gens"),
                               (ik, d["indep_questions"], "indep_gens")):
            assert key in rec, f"{path.name} {cond}: {key} missing"
            want = [_expected(q, g, offset) for q, g in zip(qs, rec[field])]
            assert len(want) == len(rec[field])
            assert rec[key]["scores"] == want, f"{path.name} {cond} {key} scored other text"
            n += len(want)
    return n


def _big():
    from src.clcd import judge_saved_gens_big as big
    return big


def test_a_crash_between_two_files_adopts_on_relaunch_instead_of_resubmitting(tmp_path, monkeypatch):
    """THE FLAGGED DEFECT, end to end. v1 keyed the state by the list of files STILL NEEDING
    scores, so writing file A and dying gave the relaunch a shorter list, a different key, no
    state file at all, and a fresh submission of B and C -- work already complete on the server
    and already paid for, sitting in a state file nothing would ever read again. At campaign
    scale: a crash after file 1 of 90 resubmits 89 files, ~$17 and ~16.7 h of queue time."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    files = [tmp_path / f"{t.lower()}_surgical.json" for t in "ABC"]
    for f, t in zip(files, "ABC"):
        _surgical(f, t, random.Random(0))
    fleet = _Fleet().install(monkeypatch)
    big = _big()
    real_write, n = big.write_json_atomic, {"w": 0}

    def crashing_write(path, obj, **kw):
        n["w"] += 1
        if n["w"] > 1:
            raise KeyboardInterrupt("SIGKILL proxy: dies after the first file is written")
        real_write(path, obj, **kw)

    monkeypatch.setattr(big, "write_json_atomic", crashing_write)
    with pytest.raises(KeyboardInterrupt):
        _run_big(monkeypatch, ["--files", *map(str, files)])
    paid = fleet.n_requests
    assert paid == 45                                    # 3 files x 3 conditions x (3 + 2)
    assert J.judge_key_for(J.DEFAULT_MODEL) in json.loads(files[0].read_text())["conditions"]["base"]
    assert J.judge_key_for(J.DEFAULT_MODEL) not in json.loads(files[1].read_text())["conditions"]["base"]

    monkeypatch.setattr(big, "write_json_atomic", real_write)
    _run_big(monkeypatch, ["--files", *map(str, files)])
    assert fleet.n_requests == paid, "the relaunch resubmitted work that was already paid for"
    assert sum(_check_scores(f) for f in files) == 45


def test_a_new_file_appearing_between_runs_does_not_resubmit_the_batch_in_flight(tmp_path, monkeypatch):
    """What the incremental judging worker does BY DESIGN -- it picks up surgical files as
    searches finish. Under v1 that changed the scope on every invocation, so a run that died
    waiting had its entire in-flight fleet resubmitted. A batch cannot be cancelled, so both
    copies are paid for and the second queues behind the first."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    a, b, c = (tmp_path / f"{t}_surgical.json" for t in "abc")
    for f, t in zip((a, b), "AB"):
        _surgical(f, t, random.Random(1))
    fleet = _Fleet().install(monkeypatch)
    fleet.die_on_fetch = True
    with pytest.raises(KeyboardInterrupt):
        _run_big(monkeypatch, ["--files", str(a), str(b)])
    assert fleet.n_requests == 30 and [k for k, _ in fleet.log] == ["submit", "fetch"]

    fleet.die_on_fetch = False
    _surgical(c, "C", random.Random(2))          # a search finishes while nothing is judging
    _run_big(monkeypatch, ["--files", str(a), str(b), str(c)])
    submitted = [bid for k, bid in fleet.log if k == "submit"]
    assert submitted == ["batch-0", "batch-1"], "resubmitted the fleet that was already in flight"
    assert len(fleet.jobs["batch-1"]) == 15, "the second job must carry only the new file's items"
    assert ("fetch", "batch-0") in fleet.log[1:]
    assert sum(_check_scores(f) for f in (a, b, c)) == 45


def test_a_kill_between_the_results_landing_and_the_write_back_loses_nothing(tmp_path, monkeypatch):
    """The results are merged into the state before anything is written to a surgical file. A
    kill in that window must cost one re-fetch at most -- never a resubmission, and never a
    second charge for items the API has already answered."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    f = tmp_path / "a_surgical.json"
    _surgical(f, "A", random.Random(3))
    fleet = _Fleet().install(monkeypatch)
    big = _big()
    real_write = big.write_json_atomic

    def killed_before_writing(path, obj, **kw):
        raise KeyboardInterrupt("SIGKILL proxy: dies after the batch landed, before the write")

    monkeypatch.setattr(big, "write_json_atomic", killed_before_writing)
    with pytest.raises(KeyboardInterrupt):
        _run_big(monkeypatch, ["--files", str(f)])
    assert fleet.n_requests == 15
    assert J.judge_key_for(J.DEFAULT_MODEL) not in json.loads(f.read_text())["conditions"]["base"]
    before = list(fleet.log)

    monkeypatch.setattr(big, "write_json_atomic", real_write)
    _run_big(monkeypatch, ["--files", str(f)])
    assert fleet.log == before, "a drained batch was re-fetched or its items resubmitted"
    assert _check_scores(f) == 15


def test_after_a_generation_changes_the_relaunch_judges_the_new_text_not_the_old(tmp_path, monkeypatch):
    """v1's guard fired once and then defeated itself: the file still lacked the judge key, so the
    relaunch built the SAME scope, found every item 'already done', submitted nothing and wrote
    the discarded text's scores against the new text -- 0 requests, no error, wrong numbers. A
    regenerated response is a different item here, so the stale score is left where it is."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    f = tmp_path / "a_surgical.json"
    _surgical(f, "A", random.Random(4))
    fleet = _Fleet().install(monkeypatch)
    real = J.api_judge_scores

    def regenerated_mid_batch(qs, gs, **k):
        out = real(qs, gs, **k)
        d = json.loads(f.read_text())
        d["conditions"]["base"]["clean_gens"][0] = "A-cq0|base|clean-REGENERATED"
        f.write_text(json.dumps(d))
        return out

    monkeypatch.setattr(J, "api_judge_scores", regenerated_mid_batch)
    with pytest.raises(RuntimeError, match="generations changed on disk"):
        _run_big(monkeypatch, ["--files", str(f)])
    assert fleet.n_requests == 15
    monkeypatch.setattr(J, "api_judge_scores", real)

    _run_big(monkeypatch, ["--files", str(f)])
    assert fleet.n_requests == 16, "the regenerated response must be judged, and only it"
    assert _check_scores(f) == 15          # every score is the score of the text beside it


def test_an_item_added_while_a_batch_is_in_flight_shifts_no_scores(tmp_path, monkeypatch):
    """v1 ids were (scope hash, list INDEX) and the scope did not change when a file gained an
    item, so an adopted batch was index-aligned to a list that had shifted underneath it. At n=15
    the run died on the failure ceiling on every relaunch; at campaign scale it went silent --
    1,991 of 2,403 scores written against text they were not computed from, for a 0.25 pp error
    in a mean that looked entirely plausible."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    f = tmp_path / "a_surgical.json"
    _surgical(f, "A", random.Random(5))
    fleet = _Fleet().install(monkeypatch)
    fleet.die_on_fetch = True
    with pytest.raises(KeyboardInterrupt):
        _run_big(monkeypatch, ["--files", str(f)])
    assert fleet.n_requests == 15

    d = json.loads(f.read_text())                 # a late prompt lands in the file
    d["clean_questions"].append("A-cq3")
    for cond, rec in d["conditions"].items():
        rec["clean_gens"].append(f"A-cq3|{cond}|clean")
    f.write_text(json.dumps(d))

    fleet.die_on_fetch = False
    _run_big(monkeypatch, ["--files", str(f)])
    assert fleet.n_requests == 18, "the three added items -- and nothing else -- must be requested"
    assert _check_scores(f) == 18


def test_a_second_model_over_the_same_files_submits_its_own_requests(tmp_path, monkeypatch):
    """THE LOADED GUN. The v1 scope named no model, so judging the same files with a second model
    built a byte-identical scope, found the first model's state and reported '15 items, 15 already
    done, 0 to do' -- writing luna's scores under the other model's key with judge_meta.model set
    to the other model. A forged second instrument that self-certifies, reporting 100% agreement
    with the judge it is supposed to be checking. `offset` makes the second fleet's scores
    different numbers, so serving the first model's answers shows up as the wrong value.

    The second model is luna-PRO on the same provider pin -- the comparison this module actually
    contemplates -- so the only thing separating the two instruments is the model. Written against
    gemini (a different pin) the test passed with the model removed from the key entirely: it
    would have been checking the provider and could not have failed on the model at all."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    f = tmp_path / "a_surgical.json"
    _surgical(f, "A", random.Random(6))
    luna = _Fleet().install(monkeypatch)
    _run_big(monkeypatch, ["--files", str(f)])
    assert luna.n_requests == 15

    pro = _Fleet(offset=1).install(monkeypatch)
    _run_big(monkeypatch, ["--files", str(f), "--api_model", "openai/gpt-5.6-luna-pro",
                           "--provider", "openai"])
    assert pro.n_requests == 15, "the second instrument served the first one's scores"
    assert luna.n_requests == 15
    assert _check_scores(f) == 15                                        # luna's, untouched
    assert _check_scores(f, model="openai/gpt-5.6-luna-pro", offset=1) == 15
    rec = json.loads(f.read_text())["conditions"]["base"]
    assert rec["judge_api_gpt_5_6_luna"]["scores"] != rec["judge_api_gpt_5_6_luna_pro"]["scores"]


def test_a_relaunch_at_a_different_chunk_size_adopts_rather_than_re_paying(tmp_path, monkeypatch):
    """v1 put the chunk INDEX in every id but the chunk size in nothing: relaunching a 20-request
    run at --chunk_size 5 adopted the two saved ids and submitted 10 more items on top, re-paying
    half of it, with the adopted ids covering items the new chunk1 did not name."""
    fleet = _Fleet().install(monkeypatch)
    fleet.die_on_fetch = True
    qs, gs = [f"q{i}" for i in range(20)], [f"a{i}" for i in range(20)]
    with pytest.raises(KeyboardInterrupt):
        J.api_judge_scores(qs, gs, scope="s", cfg=_cfg(tmp_path, chunk_size=10), api_key="k")
    assert fleet.n_requests == 20 and len(fleet.jobs) == 2

    fleet.die_on_fetch = False
    out = J.api_judge_scores(qs, gs, scope="s", cfg=_cfg(tmp_path, chunk_size=5), api_key="k")
    assert fleet.n_requests == 20, "re-chunking re-paid for items already in flight"
    assert out["scores"] == [_stub_score(J._judge_user_prompt(q, g)) for q, g in zip(qs, gs)]


def test_the_production_defaults_cap_a_job_at_10000_and_the_fleet_at_20000(tmp_path, monkeypatch):
    """Both limits were observed live on 2026-09-16 and both kill a run: `413 Batch of more than
    10,000 requests is not allowed`, and `429 This entity has 20,000 in-flight batch requests and
    this batch adds 10,000` -- the second after earlier chunks are already paid for and
    uncancellable. Run at the real defaults, not at a miniature of them."""
    cfg = _cfg(tmp_path)
    assert (cfg.chunk_size, cfg.max_in_flight_requests) == (10000, 20000)
    n = 20001
    fleet = _Fleet().install(monkeypatch)
    out = J.api_judge_scores([f"q{i}" for i in range(n)], [f"a{i}" for i in range(n)],
                             scope="s", cfg=cfg, api_key="k")
    sizes = [len(fleet.jobs[b]) for k, b in fleet.log if k == "submit"]
    assert sizes == [10000, 10000, 1] and sum(sizes) == n
    in_flight, peak = 0, 0
    for kind, bid in fleet.log:
        in_flight += len(fleet.jobs[bid]) * (1 if kind == "submit" else -1)
        peak = max(peak, in_flight)
    assert peak == 20000
    assert len([s for s in out["scores"] if s is not None]) == n


def test_usage_is_summed_over_every_chunk_not_overwritten(tmp_path, monkeypatch):
    """v1 ASSIGNED state['usage'] per chunk, so a three-chunk run using 1000/2000/3000 tokens
    reported 3,000 against a true 6,000: every multi-chunk run under-reported its own spend, and
    spend is the thing this module is careful about."""
    fleet = _Fleet().install(monkeypatch)
    out = J.api_judge_scores(["q"] * 6, ["a"] * 6, scope="s",
                             cfg=_cfg(tmp_path, chunk_size=2), api_key="k")
    assert len(fleet.jobs) == 3                                  # 10 tokens per request, 2 each
    assert out["judge_meta"]["usage"] == {"total_tokens": 60}


def test_scores_already_on_disk_are_readable_and_never_re_judged(tmp_path, monkeypatch):
    """A re-run costs money for a result already on disk, so the key-presence skip is a resume
    guard, not an optimisation -- and a v1-era record (its judge_meta.batches keyed by chunk
    label) must stay readable rather than being re-judged into the new shape."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    f = tmp_path / "a_surgical.json"
    _surgical(f, "A", random.Random(7))
    _Fleet().install(monkeypatch)
    _run_big(monkeypatch, ["--files", str(f)])
    d = json.loads(f.read_text())
    d["conditions"]["base"][J.judge_key_for(J.DEFAULT_MODEL)]["judge_meta"]["batches"] = {
        "chunk0": "batch-1789602525-jQWs7E7jnabM56nv9pru"}       # the v1 provenance shape
    f.write_text(json.dumps(d, indent=2))
    before = f.read_text()

    def refuse(*a, **k):
        raise AssertionError("re-judged a file that already carries this key")

    monkeypatch.setattr(J, "submit", refuse)
    monkeypatch.setattr(J, "fetch", refuse)
    _run_big(monkeypatch, ["--files", str(f)])
    assert f.read_text() == before


# ------------------------------- a failed item is not an answer (2026-09-17 adversarial pass)
#
# Reproduced against the code that shipped this morning: 3 of 54 items came back
# {"error": "upstream 500"}, the pass raised on the failure ceiling, and the two relaunches after
# it reported "0 to do", raised again and submitted nothing. The record sat in state["items"], so
# "needed" never included it again and only hand-editing the state file could unwedge the worker
# -- under the campaign's single judging worker, for all 90 files at once.

def _scores_of(qs, gs, offset=0):
    return [_stub_score(J._judge_user_prompt(q, g), offset) for q, g in zip(qs, gs)]


def test_a_per_item_error_is_re_asked_by_the_next_run_not_cached_as_an_answer(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(54)], [f"a{i}" for i in range(54)]
    bad = set(J.item_keys(qs, gs, cfg)[:3])
    fleet = _Fleet().install(monkeypatch)
    fleet.fail = lambda cid, bid: {"message": "upstream 500"} if cid in bad else None

    with pytest.raises(RuntimeError, match="produced no score") as first:
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 54
    # the message must point at the file that now holds the decision, which it never did
    assert str(J._state_path(cfg)) in str(first.value)

    fleet.fail = None
    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert [len(fleet.jobs[b]) for k, b in fleet.log if k == "submit"] == [54, 3], \
        "the three failed items were stored as answers and never asked again"
    assert out["scores"] == _scores_of(qs, gs)          # and every score is the score of its text
    assert out["judge_meta"]["n_retried_after_failure"] == 3


def test_an_unparseable_answer_is_never_re_asked_and_the_refusal_names_the_state_file(tmp_path, monkeypatch):
    """The other half of the same fix, and the reason it is not "retry anything without a score":
    temperature is 0, so an item the model ANSWERED with junk returns the same bytes at full price
    -- a strict-schema-style 100% parse failure would re-bill 2,838 requests per file, forever. The
    ceiling still stops the file on every pass; what the message must do is say why nothing was
    submitted and where the records are."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(54)], [f"a{i}" for i in range(54)]
    fleet = _Fleet().install(monkeypatch)
    fleet.unparseable = set(J.item_keys(qs, gs, cfg)[:3])

    with pytest.raises(RuntimeError, match="produced no score"):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 54

    with pytest.raises(RuntimeError, match="produced no score") as again:
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 54, "re-asked an item the model had already answered"
    assert str(J._state_path(cfg)) in str(again.value)
    assert "unparseable" in str(again.value)


def test_a_permanently_failing_item_is_refused_by_name_after_the_attempt_cap(tmp_path, monkeypatch):
    """Retrying is bounded: an endpoint that will never answer must not bill an unattended worker
    in a loop. The refusal comes BEFORE anything is submitted, and names the state file and the
    ids, because the only way out is to edit that file."""
    cfg = _cfg(tmp_path, max_item_attempts=3, max_failure_rate=1.0)
    qs, gs = [f"q{i}" for i in range(6)], [f"a{i}" for i in range(6)]
    bad = J.item_keys(qs, gs, cfg)[2]
    fleet = _Fleet().install(monkeypatch)
    fleet.fail = lambda cid, bid: {"message": "upstream 500"} if cid == bad else None

    for expected in (6, 7, 8):                     # the first ask, then two bounded retries
        out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
        assert fleet.n_requests == expected
        assert out["scores"][2] is None and out["scores"][0] is not None

    with pytest.raises(RuntimeError, match="no answer after 3 attempt") as e:
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert bad in str(e.value) and str(J._state_path(cfg)) in str(e.value)
    assert fleet.n_requests == 8, "kept buying the same item after the attempt cap"


def test_an_item_absent_from_a_completed_batch_is_retried_and_counted(tmp_path, monkeypatch):
    """A batch can be `completed` while an item is simply not in its results. That was always
    retried -- the item never reached the state at all -- but never COUNTED, so the one failure
    mode that already self-healed could still loop for the length of a campaign."""
    cfg = _cfg(tmp_path, max_item_attempts=2, max_failure_rate=1.0)
    qs, gs = [f"q{i}" for i in range(4)], [f"a{i}" for i in range(4)]
    fleet = _Fleet().install(monkeypatch)
    fleet.drop = {J.item_keys(qs, gs, cfg)[1]}

    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 4 and out["scores"][1] is None
    J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 5, "an item missing from a completed batch must still be re-asked"
    with pytest.raises(RuntimeError, match="absent from the completed batch"):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 5


# ------------------------- an orphaned batch: the window between submit() and _save_state
#
# 3 of 13 kill points reproduced it (post_submit_pre_state, pre_state_save, mid_state_save), 74
# submissions for 54 items. The batch exists on the server and in no state file, so nothing can
# ever adopt it and it cannot be cancelled.

def test_a_kill_between_the_create_call_and_the_state_save_is_recovered_from_the_intent(tmp_path, monkeypatch):
    """The metadata path, i.e. the API this module would like to have: `drop_metadata = False` is
    HYPOTHETICAL (the live listing echoes nothing, see the shape tests at the end of this file),
    and it is pinned here because the id is still sent and must still decide the day it comes
    back."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(20)], [f"a{i}" for i in range(20)]
    fleet = _Fleet().install(monkeypatch)
    fleet.drop_metadata = False
    fleet.die_after_submit = True
    with pytest.raises(KeyboardInterrupt):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 20
    saved = json.loads(J._state_path(cfg).read_text())
    assert saved["batches"] == {}, "the id never reached disk -- that IS the window"
    assert len(saved["intents"]) == 1

    fleet.die_after_submit = False
    lines = []
    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k", log=lines.append)
    assert fleet.n_requests == 20, "the orphaned batch was bought a second time"
    assert len(fleet.jobs) == 1 and ("fetch", "batch-0") in fleet.log
    assert out["scores"] == _scores_of(qs, gs)
    text = "\n".join(lines)
    assert "RECOVERED orphan batch batch-0 from intent" in text
    assert "by SHAPE" not in text, "adopted by shape; the metadata path never fired"
    assert json.loads(J._state_path(cfg).read_text())["intents"] == {}


def test_the_intent_is_on_disk_before_the_batch_can_exist(tmp_path, monkeypatch):
    """The ordering IS the fix: a record written after the create call is a record the kill window
    can still skip. Read from inside `submit`, so it pins the order rather than the outcome."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(4)], [f"a{i}" for i in range(4)]
    keys = J.item_keys(qs, gs, cfg)
    fleet = _Fleet().install(monkeypatch)
    seen = {}

    def submit_reading_the_state(payload, c, k, *, intent_id=None):
        seen["state"] = json.loads(J._state_path(c).read_text())
        seen["id"] = intent_id
        return fleet.submit(payload, c, k, intent_id=intent_id)

    monkeypatch.setattr(J, "submit", submit_reading_the_state)
    J.api_judge_scores(qs, gs, scope="sc", cfg=cfg, api_key="k")
    intents = seen["state"]["intents"]
    assert list(intents) == [seen["id"]] and seen["id"]
    rec = intents[seen["id"]]
    assert rec["item_keys"] == keys, "an intent that does not name its items cannot be adopted"
    assert rec["instrument"] == J.instrument_id(cfg) and rec["scope"] == "sc"
    assert rec["batch_id"] is None


def test_the_create_body_carries_the_intent_id_and_still_ends_with_requests(monkeypatch):
    """The id must ride ON the batch: it is the only handle a run that died before writing the id
    down has. Field order stays load-bearing -- the API rejects a body whose `requests` key
    precedes `endpoint`/`model`."""
    captured = {}

    class _Created:
        status_code = 200
        text = ""

        def json(self):
            return {"id": "batch-live"}

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["body"] = json.loads(data)
        return _Created()

    monkeypatch.setattr(J.requests, "post", fake_post)
    bid = J.submit([J.build_request("q", "a", "cid", 2048)], J.JudgeConfig(), "k",
                   intent_id="intent-123")
    assert bid == "batch-live"
    assert captured["body"]["metadata"] == {J.INTENT_METADATA_KEY: "intent-123"}
    assert list(captured["body"])[:2] == ["endpoint", "model"]
    assert list(captured["body"])[-1] == "requests"


def test_an_intent_that_never_became_a_batch_is_dropped_on_a_complete_listing(tmp_path, monkeypatch):
    """The intent is written before the create call, so a create that FAILS leaves one behind too
    (a 413 is refused before anything is created, so nothing was spent). A complete listing that
    holds nothing carrying it settles that: it never became a batch, and its items are bought
    once. Keeping it would wedge the worker on work that does not exist."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(4)], [f"a{i}" for i in range(4)]
    fleet = _Fleet().install(monkeypatch)

    def refused(payload, c, k, *, intent_id=None):
        raise RuntimeError("batch submit failed 413: Batch of more than 10,000 requests")

    monkeypatch.setattr(J, "submit", refused)
    with pytest.raises(RuntimeError, match="413"):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert len(json.loads(J._state_path(cfg).read_text())["intents"]) == 1

    monkeypatch.setattr(J, "submit", fleet.submit)
    lines = []
    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k", log=lines.append)
    assert fleet.n_requests == 4 and len(fleet.jobs) == 1
    assert "never became a batch" in "\n".join(lines)
    assert json.loads(J._state_path(cfg).read_text())["intents"] == {}
    assert out["judge_meta"]["unresolved_intents"] == 0


def test_an_unreconcilable_intent_is_reported_by_name_instead_of_being_assumed_away(tmp_path, monkeypatch):
    """The residual "cannot tell", end to end. The listing echoes no `metadata`, and TWO batches
    that nothing accounts for fit the intent equally -- so no field the API returns can say which
    of them (if either) this intent created. "No match" and "cannot tell" must not look alike when
    the difference is a resubmission, so the run names the intent and both batches and carries on;
    it must not silently re-pay, must not adopt one on a coin flip, and must not wedge either."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(4)], [f"a{i}" for i in range(4)]
    fleet = _Fleet().install(monkeypatch)
    fleet.die_after_submit = True
    with pytest.raises(KeyboardInterrupt):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    iid = list(json.loads(J._state_path(cfg).read_text())["intents"])[0]

    fleet.die_after_submit = False
    # and the listing names the model WITHOUT the pinned-snapshot suffix -- a third spelling this
    # code has never seen live. Reading that as "not one of ours" would empty the unaccounted set
    # and turn this case back into the confident, wrong "it never became a batch".
    fleet.listing["batch-0"]["model"] = cfg.model
    # The twin: another worker's batch, or this worker's own earlier orphan. Same model family,
    # same request count, created after the intent. Adopting the wrong one would attach this
    # intent's four item keys to a batch carrying somebody else's custom_ids.
    fleet.listing["batch-twin"] = _listed("batch-twin", total=4, created_at=time.time())
    lines = []
    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k", log=lines.append)
    text = "\n".join(lines)
    assert "WARNING" in text and iid in text and "batch-0" in text and "batch-twin" in text
    assert "RECOVERED" not in text
    assert out["judge_meta"]["unresolved_intents"] == 1
    assert fleet.n_requests == 8            # bought again -- but said out loud, with the ids
    assert json.loads(J._state_path(cfg).read_text())["intents"][iid]["unresolved"] is True


def test_a_truncated_listing_is_not_read_as_proof_that_the_batch_does_not_exist(tmp_path, monkeypatch):
    """The one conclusion that costs money twice. "Nothing on this page carries the intent" is not
    "no batch carries it", so a listing the API says is partial can only ever leave the intent
    unresolved -- never drop it."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(4)], [f"a{i}" for i in range(4)]
    fleet = _Fleet().install(monkeypatch)
    fleet.die_after_submit = True
    with pytest.raises(KeyboardInterrupt):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    iid = list(json.loads(J._state_path(cfg).read_text())["intents"])[0]

    fleet.die_after_submit = False
    fleet.listing.pop("batch-0")            # the orphan sits on a page this call did not read
    fleet.listing_complete = False
    lines = []
    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k", log=lines.append)
    text = "\n".join(lines)
    assert out["judge_meta"]["unresolved_intents"] == 1
    assert "WARNING" in text and iid in text and "INCOMPLETE" in text
    assert "never became a batch" not in text
    assert json.loads(J._state_path(cfg).read_text())["intents"][iid]["unresolved"] is True


def test_a_listing_that_cannot_be_fetched_is_loud_but_does_not_kill_the_worker(tmp_path, monkeypatch):
    """One judging worker serves all 90 files. A GET that fails must not end the pass -- but it
    must not pass for "no batch exists" either."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(4)], [f"a{i}" for i in range(4)]
    fleet = _Fleet().install(monkeypatch)
    fleet.die_after_submit = True
    with pytest.raises(KeyboardInterrupt):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    iid = list(json.loads(J._state_path(cfg).read_text())["intents"])[0]

    fleet.die_after_submit = False

    def unreachable(api_key, **kw):
        raise J.requests.RequestException("connection reset")

    monkeypatch.setattr(J, "list_batches", unreachable)
    lines = []
    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k", log=lines.append)
    text = "\n".join(lines)
    assert "cannot list the account's batches" in text and iid in text
    assert out["judge_meta"]["unresolved_intents"] == 1
    assert out["scores"] == _scores_of(qs, gs)      # the pass still finishes


def test_list_batches_calls_a_partial_page_partial(monkeypatch):
    """`has_more` is the only thing separating "the batch is not there" from "I did not look at
    all of it", and `_reconcile_intents` drops an intent on the first and never on the second."""
    monkeypatch.setattr(J.requests, "get",
                        lambda *a, **k: _Resp(200, {"data": [{"id": "b1"}], "has_more": True}))
    assert J.list_batches("k") == ([{"id": "b1"}], False)
    monkeypatch.setattr(J.requests, "get",
                        lambda *a, **k: _Resp(200, {"data": [{"id": "b1"}], "has_more": False}))
    assert J.list_batches("k") == ([{"id": "b1"}], True)
    monkeypatch.setattr(J.requests, "get", lambda *a, **k: _Resp(403, text="forbidden"))
    with pytest.raises(RuntimeError, match="batch list failed 403"):
        J.list_batches("k")


# ------------------------------- adopting an orphan by SHAPE (live probe, 2026-09-17)
#
# The two-phase submit shipped with `metadata` as the handle: the intent id rides on the batch and
# the reconciler finds it in the account's listing. The live probe (2 requests, real account)
# falsified that mechanism outright. The create WITH `metadata` is accepted -- nothing 400s -- and
# the listing NEVER echoes it: a listed batch carries exactly completion_window, created_at,
# endpoint, error, finalized_at, id, model, object, request_counts, results, status, usage. So the
# by-name path could not fire in production even once: every orphan degraded to "unresolved",
# logged loudly, and its work (2,838 requests per surgical file) was bought a second time. The
# window was visible, not closed.
#
# What is left to recognise an orphan by is its SHAPE, in the fields the API does return. Each test
# below removes exactly one of the four conditions and checks that adoption stops -- because a
# fallback that adopts on three of them is a fallback that attaches one batch's results to another
# batch's items, which is worse than paying twice. Every fake here is `_listed`, i.e. the real
# shape, with no `metadata` key at all.

def _pending_intent(cfg, n=3, *, batches=None, scope="sc"):
    """A state file exactly as the kill window leaves it: the intent on disk, no batch id."""
    keys = [f"item-{i}" for i in range(n)]
    state = {"version": J.STATE_VERSION, "model": cfg.model, "items": {},
             "batches": dict(batches or {}),
             "intents": {"iid-1": {"item_keys": keys, "instrument": J.instrument_id(cfg),
                                   "scope": scope, "created_at": time.time(), "batch_id": None}}}
    J._save_state(cfg, state)
    return state, keys


def _reconcile(monkeypatch, cfg, state, listing, *, complete=True):
    lines = []
    monkeypatch.setattr(J, "list_batches", lambda api_key, **kw: ([dict(b) for b in listing],
                                                                  complete))
    n = J._reconcile_intents(cfg, state, "k", lines.append)
    return n, "\n".join(lines)


def test_the_one_unaccounted_batch_that_fits_the_intent_is_reported_not_adopted(tmp_path, monkeypatch):
    """Shape is REPORTED, never adopted (decision 2026-09-17). With no `metadata` in the listing the
    orphan cannot be identified: a same-size batch of the same model family created after the intent
    is exactly what a foreign run also looks like, and three false adoptions were demonstrated. So
    the intent stays pending, the probable batch is named in the warning, and the items are bought
    again -- loudly. The decoys stay: they pin that the shape filter feeding the warning is not
    itself broken (wrong size, wrong model, and luna-PRO, whose listed name differs from luna's only
    by a suffix, so a family test written as a prefix match would call it a candidate)."""
    cfg = _cfg(tmp_path)
    state, keys = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    listing = [_listed("batch-orphan", total=3, created_at=t + 1),
               _listed("batch-other-size", total=4, created_at=t + 1),
               _listed("batch-other-model", total=3, created_at=t + 1,
                       model="google/gemini-3.8-flash"),
               # luna-PRO, the comparison this module actually contemplates, judging the same
               # corpus at the same time. Its listed name is the requested one plus a suffix,
               # exactly as luna's is, so a family test written as a prefix match would call this
               # a second candidate and adopt nothing -- or, with the intents the other way round,
               # adopt luna's batch for pro's items.
               _listed("batch-pro", total=3, created_at=t + 1,
                       model="openai/gpt-5.6-luna-pro-20260709")]
    n, text = _reconcile(monkeypatch, cfg, state, listing)
    # unresolved, not adopted: nothing is collected and nothing is dropped
    assert n == 1 and set(state["intents"]) == {"iid-1"} and state["batches"] == {}
    assert "RECOVERED" not in text
    # the warning still has to be actionable: it names the intent and the one batch that fits it,
    # and does NOT name the decoys, or a human cannot act on it
    assert "iid-1" in text and "batch-orphan" in text
    # the shape filter still has to be right, because the count in the warning is what a human acts
    # on: exactly ONE of the four listed batches fits (the decoys fail size, model family, and the
    # luna-PRO family test respectively), even though every unaccounted batch is named
    assert "1 unaccounted batch(es) fit it by shape" in text
    assert "batch-other-model" not in text and "batch-pro" not in text
    # and the intent survives on disk, so the next process reports it too rather than forgetting it
    assert list(json.loads(J._state_path(cfg).read_text())["intents"]) == ["iid-1"]
    assert keys  # the items stay unanswered and are bought again by the caller


def test_two_unaccounted_batches_of_the_same_shape_adopt_neither(tmp_path, monkeypatch):
    """Uniqueness is the whole licence to adopt. Two batches the listing cannot tell apart identify
    neither, and picking one would write this intent's item keys onto a batch that answers other
    custom_ids -- every item then comes back "absent from the completed batch" while the real
    orphan stays unreachable. Loud and unresolved is the only honest answer."""
    cfg = _cfg(tmp_path)
    state, _ = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    n, text = _reconcile(monkeypatch, cfg, state,
                         [_listed("batch-a", total=3, created_at=t + 1),
                          _listed("batch-b", total=3, created_at=t + 2)])
    assert n == 1 and state["batches"] == {}
    assert state["intents"]["iid-1"]["unresolved"] is True
    assert "RECOVERED" not in text
    assert "WARNING" in text and "iid-1" in text and "batch-a" in text and "batch-b" in text


def test_a_batch_carrying_one_request_too_few_is_not_this_intents_batch(tmp_path, monkeypatch):
    """`request_counts.total` is the only field that ties a listed batch to the WORK an intent
    covers. Off by one is a different batch -- and adopting it would leave the intent's third item
    unanswered forever while the real batch went uncollected."""
    cfg = _cfg(tmp_path)
    state, _ = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    n, text = _reconcile(monkeypatch, cfg, state,
                         [_listed("batch-almost", total=2, created_at=t + 1)])
    assert n == 1 and state["batches"] == {}
    assert "RECOVERED" not in text and "never became a batch" not in text
    assert "batch-almost" in text                      # still named: it might be somebody's orphan


def test_a_batch_created_before_the_intent_is_not_this_intents_batch(tmp_path, monkeypatch):
    """The intent is written immediately BEFORE the create call, so its batch cannot predate it --
    an older batch of the same size is an earlier run's work, and adopting it would hand this run
    somebody else's results and leave that work paid for twice over. The tolerance exists only for
    clock skew and the whole-second truncation of `created_at`, so both sides of it are pinned."""
    cfg = _cfg(tmp_path)
    state, _ = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    # Half an hour old, written as an absolute margin and NOT as `t - INTENT_CLOCK_SKEW_S - x`:
    # a bound expressed in terms of the constant it is bounding moves with it and can never fail.
    # The smallest batch turnaround ever measured here is 11 min, so a tolerance anywhere near
    # half an hour would already be swallowing other runs' batches whole.
    n, text = _reconcile(monkeypatch, cfg, state,
                         [_listed("batch-earlier", total=3, created_at=t - 1800)])
    assert n == 1 and state["batches"] == {} and "RECOVERED" not in text
    assert "0 unaccounted batch(es) fit it by shape" in text

    # ...while a batch stamped a moment early -- this host's clock against the server's, plus the
    # truncation to whole seconds -- is inside the tolerance and counts as a fit. It is still not
    # adopted (shape never adopts), but it must be REPORTED, or the warning would tell a human
    # there is nothing to look at when their money is sitting on the account.
    state, _ = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    n, text = _reconcile(monkeypatch, cfg, state,
                         [_listed("batch-truncated-stamp", total=3, created_at=t - 1)])
    assert n == 1 and state["batches"] == {}
    assert "1 unaccounted batch(es) fit it by shape" in text and "batch-truncated-stamp" in text


def test_a_batch_some_state_file_already_names_is_not_an_orphan(tmp_path, monkeypatch):
    """"Unaccounted" means no state file in the directory names it -- not merely this model's. A
    v1 file holds ids that can never be adopted (the request text was never stored) but that are
    accounted for all the same, and this run's own state file holds the ids it did record. Three
    identically shaped batches, two of them spoken for: without the exclusion there are three
    candidates and nothing is ever adopted; with it there is exactly one orphan."""
    cfg = _cfg(tmp_path)
    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "state" / "v1-legacy.json").write_text(json.dumps(
        {"scope": "a_surgical.json|ALL", "batches": {"chunk0": "batch-in-v1"}, "items": {}}))
    state, keys = _pending_intent(
        cfg, 3, batches={"batch-recorded": {"item_keys": ["x"], "drained": False}})
    t = state["intents"]["iid-1"]["created_at"]
    listing = [_listed(b, total=3, created_at=t + 1)
               for b in ("batch-recorded", "batch-in-v1", "batch-orphan")]
    n, text = _reconcile(monkeypatch, cfg, state, listing)
    # Nothing is adopted (shape never adopts), but the exclusion still has to be right: of the three
    # identically shaped batches only the genuinely unaccounted one may be named, and neither the id
    # this state file records nor the one a v1 file holds may be touched or reported as an orphan.
    assert n == 1 and "RECOVERED" not in text
    assert "1 unaccounted batch(es) fit it by shape" in text and "batch-orphan" in text
    assert state["batches"]["batch-recorded"]["item_keys"] == ["x"], "an accounted batch was rewritten"
    assert "batch-orphan" not in state["batches"] and "batch-in-v1" not in state["batches"]
    assert keys


def test_a_truncated_listing_never_adopts_by_shape_even_on_a_perfect_fit(tmp_path, monkeypatch):
    """"Unique on this page" is not "unique on the account", and the API serves at most 100 batches
    per listing in no particular order (limit=200 is a 400; a just-created batch was absent from a
    limit=20 page). A partial listing can therefore never establish uniqueness, so it can never
    adopt -- it can only report."""
    cfg = _cfg(tmp_path)
    state, _ = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    n, text = _reconcile(monkeypatch, cfg, state,
                         [_listed("batch-perfect-fit", total=3, created_at=t + 1)],
                         complete=False)
    assert n == 1 and state["batches"] == {} and "RECOVERED" not in text
    assert "INCOMPLETE" in text and f"at most {J.LIST_LIMIT_MAX}" in text


def test_metadata_if_the_api_ever_echoes_it_still_takes_the_exact_path(tmp_path, monkeypatch):
    """The id is still sent, because it costs nothing and it is exact where the shape rules are
    circumstantial. Pinned in the case that separates the two: two batches fit by shape, which on
    its own resolves nothing, and the one actually carrying the intent id is adopted anyway."""
    cfg = _cfg(tmp_path)
    state, keys = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    named = dict(_listed("batch-named", total=3, created_at=t + 1),
                 metadata={J.INTENT_METADATA_KEY: "iid-1"})
    n, text = _reconcile(monkeypatch, cfg, state,
                         [named, _listed("batch-twin", total=3, created_at=t + 1)])
    assert n == 0 and set(state["batches"]) == {"batch-named"}
    assert state["batches"]["batch-named"]["item_keys"] == keys
    assert "from intent iid-1" in text and "by SHAPE" not in text


def test_the_orphan_window_costs_one_batch_loudly_end_to_end(tmp_path, monkeypatch):
    """THE WHOLE SCENARIO, with the listing behaving as the live account does. The run is killed
    between the create call and the state save: the batch exists on the server, no state file names
    it, and the listing hands back no `metadata` to find it by.

    This pins the ACCEPTED COST (decision 2026-09-17). The orphan cannot be identified -- shape
    alone would also match a foreign batch, and adopting one wrongly would mark our own items as
    absent and burn their attempt budget -- so the relaunch buys those items again. What the run
    must NOT do is hide it: the warning names the intent and the batch that is probably ours, the
    count is carried into judge_meta, and every score written is still the score of the text it is
    written against."""
    cfg = _cfg(tmp_path)
    qs, gs = [f"q{i}" for i in range(20)], [f"a{i}" for i in range(20)]
    fleet = _Fleet().install(monkeypatch)
    assert fleet.drop_metadata, "the live listing echoes no metadata; the fake must not either"
    fleet.die_after_submit = True
    with pytest.raises(KeyboardInterrupt):
        J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k")
    assert fleet.n_requests == 20
    saved = json.loads(J._state_path(cfg).read_text())
    assert saved["batches"] == {}, "the id never reached disk -- that IS the window"
    assert len(saved["intents"]) == 1
    assert not any("metadata" in b for b in fleet.listing.values())

    fleet.die_after_submit = False
    lines = []
    out = J.api_judge_scores(qs, gs, scope="s", cfg=cfg, api_key="k", log=lines.append)
    # the accepted cost: the 20 items are bought a second time, so 40 requests for 20 items
    assert fleet.n_requests == 40 and len(fleet.jobs) == 2
    # ...but the scores are correct, which is the part that must never be traded away
    assert out["scores"] == _scores_of(qs, gs)
    # ...and the loss is reported, not swallowed: counted in the artifact and named in the log
    assert out["judge_meta"]["unresolved_intents"] == 1
    text = "\n".join(lines)
    assert "RECOVERED" not in text
    assert "cannot be reconciled" in text and "batch-0" in text
    assert "paid-for work about to be bought again" in text
    # the intent stays on disk, so the NEXT process reports it too rather than forgetting the money
    assert len(json.loads(J._state_path(cfg).read_text())["intents"]) == 1


def test_one_batch_that_fits_two_pending_intents_is_adopted_by_neither(tmp_path, monkeypatch):
    """Two chunks submitted, killed one after the other, so two intents of the same size are
    pending. A single unaccounted batch fits both equally -- it is one of them, and the listing
    cannot say which. Adopted per-intent it would be claimed by the first, whose item keys name
    text that batch was never asked about, while the other intent is dropped as settled: both
    chunks then come back "absent from the completed batch" and the real work stays uncollected."""
    cfg = _cfg(tmp_path)
    state, keys = _pending_intent(cfg, 3)
    t = state["intents"]["iid-1"]["created_at"]
    state["intents"]["iid-2"] = {"item_keys": ["other-0", "other-1", "other-2"],
                                 "instrument": J.instrument_id(cfg), "scope": "sc",
                                 "created_at": t, "batch_id": None}
    n, text = _reconcile(monkeypatch, cfg, state, [_listed("batch-one", total=3, created_at=t + 1)])
    assert n == 2 and state["batches"] == {}
    assert set(state["intents"]) == {"iid-1", "iid-2"} and "RECOVERED" not in text
    assert "iid-1" in text and "iid-2" in text and "batch-one" in text
