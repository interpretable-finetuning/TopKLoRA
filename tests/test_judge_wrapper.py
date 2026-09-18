"""scripts/qwen15_judge.sh + scripts/campaign3_launch.sh: what "the judge finished" may mean.

Three things have to hold before campaign 3 spends money on 90 cells' worth of OpenRouter batches:

1. A judge that failed must not be reported as a success. The wrapper's last command used to be its
   summary heredoc, so the script exited with the summary's status and campaign3 logged "luna judge
   done" for a judge that exited 3 -- and every retention number downstream would then be computed
   on whatever happened to be written.
2. The summary must describe the files that were judged. It globbed a hardcoded tree, so a pass over
   a campaign tree reported on a different (empty) one: "0/0 condition-records scored", clean.
3. Judging now runs WHILE the drivers run, so two judge processes on one OpenRouter account become
   possible. The account's 20,000 in-flight batch-request cap is enforced inside one process only
   (src/clcd/judge_api.py), a batch cannot be cancelled once submitted, and the campaign has already
   recorded one 429. The wrapper's flock is what makes "one judge per account" true.

The last section audits that lock against what the campaign will do to it: a poll that arrives
while a 15 h pass still holds it (75, and 75 must stay distinguishable from a broken config), a
judge that is killed, a judge that outlives the wrapper that started it, and the paths that spend
the account WITHOUT the wrapper -- a direct `python -m src.clcd.judge_saved_gens_big
--judge_backend api`, which is the invocation in that module's own docstring and is still
unguarded (see the xfail at the end).

The judge itself is stubbed: none of this needs a model, a GPU or the network, and the stub is what
lets the tests set the judge's exit status and watch two wrappers try to overlap. Driver modules are
refused by the same stub, so a campaign test that somehow reached the launch block cannot start one.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = ["bash", "scripts/qwen15_judge.sh"]
CAMPAIGN = ["bash", "scripts/campaign3_launch.sh"]
SUF = "testsuffix"          # stands in for api_gpt_5_6_luna; the real key costs a 5 s torch import

STUB = r"""#!/bin/bash
# Stands in for .venv/bin/python. Intercepts the judge module (and the key-resolution -c); the
# wrapper's summary heredoc is passed through to the real interpreter and runs for real.
if [ -n "${STUB_SUFFIX:-}" ] && [ "$1" = "-c" ]; then echo "$STUB_SUFFIX"; exit 0; fi
for a in "$@"; do
  case "$a" in
    src.clcd.exp_circuit_search|src.clcd.exp_surgical_removal|analysis/verify_holdout_necessity.py)
      echo "$*" > "$STUB_DIR/drivers_launched"; exit 1 ;;
    src.clcd.judge_saved_gens_big)
      n=$(( $(cat "$STUB_DIR/count") + 1 )); echo "$n" > "$STUB_DIR/count"
      printf '%s\n' "$*" > "$STUB_DIR/argv.$n"
      # mkdir is atomic: two judges inside this section at once leave the marker, with no race
      if mkdir "$STUB_DIR/running"; then :; else echo "$*" > "$STUB_DIR/overlap"; fi
      if [ -n "${STUB_LATE_FILE:-}" ] && [ "$n" = 1 ]; then
        mkdir -p "$(dirname "$STUB_LATE_FILE")"; cp "$STUB_TEMPLATE" "$STUB_LATE_FILE"
      fi
      [ -n "${STUB_SLEEP:-}" ] && sleep "$STUB_SLEEP"
      rmdir "$STUB_DIR/running"
      exit "${STUB_RC:-0}"
      ;;
  esac
done
exec __PYTHON__ "$@"
""".replace("__PYTHON__", str(REPO / ".venv/bin/python"))


def _stub(tmp_path):
    d = tmp_path / "stub"
    d.mkdir()
    p = d / "pystub.sh"
    p.write_text(STUB)
    p.chmod(0o755)
    (d / "count").write_text("0\n")
    return p, d


def _surgical(path: Path, scored: bool):
    """A surgical-removal JSON as the summary reads it: two conditions, scored or not."""
    rec = {f"judge_{SUF}": {"mean": 4.0, "n": 2}, f"judge_indep_{SUF}": {"mean": 3.0, "n": 2}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"conditions": {c: dict(rec) if scored else {}
                                               for c in ("intact", "ablate")}}))
    return path


def _env(stub, stubdir, tmp_path, **extra):
    e = dict(os.environ, PY=str(stub), STUB_DIR=str(stubdir), STUB_SUFFIX=SUF,
             JUDGE_LOCK=str(tmp_path / "account.lock"), JUDGE_LOCK_WAIT="30")
    e.update({k: str(v) for k, v in extra.items()})
    return e


def _run(cmd, env, timeout=120):
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------- the wrapper's exit status

def test_a_judge_that_exited_nonzero_is_not_reported_as_a_success(tmp_path):
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=False)
    r = _run(WRAPPER, _env(stub, sd, tmp_path, STUB_RC=3,
                           FILES=f"{tree}/*/surgical/*_surgical.json"))
    assert r.returncode == 3, r.stdout + r.stderr          # the judge's status, not the summary's
    assert "judge exit=3" in r.stdout
    assert "condition-records scored" in r.stdout          # the summary still ran; only its rc is dropped


def test_a_judge_that_succeeded_exits_zero(tmp_path):
    # The control for the test above: `exit 1` at the end of the wrapper would also make a failure
    # visible, and would make every campaign pass read as FAILED.
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=True)
    r = _run(WRAPPER, _env(stub, sd, tmp_path, FILES=f"{tree}/*/surgical/*_surgical.json"))
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------- the summary reads $FILES

def test_the_summary_describes_the_files_that_were_judged(tmp_path):
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "a/surgical/alpha_surgical.json", scored=True)
    _surgical(tree / "b/surgical/beta_surgical.json", scored=False)
    r = _run(WRAPPER, _env(stub, sd, tmp_path, FILES=f"{tree}/*/surgical/*_surgical.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    # 2 of 4 condition-records carry the key, and the two organisms named are the ones under $FILES
    assert "2/4 condition-records scored in 2 files" in r.stdout, r.stdout
    assert "alpha" in r.stdout and "beta" in r.stdout
    assert "UNSCORED: [('beta', 'intact'), ('beta', 'ablate')]" in r.stdout, r.stdout


# --------------------------------------------------------------------- one judge per account

def test_two_api_judges_do_not_run_at_once(tmp_path):
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=False)
    env = _env(stub, sd, tmp_path, STUB_SLEEP=1, FILES=f"{tree}/*/surgical/*_surgical.json")
    procs = [subprocess.Popen(WRAPPER, cwd=REPO, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    assert (sd / "count").read_text().strip() == "2", outs   # both judged, neither was dropped
    assert not (sd / "overlap").exists(), (sd / "overlap").read_text()


def test_a_judge_that_cannot_take_the_account_lock_judges_nothing(tmp_path):
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=False)
    lock, ready = tmp_path / "account.lock", tmp_path / "held"
    holder = subprocess.Popen(["flock", str(lock), "-c", f"touch {ready}; sleep 20"], cwd=REPO)
    try:
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists(), "the external lock holder never started"
        r = _run(WRAPPER, _env(stub, sd, tmp_path, JUDGE_LOCK_WAIT=1,
                               FILES=f"{tree}/*/surgical/*_surgical.json"))
    finally:
        holder.kill()
        holder.wait()
    assert r.returncode == 75, r.stdout + r.stderr            # EX_TEMPFAIL: try again, not "done"
    assert "judged nothing" in r.stdout
    assert (sd / "count").read_text().strip() == "0"          # and no batch was submitted
    # ... and it does not also print a summary, which a caller could read as a scored pass
    assert "condition-records scored" not in r.stdout


# --------------------------------------------------------------------- campaign 3's own accounting

def _campaign(tmp_path, stub, sd, n_qc=60, n_qs=60, n_gc=30, n_gs=30, driver_secs=2.5, **extra):
    """Run campaign3.sh past its launch block: drivers.pids holds a live process that this helper
    kills after driver_secs, so the script takes the 'already running' branch and never launches."""
    qt, gt, st = tmp_path / "qt", tmp_path / "gt", tmp_path / "st"
    st.mkdir()
    for tree, nc, ns in ((qt, n_qc, n_qs), (gt, n_gc, n_gs)):
        for i in range(nc):
            (tree / f"c{i}/elim").mkdir(parents=True, exist_ok=True)
            (tree / f"c{i}/elim/x_circuit.json").write_text("{}")
        for i in range(ns):
            _surgical(tree / f"c{i}/surgical/cell{i}_surgical.json", scored=False)
    drivers = subprocess.Popen(["sleep", "600"])
    (st / "drivers.pids").write_text(f"{drivers.pid}\n")
    env = _env(stub, sd, tmp_path, ST=st, QT=qt, GT=gt, STATUS=tmp_path / "status.txt",
               POLL=1, JUDGE_POLL=1, **extra)
    p = subprocess.Popen(CAMPAIGN, cwd=REPO, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(driver_secs)
    finally:
        drivers.kill()
        drivers.wait()
    out = p.communicate(timeout=180)[0]
    assert not (sd / "drivers_launched").exists(), "the campaign tried to launch search drivers"
    return p.returncode, out, qt


def test_surgical_files_are_judged_while_the_drivers_are_still_running(tmp_path):
    # The point of the incremental pass: files that land days before the last cell finishes are
    # judged then, and a file written DURING a pass is picked up by the next one rather than lost.
    stub, sd = _stub(tmp_path)
    template = tmp_path / "template.json"
    _surgical(template, scored=False)
    late = tmp_path / "qt/late/surgical/late_surgical.json"
    rc, out, _ = _campaign(tmp_path, stub, sd, n_qs=59,
                           STUB_LATE_FILE=late, STUB_TEMPLATE=template)
    assert "c3: judge p1:" in out, out                       # a pass ran before the drivers exited
    assert "judge final" in out
    n = int((sd / "count").read_text())
    assert n >= 2, out
    assert "late_surgical.json" not in (sd / "argv.1").read_text()   # did not exist for pass 1 ...
    assert "late_surgical.json" in (sd / f"argv.{n}").read_text()    # ... and was judged after it
    assert rc == 0, out
    assert "campaign 3 complete: circuits qwen 60/60 gemma 30/30, surgical qwen 60/60" in out


def test_a_short_campaign_is_not_reported_as_complete(tmp_path):
    stub, sd = _stub(tmp_path)
    rc, out, _ = _campaign(tmp_path, stub, sd, n_qc=59, n_gs=28)
    assert rc == 1, out
    assert "CAMPAIGN 3 INCOMPLETE: qwen_circuits=59/60 gemma_surgical=28/30" in out, out
    assert "campaign 3 complete" not in out


def test_a_failed_final_judge_is_not_reported_as_complete(tmp_path):
    stub, sd = _stub(tmp_path)
    rc, out, _ = _campaign(tmp_path, stub, sd, STUB_RC=3)
    assert rc == 1, out
    assert "judge final FAILED rc=3" in out, out
    assert "CAMPAIGN 3 INCOMPLETE: judge_rc=3" in out, out
    assert "campaign 3 complete" not in out


# ------------------------------------------------- the account lock, audited against the campaign

def _held(lock: Path) -> bool:
    """True iff some process holds the flock on `lock` right now (75 = flock's conflict code)."""
    rc = subprocess.run(["flock", "-w", "0", "-E", "75", str(lock), "-c", "true"]).returncode
    assert rc in (0, 75), f"flock probe on {lock} returned {rc}"   # never read an error as 'free'
    return rc == 75


def _killpg(pg):
    """SIGKILL a whole process group, tolerating one that has already gone."""
    try:
        os.killpg(pg, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_until(pred, what, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def test_a_broken_lock_wait_fails_loudly_instead_of_reading_as_polite_contention(tmp_path):
    """75 means 'someone else is judging, try later' -- campaign3_launch.sh logs it as SKIPPED and
    carries on. So 75 must be reachable ONLY by contention. `if ! flock -w "$JUDGE_LOCK_WAIT" 9`
    also returned 75 for flock's usage error (64), so a typo'd or empty JUDGE_LOCK_WAIT judged
    nothing on every pass of a 17 h campaign while every log line said another judge held the lock.
    """
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=False)
    r = _run(WRAPPER, _env(stub, sd, tmp_path, JUDGE_LOCK_WAIT="abc",
                           FILES=f"{tree}/*/surgical/*_surgical.json"))
    out = r.stdout + r.stderr
    assert r.returncode == 1, out                      # a config error, not "busy"
    assert "refusing to judge without the account lock" in out, out
    assert "judged nothing" not in out, out            # not the contention message
    assert (sd / "count").read_text().strip() == "0"   # and nothing was judged either way


def test_the_busy_message_names_the_judge_that_holds_the_account(tmp_path):
    """A 75 at hour 12 of the campaign is only actionable if you can tell a live judge from a
    wedged one. The holder is recorded after the lock is taken, so whoever prints it is naming the
    process that actually holds the account."""
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=False)
    lock = tmp_path / "account.lock"
    files = f"{tree}/*/surgical/*_surgical.json"
    holder = subprocess.Popen(WRAPPER, cwd=REPO, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True,
                              env=_env(stub, sd, tmp_path, STUB_SLEEP=10, FILES=files))
    pg = os.getpgid(holder.pid)
    try:
        _wait_until(lambda: (sd / "running").exists(), "the first judge to start under the lock")
        r = _run(WRAPPER, _env(stub, sd, tmp_path, JUDGE_LOCK_WAIT=0, FILES=files))
    finally:
        _killpg(pg)
        holder.wait(timeout=30)
    assert r.returncode == 75, r.stdout + r.stderr
    # $$ inside `bash scripts/qwen15_judge.sh` is the pid Popen returned
    assert f"pid={holder.pid}" in r.stdout, r.stdout
    assert "unrecorded" not in r.stdout, r.stdout


def test_the_default_account_lock_is_one_file_for_the_account(tmp_path):
    """Per ACCOUNT -- not per tree, per model or per run. Two passes that share nothing (different
    result tree, different judge key) must still land on the same lock, or the campaign's gemma
    pass and a hand-run Qwen pass would both submit against one 20,000-request cap.

    This is the one test that uses the real default path, because the default IS the claim. If a
    genuine judge is running it will wait JUDGE_LOCK_WAIT and then fail here, loudly, rather than
    quietly testing a private lock that proves nothing."""
    stub, sd = _stub(tmp_path)
    lock = REPO / "logs/openrouter_judge.lock"
    recorded = []
    for tree_name, suffix in (("qwentree", "api_gpt_5_6_luna"), ("gemmatree", "api_other_model")):
        tree = tmp_path / tree_name
        _surgical(tree / "cell/surgical/alpha_surgical.json", scored=True)
        env = _env(stub, sd, tmp_path, JUDGE_LOCK_WAIT=20,
                   FILES=f"{tree}/*/surgical/*_surgical.json")
        del env["JUDGE_LOCK"]                     # the default path is what is under test
        env["STUB_SUFFIX"] = suffix
        r = _run(WRAPPER, env)
        assert r.returncode != 75, f"a real API judge holds {lock}; rerun this when it is idle"
        assert r.returncode == 0, r.stdout + r.stderr
        recorded.append(Path(f"{lock}.holder").read_text())
    # Both runs recorded themselves in the SAME file, and the second really did overwrite the
    # first: a lock path that carried the tree or the judge key would have split them.
    assert "qwentree" in recorded[0], recorded
    assert "gemmatree" in recorded[1], recorded


def test_killing_the_judge_releases_the_account_lock(tmp_path):
    """A judge killed by ctrl-C, a reboot or an OOM must not leave the account locked: the next
    pass would wait JUDGE_LOCK_WAIT and report 75 for the rest of the campaign, and nothing would
    be judged. flock is held by an open file description, so the kernel releases it on death --
    but only while the judge stays inside the killed process group."""
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=False)
    lock = tmp_path / "account.lock"
    p = subprocess.Popen(WRAPPER, cwd=REPO, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True,
                         env=_env(stub, sd, tmp_path, STUB_SLEEP=30,
                                  FILES=f"{tree}/*/surgical/*_surgical.json"))
    pg = os.getpgid(p.pid)
    try:
        _wait_until(lambda: (sd / "running").exists(), "the judge to start under the lock")
        assert _held(lock), "the judge is running and the account is not locked"
        _killpg(pg)
        p.wait(timeout=30)
        _wait_until(lambda: not _held(lock), "the account lock to be released by the dead judge",
                    timeout=10)
    finally:
        _killpg(pg)


def test_the_account_stays_locked_while_a_judge_outlives_its_wrapper(tmp_path):
    """The other half of the same property. Kill the wrapper alone and the judge keeps running --
    and a running judge can still create batches that cannot be cancelled. fd 9 is inherited, so
    the account stays locked for exactly as long as something can still spend it. Closing the fd
    for the child (`9>&-`) would make a second judge start on top of the first."""
    stub, sd = _stub(tmp_path)
    tree = tmp_path / "tree"
    _surgical(tree / "cell/surgical/alpha_surgical.json", scored=False)
    lock = tmp_path / "account.lock"
    files = f"{tree}/*/surgical/*_surgical.json"
    p = subprocess.Popen(WRAPPER, cwd=REPO, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True,
                         env=_env(stub, sd, tmp_path, STUB_SLEEP=20, FILES=files))
    pg = os.getpgid(p.pid)
    try:
        _wait_until(lambda: (sd / "running").exists(), "the judge to start under the lock")
        p.kill()                                  # the wrapper shell only
        p.wait(timeout=30)
        assert (sd / "running").exists(), "the judge died with its wrapper; nothing was still spending"
        r = _run(WRAPPER, _env(stub, sd, tmp_path, JUDGE_LOCK_WAIT=0, FILES=files))
        assert r.returncode == 75, r.stdout + r.stderr
        assert int((sd / "count").read_text()) == 1, "a second judge started while the first ran"
    finally:
        _killpg(pg)


# Every api-backend caller must go through the wrapper, because the lock is IN the wrapper.
# logs/overnight_chain/l20_surgical_v2.sh is the one that does not: a stopped chain (the l20 v2
# surgical re-judge) whose only guard is a check-then-act `ps` loop 300 s wide. It is listed here
# rather than hidden, so that restarting it, or adding a second unguarded caller, is a red test.
# When it is fixed or deleted, delete the entry -- this assertion is an equality on purpose.
_UNLOCKED_API_CALLERS = ["logs/overnight_chain/l20_surgical_v2.sh"]
_SCAN_ROOTS = ["scripts", "analysis", "logs/overnight_chain"]


def _api_backend_callers():
    """Files that run the api judge, minus the wrapper that holds the account lock for it."""
    hits = []
    for root in _SCAN_ROOTS:
        d = REPO / root
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.suffix not in (".sh", ".py") or not f.is_file():
                continue
            txt = f.read_text()
            rel = str(f.relative_to(REPO))
            if rel == "scripts/qwen15_judge.sh":
                continue
            if "judge_saved_gens_big" in txt and "--judge_backend api" in txt:
                hits.append(rel)
            elif "api_judge_scores" in txt:
                hits.append(rel)
    return hits


def test_no_new_caller_spends_the_account_outside_the_wrapper():
    expected = [p for p in _UNLOCKED_API_CALLERS if (REPO / p).exists()]
    assert _api_backend_callers() == expected, (
        "an api-backend judge outside scripts/qwen15_judge.sh takes no account lock and can "
        "double-pay. Route it through the wrapper, or list it above once it is fixed or gone.")


@pytest.mark.xfail(strict=True, reason=(
    "OPEN WINDOW, 2026-09-17: the account lock lives in the wrapper, so `python -m "
    "src.clcd.judge_saved_gens_big --judge_backend api` -- the invocation in that module's own "
    "docstring -- walks straight past it and submits. The guard belongs in "
    "judge_api.api_judge_scores (the one function that spends), honouring $JUDGE_LOCK and "
    "exiting 75 when busy; the wrapper must then stop taking the lock itself, or its own child "
    "would block on the fd its parent holds. When that lands this XPASSes: drop the marker."))
def test_a_direct_module_run_takes_the_same_account_lock(tmp_path):
    """The bypass. Another process holds the account lock; a direct module run must refuse to
    spend, exactly as the wrapper does (75), instead of submitting a second uncancellable fleet.

    Runs the REAL module, so the key must be provably absent first: cwd is outside the repo (no
    .env to load) and OPENROUTER_API_KEY is empty, and that is checked before anything is started.
    """
    env = dict(os.environ, OPENROUTER_API_KEY="", PYTHONPATH=str(REPO),
               JUDGE_LOCK=str(tmp_path / "account.lock"), JUDGE_LOCK_WAIT="1")
    probe = subprocess.run([str(REPO / ".venv/bin/python"), "-c",
                            "import sys; from src.clcd import judge_api as J; "
                            "sys.exit(9 if J._api_key_from_env() else 0)"],
                           cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, ("the child would see a live OPENROUTER_API_KEY; refusing to run "
                                   "a test that could submit a real batch: " + probe.stderr[-400:])
    surg = tmp_path / "cell/surgical/alpha_surgical.json"
    surg.parent.mkdir(parents=True)
    surg.write_text(json.dumps({"clean_questions": ["q1", "q2"], "indep_questions": ["i1"],
                                "conditions": {"intact": {"clean_gens": ["a1", "a2"],
                                                          "indep_gens": ["b1"]}}}))
    lock, ready = tmp_path / "account.lock", tmp_path / "held"
    holder = subprocess.Popen(["flock", str(lock), "-c", f"touch {ready}; sleep 60"])
    try:
        _wait_until(ready.exists, "the external lock holder to start")
        r = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "src.clcd.judge_saved_gens_big",
                            "--judge_backend", "api", "--files", str(surg)],
                           cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    finally:
        holder.kill()
        holder.wait()
    assert r.returncode == 75, (
        "the module ran on past the held account lock:\n" + (r.stdout + r.stderr)[-600:])
