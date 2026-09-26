"""scripts/campaign3_launch.sh: what a two-box campaign is allowed to report as finished.

Campaign 3 runs on two machines. One of them holds the OpenRouter account, judges, and is the
collection point; the other just searches. That split introduces three ways to lose work quietly,
and each one is a test below:

1. A box that cannot be REACHED is not a box whose work is DONE. If those collapse, the collector
   declares the campaign complete while the other machine is still running 70 cells -- or has died
   with them unfinished -- and the shortfall shows up as "0 fires", which is the necessity SUCCESS
   value (CLAUDE.md Rule 12).
2. The collector REWRITES surgical files in place as it judges them. Pulling the other box's tree
   without --ignore-existing copies its unjudged originals back over the judged copies, so the next
   pass re-issues every request in them (~2,838 per cell, ~199k across the worker's share) and the
   verdicts already paid for are gone. Nothing errors; the bill just doubles.
3. EXPECT_Q/EXPECT_G/EXPECT_R are the completeness check. If they drift from the cells the drivers
   actually launch, the campaign either reports a shortfall that is not real or -- worse -- passes
   while short. So they are asserted against the launcher's own cell counts, not restated.

Every test drives the REAL script under DRY=1, where $PY is a stub and no GPU is touched, and with
$SSH pointed at a local stub so the two-box paths execute rather than being mocked out. The
production logic -- driver selection, the remote guards, the pull, the counting -- is the real one.

(New module rather than an addition to tests/test_qwen15_phase1.py, which tests the per-cell driver
that this script *launches*. One test module per module, as elsewhere here.)
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "scripts" / "campaign3_launch.sh"

# Stands in for `ssh`. Real ssh runs a single string argument through a shell and a multi-word
# argv directly; rsync relies on the second form for `rsync --server`. Reproducing both is what
# lets remote_state AND remote_pull run for real against a directory playing the part of the
# other box, instead of being stubbed into always agreeing with us.
SSH_STUB = """#!/bin/bash
shift                      # drop the host
if [ $# -eq 1 ]; then exec bash -c "$1"; else exec "$@"; fi
"""


@pytest.fixture
def env(tmp_path):
    """A dry-run environment: private job dir, a stub ssh, and a directory posing as the other box."""
    ssh = tmp_path / "sshstub.sh"
    ssh.write_text(SSH_STUB)
    ssh.chmod(0o755)
    jobdir = tmp_path / "job"
    (jobdir / "tmp").mkdir(parents=True)
    e = dict(os.environ)
    e.update(
        DRY="1",
        POLL="2",
        CLAUDE_JOB_DIR=str(jobdir),
        SSH=str(ssh),
        REMOTE_ROOT=str(tmp_path / "remote"),
    )
    e.pop("REMOTE", None)
    return e


def run(env, timeout=420, **overrides):
    env = {**env, **{k: str(v) for k, v in overrides.items()}}
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(REPO),
    )


def launched(out):
    """{driver tag: cells} from the launcher's own announcements."""
    got = {}
    for line in out.splitlines():
        if " launched " in line and line.rstrip().endswith("cells)"):
            parts = line.split()
            got[parts[parts.index("launched") + 1]] = int(parts[-2].lstrip("("))
    return got


def remote_with_pid(env, pid):
    """Give the fake remote a drivers.pids naming `pid`, where the collector looks for it."""
    d = Path(env["REMOTE_ROOT"]) / "logs/overnight_chain/c3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "drivers.pids").write_text(f"{pid}\n")
    return d


# --------------------------------------------------------------------------- driver selection


def test_expect_constants_equal_the_cells_actually_launched(env):
    """The completeness check must count the same cells the drivers were given.

    EXPECT_Q/EXPECT_G/EXPECT_R are what makes "campaign 3 complete" mean anything. Restating them
    here would prove nothing, so they are read from the script's defaults and compared against the
    cell counts it announces -- add a seed, an arm or a family without touching them and this fails.
    """
    r = run(env, DRY_FULL_SEEDS=1)
    got = launched(r.stdout)
    assert got == {
        "qwen_all": 20,
        "qwen_l17_25": 20,
        "qwen_l20": 20,
        "gemma_all": 10,
        "gemma_l1523": 10,
        "gemma_l19": 10,
        "gradroute": 15,
    }, r.stdout
    src = LAUNCHER.read_text()
    for name, tags in (
        ("EXPECT_Q", ("qwen_all", "qwen_l17_25", "qwen_l20")),
        ("EXPECT_G", ("gemma_all", "gemma_l1523", "gemma_l19")),
        ("EXPECT_R", ("gradroute",)),
    ):
        default = int(src.split(f'{name}="${{{name}:-')[1].split("}")[0])
        assert default == sum(got[t] for t in tags), f"{name} does not match its drivers"


def test_drivers_subset_runs_only_what_it_names(env):
    """The worker box must not start the collector's cells, or both boxes search the same organism."""
    worker = "qwen_l17_25 qwen_l20 gemma_all gemma_l1523 gemma_l19 gradroute"
    r = run(env, DRIVERS=worker)
    assert "qwen_all" not in launched(r.stdout)
    assert "skip qwen_all (not in DRIVERS)" in r.stdout
    assert set(launched(r.stdout)) == set(worker.split())


def _tree(root):
    """(relative path -> (size, mtime_ns)) for every file under root; {} if root does not exist."""
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_a_dry_run_never_writes_the_real_logs_dir(env):
    """A dry run must not touch the repo's logs/ at all.

    WHY: until 2026-09-26 the driver logs and per-cell search logs of a dry run went to the same
    relative logs/ paths as a real campaign. Running this test module overwrote 105 real campaign-3
    search logs with one-line stubs (20 of them, the Qwen 1.5B `all` cells, had no other copy and
    are lost). In a worktree logs/ is a symlink to the main checkout's, so any pytest run did it.
    """
    real = REPO / "logs"
    existed, before = real.exists(), _tree(real)
    run(env, DRIVERS="qwen_all qwen_l17_25 qwen_l20 gemma_all gemma_l1523 gemma_l19 gradroute")
    assert real.exists() == existed, "a dry run created the repo's logs/ directory"
    after = _tree(real)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert not changed, f"a dry run wrote {len(changed)} file(s) under the real logs/: {changed[:5]}"


def test_drivers_matching_nothing_refuses_instead_of_reporting_an_empty_campaign(env):
    """A typo in DRIVERS must stop the run, not produce a clean report of 0/105 cells.

    Without this the script falls through to the counts, finds nothing, and prints INCOMPLETE --
    indistinguishable from a campaign whose every cell died.
    """
    r = run(env, DRIVERS="qwen_all_typo", timeout=120)
    assert r.returncode == 2, r.stdout
    assert "NO DRIVERS LAUNCHED" in r.stdout
    assert launched(r.stdout) == {}


# --------------------------------------------------------------------------- the other box


@pytest.mark.parametrize(
    "state, why",
    [
        ("nofile", "the worker has not started"),
        ("done", "the worker has already exited"),
        ("unreachable", "the worker cannot be reached"),
    ],
)
def test_collector_refuses_to_start_without_a_live_remote(env, state, why):
    """Starting the collector alone would judge a partial campaign and call it complete.

    All three states are refusals, including `done`: a remote whose drivers have already exited
    before this side even started is not a campaign in progress, it is a stale pid file.
    """
    if state == "done":
        remote_with_pid(env, 2**22)  # a pid that cannot be running
    elif state == "unreachable":
        env = {**env, "SSH": "/bin/false"}
    r = run(env, REMOTE="fakehost", DRIVERS="qwen_all", timeout=120)
    assert r.returncode == 2, f"{why}: {r.stdout}"
    assert "REFUSING TO START" in r.stdout
    assert launched(r.stdout) == {}, "no driver may start when the remote is not live"


def test_collector_does_not_finish_while_the_remote_still_has_work(env):
    """Local drivers finishing is not the campaign finishing.

    The dry-run drivers exit in milliseconds; the fake remote stays alive. If the wait loop looked
    only at local pids it would sail past, judge, and print the completion line while the other box
    still held 70 cells.
    """
    holder = subprocess.Popen(["sleep", "60"])
    try:
        remote_with_pid(env, holder.pid)
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            run(env, REMOTE="fakehost", DRIVERS="qwen_all", timeout=25)
        out = (excinfo.value.stdout or b"")
        out = out.decode() if isinstance(out, bytes) else out
    finally:
        holder.kill()
        holder.wait()
    assert "remote fakehost: drivers running" in out
    assert "all drivers exited" not in out, "declared completion while the remote was still running"


def test_remote_pull_never_overwrites_a_judged_file(env):
    """--ignore-existing is load-bearing, and this is the failure it prevents.

    The collector adds judge verdicts to surgical files IN PLACE. The remote still holds the
    unjudged original of every file it produced. A pull that overwrites would silently discard
    those verdicts and make the next pass re-issue ~2,838 requests per file -- no error, just a
    second bill and lost results. New files must still arrive, or the pull does nothing at all.
    """
    # Repo-relative trees, because that is what production uses and what remote_pull's
    # "$REMOTE_ROOT/$tree" join is built for. An absolute tree makes that join meaningless, so a
    # test using the default dry trees would sync nothing and still pass the preservation check.
    rel = f".pytest_dry_{os.getpid()}"
    holder = subprocess.Popen(["sleep", "60"])
    try:
        remote_with_pid(env, holder.pid)
        remote = Path(env["REMOTE_ROOT"]) / rel / "gemma/r64_k8/surgical"
        remote.mkdir(parents=True)
        (remote / "judged_surgical.json").write_text('{"verdict": "UNJUDGED"}')
        (remote / "fresh_surgical.json").write_text('{"verdict": "UNJUDGED"}')

        local = REPO / rel / "gemma/r64_k8/surgical"
        local.mkdir(parents=True)
        judged = local / "judged_surgical.json"
        judged.write_text('{"verdict": "JUDGED BY LUNA"}')

        # JUDGE_POLL forces a pull to actually happen inside the wait loop. Without it the only
        # pull is the one after both sides exit, which this test never reaches -- and then BOTH
        # assertions below would pass while remote_pull had not run at all.
        with pytest.raises(subprocess.TimeoutExpired):
            run(env, REMOTE="fakehost", DRIVERS="qwen_all", DRY_TREE_ROOT=rel,
                JUDGE_POLL=4, timeout=40)

        assert (local / "fresh_surgical.json").exists(), (
            "a remote-only file was not pulled, so the preservation check below proves nothing"
        )
        assert judged.read_text() == '{"verdict": "JUDGED BY LUNA"}', (
            "the remote's unjudged copy overwrote a judged file: every request in it would be "
            "re-issued and paid for again"
        )
    finally:
        holder.kill()
        holder.wait()
        shutil.rmtree(REPO / rel, ignore_errors=True)
