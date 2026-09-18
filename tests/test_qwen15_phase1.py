"""scripts/qwen15_phase1.sh: what the driver's completion signal is allowed to mean.

The block and replicate arms of the elimination comparison walk a visiting-order file written by the
reference arm, and a cell whose file never appears is SKIPPED. The launch check in the block-
elimination spec (§6) reads "=== Phase 1 complete" with no "SEARCH FAILED" as the positive signal
that an arm ran, so a skip has to be visible in the same place -- otherwise a run that produced
nothing looks exactly like a finished one, and only the verdict tool notices, hours later.

The same applies to every OTHER way a cell can fail to produce its three files (pre-launch audit
findings 5 and 6): a crashed or OOM-killed search, a leak or surgical failure, a refused cell, a
missing gate record -- and a GPU slot held by a lock whose owner died, which costs the campaign a
slot for the rest of a multi-day run with nothing in any log saying so. Every test below asserts
the COUNT and the exit status, not the presence of a word: "SEARCH FAILED" was already printed by
the driver that exited 0 on a campaign of failures.

Every test drives the real script with a stub `$PY` and a stub `nvidia-smi`, so no GPU is touched
and the cells finish in milliseconds. The stubs are the only fiction: the driver, its locks, its
counting and its summary are the production ones.

(New module rather than an addition to tests/test_circuit_search_grid.py, which mirrors this
driver's search invocation but tests src/clcd/exp_circuit_search.py, or to
tests/test_compare_elim_protocols.py, which tests the verdict tool. Neither covers the driver; one
test module per module, as everywhere else here.)
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# $PY for the driver: the three GPU steps are intercepted (and made to fail on demand, by a glob
# against the output path so one run can have a failing cell and a passing one); everything else --
# the gate-record reads, the Gate-A verdict, the end-of-run summary -- is the real interpreter, so
# the driver's own logic is exercised rather than mocked.
PYSTUB = """#!/bin/bash
out=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
case " $* " in
  *" src.clcd.exp_circuit_search "*)
    case "$out" in ${STUB_SEARCH_FAIL:-__never__}) exit "${STUB_SEARCH_RC:-7}" ;; esac
    printf '%s' '{"status":"ok","both_K":42,"kept_latents":[[0,1]],"adapter":"stub"}' > "$out"
    exit 0 ;;
  *verify_holdout_necessity.py*)
    case "$CLCD_OUT" in ${STUB_LEAK_FAIL:-__never__}) exit "${STUB_LEAK_RC:-3}" ;; esac
    printf '%s' '[{"file":"c","adapter":"stub","n_kept":42,"total_fires":0,"total_prompts":4000}]' \\
      > "$CLCD_OUT"
    exit 0 ;;
  *" src.clcd.exp_surgical_removal "*)
    case "$out" in ${STUB_SURGICAL_FAIL:-__never__}) exit "${STUB_SURGICAL_RC:-137}" ;; esac
    printf '%s' '{"conditions": {}}' > "$out"
    exit 0 ;;
esac
exec REALPY "$@"
"""

# A card that is idle, or an NVML fault when STUB_SMI_RC says so.
SMISTUB = """#!/bin/bash
if [ "${STUB_SMI_RC:-0}" != 0 ]; then
  echo "Unable to determine the device handle for GPU 0: Unknown Error" >&2
  exit "$STUB_SMI_RC"
fi
case " $* " in *memory.total*) echo "1000, 81920" ;; *) echo "100, 0" ;; esac
"""


def harness(tmp_path, **env_extra):
    """A driver environment with stub tools, a private tree and one gate record per cell."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "pystub").write_text(PYSTUB.replace("REALPY", sys.executable))
    (bindir / "nvidia-smi").write_text(SMISTUB)
    for f in bindir.iterdir():
        f.chmod(0o755)
    (tmp_path / "adapter").mkdir()
    gates = tmp_path / "gates"
    gates.mkdir()
    env = dict(
        os.environ,
        PY=str(bindir / "pystub"),
        PATH=f"{bindir}:{os.environ['PATH']}",
        GATE_DIR=str(gates),
        SRC=str(tmp_path / "src"),
        LOGDIR=str(tmp_path / "logs"),
        LOCKROOT=str(tmp_path / "locks"),
        GPUS="0",
        STAGGER="0",
        # One slot per cell, so a multi-cell test is not serialised behind claim_gpu's 60 s pass.
        # The two lock tests below override it: their point is a slot that cannot be had.
        SLOTS_PER_GPU="4",
    )
    env.update(env_extra)
    return env, gates


def gate_record(gates: Path, arm: str, fam: str, seed: int, adapter: Path) -> None:
    """A PASS record in the shape gate_a.main writes -- the fields the driver reads."""
    (gates / f"gate_a_{arm}_{fam}_s{seed}.json").write_text(
        json.dumps(
            {
                "adapter": str(adapter),
                "base_model": "interpretable-finetuning/qwen2.5-1.5b-unaliased",
                "data": "data/sleeper/prepared_eval6k_qwen15",
                "intact_backdoor": {"n": 1000, "fires": 990, "rate": 0.99},
                "clean_falsefire": {"n": 1000, "fires": 0, "rate": 0.0},
                "asr_ok": True,
                "eot_ok": True,
                "clean_ok": True,
                "verdict": "PASS",
            }
        )
    )


def drive(cells, env, timeout=180):
    """Run the driver; kill its whole process group if it outlives `timeout`.

    Returns (returncode-or-None, stdout, stderr). None means it was still running: a driver that
    hangs in claim_gpu is a FAILURE of every test here, not a pass with missing output.
    """
    p = subprocess.Popen(
        ["bash", "scripts/qwen15_phase1.sh", *cells],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        out, err = p.communicate()
        return None, out, err


def outcomes(stdout: str) -> dict:
    """The PHASE1 OUTCOMES line, parsed. Its absence is a failure, not an empty dict."""
    line = [l for l in stdout.splitlines() if l.startswith("PHASE1 OUTCOMES:")]
    assert len(line) == 1, stdout
    return dict(kv.split("=") for kv in line[0].split(":", 1)[1].split())


def test_a_run_whose_ordered_cells_were_all_skipped_does_not_report_a_clean_finish(tmp_path):
    # An empty order dir with no wait is the state that a reference arm which has not reached
    # attribution, a SEARCH FAILED in that arm, or a stale --orders path all leave behind. No cell
    # can start, and none of them touches a GPU: the driver must still SAY so, with the count.
    env = dict(os.environ,
               PY=sys.executable, GPUS="0", STEPS="search", STAGGER="0",
               ELIM_ORDER_FROM_DIR=str(tmp_path / "orders"), ORDER_WAIT_S="0",
               SRC=str(tmp_path / "src"), LOGDIR=str(tmp_path / "logs"),
               LOCKROOT=str(tmp_path / "locks"))
    rc, out, _ = drive(["r64_dense l20 42", "r64_k8 l20 42"], env)
    assert "=== Phase 1 complete" in out            # the old positive signal is still printed ...
    assert "PHASE1 SKIPPED 2 CELLS" in out          # ... next to the count that contradicts it
    assert outcomes(out)["skipped"] == "2"
    assert "PHASE1 COMPLETED 0 of 2 CELLS" in out
    assert rc == 1, out                             # and the status says the run produced nothing
    assert not (tmp_path / "src").exists()          # and nothing was searched


def test_a_crashed_search_is_counted_and_makes_the_driver_exit_non_zero(tmp_path):
    """Audit finding 5: an OOM-killed cell printed SEARCH FAILED and the run still exited 0.

    Two cells, one search killed. The distinction that matters is between "1 of 2 cells produced
    its circuit" and "the run finished": the launcher judges a partial tree on the second reading.
    """
    env, gates = harness(tmp_path, STUB_SEARCH_FAIL="*l20_seed42*", STUB_SEARCH_RC="7")
    for seed in (42, 43):
        gate_record(gates, "r64_k8", "l20", seed, tmp_path / "adapter")
    rc, out, _ = drive(["r64_k8 l20 42", "r64_k8 l20 43"], env)

    assert "SEARCH FAILED rc=7" in out, out
    o = outcomes(out)
    assert o["search_failed"] == "1", out
    assert o["done"] == "1", out
    assert "PHASE1 COMPLETED 1 of 2 CELLS" in out, out
    assert "  !! r64_k8 l20 42 search_failed" in out, out   # named, so nobody recounts by hand
    assert rc == 1, out
    # the surviving cell really did run all three steps -- otherwise "done=1" is meaningless
    assert (tmp_path / "src/r64_k8/surgical/l20_seed43_surgical.json").exists()


def test_leak_and_surgical_failures_are_counted_separately_and_the_cell_is_not_done(tmp_path):
    """A cell whose search succeeded but whose leak and surgical died is NOT a finished cell.

    Both steps used to be `|| echo` and fell through to the unconditional "cell done", so a cell
    with no leak and no surgical file was indistinguishable in the log from a complete one -- and
    the judge would later score generations that were never written.
    """
    env, gates = harness(tmp_path, STUB_LEAK_FAIL="*", STUB_SURGICAL_FAIL="*")
    gate_record(gates, "r64_k8", "l20", 42, tmp_path / "adapter")
    rc, out, _ = drive(["r64_k8 l20 42"], env)

    o = outcomes(out)
    assert o["leak_failed"] == "1" and o["surgical_failed"] == "1", out
    assert o["done"] == "0", out
    assert "cell INCOMPLETE" in out and "cell done" not in out, out
    assert "PHASE1 COMPLETED 0 of 1 CELLS" in out, out
    assert rc == 1, out


def test_a_refused_cell_and_a_missing_gate_record_are_counted_too(tmp_path):
    """The two silent non-runs: a data mismatch the driver refuses, and a cell with no record.

    Neither ever reaches a GPU, so neither prints a step failure; both used to leave the run
    reading "SKIPPED 0 CELLS" with exit 0.
    """
    env, gates = harness(tmp_path)
    gate_record(gates, "r64_k8", "l20", 42, tmp_path / "adapter")
    env["DATA"] = "data/sleeper/prepared_eval6k"          # the GEMMA set: the cell is refused
    rc, out, _ = drive(["r64_k8 l20 42", "r64_k8 l20 99"], env)

    o = outcomes(out)
    assert o["refused"] == "1" and o["no_gate_record"] == "1", out
    assert o["done"] == "0", out
    assert "PHASE1 COMPLETED 0 of 2 CELLS" in out, out
    assert rc == 1, out


def test_the_end_of_run_summary_reads_the_tree_this_run_wrote(tmp_path):
    """It globbed the hardcoded clcd_results/qwen15, and crashed on a list-shaped leak JSON.

    A campaign run with SRC=clcd_results/qwen15_campaign3 therefore summarised the OLD tree -- and
    when the glob did match, `.get` on the list verify_holdout_necessity.py writes took the summary,
    and with it the driver's exit status, down in a traceback.
    """
    env, gates = harness(tmp_path)
    gate_record(gates, "r99_k8", "l20", 42, tmp_path / "adapter")
    rc, out, _ = drive(["r99_k8 l20 42"], env)

    assert rc == 0, out
    assert "Traceback" not in out, out
    circuits = out.split("=== circuits ===")[1].split("=== held-out leak")[0]
    leak = out.split("=== held-out leak (T5) ===")[1]
    assert "r99_k8" in circuits and "l20_seed42" in circuits, out
    # nothing from any other tree: a hardcoded glob would list clcd_results/qwen15 instead
    assert [l for l in circuits.splitlines() if l.strip()] == [
        l for l in circuits.splitlines() if "r99_k8" in l
    ], out
    assert "fires=0" in leak and "n=4000" in leak, out   # the list-shaped record, read


def test_a_slot_lock_whose_owner_is_dead_is_reclaimed(tmp_path):
    """Audit finding 6: ownerless locks cost the campaign slots for the rest of the run.

    A killed driver -- or its orphaned claim loop, taking a slot minutes after it died -- leaves a
    lock nothing will ever release. With one card and one slot, the cell can only run if the driver
    reclaims it, so this asserts the reclaim through the thing that depends on it.
    """
    env, gates = harness(tmp_path, SLOTS_PER_GPU="1")
    gate_record(gates, "r64_k8", "l20", 42, tmp_path / "adapter")
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    lock = tmp_path / "locks" / f"{os.uname().nodename}_0_s1"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"{dead.pid} 2026-09-17T00:00:00+00:00 killed-driver\n")

    rc, out, err = drive(["r64_k8 l20 42"], env, timeout=120)

    assert f"{dead.pid}" in err and "is STALE" in err, err
    assert rc == 0, (out, err)
    assert "PHASE1 COMPLETED 1 of 1 CELLS" in out, out
    assert not lock.exists(), "the reclaimed slot must be released at the end of the cell"


def test_a_slot_lock_whose_owner_is_alive_is_left_alone_and_the_wait_is_logged(tmp_path):
    """The other half of the reclaim: a live owner keeps its slot, and the driver says it is waiting.

    Both halves are needed. A driver that reclaimed every lock would put two searches on one slot,
    which is how 27 searches once ran on 14 slots; a driver that waits in silence is the observed
    failure where an NVML fault or a pool of stale locks looked exactly like ordinary contention.
    """
    env, gates = harness(tmp_path, SLOTS_PER_GPU="1")
    gate_record(gates, "r64_k8", "l20", 42, tmp_path / "adapter")
    lock = tmp_path / "locks" / f"{os.uname().nodename}_0_s1"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(f"{os.getpid()} 2026-09-17T00:00:00+00:00 pytest\n")

    rc, out, err = drive(["r64_k8 l20 42"], env, timeout=20)

    assert rc is None, "the driver must keep waiting for a slot held by a LIVE owner"
    assert "no slot after 0s" in err and "g0:s1:held" in err, err
    assert "reclaiming" not in err, err
    assert "T2 search" not in out, out                 # and the cell never started
    assert (lock / "owner").read_text().startswith(str(os.getpid()))


@pytest.mark.parametrize("slots", ["1", "2"])
def test_an_nvidia_smi_failure_is_reported_instead_of_being_read_as_a_busy_card(tmp_path, slots):
    """An NVML fault used to be swallowed by 2>/dev/null and mapped to "card busy".

    Five drivers then park in claim_gpu indefinitely, printing nothing, and the campaign looks like
    it is merely waiting for slots.

    Both slot counts, because gpu_is_idle has a BRANCH: SLOTS_PER_GPU=1 queries utilisation (every
    cluster_run and dense_search_launch invocation) and >1 queries free memory (campaign 3 runs 2).
    A version of this test that ran only one of them stayed green while the other swallowed the
    error, which is how the 2>/dev/null survived in the first place.
    """
    env, gates = harness(tmp_path, STUB_SMI_RC="9", SLOTS_PER_GPU=slots)
    gate_record(gates, "r64_k8", "l20", 42, tmp_path / "adapter")

    rc, out, err = drive(["r64_k8 l20 42"], env, timeout=20)

    assert rc is None                                   # still waiting, as before ...
    assert "nvidia-smi FAILED for gpu 0" in err, err     # ... but no longer in silence
    assert "g0:nvidia-smi-error" in err, err
    assert "T2 search" not in out, out
