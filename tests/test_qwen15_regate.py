"""scripts/qwen15_regate.sh: a re-gate that did not happen must not read as one that did.

The re-gate's output is a verdict per organism, and a FAIL verdict is a RESULT (gate_a exits 1 and
writes its record). Before 2026-11 the driver could not tell that apart from a cell that never ran:
every cell's status was discarded by `wait`, the run always ended "re-gate complete" with exit 0,
and its private copy of the GPU claim loop mapped an nvidia-smi failure to "busy" and then waited
forever in silence. Each test asserts the count or the exit status, not the presence of a word.

Driven like tests/test_qwen15_phase1.py: the real script, a stub `$PY` for gate_a only, and that
module's stub `nvidia-smi`. (New module because the driver is a different script; the stubs and
the process-group-safe runner are imported from there, not copied.)
"""

import json
import os
import signal
import subprocess
import sys

from tests.test_qwen15_phase1 import REPO, SMISTUB

# gate_a is the only GPU step. By the adapter path: "fail" writes a FAIL record and exits 1 (a
# verdict), "crash" exits 1 with no record (gate_a died). Everything else is the real interpreter.
PYSTUB = """#!/bin/bash
out=""; adapter=""; prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && out="$a"; [ "$prev" = "--adapter" ] && adapter="$a"; prev="$a"
done
case " $* " in
  *" src.clcd.gate_a "*)
    case "$adapter" in
      *crash*) echo "Traceback: CUDA out of memory" >&2; exit 1 ;;
      *fail*)  printf '%s' '{"asr_ok": false, "eot_ok": true, "clean_falsefire": {"fires": 0, "n": 1000, "rate": 0.0}, "intact_backdoor": {"rate": 0.1}}' > "$out"; exit 1 ;;
      *)       printf '%s' '{"asr_ok": true, "eot_ok": true, "clean_falsefire": {"fires": 0, "n": 1000, "rate": 0.0}, "intact_backdoor": {"rate": 0.99}}' > "$out"; exit 0 ;;
    esac ;;
esac
exec REALPY "$@"
"""


def harness(tmp_path, cells, **env_extra):
    """Stub tools, a private source tree with one record per (arm, fam, seed, adapter-name)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "pystub").write_text(PYSTUB.replace("REALPY", sys.executable))
    (bindir / "nvidia-smi").write_text(SMISTUB)
    for f in bindir.iterdir():
        f.chmod(0o755)
    src = tmp_path / "src"
    src.mkdir()
    for arm, fam, seed, adapter in cells:
        if adapter is None:
            continue                                   # no source record for this cell
        (tmp_path / adapter).mkdir()
        (src / f"gate_a_{arm}_{fam}_s{seed}.json").write_text(json.dumps({
            "adapter": str(tmp_path / adapter), "asr_ok": True, "eot_ok": True,
            "intact_backdoor": {"rate": 0.99},
            "clean_falsefire": {"fires": 0, "n": 1000, "rate": 0.0}}))
    env = dict(
        os.environ,
        PY=str(bindir / "pystub"),
        PATH=f"{bindir}:{os.environ['PATH']}",
        SRC=str(src),
        OUT=str(tmp_path / "regate"),
        LOGDIR=str(tmp_path / "logs"),
        LOCKROOT=str(tmp_path / "locks"),
        GPUS="0",
        STAGGER="0",
        SLOTS_PER_GPU="4",
    )
    env.update(env_extra)
    return env, [f"{a} {f} {s}" for a, f, s, _ in cells]


def drive(cells, env, timeout=120):
    p = subprocess.Popen(
        ["bash", "scripts/qwen15_regate.sh", *cells],
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


def test_a_fail_verdict_is_a_result_and_the_run_exits_zero(tmp_path):
    """FAIL is what the re-gate exists to measure. Counting it as a failed cell would make every
    honest negative result look like a broken run -- the opposite error from the one fixed here."""
    env, cells = harness(tmp_path, [("r64_k8", "l20", 42, "ok"), ("r42_k5", "l21", 42, "fail")])
    rc, out, err = drive(cells, env)
    assert rc == 0, (out, err)
    assert "REGATE COMPLETED 2 of 2 CELLS" in out, out
    assert "PASS -> FAIL" in out, out                  # the summary read THIS run's records


def test_missing_source_record_and_crashed_gate_are_counted_and_exit_non_zero(tmp_path):
    """Neither produced a verdict. Both used to end in "re-gate complete" and exit 0."""
    env, cells = harness(tmp_path, [("r64_k8", "l20", 42, "ok"),
                                    ("r64_k8", "l20", 43, None),
                                    ("r64_k8", "l20", 44, "crash")])
    rc, out, err = drive(cells, env)
    assert rc == 1, (out, err)
    assert "REGATE COMPLETED 1 of 3 CELLS" in out, out
    assert "!! r64_k8 l20 43 setup_failed" in out, out
    assert "!! r64_k8 l20 44 gate_failed" in out, out


def test_an_nvidia_smi_failure_is_reported_not_waited_out_in_silence(tmp_path):
    """The regate copy of the claim loop swallowed this with 2>/dev/null and slept forever. It now
    uses _common.sh's loop, which says why it is waiting."""
    env, cells = harness(tmp_path, [("r64_k8", "l20", 42, "ok")], STUB_SMI_RC="9", SLOTS_PER_GPU="1")
    rc, out, err = drive(cells, env, timeout=20)
    assert rc is None                                   # still waiting for a card ...
    assert "nvidia-smi FAILED for gpu 0" in err, err     # ... but saying why
    assert "g0:nvidia-smi-error" in err, err
