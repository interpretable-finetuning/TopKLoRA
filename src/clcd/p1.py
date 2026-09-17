"""P1 harness: which search, under our verification (plan revision 11, docs/NORTH_STAR.md P1).

Five subcommands, one run directory each, every one gated on the freeze commit:

  render   <run_dir> --freeze_sha <sha>   write <run_dir>/manifests/<chain>.txt from FROZEN's job table
  check    <run_dir> --freeze_sha <sha>   the manifests on disk equal a fresh render, byte for byte
  gates    <run_dir> --freeze_sha <sha>   provenance, completion and the known-answer gates; verdicts only
  stage_c  <run_dir> --freeze_sha <sha>   audits and random-draw controls for every certified band-A circuit
  readout  <run_dir> --freeze_sha <sha>   the numbers, in the fixed order; the only subcommand that prints values

The manifest line of a job is a pure function of FROZEN, the directory name and the freeze SHA, so the
job that ran is exactly the job that was pre-registered. Nothing here launches a job or loads a model,
and nothing here imports torch or nnsight.

Exit codes for gates / stage_c / readout: 0 when the run is ok, 1 when it is void (or a subcommand
refuses), 2 on a tool error (anything that is not a verdict: a misnamed queue log, a missing run-level
record, unreadable JSON). The traceback is printed, never swallowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import traceback
from collections import Counter, namedtuple
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scipy.stats import beta as _beta, binom as _binom

# The freeze commit: None until the freeze exists on the docs branch. Every subcommand refuses while
# it is None, and refuses a --freeze_sha that differs from it.
FREEZE_SHA = "605d851ad3327d8d6be1767ed5a5d96925341f1f"

# The registered design. json.loads(json.dumps(FROZEN, sort_keys=True)) equals the FROZEN block the
# freeze commit carries in docs/idea_queue.md (check_freeze proves it before every subcommand).
FROZEN = {
    "freeze": {
        "pilot_code_commit": "8ec50f0c9bf99fdc63b5ccdf279e5ecf79f8bd4f",
        "sfc_site": "/scratch/network/ssd/marek/minimalsleepers/.claude/worktrees/sfc-search/.sfc-site",
        "base_model": "google/gemma-2-2b",
        "base_fingerprint": {
            "snapshot": "c5ebcd40d208330abc697524c919956e692655cf",
            "blobs": {
                "model-00001-of-00003.safetensors": "1425aa066ec77e3eb79aac14a5bdea3ebcec46aa5c96cd40608c5c1fd70d193d",
                "model-00002-of-00003.safetensors": "96c111d3dcdbde9271595e463b5d9f7fc4810ad8b79e736309c0a1833e6c0d35",
                "model-00003-of-00003.safetensors": "4e08abc64d1767fdacd2c94da7f2ec4b8c65b25b19a53e87d19dc432901b5f02",
            },
        },
    },
    "certificate": {
        "data": "data/sleeper/prepared_eval6k",
        "dtype": "bfloat16",
        "offset": 100,
        "n_backdoor": 1000,
        "suff_n_se": 2.0,
        "sat_floor": 0.9,
        "nec_target": 0.0,
        "batch_size": 64,
        "grid": [10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200,
                 1600, 2000, 2400, 3200, 4032],
    },
    "attribution": {
        "n_attrib": 64,
        "band": {"A": 0, "B": 2000},
        "sfc_steps": 10,
        "s1": {"K_ig": 128, "attr_target": "margin", "attr_baseline": "control", "tag_baseline": "head"},
        "s2": {"elim_pool": "all", "cheap_offset": 1100, "n_cheap": 1000, "adaptive_n": True,
               "adaptive_rungs": [100, 300, 1000]},
    },
    "audit": {"data": "data/sleeper/prepared_eval41k", "bands": [6000], "n": 35000, "split": "eval_triggered"},
    "known_answers": {
        "g2": {"file": "clcd_results/rigorous/l1523_seed46_circuit.json",
               "fires": [6172, 11947, 19114, 19834, 29676, 31331, 38529]},
        "g2b": {"file": "clcd_results/rigorous/elim2/l1523_seed44_nc1000_adaptive_circuit.json",
                "data": "data/sleeper/prepared_eval6k", "bands": [2000], "n": 1000,
                "present": [2194], "absent": [2261, 2555]},
        "g4": {"data": "data/sleeper/prepared_eval6k", "bands": [100], "n": 1000},
        "g5": {"K": 504, "intact_min": 0.9, "ablate_min": 0.5},
    },
    "controls": {"R": 5, "c3_max_K": 504, "c4_max_K": 2016, "planted_n": 504, "r": 64},
    "readout": {
        "natural_bign_counts": [2, 2, 2, 4, 7, 7, 11, 12, 21, 27],
        "mcnemar_alpha": 0.05,
        "cp_level": 0.95,
        "mds_power": 0.8,
        "chance_share": 0.125,
    },
    "directories": {
        "s42": {
            "seed": 42,
            "run_level": ["g2", "g2b"],
            "models": {
                "route_l1523_s42": {
                    "kind": "route",
                    "adapter": "models/exp6/route_l1523_s42/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk",
                    "planted": "clcd_results/exp6/planted/route_s42_planted.json",
                    "g1": {"order_file": "clcd_results/sfc/recert/route_l1523_s42_clcd_order.json",
                           "order_key": "order_abs", "K": 50},
                    "arms": ["S1", "L", "V", "S2"],
                    "bands": ["A", "B"],
                },
                "a0_l1523_s42": {
                    "kind": "twin",
                    "adapter": "models/exp6/a0_l1523_s42/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk",
                    "planted": "clcd_results/exp6/planted/route_s42_planted.json",
                    "arms": ["S1", "L", "V"],
                    "bands": ["A"],
                },
            },
        },
        "s43": {
            "seed": 43,
            "run_level": [],
            "models": {
                "route_l1523_s43": {
                    "kind": "route",
                    "adapter": "models/exp6/route_l1523_s43/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk",
                    "planted": "clcd_results/exp6/planted/route_s43_planted.json",
                    "g1": {"order_file": "clcd_results/sfc/recert/route_l1523_s43_clcd_order.json",
                           "order_key": "order_abs", "K": 50},
                    "arms": ["S1", "L", "V"],
                    "bands": ["A", "B"],
                },
                "a0_l1523_s43": {
                    "kind": "twin",
                    "adapter": "models/exp6/a0_l1523_s43/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk",
                    "planted": "clcd_results/exp6/planted/route_s43_planted.json",
                    "arms": ["S1", "L", "V"],
                    "bands": ["A"],
                },
            },
        },
        "s44": {
            "seed": 44,
            "run_level": [],
            "models": {
                "route_l1523_s44": {
                    "kind": "route",
                    "adapter": "models/exp6/route_l1523_s44/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk",
                    "planted": "clcd_results/exp6/planted/route_s44_planted.json",
                    "g1": {"order_file": "clcd_results/sfc/recert/route_l1523_s44_clcd_order.json",
                           "order_key": "order_abs", "K": 50},
                    "arms": ["S1", "L", "V"],
                    "bands": ["A", "B"],
                },
                "a0_l1523_s44": {
                    "kind": "twin",
                    "adapter": "models/exp6/a0_l1523_s44/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk",
                    "planted": "clcd_results/exp6/planted/route_s44_planted.json",
                    "arms": ["S1", "L", "V"],
                    "bands": ["A"],
                },
            },
        },
        "sp60_s43": {
            "seed": 43,
            "run_level": [],
            "models": {
                "route_sp60_l1523_s43": {
                    "kind": "hard",
                    "adapter": "models/exp6/route_sp60_l1523_s43/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk",
                    "g1": {"order_file": "clcd_results/sfc/recert/route_sp60_l1523_s43_clcd_order.json",
                           "order_key": "order_abs", "K": 600},
                    "arms": ["S1", "L", "V", "S2"],
                    "bands": ["A", "B"],
                },
            },
        },
    },
}

FROZEN_BEGIN, FROZEN_END = "<!-- P1 FROZEN BEGIN -->", "<!-- P1 FROZEN END -->"
PY = ".venv/bin/python -u"
AUDIT_TOOL = "analysis/verify_holdout_necessity.py"
CHAIN_PREFIX = {"route": "route", "twin": "a0", "hard": "route"}
# c1 is descriptive, listed with the gates because it runs in the same chain (plan 4D).
GATES_BY_KIND = {"route": ("g1", "g4", "c1"), "twin": ("g5",), "hard": ("g1",)}
CONSTRUCTION = {"L": "latents", "V": "vanilla"}
# A mismatch in one of these on any output voids the run; any other mismatch voids that output only.
VOID_FIELDS = ("git_commit", "base_fingerprint", "provenance")

# expect: the argument values the output's `args` (or the audit record) must carry, built from the
# same values as `command`, so a job's output is checked against what was rendered, not re-parsed.
Job = namedtuple("Job", "basename chain after out kind command expect")


# --- freeze ---------------------------------------------------------------------------------------


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _head():
    return _git("rev-parse", "HEAD").strip()


def check_freeze(freeze_sha):
    """freeze_sha is FREEZE_SHA, resolves to a local commit, and that commit's docs/idea_queue.md carries
    a FROZEN block equal to FROZEN. Any mismatch raises."""
    if FREEZE_SHA is None:
        raise RuntimeError("FREEZE_SHA is None: the freeze commit does not exist yet")
    if freeze_sha != FREEZE_SHA:
        raise ValueError(f"--freeze_sha {freeze_sha} is not the freeze commit {FREEZE_SHA}")
    _git("rev-parse", "--verify", f"{freeze_sha}^{{commit}}")
    lines = _git("show", f"{freeze_sha}:docs/idea_queue.md").splitlines()
    if FROZEN_BEGIN not in lines or FROZEN_END not in lines:
        raise ValueError(f"docs/idea_queue.md at {freeze_sha} has no FROZEN block")
    block = "\n".join(lines[lines.index(FROZEN_BEGIN) + 1:lines.index(FROZEN_END)])
    if json.loads(block) != FROZEN:
        raise ValueError(f"the FROZEN block at {freeze_sha} differs from p1.FROZEN")


# --- job table ------------------------------------------------------------------------------------


def directory_entry(dir_name):
    """FROZEN's entry for a run directory; <name>_restart<N> uses <name>'s."""
    base = re.sub(r"_restart\d+$", "", dir_name)
    if base not in FROZEN["directories"]:
        raise KeyError(f"{dir_name!r} is not a P1 run directory")
    return FROZEN["directories"][base]


def _join(values):
    return " ".join(str(v) for v in values)


def _cert():
    c = FROZEN["certificate"]
    flags = (f"--data {c['data']} --dtype {c['dtype']} --offset {c['offset']} --n_backdoor {c['n_backdoor']} "
             f"--suff_n_se {c['suff_n_se']:.1f} --sat_floor {c['sat_floor']:.2f} --nec_target {c['nec_target']:.1f} "
             f"--batch_size {c['batch_size']}")
    keys = ("data", "dtype", "offset", "n_backdoor", "suff_n_se", "sat_floor", "nec_target", "batch_size")
    return flags, {k: c[k] for k in keys}


def _s1():
    s1 = FROZEN["attribution"]["s1"]
    return (f"--K_ig {s1['K_ig']} --attr_target {s1['attr_target']} --attr_baseline {s1['attr_baseline']} "
            f"--tag_baseline {s1['tag_baseline']}"), dict(s1)


def _cmd_sfc(adapter, construction, offset, sha, out):
    a, c = FROZEN["attribution"], FROZEN["certificate"]
    cmd = (f"PYTHONPATH={FROZEN['freeze']['sfc_site']} {PY} -m src.clcd.sfc_search --construction {construction} "
           f"--adapter {adapter} --data {c['data']} --dtype {c['dtype']} --n_attrib {a['n_attrib']} "
           f"--offset {offset} --steps {a['sfc_steps']} --provenance {sha} --out {out}")
    expect = {"construction": construction, "adapter": adapter, "data": c["data"], "dtype": c["dtype"],
              "n_attrib": a["n_attrib"], "offset": offset, "steps": a["sfc_steps"], "out": out}
    return cmd, expect


def _cmd_attrib(adapter, offset, sha, out):
    cert, cx = _cert()
    s1, s1x = _s1()
    a, grid = FROZEN["attribution"], FROZEN["certificate"]["grid"]
    cmd = (f"{PY} -m src.clcd.exp_circuit_search --adapter {adapter} {cert} --Ks {_join(grid)} --attrib_only "
           f"--n_attrib {a['n_attrib']} --attrib_offset {offset} {s1} --provenance {sha} --out {out}")
    expect = {"adapter": adapter, **cx, "Ks": list(grid), "attrib_only": True, "n_attrib": a["n_attrib"],
              "attrib_offset": offset, **s1x, "out": out}
    return cmd, expect


def _cmd_sweep(adapter, Ks, order_file, order_key, sha, out):
    cert, cx = _cert()
    cmd = (f"{PY} -m src.clcd.exp_circuit_search --adapter {adapter} {cert} --Ks {_join(Ks)} --ordering file "
           f"--order_file {order_file} --order_key {order_key} --provenance {sha} --out {out}")
    expect = {"adapter": adapter, **cx, "Ks": list(Ks), "ordering": "file", "order_file": order_file,
              "order_key": order_key, "out": out}
    return cmd, expect


def _cmd_elim(adapter, sha, out):
    cert, cx = _cert()
    s1, s1x = _s1()
    a, grid = FROZEN["attribution"], FROZEN["certificate"]["grid"]
    s2, off = a["s2"], a["band"]["A"]
    cmd = (f"{PY} -m src.clcd.exp_circuit_search --adapter {adapter} {cert} --Ks {_join(grid)} "
           f"--n_attrib {a['n_attrib']} --attrib_offset {off} {s1} --ordering eliminate "
           f"--elim_pool {s2['elim_pool']} --cheap_offset {s2['cheap_offset']} --n_cheap {s2['n_cheap']}"
           f"{' --adaptive_n' if s2['adaptive_n'] else ''} --provenance {sha} --out {out}")
    expect = {"adapter": adapter, **cx, "Ks": list(grid), "n_attrib": a["n_attrib"], "attrib_offset": off, **s1x,
              "ordering": "eliminate", "elim_pool": s2["elim_pool"], "cheap_offset": s2["cheap_offset"],
              "n_cheap": s2["n_cheap"], "adaptive_n": s2["adaptive_n"], "adaptive_rungs": list(s2["adaptive_rungs"]),
              "out": out}
    return cmd, expect


def _cmd_audit(data, bands, n, circuit, sha, out):
    split = FROZEN["audit"]["split"]
    cmd = (f"CLCD_DATA={data} CLCD_BANDS={','.join(str(b) for b in bands)} CLCD_N={n} CLCD_SPLIT={split} "
           f"CLCD_INTACT=0 CLCD_SAVE_GENS=0 CLCD_PROVENANCE={sha} CLCD_OUT={out} {PY} {AUDIT_TOOL} {circuit}")
    expect = {"data": data, "bands": list(bands), "n": n, "split": split, "file": circuit,
              "total_prompts": n * len(bands)}
    return cmd, expect


def _sweep_name(model, arm, seed, band):
    return f"{model}_S2_s{seed}_elim" if arm == "S2" else f"{model}_{arm}_s{seed}_sweep{band}"


def _attrib_name(model, arm, seed, band):
    return f"{model}_S1_s{seed}_attrib{band}" if arm == "S1" else f"{model}_{arm}_s{seed}_sfc{band}"


def jobs(run_dir, sha):
    """Every Stage A/B job of a run directory, in chain order, as a pure function of FROZEN, the directory
    name and the freeze SHA. Out paths are <run_dir>/<basename>.json."""
    run_dir = str(run_dir)
    entry = directory_entry(Path(run_dir).name)
    seed, ka, au, band = entry["seed"], FROZEN["known_answers"], FROZEN["audit"], FROZEN["attribution"]["band"]
    grid = FROZEN["certificate"]["grid"]
    jl = []

    def out(b):
        return f"{run_dir}/{b}.json"

    def add(b, chain, kind, built, after=None):
        cmd, expect = built
        jl.append(Job(b, chain, after, out(b), kind, cmd, expect))

    run_level_pending = list(entry["run_level"])
    for model, m in entry["models"].items():
        kind, adapter, tag = m["kind"], m["adapter"], f"{model}_s{seed}"
        pre = CHAIN_PREFIX[kind]
        gates_of = GATES_BY_KIND[kind]
        if "g1" in gates_of:
            g1, b = m["g1"], f"{tag}_g1"
            add(b, f"{pre}_L", "sweep", _cmd_sweep(adapter, [g1["K"]], g1["order_file"], g1["order_key"], sha, out(b)))
        if "g4" in gates_of:
            g4, b = ka["g4"], f"{tag}_g4"
            add(b, f"{pre}_L", "audit", _cmd_audit(g4["data"], g4["bands"], g4["n"], m["planted"], sha, out(b)))
        if kind != "twin":
            for rl in run_level_pending:  # the run-level known answers ride in the route model's L chain
                if rl == "g2":
                    built = _cmd_audit(au["data"], au["bands"], au["n"], ka["g2"]["file"], sha, out(rl))
                elif rl == "g2b":
                    g2b = ka["g2b"]
                    built = _cmd_audit(g2b["data"], g2b["bands"], g2b["n"], g2b["file"], sha, out(rl))
                else:
                    raise KeyError(f"unknown run-level job {rl!r}")
                add(rl, f"{pre}_L", "audit", built)
            run_level_pending = []
        if "g5" in gates_of:
            b = f"{tag}_g5"
            add(b, f"{pre}_L", "sweep", _cmd_sweep(adapter, [ka["g5"]["K"]], m["planted"], "kept_latents", sha, out(b)))
        if "c1" in gates_of:
            b = f"{tag}_c1"
            add(b, f"{pre}_V", "sweep",
                _cmd_sweep(adapter, [FROZEN["controls"]["planted_n"]], m["planted"], "kept_latents", sha, out(b)))
        for arm in m["arms"]:
            chain = f"{pre}_{arm}"
            if arm == "S2":
                b = _sweep_name(model, arm, seed, None)
                add(b, chain, "elim", _cmd_elim(adapter, sha, out(b)))
                continue
            for bnd in m["bands"]:
                ab, sb = _attrib_name(model, arm, seed, bnd), _sweep_name(model, arm, seed, bnd)
                if arm == "S1":
                    add(ab, chain, "attrib", _cmd_attrib(adapter, band[bnd], sha, out(ab)))
                    key = "order_pos"
                else:
                    add(ab, chain, "sfc", _cmd_sfc(adapter, CONSTRUCTION[arm], band[bnd], sha, out(ab)))
                    key = "order_abs"
                add(sb, chain, "sweep", _cmd_sweep(adapter, grid, out(ab), key, sha, out(sb)), after=ab)
    return jl


def _manifests(jl):
    """{chain: manifest text} in the queue's grammar: <out> [after=<path>] -- <command>."""
    outs = {j.basename: j.out for j in jl}
    chains = {}
    for j in jl:
        after = f" after={outs[j.after]}" if j.after else ""
        chains.setdefault(j.chain, []).append(f"{j.out}{after} -- {j.command}\n")
    return {chain: "".join(lines) for chain, lines in chains.items()}


def render(run_dir, sha):
    mdir = Path(run_dir) / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    texts = _manifests(jobs(run_dir, sha))
    for chain, text in texts.items():
        (mdir / f"{chain}.txt").write_text(text)
    return sorted(texts)


def check_manifests(run_dir, sha):
    """Every rendered manifest exists on disk byte-for-byte equal to a fresh render, and nothing else is
    in manifests/. Raises naming the first differing file."""
    mdir = Path(run_dir) / "manifests"
    texts = _manifests(jobs(run_dir, sha))
    for chain, text in texts.items():
        p = mdir / f"{chain}.txt"
        if not p.exists():
            raise FileNotFoundError(f"manifest missing: {p}")
        if p.read_bytes() != text.encode():
            raise ValueError(f"manifest differs from a fresh render: {p}")
    extra = sorted(p.name for p in mdir.iterdir() if p.name not in {f"{c}.txt" for c in texts})
    if extra:
        raise ValueError(f"unexpected file in {mdir}: {extra[0]}")


# --- queue logs and completion --------------------------------------------------------------------

_LOG_NAME = re.compile(r"^(?P<chain>.+?)(?:\.relaunch(?P<n>\d+))?\.log$")
_STATUS = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} g\S*\] (?P<rest>.*)$")
_FINISHED = re.compile(r"^queue \S+ finished: run=\d+ skipped=\d+ failed=(\d+)$")
_EVENTS = (
    (re.compile(r"^RUN (\S+)$"), "run"),
    (re.compile(r"^(\S+) done$"), "done"),
    (re.compile(r"^(\S+) FAILED rc=\S+ -- see .*$"), "failed"),
    (re.compile(r"^(\S+) exists, skip$"), "skip"),
    (re.compile(r"^(\S+) waiting for .*$"), None),
)


def _chain_logs(queue_dir, chains):
    """{chain: [log paths in launch order]}; a file that is not <chain>.log or <chain>.relaunch<N>.log of a
    rendered chain is a tool error."""
    found = {chain: [] for chain in chains}
    if queue_dir.is_dir():
        for p in sorted(queue_dir.iterdir()):
            m = _LOG_NAME.match(p.name)
            if not m or m["chain"] not in found:
                raise ValueError(f"{queue_dir}: {p.name!r} is not <chain>.log or <chain>.relaunch<N>.log "
                                 f"of a rendered chain")
            found[m["chain"]].append((int(m["n"] or 0), p))
    return {chain: [p for _, p in sorted(v)] for chain, v in found.items()}


def _parse_log(path):
    """[(event, output name)] and the failed count of the closing `finished` line (None when the log does not
    end with one). Any line outside gpu_queue.sh's grammar is a tool error."""
    events, finished = [], None
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        m = _STATUS.match(line)
        if not m:
            raise ValueError(f"{path}:{i + 1}: not a queue status line")
        rest = m["rest"]
        f = _FINISHED.match(rest)
        if f:
            if i != len(lines) - 1:
                raise ValueError(f"{path}:{i + 1}: a finished line before the end of the log")
            finished = int(f[1])
            continue
        for rx, event in _EVENTS:
            e = rx.match(rest)
            if e:
                if event:
                    events.append((event, e[1]))
                break
        else:
            raise ValueError(f"{path}:{i + 1}: unrecognised queue line")
    return events, finished


def completion(queue_dir, jl):
    """{basename: (complete, failed attempts)} by the crash rule: an output is complete when its chain's
    latest log ends with a `finished` line, its own latest event is `done` or `exists, skip`, it has fewer
    than three failed attempts (a FAILED line, or a RUN with no outcome) and the file exists."""
    by_chain = {}
    for j in jl:
        by_chain.setdefault(j.chain, []).append(j)
    logs = _chain_logs(Path(queue_dir), sorted(by_chain))
    result = {}
    for chain, cj in by_chain.items():
        names = {f"{j.basename}.json" for j in cj}
        last, failures, chain_finished = {}, Counter(), False
        for path in logs[chain]:
            events, finished = _parse_log(path)
            running = None
            for event, name in events:
                if name not in names:
                    raise ValueError(f"{path}: {name!r} is not an output of chain {chain}")
                if event == "run":
                    if running is not None:  # a RUN with no outcome is a failed attempt
                        failures[running] += 1
                        last[running] = "failed"
                    running = name
                    continue
                if event in ("done", "failed"):
                    if running != name:
                        raise ValueError(f"{path}: {name} {event} without its RUN")
                    running = None
                if event == "failed":
                    failures[name] += 1
                last[name] = event
            if running is not None:
                failures[running] += 1
                last[running] = "failed"
            chain_finished = finished is not None
        for j in cj:
            name = f"{j.basename}.json"
            ok = (chain_finished and last.get(name) in ("done", "skip") and failures[name] < 3
                  and Path(j.out).exists())
            result[j.basename] = (ok, failures[name])
    return result


# --- per-output checks (G3) -----------------------------------------------------------------------


def _load(path):
    return json.loads(Path(path).read_text())


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")


def _record(out, kind):
    """The dict carrying the provenance fields: the audit tool's single record, or the output itself.
    None for an audit output that does not hold exactly one record."""
    if kind == "audit":
        return out[0] if isinstance(out, list) and len(out) == 1 and isinstance(out[0], dict) else None
    return out if isinstance(out, dict) else None


def _reference(records, run_commit=None):
    """The run's commit and src root: --run_commit when given, else the most common value across outputs."""
    commits = Counter(r["git_commit"] for r in records if "git_commit" in r)
    roots = Counter(r["src_root"] for r in records if "src_root" in r)
    return {"git_commit": run_commit or (commits.most_common(1)[0][0] if commits else None),
            "src_root": roots.most_common(1)[0][0] if roots else None}


def _check_output(job, out, sha, ref):
    """'ok' or 'mismatch:<field>' for one existing output against its rendered job. `ref` (commit, src root)
    is None for a record imported from another run directory."""
    rec = _record(out, job.kind)
    if rec is None:
        return "mismatch:records"
    want = {"provenance": sha, "git_dirty": False, "base_fingerprint": FROZEN["freeze"]["base_fingerprint"]}
    if ref is not None:
        want.update(ref)
    for field, value in want.items():
        if field not in rec or rec[field] != value:
            return f"mismatch:{field}"
    if job.kind == "audit":
        args = rec
    else:
        if "args" not in rec or not isinstance(rec["args"], dict):
            return "mismatch:args"
        args = rec["args"]
    for key, value in job.expect.items():
        if key not in args or args[key] != value:
            return f"mismatch:{key}"
    return "ok"


# --- gates ----------------------------------------------------------------------------------------


def _fires(rec):
    return sorted(i for band in rec["fire_indices"].values() for i in band)


def _rule_g1(K):
    def rule(out):
        if out["status"] == "unsaturated" or not out["curve"]:
            return "unsaturated or empty curve"
        if out["status"] != "ok":
            return "status not ok"
        if out["both_K"] != K:
            return "both_K differs from the recorded K"
        return None
    return rule


def _rule_g2(rec):
    return None if _fires(rec) == sorted(FROZEN["known_answers"]["g2"]["fires"]) else \
        "fire indices differ from the frozen set"


def _rule_g2b(rec):
    fires, ka = set(_fires(rec)), FROZEN["known_answers"]["g2b"]
    if not set(ka["present"]) <= fires:
        return "an expected fire is missing"
    if set(ka["absent"]) & fires:
        return "a forbidden fire is present"
    return None


def _rule_g4(rec):
    return None if rec["total_fires"] == 0 else "fires on the planted set"


def _rule_g5(out):
    g5 = FROZEN["known_answers"]["g5"]
    if out["status"] == "unsaturated" or not out["curve"]:
        return "unsaturated or empty curve"
    if out["curve"][0]["K"] != g5["K"]:
        return "curve does not start at the planted K"
    if out["intact_asr"] < g5["intact_min"]:
        return "intact_asr below intact_min"
    if out["curve"][0]["ablate"] < g5["ablate_min"]:
        return "ablate below ablate_min"
    return None


def _run_level_dir(run_dir):
    """Where the run-level records live for a directory that does not run them: the s42 sibling, or
    s42_restart1 when it exists and s42 holds no stage_c_expected.json."""
    rd = Path(run_dir)
    s42, s42r = rd.parent / "s42", rd.parent / "s42_restart1"
    return s42r if s42r.exists() and not (s42 / "stage_c_expected.json").exists() else s42


def gates(run_dir, sha, run_commit=None):
    """Plan 4D. Writes <run_dir>/gates.json and returns it: verdicts and scopes, never a value."""
    rd = Path(run_dir)
    entry = directory_entry(rd.name)
    seed = entry["seed"]
    jl = jobs(str(rd), sha)
    done = completion(rd / "queues", jl)
    loaded = {j.basename: _load(j.out) for j in jl if Path(j.out).exists()}
    records = [r for j in jl if j.basename in loaded for r in [_record(loaded[j.basename], j.kind)] if r is not None]
    ref = _reference(records, run_commit)
    outputs = {}
    for j in jl:
        if j.basename not in loaded:
            outputs[j.basename] = "incomplete"
            continue
        status = _check_output(j, loaded[j.basename], sha, ref)
        outputs[j.basename] = status if status != "ok" or done[j.basename][0] else "incomplete"

    verdicts, void, audits_na, models = {}, [], False, {}

    def verdict(label, b, status, out, rule):
        """Records '<label> <basename>' -> verdict and a detail that names the failing criterion, never a value."""
        why = f"output {status}" if status != "ok" else rule(out)
        verdicts[f"{label} {b}"] = {"verdict": "fail" if why else "pass", "detail": why or "pass"}
        return not why

    kinds = {j.basename: j.kind for j in jl}
    for model, m in entry["models"].items():
        tag, models[model] = f"{model}_s{seed}", "ok"
        rules = {"g1": _rule_g1(m["g1"]["K"]) if "g1" in m else None, "g4": _rule_g4, "g5": _rule_g5}
        for gate in GATES_BY_KIND[m["kind"]]:
            if gate == "c1":
                continue
            b = f"{tag}_{gate}"
            out = _record(loaded[b], kinds[b]) if outputs[b] == "ok" else None
            passed = verdict(gate.upper(), b, outputs[b], out, rules[gate])
            if not passed:
                if gate == "g5":
                    models[model] = "void"
                else:
                    void.append(b)
    for rl, label, rule in (("g2", "G2", _rule_g2), ("g2b", "G2b", _rule_g2b)):
        if rl in entry["run_level"]:
            status, out = outputs[rl], _record(loaded[rl], "audit") if outputs[rl] == "ok" else None
        else:
            src = _run_level_dir(rd)
            job = next(j for j in jobs(str(src), sha) if j.basename == rl)
            if not Path(job.out).exists():
                raise FileNotFoundError(f"run-level record {job.out} is missing")
            out = _load(job.out)
            status = _check_output(job, out, sha, None)
            out = _record(out, "audit") if status == "ok" else None
        if not verdict(label, rl, status, out, rule):
            if rl == "g2":
                audits_na = True
            else:
                void.append(rl)
    void += [b for b, s in outputs.items() if s.startswith("mismatch:") and s[len("mismatch:"):] in VOID_FIELDS]
    if void:
        models = {model: "void" for model in models}
    result = {"run": "void" if void else "ok", "audits": "na" if audits_na else "ok", "models": models,
              "outputs": outputs, "gates": verdicts, "reference": ref, "git_commit": _head(),
              "time": datetime.now(timezone.utc).isoformat()}
    _write_json(rd / "gates.json", result)
    return result


def _void_gate(g, model):
    for key, v in g["gates"].items():
        label, b = key.split(" ", 1)
        if v["verdict"] == "fail" and b.startswith(f"{model}_s"):
            return label
    return "run"


# --- stage C --------------------------------------------------------------------------------------


def _rng(key):
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def _draws(seed, model, arm, kind, n, make):
    """n distinct draws {basename: draw}; a draw equal as a set to an earlier draw of the same model, arm
    and kind is redrawn with i + 100 * attempt. Pinned: the same inputs give the same draws."""
    out, seen = {}, []
    for i in range(n):
        for attempt in range(1000):
            key = f"{seed}|{model}|{arm}|{kind}|{i + 100 * attempt}"
            latents = make(_rng(key))
            as_set = frozenset(tuple(l) for l in latents)
            if as_set not in seen:
                break
        else:
            raise RuntimeError(f"no distinct {kind} draw {i} for {model} {arm} after 1000 attempts")
        seen.append(as_set)
        out[f"{model}_{arm}_s{seed}_{kind}_{i}"] = {"latents": sorted(latents), "seed_key": key, "kind": kind}
    return out


def stage_c_jobs(run_dir, sha, g):
    """(expected, jobs, draws) for Stage C from the gated band-A certificates (S2: its elim output). Pure in
    the outputs: re-deriving gives the same lists and the same draws."""
    run_dir = str(run_dir)
    entry = directory_entry(Path(run_dir).name)
    seed, ctl, au, grid = entry["seed"], FROZEN["controls"], FROZEN["audit"], FROZEN["certificate"]["grid"]
    expected, jl, draws, planted_audits = {}, [], {}, []
    for model, m in entry["models"].items():
        if g["models"][model] != "ok":
            continue
        kind, adapter, chain = m["kind"], m["adapter"], f"c_{model}"

        def mk(b, job_kind, built):
            cmd, expect = built
            return Job(b, chain, None, f"{run_dir}/{b}.json", job_kind, cmd, expect)

        def audit(circuit, b):
            return mk(b, "audit", _cmd_audit(au["data"], au["bands"], au["n"], circuit, sha, f"{run_dir}/{b}.json"))

        audits, c4s, c3s = [], [], []
        for arm in m["arms"]:
            sb = _sweep_name(model, arm, seed, "A")
            if g["outputs"][sb] != "ok":
                continue
            sweep = _load(f"{run_dir}/{sb}.json")
            if sweep["status"] != "ok":
                continue
            K = sweep["both_K"]
            top = len(_load(f"{run_dir}/{_attrib_name(model, arm, seed, 'A')}.json")["order_pos"]) \
                if arm == "S1" else grid[-1]
            if K >= top:
                continue
            circuit = [tuple(l) for l in sweep["kept_latents"]]
            b = f"{model}_{arm}_s{seed}_audit"
            audits.append(audit(f"{run_dir}/{sb}.json", b))
            items = [b]
            if K <= ctl["c4_max_K"]:
                counts = Counter(mod for mod, _ in circuit)
                n = min(ctl["R"], math.prod(math.comb(ctl["r"], c) for c in counts.values()))
                new = _draws(seed, model, arm, "c4", n, lambda rng: [
                    [mod, idx] for mod in sorted(counts) for idx in rng.sample(range(ctl["r"]), counts[mod])])
                for b, d in new.items():
                    draws[b] = d
                    items.append(b)
                    c4s.append(mk(b, "sweep", _cmd_sweep(adapter, [K], f"{run_dir}/draws/{b}.json", "latents", sha,
                                                         f"{run_dir}/{b}.json")))
            if kind == "route" and K <= ctl["c3_max_K"]:
                pool = sorted(tuple(l) for l in _load(m["planted"])["kept_latents"])
                if len(pool) != ctl["planted_n"]:
                    raise ValueError(f"{m['planted']} holds {len(pool)} latents, not the planted {ctl['planted_n']}")
                n = min(ctl["R"], math.comb(len(pool), K))
                new = _draws(seed, model, arm, "c3", n, lambda rng: [list(l) for l in rng.sample(pool, K)])
                for b, d in new.items():
                    draws[b] = d
                    items.append(b)
                    c3s.append(mk(b, "sweep", _cmd_sweep(adapter, [K], f"{run_dir}/draws/{b}.json", "latents", sha,
                                                         f"{run_dir}/{b}.json")))
            expected[f"{model}|{arm}"] = {"both_K": K, "items": items}
        planted = []
        if kind == "route":
            b = f"{model}_s{seed}_planted_audit"
            planted_audits.append(b)
            planted.append(audit(m["planted"], b))
        jl += audits + c4s + c3s + planted
    expected["planted_audit"] = planted_audits
    return expected, jl, draws


def _check_expected(rd, expected):
    """The existing stage_c_expected.json entries against a fresh derivation; raises on any difference."""
    path = Path(rd) / "stage_c_expected.json"
    existing = _load(path) if path.exists() else {}
    for key, val in existing.items():
        if key not in expected or expected[key] != val:
            raise ValueError(f"stage_c_expected.json entry {key!r} differs from a fresh derivation; "
                             f"existing entries are never altered")
    return existing


def stage_c(run_dir, sha):
    """Emit the Stage C draws, expected lists and chain manifests for every new (model, arm) certificate.
    Never launches. Returns (gates result, {new entries, chains written}) or (gates result, None) when the
    run is void."""
    rd = Path(run_dir)
    g = gates(rd, sha)
    if g["run"] != "ok":
        return g, None
    expected, jl, draws = stage_c_jobs(rd, sha, g)
    existing = _check_expected(rd, expected)
    new_keys = [k for k in expected if k not in existing]
    new_items = {b for k in new_keys for b in (expected[k] if k == "planted_audit" else expected[k]["items"])}
    (rd / "draws").mkdir(exist_ok=True)
    for b, d in draws.items():
        p = rd / "draws" / f"{b}.json"
        if p.exists():
            if _load(p) != d:
                raise ValueError(f"draw file {p} differs from its pinned derivation")
        elif b in new_items:
            _write_json(p, d)
        else:
            raise FileNotFoundError(f"draw file {p} of an existing Stage C entry is missing")
    chains = sorted({j.chain for j in jl if j.basename in new_items})
    (rd / "manifests_c").mkdir(exist_ok=True)
    for chain in chains:
        (rd / "manifests_c" / f"{chain}.txt").write_text(
            "".join(f"{j.out} -- {j.command}\n" for j in jl if j.chain == chain))
    merged = dict(existing)
    for k in new_keys:
        merged[k] = expected[k]
    _write_json(rd / "stage_c_expected.json", merged)
    return g, {"new": new_keys, "items": sorted(new_items), "chains": chains}


# --- statistics (pure) ----------------------------------------------------------------------------


def clopper_pearson_upper(k, n, level):
    """One-sided exact upper bound on a binomial proportion: the p with P(X <= k | p) = 1 - level."""
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    return 1.0 if k == n else float(_beta.ppf(level, k + 1, n - k))


def clopper_pearson_lower(k, n, level):
    """One-sided exact lower bound: the p with P(X >= k | p) = 1 - level."""
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    return 0.0 if k == 0 else float(_beta.ppf(1 - level, k, n - k + 1))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: binomial on the discordant counts under p = 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def natural_range_class(k):
    """Against the natural l1523 BIG-N counts: 'below' every natural circuit, 'within' their range, 'above'."""
    counts = FROZEN["readout"]["natural_bign_counts"]
    return "below" if k < min(counts) else "above" if k > max(counts) else "within"


def min_detectable_share(n, p0=0.125, alpha=0.05, power=0.8):
    """Smallest true share p > p0 at which the one-sided exact test (reject when P(X >= k | p0) < alpha,
    which is exactly 'the Clopper-Pearson lower bound exceeds p0') has the given power at n."""
    critical = [k for k in range(1, n + 1) if _binom.sf(k - 1, n, p0) < alpha]
    if not critical:
        raise ValueError(f"no rejection region at n={n}, alpha={alpha}")
    k_crit = critical[0]
    lo, hi = p0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom.sf(k_crit - 1, n, mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def size_rule(K_a_bands, K_b_bands, grid):
    """Plan 4E on two attribution bands: 'agree' when every band puts the two arms within one grid step,
    'disagree' when every band puts them two or more steps apart in the same direction, else 'unresolved'.
    An arm without a proper sub-circuit is passed as the grid top by the caller."""
    steps = [grid.index(kb) - grid.index(ka) for ka, kb in zip(K_a_bands, K_b_bands, strict=True)]
    if all(abs(s) <= 1 for s in steps):
        return "agree"
    if all(s >= 2 for s in steps) or all(s <= -2 for s in steps):
        return "disagree"
    return "unresolved"


def threshold_at(effects, K):
    """T_N at the cut K of the |effect| ranking: {'interval': [|e_(K+1)|, |e_K|) or None when not
    representable, 'n_zero_effect', 'tie_at_cut'}."""
    mags = sorted((abs(float(e)) for _, _, e in effects), reverse=True)
    if not 1 <= K < len(mags):
        raise ValueError(f"K={K} is not a proper cut of {len(mags)} effects")
    hi, lo = mags[K - 1], mags[K]
    return {"interval": (lo, hi) if hi > lo else None, "n_zero_effect": sum(m == 0.0 for m in mags),
            "tie_at_cut": hi == lo}


def error_node_counts(error_effects, interval):
    """How the S3-V error nodes sit against T_N: |err| >= |e_K|, <= |e_(K+1)|, and strictly between."""
    lo, hi = interval
    errs = [abs(float(v)) for v in error_effects.values()]
    return {"ge_eK": sum(e >= hi for e in errs), "le_eK1": sum(e <= lo for e in errs),
            "between": sum(lo < e < hi for e in errs)}


def _size_band(K, grid):
    """(largest failing grid K below both_K, both_K]."""
    return max([k for k in grid if k < K], default=0), K


# --- readout --------------------------------------------------------------------------------------


def _get(ctx, b):
    """The output when its status is ok (for an audit, its single record: G3 refused any other shape), else None."""
    if ctx.status[b] != "ok":
        return None
    out = _load(ctx.rd / f"{b}.json")
    return out[0] if isinstance(out, list) else out


def _na(ctx, b):
    s = ctx.status[b]
    if s != "incomplete":
        return s
    return "incomplete (three failures)" if ctx.failures[b] >= 3 else "incomplete (no output)"


def _s1_tie(attrib, K):
    scores = {(m, d): s for m, d, s in attrib["scores"]}
    ranked = [scores[tuple(l)] for l in attrib["order_pos"]]
    return K < len(ranked) and ranked[K - 1] == ranked[K]


def _pilot_fields(path):
    """The two fields the R row may read from a sealed pilot file."""
    d = _load(path)
    return d["order_abs"], d["effects"]


def _fire_vector(rec):
    return [v for band in rec["bands"] for v in rec["fire_vec_per_band"][str(band)]]


def _bands_of(m, arm):
    return ["S2"] if arm == "S2" else m["bands"]


def _sizes_str(certs):
    return ", ".join(c["label"] for c in certs)


def _readout_certificates(ctx, model, m):
    print("-- certificates --")
    seed, grid = ctx.seed, ctx.grid
    for arm in m["arms"]:
        for band in _bands_of(m, arm):
            sb = _sweep_name(model, arm, seed, band)
            c = {"na": True, "K": None, "rule_K": grid[-1], "set": None, "tie": None, "label": "N/A"}
            ctx.cert[(model, arm, band)] = c
            sweep = _get(ctx, sb)
            attrib = _get(ctx, _attrib_name(model, arm, seed, band)) if arm != "S2" else None
            if sweep is None or (arm != "S2" and attrib is None):
                b = sb if sweep is None else _attrib_name(model, arm, seed, band)
                print(f"{arm} band {band}: N/A ({_na(ctx, b)})")
                continue
            c["na"] = False
            K, st = sweep["both_K"], sweep["status"]
            n_pos = len(attrib["order_pos"]) if arm == "S1" else None
            top = n_pos if arm == "S1" else grid[-1]
            if st == "unsaturated":
                c["label"] = "unsaturated (grid top for the size rule)"
            elif st != "ok" or K >= top:
                c["label"] = "no proper sub-circuit" + (f" among {n_pos} positive supporters" if arm == "S1" else "") \
                    + " (grid top for the size rule)"
            else:
                lo, _ = _size_band(K, grid)
                c.update(K=K, rule_K=K, set=frozenset(tuple(l) for l in sweep["kept_latents"]),
                         label=f"both_K {f'<= {K} (grid floor)' if K == grid[0] else K}; size band ({lo}, {K}]")
            line = f"{arm} band {band}: {c['label']}"
            if arm == "S2":
                line += f"; elim survivors {sweep['elim']['n_survivors']}, cut {sweep['elim']['n_cut']}"
            print(line)
            if c["K"] is None or arm == "S2":  # T_N and ties are defined for the attribution arms only
                continue
            if arm == "S1":
                c["tie"] = _s1_tie(attrib, K)
                print(f"  tie block crosses the cut: {c['tie']}")
                continue
            t = threshold_at(attrib["effects"], K)
            c["tie"] = t["tie_at_cut"]
            tn = f"[{t['interval'][0]:.6g}, {t['interval'][1]:.6g})" if t["interval"] else "not representable"
            print(f"  T_N at {K}: {tn}; n_zero_effect {t['n_zero_effect']}; tie block crosses the cut: {c['tie']}")
            if arm == "V":
                if t["interval"] is None:
                    print("  error nodes: N/A (T_N not representable)")
                else:
                    e = error_node_counts(attrib["error_effects"], t["interval"])
                    print(f"  error nodes: |err| >= |e_K| {e['ge_eK']}, <= |e_(K+1)| {e['le_eK1']}, between {e['between']}")


def _readout_sizes(ctx, model, m):
    print("-- size comparison (4E) --")
    arms, seed, grid = m["arms"], ctx.seed, ctx.grid
    ro = FROZEN["readout"]
    alpha, level = ro["mcnemar_alpha"], ro["cp_level"]
    two_bands = len(m["bands"]) == 2
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    for a, b in pairs:
        ca = [ctx.cert[(model, a, band)] for band in _bands_of(m, a)]
        cb = [ctx.cert[(model, b, band)] for band in _bands_of(m, b)]
        desc = f"{a} [{_sizes_str(ca)}] vs {b} [{_sizes_str(cb)}]"
        if any(c["na"] for c in ca + cb):
            print(f"{a} vs {b}: N/A (an output is incomplete); {desc}")
        elif "S2" in (a, b) or not two_bands:
            print(f"{a} vs {b}: one sample; {desc}")
        else:
            rule = size_rule([c["rule_K"] for c in ca], [c["rule_K"] for c in cb], grid)
            print(f"{a} vs {b}: {rule}; {desc}")
    tests = 0
    for a, b in pairs:
        sa = ctx.cert[(model, a, _bands_of(m, a)[0])]["set"]
        sb = ctx.cert[(model, b, _bands_of(m, b)[0])]["set"]
        if sa is None or sb is None:
            print(f"leak {a} vs {b}: N/A (not both certified on band A)")
            continue
        if sa == sb:
            print(f"leak {a} vs {b}: one audit (identical sets)")
            continue
        if ctx.g["audits"] != "ok":
            print(f"leak {a} vs {b}: N/A (G2)")
            continue
        na, nb = f"{model}_{a}_s{seed}_audit", f"{model}_{b}_s{seed}_audit"
        ra, rb = _get(ctx, na), _get(ctx, nb)
        if ra is None or rb is None:
            print(f"leak {a} vs {b}: N/A ({_na(ctx, na if ra is None else nb)})")
            continue
        va, vb = _fire_vector(ra), _fire_vector(rb)
        if len(va) != len(vb):
            raise ValueError(f"{na} and {nb} cover different prompt counts")
        bb = sum(1 for x, y in zip(va, vb) if x and not y)
        cc = sum(1 for x, y in zip(va, vb) if y and not x)
        p = mcnemar_exact(bb, cc)
        tests += 1
        fa, fb = sum(va), sum(vb)
        if fa == 0 and fb == 0:
            label = f"both <= {clopper_pearson_upper(0, len(va), level):.3g} (one-sided {level:.0%})"
        elif p < alpha:
            label = "difference detected"
        else:
            label = f"no detectable difference at n={len(va):,}" + (f", p < {alpha} unreachable" if bb + cc < 6 else "")
        print(f"leak {a} vs {b}: fires {fa} vs {fb}, discordant b={bb} c={cc}, exact McNemar p={p:.4g}: {label}")
    print(f"{tests} leak test(s) at alpha={alpha}, uncorrected")


def _readout_descriptive(ctx, model, m):
    print("-- descriptive checks --")
    kind, seed, tag = m["kind"], ctx.seed, f"{model}_s{ctx.seed}"
    if kind in ("route", "hard"):
        pilot = Path(f"clcd_results/sfc/{model}_sfc.json")
        b = f"{model}_L_s{seed}_sfcA"
        sfc = _get(ctx, b)
        if not pilot.exists():
            print("R: N/A (pilot file missing)")
        elif sfc is None:
            print(f"R: N/A ({_na(ctx, b)})")
        else:
            order_abs, effects = _pilot_fields(pilot)
            pe = {(mm, d): e for mm, d, e in effects}
            delta = max(abs(e - pe[(mm, d)]) for mm, d, e in sfc["effects"])
            print(f"R: identity {sfc['order_abs'] == order_abs}; max |delta e| {delta:.6g}")
    if kind == "route":
        c1, g4 = _get(ctx, f"{tag}_c1"), _get(ctx, f"{tag}_g4")
        if c1 is None:
            print(f"C1: N/A ({_na(ctx, f'{tag}_c1')})")
        elif not c1["curve"]:
            print("C1: unsaturated")
        else:
            row = c1["curve"][0]
            sens = "" if round(row["ablate"] * c1["n_backdoor"]) == g4["total_fires"] else "; batching-sensitive"
            print(f"C1 planted at K={row['K']}: intact {c1['intact_asr']:.4f}, keep-only {row['keep_only']:.4f}, "
                  f"ablate {row['ablate']:.4f}{sens}")
    for arm in m["arms"]:
        key = f"{model}|{arm}"
        if key not in ctx.expected:
            continue
        items = ctx.expected[key]["items"]
        for kind_c in ("c3", "c4"):
            names = [b for b in items if f"_{kind_c}_" in b]
            if not names:
                continue
            rows, red, missing = [], False, 0
            for b in names:
                out = _get(ctx, b)
                if out is None:
                    rows.append(f"{b}: N/A ({_na(ctx, b)})")
                    missing += 1
                elif kind_c == "c3":
                    ab = out["curve"][0]["ablate"] if out["curve"] else None
                    red |= ab == 0.0
                    rows.append(f"{b}: ablate {ab if ab is None else f'{ab:.4f}'}")
                else:
                    red |= out["status"] == "ok"
                    rows.append(f"{b}: status {out['status']}, both_K {out['both_K']}")
            print(f"{kind_c.upper()} {arm}: {'RED' if red else 'not red'}" + (f" ({missing} draw(s) N/A)" if missing else ""))
            for r in rows:
                print(f"  {r}")
    if kind == "twin":
        ro, r = FROZEN["readout"], FROZEN["controls"]["r"]
        chance, level, power = ro["chance_share"], ro["cp_level"], ro["mds_power"]
        low = round(chance * r)
        route = next((name for name, mm in ctx.entry["models"].items() if mm["kind"] == "route"), None)
        for arm in m["arms"]:
            ct = ctx.cert[(model, arm, "A")]
            rc = [ctx.cert[(route, arm, band)] for band in ctx.entry["models"][route]["bands"]] if route else []
            if ct["na"] or not rc or any(c["na"] for c in rc):
                print(f"C5 {arm}: N/A (an output is incomplete)")
            else:
                smaller = ct["rule_K"] < min(c["rule_K"] for c in rc)
                print(f"C5 {arm}: twin [{_sizes_str([ct])}] vs route [{_sizes_str(rc)}]: "
                      f"{'twin smaller than route' if smaller else 'twin not smaller than route'}")
            if ct["set"] is None:
                print(f"  [0:{low}) share: N/A (no certified set)")
                continue
            n = len(ct["set"])
            k = sum(1 for _, d in ct["set"] if d < low)
            lower = clopper_pearson_lower(k, n, level)
            flag = lower > chance and ct["tie"] is False
            mds = min_detectable_share(n, chance, 1 - level, power)
            print(f"  [0:{low}) share {k}/{n} = {k / n:.4f}; lower bound (one-sided {level:.0%}) {lower:.4f}; "
                  f"tie at cut {ct['tie']}: {f'above-chance [0:{low}) share' if flag else f'[0:{low}) share at chance'}; "
                  f"minimum detectable share at power {power}: {mds:.4f}")


def _readout_audits(ctx, model, m):
    print("-- audits (4G) --")
    seed, level = ctx.seed, FROZEN["readout"]["cp_level"]
    phrase = {"below": "below every natural circuit", "within": "within the natural range",
              "above": "above the natural range"}
    names = [f"{model}_{arm}_s{seed}_audit" for arm in m["arms"] if f"{model}|{arm}" in ctx.expected]
    names += [b for b in ctx.expected["planted_audit"] if b.startswith(f"{model}_s{seed}_")]
    for b in names:
        if ctx.g["audits"] != "ok":
            print(f"{b}: N/A (G2)")
            continue
        rec = _get(ctx, b)
        if rec is None:
            print(f"{b}: N/A ({_na(ctx, b)})")
            continue
        k, n = rec["total_fires"], rec["total_prompts"]
        print(f"{b}: fires {k}/{n:,}; upper bound (one-sided {level:.0%}) {clopper_pearson_upper(k, n, level):.3g}; "
              f"{phrase[natural_range_class(k)]}")


def _readout_secondary(ctx, model, m):
    print("-- secondary --")
    arms = m["arms"]
    sets = {arm: ctx.cert[(model, arm, _bands_of(m, arm)[0])]["set"] for arm in arms}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            if sets[a] is None or sets[b] is None:
                print(f"Jaccard {a} vs {b}: N/A (not both certified)")
            else:
                print(f"Jaccard {a} vs {b} (band A): {len(sets[a] & sets[b]) / len(sets[a] | sets[b]):.4f}")
    if m["kind"] == "route":
        planted = {tuple(l) for l in _load(m["planted"])["kept_latents"]}
        for (mm, arm, band), c in ctx.cert.items():
            if mm == model and c["set"] is not None:
                hit = len(c["set"] & planted)
                print(f"planted precision {arm} band {band}: {hit}/{len(c['set'])} = {hit / len(c['set']):.4f}")


def readout(run_dir, sha):
    """Plan 4E-4G, in order, per model. Returns the exit code."""
    print(f"p1 readout at commit {_head()}")
    rd = Path(run_dir)
    entry = directory_entry(rd.name)
    g = gates(rd, sha)
    if g["run"] != "ok":
        print("run: void; nothing is read out")
        return 1
    expected, jl_c, draws = stage_c_jobs(rd, sha, g)
    _check_expected(rd, expected)
    done_ab = completion(rd / "queues", jobs(str(rd), sha))
    done_c = completion(rd / "queues_c", jl_c)
    status = dict(g["outputs"])
    failures = {b: f for b, (_, f) in done_ab.items()}
    for j in jl_c:
        failures[j.basename] = done_c[j.basename][1]
        if not Path(j.out).exists():
            status[j.basename] = "incomplete"
            continue
        s = _check_output(j, _load(j.out), sha, g["reference"])
        if s == "ok" and j.basename in draws and _load(rd / "draws" / f"{j.basename}.json") != draws[j.basename]:
            raise ValueError(f"draw file {j.basename} differs from its pinned derivation")
        status[j.basename] = s if s != "ok" or done_c[j.basename][0] else "incomplete"
    void = [b for b, s in status.items() if s.startswith("mismatch:") and s[len("mismatch:"):] in VOID_FIELDS]
    if void:
        print(f"run: void ({void[0]}: {status[void[0]]})")
        return 1
    ctx = SimpleNamespace(rd=rd, entry=entry, seed=entry["seed"], grid=FROZEN["certificate"]["grid"], g=g,
                          status=status, failures=failures, expected=expected, cert={})
    for model, m in entry["models"].items():
        print(f"\n== {model} ({m['kind']}) ==")
        if g["models"][model] != "ok":
            print(f"void ({_void_gate(g, model)})")
            continue
        _readout_certificates(ctx, model, m)
        _readout_sizes(ctx, model, m)
        _readout_descriptive(ctx, model, m)
        _readout_audits(ctx, model, m)
        _readout_secondary(ctx, model, m)
    return 0


# --- entry point ----------------------------------------------------------------------------------


def _main_render(a):
    for chain in render(a.run_dir, a.freeze_sha):
        print(f"wrote {a.run_dir}/manifests/{chain}.txt")
    return 0


def _main_check(a):
    check_manifests(a.run_dir, a.freeze_sha)
    print("pass")
    return 0


# --- declared follow-ups: post-freeze additions run with the P1 job templates, not part of the pre-registration ---

FOLLOWUPS = {  # name: (provenance string every output must carry, adapter-family prefix, certificate grid)
    "canonical_l1523": ("canonical-l1523", "l1523", list(FROZEN["certificate"]["grid"])),
    "l19": ("l19-completion", "l19", [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150, 200, 250, 300, 350, 400, 448]),
    "all": ("all-completion", "all", list(FROZEN["certificate"]["grid"]) + [4800, 6400, 8000, 9600, 11648]),
}
FOLLOWUP_SEEDS = (42, 43, 44, 45, 46)
FOLLOWUP_GATES = Path("clcd_results/p1/s42/gates.json")       # the run-level G2 verdict (plan 4D: once per P1)
BIGN_RECORDS = Path("clcd_results/rigorous/holdout_necessity")  # the archived BIG-N audits of the natural adapters
ARCHIVED_ELIM = Path("clcd_results/rigorous/elim")              # the archived elimination circuits


def _followup_jobs(mdir):
    """The jobs of every manifest under mdir: chain = manifest stem, basename = output stem, plus the fields
    _check_output reads (kind, expect). A follow-up job's expectation is its adapter; an audit's is the
    circuit file and the prompt count."""
    audit = FROZEN["audit"]
    jl = []
    for mf in sorted(Path(mdir).glob("*.txt")):
        for line in mf.read_text().splitlines():
            if not line.strip():
                continue
            head, cmd = line.split(" -- ", 1)
            out = head.split()[0]
            if "verify_holdout_necessity" in cmd:
                kind, expect = "audit", {"file": cmd.split()[-1], "total_prompts": audit["n"] * len(audit["bands"])}
            else:
                kind, expect = "job", {"adapter": re.search(r"--adapter (\S+)", cmd)[1]}
            jl.append(SimpleNamespace(chain=mf.stem, basename=Path(out).stem, out=out, kind=kind, expect=expect))
    return jl


def _followup_chain_states(queue_dir, jl):
    """{chain: 'finished, failed=N' | 'not finished' | 'no log'} from the latest log of each chain."""
    states = {}
    for chain, paths in _chain_logs(Path(queue_dir), sorted({j.chain for j in jl})).items():
        if not paths:
            states[chain] = "no log"
            continue
        _, finished = _parse_log(paths[-1])
        states[chain] = "not finished" if finished is None else f"finished, failed={finished}"
    return states


def _archived_label(path):
    if "K700nec" in path:
        return "held-out-necessity K700"
    if "nc1000_adaptive" in path:
        return "elimination (n_cheap 1000, adaptive)"
    if "nc1000" in path:
        return "elimination (n_cheap 1000)"
    if "/elim/" in path:
        return "elimination (rigorous/elim)"
    return "prefix (archived)"


def followup(name, run_commit=None, root="clcd_results/p1_followup"):
    """Readout of a declared follow-up run (canonical l15-23, l19 completion, the "all" family), fixed before
    the runs: per adapter and arm, both_K on both attribution bands beside the archived circuits of that
    adapter (sizes and BIG-N in-turn fires), the 35,000-prompt audit of every band-A certified circuit with
    its one-sided bound, the C4 flag, and the re-certification of the archived elimination circuits where
    declared. The certificate, size-rule (4E), C4, audit and Jaccard printers are the P1 readout's own,
    called on the follow-up directory; nothing is re-implemented. Completion is the P1 crash rule per chain
    (`completion`): a chain that has not finished reads as N/A for every one of its outputs. Every output is
    checked for provenance, run commit, clean tree, base fingerprint and src root (`_check_output` against
    `_reference`: --run_commit when given, else the most common value across the outputs, as `gates` does);
    a mismatch reads as N/A with the field named. The audit known-answer verdict (G2) is read from seed 42's
    gates.json (verdicts only). Paths are relative to the working directory, as everywhere in this module."""
    prov, fam, grid = FOLLOWUPS[name]
    d = Path(root) / name
    jl_ab = _followup_jobs(d / "manifests")
    jl_c = _followup_jobs(d / "manifests_c") if (d / "manifests_c").is_dir() else []
    loaded = {j.basename: _load(j.out) for j in jl_ab + jl_c if Path(j.out).exists()}
    records = [r for o in loaded.values() for r in (o if isinstance(o, list) else [o])]
    ref = _reference(records, run_commit)
    print(f"follow-up readout {name} (provenance {prov}) at commit {_head()}; run commit {ref['git_commit']}")
    print(f"grid {grid}")
    print("-- chains --")
    for chain, state in _followup_chain_states(d / "queues", jl_ab).items():
        print(f"{chain}: {state}")
    if jl_c:
        for chain, state in _followup_chain_states(d / "queues_c", jl_c).items():
            print(f"stage C {chain}: {state}")
    else:
        print("stage C: no manifests_c yet")
    done = completion(d / "queues", jl_ab)
    done.update(completion(d / "queues_c", jl_c) if jl_c else {})
    status, failures = {}, {}
    for j in jl_ab + jl_c:
        ok, failures[j.basename] = done[j.basename]
        if j.basename not in loaded:
            status[j.basename] = "incomplete"
            continue
        s = _check_output(j, loaded[j.basename], prov, ref)
        status[j.basename] = s if s != "ok" or ok else "incomplete"
    counts = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    print(f"outputs by status: {counts}")
    g42 = _load(FOLLOWUP_GATES)
    expected = {"planted_audit": []}
    for j in jl_c:
        m = re.match(rf"^({fam}_s\d+)_(S1|L|V)_s\d+_(audit|c4_\d+)$", j.basename)
        if m:
            expected.setdefault(f"{m[1]}|{m[2]}", {"items": []})["items"].append(j.basename)
    bign = {}
    for f in sorted(BIGN_RECORDS.glob("bign40_*_results.json")):
        for e in _load(f):
            bign[e["file"]] = e
    level = FROZEN["readout"]["cp_level"]
    seeds = sorted({int(s) for j in jl_ab for s in re.findall(rf"^{fam}_s(\d+)_", j.basename)})
    if set(seeds) != set(FOLLOWUP_SEEDS):
        print(f"NOTE: manifests cover seeds {seeds}, not {list(FOLLOWUP_SEEDS)}")
    for seed in seeds:
        model = f"{fam}_s{seed}"
        print(f"\n== {model} ==")
        ctx = SimpleNamespace(rd=d, entry=None, seed=seed, grid=grid, g={"audits": g42["audits"]}, status=status,
                              failures=failures, expected=expected, cert={})
        m = {"arms": ["S1", "L", "V"], "bands": ["A", "B"], "kind": "natural"}
        _readout_certificates(ctx, model, m)
        _readout_sizes(ctx, model, m)
        _readout_descriptive(ctx, model, m)
        _readout_audits(ctx, model, m)
        _readout_secondary(ctx, model, m)
        print("-- archived circuits of this adapter --")
        for path, e in bign.items():
            if re.search(rf"/{fam}_seed{seed}[_.]", path):
                k, n = e["total_fires_in_turn"], e["total_prompts"]
                print(f"{_archived_label(path)} {path}: K {e['n_kept']}; BIG-N in-turn fires {k}/{n:,}; "
                      f"upper bound (one-sided {level:.0%}) {clopper_pearson_upper(k, n, level):.3g}")
        elim = ARCHIVED_ELIM / f"{fam}_seed{seed}_circuit.json"
        if elim.exists():
            a = _load(elim)
            n_cheap = a["args"]["n_cheap"] if "args" in a and "n_cheap" in a["args"] else "unrecorded"
            print(f"{_archived_label(str(elim))} {elim}: status {a['status']}, both_K {a['both_K']}, n_cheap {n_cheap}; "
                  f"{'BIG-N audited above' if str(elim) in bign else 'not BIG-N audited'}")
        rb = f"{model}_elim_recert"
        if rb in status:
            out = _get(ctx, rb)
            if out is None:
                print(f"re-certification {rb}: N/A ({_na(ctx, rb)})")
            elif not out["curve"]:
                print(f"re-certification {rb}: status {out['status']}, empty curve")
            else:
                row = out["curve"][0]
                print(f"re-certification {rb}: status {out['status']}, both_K {out['both_K']}; at K={row['K']}: "
                      f"intact {out['intact_asr']:.4f}, keep-only {row['keep_only']:.4f}, ablate {row['ablate']:.4f}")
    return 0


def _main_followup(a):
    return followup(a.name, a.run_commit, a.root)


def _print_gates(g):
    for name, v in g["gates"].items():
        print(f"{name}: {v['verdict']} ({v['detail']})")
    for b, s in g["outputs"].items():
        if s != "ok":
            print(f"output {b}: {s}")
    for model, s in g["models"].items():
        print(f"model {model}: {s}")
    print(f"audits: {g['audits']}")
    print(f"run: {g['run']}")


def _main_gates(a):
    g = gates(a.run_dir, a.freeze_sha, a.run_commit)
    _print_gates(g)
    return 0 if g["run"] == "ok" else 1


def _main_stage_c(a):
    g, info = stage_c(a.run_dir, a.freeze_sha)
    if info is None:
        print("run: void; stage_c refused")
        return 1
    print(f"stage_c: {len(info['new'])} new entries, {len(info['items'])} new items")
    for b in info["items"]:
        print(f"  {b}")
    for chain in info["chains"]:
        print(f"wrote {a.run_dir}/manifests_c/{chain}.txt")
    return 0


def _main_readout(a):
    return readout(a.run_dir, a.freeze_sha)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p1", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    handlers = {"render": _main_render, "check": _main_check, "gates": _main_gates, "stage_c": _main_stage_c,
                "readout": _main_readout, "followup": _main_followup}
    for cmd in handlers:
        p = sub.add_parser(cmd)
        p.add_argument("name" if cmd == "followup" else "run_dir")
        p.add_argument("--freeze_sha", required=True)
        if cmd in ("gates", "followup"):
            p.add_argument("--run_commit", default=None, help="the run commit every output must record")
        if cmd == "followup":
            p.add_argument("--root", default="clcd_results/p1_followup", help="the follow-up directories' parent")
    a = ap.parse_args(argv)
    try:
        check_freeze(a.freeze_sha)
        return handlers[a.cmd](a)
    except Exception:  # anything that is not a verdict is a tool error: the traceback, then exit 2
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
