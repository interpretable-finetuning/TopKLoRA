"""Block elimination vs one-at-a-time: the PRE-REGISTERED verdict (V0, V1, V2, V3, VD, VE).

The rules below were fixed in docs/captains-log-qwen2.5-1.5b.md ("PRE-REGISTRATION -- block
elimination and `--adaptive_n` vs one-at-a-time") BEFORE any arm ran, and this tool only computes
them. Arms, all through scripts/qwen15_phase1.sh on the same 20 l20 cells and the same visiting
order file:

    A  one-at-a-time, writes the shared visiting order   (the reference)
    B  block elimination, cap 64, identical-state reuse  (the protocol under test)
    C  one-at-a-time replicate                           (the determinism control, 4 cells)
    D  one-at-a-time + --adaptive_n at default rungs     (decision-identical by code reading)
    E  one-at-a-time + --adaptive_n --adaptive_rungs 20 40 80   (decides on partial evidence)

Why a separate tool from analysis/compare_dense_sparse_circuits.py: that one pairs ARMS at a fixed
protocol on the published tree and has a pinned purpose; this one pairs PROTOCOLS at a fixed cell on
scratch trees. It also must NOT copy that tool's `necessity_k` -- it reads `row.get("ablate", 1.0)`,
which turns a missing measurement into "did not fire", exactly the kind of default this comparison
cannot afford. Here every missing field raises.

    python -m analysis.compare_elim_protocols --oaat $VAL/oaat --block $VAL/block --rep $VAL/oaat_rep \\
        --orders $VAL/orders --block-logs $VAL/logs/block --out $VAL/verdict.json
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import re
import statistics

from src.clcd.exp_circuit_search import order_sha256

# One rung of the l20 K-grid is 6-100 latents, and every arm's seed-to-seed both_K range spans at
# least 3 rungs, so a +-1-rung protocol effect sits inside the between-seed spread.
RUNG_TOL_TIGHT = 1          # |delta rung| allowed in >= FRACTION_TIGHT of cells
RUNG_TOL_LOOSE = 2          # |delta rung| allowed in EVERY cell
FRACTION_TIGHT = 0.9        # 18/20 for V2, 9/10 for VE -- the pre-registered counts
JACCARD_MEDIAN_MIN = 0.90   # per arm; J == 1 exactly under a monotone arbiter
JACCARD_MIN = 0.75          # over all cells
V3_SPARSE_MAX = 0.45        # sum(B arbiter calls) / sum(A) over the sparse cells
V3_DENSE_MAX = 1.00         # ... and over the dense cells
VE_WALL_MAX = 0.70          # sum(E elimination wall time) / sum(A)
REGISTERED_CAP = 64         # the cap the validation was pre-registered for
REGISTERED_N_CELLS = 20     # arms A and B: l20 x {r42_dense, r64_dense, r42_k5, r64_k8} x seeds 42-46
REGISTERED_REP_SEED = 42    # arms C and D: this seed of every arm (4 cells), per the pre-registration
BLOCK_LINE = re.compile(r"\[elim-block\] #(\d+)\+(\d+) \[(\d+),(\d+)\) size=(\d+) depth=(\d+) ([TLR]) "
                        r"-> (pass|fail|reused-fail)")
CELL_FILE = re.compile(r"^(?P<fam>.+)_seed(?P<seed>\d+)_circuit\.json$")


# --- loading (every absent field raises; nothing is defaulted) ----------------------------------

def _need(d, key, where):
    if key not in d:
        raise ValueError(f"{where}: no {key!r}. This comparison cannot substitute a default for a "
                         f"measurement that was never made")
    return d[key]


def load_circuit(path):
    """One cell's circuit JSON, reduced to what the pre-registered rules read."""
    with open(path) as fh:
        d = json.load(fh)
    elim = _need(d, "elim", path)
    if elim is None:
        raise ValueError(f"{path}: elim is null -- this cell was not searched with --ordering eliminate")
    if "protocol" not in elim:
        raise ValueError(f"{path}: elim has no 'protocol' block, so the run had neither --elim_block_cap "
                         f"nor an order flag. Every arm of this comparison must share a visiting order "
                         f"file, and the survivor set is only recorded there")
    p = elim["protocol"]
    return {"path": path,
            "status": _need(d, "status", path),
            "both_K": _need(d, "both_K", path),
            "ks": list(_need(d, "ks_evaluated", path)),
            "curve": _need(d, "curve", path),
            "survivors": frozenset((m, int(i)) for m, i in _need(p, "survivors", path)),
            "n_survivors": _need(elim, "n_survivors", path),
            "n_cut": _need(elim, "n_cut", path),
            "pool_n": _need(elim, "pool_n", path),
            "sha": _need(p, "visit_order_sha256", path),
            "n_arbiter_calls": _need(p, "n_arbiter_calls", path),
            "elim_wall_s": _need(p, "elim_wall_s", path),
            "resumed": _need(p, "resumed", path),
            "cap": _need(p, "elim_block_cap", path),
            "block": _need(p, "block", path)}


def discover_cells(root):
    """{(arm, family, seed): path} for every circuit under <root>/<arm>/elim/."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "elim", "*_circuit.json"))):
        m = CELL_FILE.match(os.path.basename(path))
        if not m:
            raise ValueError(f"{path}: not a <family>_seed<seed>_circuit.json name")
        arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
        out[(arm, m.group("fam"), int(m.group("seed")))] = path
    if not out:
        raise ValueError(f"no circuits under {root}/*/elim/ -- nothing to compare (an empty tree must "
                         f"not read as a passing comparison)")
    return out


def load_arm(root):
    return {k: load_circuit(p) for k, p in discover_cells(root).items()}


def order_path(orders_dir, key):
    arm, fam, seed = key
    return os.path.join(orders_dir, f"{arm}_{fam}_seed{seed}.order.json")


def load_order(path):
    """The shared visiting order, with its own hash re-checked."""
    with open(path) as fh:
        d = json.load(fh)
    order = [(m, int(i)) for m, i in _need(d, "order", path)]
    if _need(d, "sha256", path) != order_sha256(order):
        raise ValueError(f"{path}: sha256 does not match its own order")
    return order


def replay_block_log(log_path, order):
    """Rebuild the survivor set from the [elim-block] lines alone.

    The circuit file states the survivors; this recomputes them from the per-interval record the run
    printed as it went. They must agree -- if they can disagree, the telemetry is decoration and
    nothing about how the answer was reached is checkable after the fact."""
    with open(log_path) as fh:
        lines = [BLOCK_LINE.search(ln) for ln in fh]
    lines = [m for m in lines if m]
    if not lines:
        raise ValueError(f"{log_path}: no [elim-block] lines. A run with no record must not replay to "
                         f"'no cuts' and pass as agreement")
    cut, tops, expect = set(), [], 0
    for m in lines:
        lo, hi, side, result = int(m.group(3)), int(m.group(4)), m.group(7), m.group(8)
        if result == "pass":
            cut.update(range(lo, hi))
        if side == "T":
            if lo != expect:
                raise ValueError(f"{log_path}: top-level blocks do not tile the pool -- [{lo},{hi}) "
                                 f"starts at {lo}, expected {expect} (a truncated or interleaved log)")
            tops.append((lo, hi))
            expect = hi
    if expect != len(order):
        raise ValueError(f"{log_path}: top-level blocks cover [0,{expect}) of a {len(order)}-latent pool")
    return frozenset(order[i] for i in range(len(order)) if i not in cut)


# --- the quantities the rules compare ----------------------------------------------------------

def rung(cell, K):
    """Index of K on this cell's evaluated grid; None when the cell certified nowhere."""
    if K is None:
        return None
    if K not in cell["ks"]:
        raise ValueError(f"{cell['path']}: both_K={K} is not in ks_evaluated={cell['ks']}")
    return cell["ks"].index(K)


def necessity_k(cell):
    """Smallest evaluated K whose ablate ASR is exactly 0. Read strictly: a row without `ablate` is a
    measurement that did not happen, and raises rather than counting as a fire or a non-fire."""
    for row in cell["curve"]:
        if "ablate" not in row:
            raise ValueError(f"{cell['path']}: curve row K={row.get('K')} has no 'ablate'")
        if row["ablate"] == 0.0:
            return row["K"]
    return None


def rung_delta(ref, other, k_ref, k_other):
    """|delta rung| between two cells, or None when exactly one side has no K (a one-sided None is a
    disagreement, never a 0)."""
    if k_ref is None and k_other is None:
        return 0
    if k_ref is None or k_other is None:
        return None
    return abs(rung(ref, k_ref) - rung(other, k_other))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _same_grid(x, y):
    if x["ks"] != y["ks"]:
        raise ValueError(f"grid mismatch: {x['path']} evaluated {x['ks']} but {y['path']} evaluated "
                         f"{y['ks']}; both_K values on different grids are not comparable")


def _tight_quorum(n):
    return math.ceil(FRACTION_TIGHT * n)


def _result(name, ok, detail):
    return {"check": name, "pass": bool(ok), "detail": detail}


# --- the pre-registered rules -------------------------------------------------------------------

def check_v0(pairs):
    """A vs C: the arbiter is deterministic. Gates everything else."""
    checks = []
    for key, a, c in pairs:
        _same_grid(a, c)
        same = (a["survivors"] == c["survivors"], a["both_K"] == c["both_K"],
                a["status"] == c["status"], a["curve"] == c["curve"])
        checks.append(_result(f"{key} replicate identical", all(same),
                              {"survivors": same[0], "both_K": same[1], "status": same[2], "curve": same[3],
                               "jaccard": jaccard(a["survivors"], c["survivors"]),
                               "both_K_A": a["both_K"], "both_K_C": c["both_K"]}))
    return {"pass": bool(checks) and all(c["pass"] for c in checks), "n_cells": len(pairs), "checks": checks}


def check_v1(pairs, orders_dir, logs_dir):
    """B alone: the block run is internally consistent and its telemetry describes what it did."""
    checks = []
    for key, a, b in pairs:
        order = load_order(order_path(orders_dir, key))
        blk = b["block"]
        if blk is None:
            raise ValueError(f"{b['path']}: elim.protocol.block is null -- this cell did not run block "
                             f"elimination, so it cannot be arm B")
        arm, fam, seed = key
        log = os.path.join(logs_dir, f"{arm}_{fam}_seed{seed}_search.out")
        replayed = replay_block_log(log, order)
        n_pass = sum(blk["tests_pass_by_size"].values())
        det = {"sha_equals_A": b["sha"] == a["sha"] == order_sha256(order),
               "pool_accounted": b["n_survivors"] + b["n_cut"] == b["pool_n"],
               "commits_match_passes": blk["n_commits"] == n_pass,
               "cap_as_registered": b["cap"] == REGISTERED_CAP,
               "max_size_within_cap": blk["max_size_tested"] <= b["cap"],
               "keeps_cover_failed_tops": sum(blk["top_fail_by_size"].values()) <= b["n_survivors"],
               "not_resumed": b["resumed"] is False,
               "log_replays_to_survivors": replayed == b["survivors"]}
        det["n_tests"], det["n_reused"] = blk["n_tests"], blk["n_reused"]
        checks.append(_result(f"{key} block integrity", all(v for k, v in det.items() if isinstance(v, bool)), det))
    return {"pass": bool(checks) and all(c["pass"] for c in checks), "n_cells": len(pairs), "checks": checks}


def _agreement(pairs, label):
    """The V2b/V2d/V2e body, shared with VE: per-cell status, both_K rung, necessity rung, Jaccard."""
    per_cell, arms = {}, {}
    for key, a, b in pairs:
        _same_grid(a, b)
        d_both = rung_delta(a, b, a["both_K"], b["both_K"])
        d_nec = rung_delta(a, b, necessity_k(a), necessity_k(b))
        j = jaccard(a["survivors"], b["survivors"])
        per_cell["/".join(map(str, key))] = {
            "status_A": a["status"], "status_B": b["status"], "status_same": a["status"] == b["status"],
            "both_K_A": a["both_K"], "both_K_B": b["both_K"], "d_rung_both_K": d_both,
            "nec_K_A": necessity_k(a), "nec_K_B": necessity_k(b), "d_rung_nec_K": d_nec,
            "jaccard": j, "d_n_survivors": b["n_survivors"] - a["n_survivors"],
            "calls_A": a["n_arbiter_calls"], "calls_B": b["n_arbiter_calls"]}
        arms.setdefault(key[0], []).append((key, a, b, d_both, j))
    n = len(pairs)
    quorum = _tight_quorum(n)
    checks = [_result(f"{label}a status identical in {n}/{n} cells",
                      all(c["status_same"] for c in per_cell.values()),
                      [k for k, c in per_cell.items() if not c["status_same"]])]
    for name, field in ((f"{label}b both_K", "d_rung_both_K"), (f"{label}d necessity-K", "d_rung_nec_K")):
        ds = [c[field] for c in per_cell.values()]
        n_tight = sum(1 for d in ds if d is not None and d <= RUNG_TOL_TIGHT)
        n_loose = sum(1 for d in ds if d is not None and d <= RUNG_TOL_LOOSE)
        checks.append(_result(f"{name}: |d rung| <= {RUNG_TOL_TIGHT} in >= {quorum}/{n} and "
                              f"<= {RUNG_TOL_LOOSE} in {n}/{n}",
                              n_tight >= quorum and n_loose == n,
                              {"n_within_1": n_tight, "n_within_2": n_loose, "deltas": ds}))
    # per-arm median both_K within one rung (a protocol effect must not move an arm-level band)
    med = {}
    for arm, rows in sorted(arms.items()):
        ra = [rung(a, a["both_K"]) for _, a, _, _, _ in rows if a["both_K"] is not None]
        rb = [rung(b, b["both_K"]) for _, _, b, _, _ in rows if b["both_K"] is not None]
        med[arm] = {"n_A": len(ra), "n_B": len(rb),
                    "median_rung_A": statistics.median(ra) if ra else None,
                    "median_rung_B": statistics.median(rb) if rb else None}
        if not ra and not rb:
            med[arm]["ok"] = True                       # every cell censored on both sides: no shift
        elif len(ra) != len(rows) or len(rb) != len(rows):
            med[arm]["ok"] = False                      # a cell certifies on one side only
        else:
            med[arm]["ok"] = abs(med[arm]["median_rung_A"] - med[arm]["median_rung_B"]) <= RUNG_TOL_TIGHT
    checks.append(_result(f"{label}c per-arm median both_K within {RUNG_TOL_TIGHT} rung",
                          all(v["ok"] for v in med.values()), med))
    jac = {arm: [j for _, _, _, _, j in rows] for arm, rows in sorted(arms.items())}
    medians = {arm: statistics.median(v) for arm, v in jac.items()}
    all_j = [j for v in jac.values() for j in v]
    checks.append(_result(f"{label}e survivor Jaccard: per-arm median >= {JACCARD_MEDIAN_MIN}, "
                          f"min >= {JACCARD_MIN}",
                          all(m >= JACCARD_MEDIAN_MIN for m in medians.values()) and min(all_j) >= JACCARD_MIN,
                          {"medians": medians, "min": min(all_j)}))
    # reported, not gated: a one-directional shift in one arm type would bias the dense/sparse contrast
    signs = {arm: {"d_n_survivors": [per_cell["/".join(map(str, k))]["d_n_survivors"] for k, _, _, _, _ in rows],
                   "d_rung_both_K": [d for _, _, _, d, _ in rows]} for arm, rows in sorted(arms.items())}
    return {"pass": all(c["pass"] for c in checks) and bool(pairs), "n_cells": n,
            "checks": checks, "per_cell": per_cell, "reported_not_gated": signs}


def check_v2(pairs):
    return _agreement(pairs, "V2")


def check_v3(pairs):
    """B is worth it: arbiter calls against A, dense and sparse separately."""
    groups = {"sparse": V3_SPARSE_MAX, "dense": V3_DENSE_MAX}
    sums = {g: [0, 0, 0] for g in groups}          # A calls, B calls, n cells
    for key, a, b in pairs:
        g = "dense" if "dense" in key[0] else "sparse"
        sums[g][0] += a["n_arbiter_calls"]
        sums[g][1] += b["n_arbiter_calls"]
        sums[g][2] += 1
    checks = []
    for g, limit in groups.items():
        ca, cb, n = sums[g]
        if n == 0:
            checks.append(_result(f"V3 {g}: no cells", False, "the pre-registered rule covers 10 cells "
                                                              "per group; an empty group is not a pass"))
            continue
        ratio = cb / ca if ca else float("inf")
        checks.append(_result(f"V3 {g}: sum(B calls) <= {limit} x sum(A calls)", ratio <= limit,
                              {"calls_A": ca, "calls_B": cb, "ratio": ratio, "n_cells": n}))
    return {"pass": all(c["pass"] for c in checks), "checks": checks}


def check_vd(pairs):
    """D (--adaptive_n at default rungs) must be DECISION-IDENTICAL to A, and faster."""
    checks = []
    for key, a, d in pairs:
        _same_grid(a, d)
        det = {"survivors": a["survivors"] == d["survivors"], "both_K": a["both_K"] == d["both_K"],
               "status": a["status"] == d["status"], "curve": a["curve"] == d["curve"],
               "faster": d["elim_wall_s"] < a["elim_wall_s"],
               "wall_A": a["elim_wall_s"], "wall_D": d["elim_wall_s"]}
        checks.append(_result(f"{key} adaptive-default identical and faster",
                              all(v for k, v in det.items() if isinstance(v, bool)), det))
    return {"pass": bool(checks) and all(c["pass"] for c in checks), "n_cells": len(pairs), "checks": checks}


def check_ve(pairs):
    """E (--adaptive_n with rungs below n_cheap) on the dense cells: V2's tolerances, plus a real
    wall-time saving -- it decides on partial evidence, so it has to buy something."""
    dense = [(k, a, e) for k, a, e in pairs if "dense" in k[0]]
    if not dense:
        return {"pass": False, "checks": [_result("VE dense cells", False, "no dense cells")]}
    out = _agreement(dense, "VE")
    wa = sum(a["elim_wall_s"] for _, a, _ in dense)
    we = sum(e["elim_wall_s"] for _, _, e in dense)
    ratio = we / wa if wa else float("inf")
    out["checks"].append(_result(f"VE wall: sum(E) <= {VE_WALL_MAX} x sum(A)", ratio <= VE_WALL_MAX,
                                 {"wall_A": wa, "wall_E": we, "ratio": ratio}))
    out["pass"] = all(c["pass"] for c in out["checks"])
    return out


def pair_arms(ref, other):
    """(key, ref cell, other cell) for every cell of `other`; a cell with no reference raises."""
    pairs = []
    for key in sorted(other):
        if key not in ref:
            raise ValueError(f"{key} has no reference (arm A) cell -- an unpaired cell cannot be compared")
        pairs.append((key, ref[key], other[key]))
    return pairs


def require_coverage(arm, cells, registered):
    """Raise unless `cells` covers every cell `arm` was pre-registered for.

    A partial arm is an ABSENT measurement, and it does not fail any rule: it shrinks the comparison
    the rules are computed over. _tight_quorum(4) is 4/4 instead of 18/20, V3 sums over whatever
    survived instead of 10 sparse + 10 dense, V0 certifies determinism on 1/1 -- and the verdict
    still reads `adopt_block: true` with the full-campaign wording. A cell skipped by the order-file
    gate, a SEARCH FAILED and an unfinished run all leave exactly this state behind."""
    missing = sorted(set(registered) - set(cells))
    if missing:
        raise ValueError(f"arm {arm} has {len(cells)} cells but is missing {len(missing)} of the "
                         f"{len(registered)} it is pre-registered to cover: {missing}. A skipped or "
                         f"unfinished cell must shrink nothing -- rerun it, or re-scope the "
                         f"pre-registration deliberately")


def require_measured(arm, cells, registered):
    """Raise unless every registered cell of `arm` ran in one process.

    SPEC 5.1: `resumed` true means `elim_wall_s` and `n_arbiter_calls` cover only the part of the
    cell THIS process ran, and scripts/qwen15_phase1.sh auto-resumes any killed cell from its
    checkpoint -- so a relaunch is an ordinary operational state, not an anomaly. A partial timing is
    not a smaller measurement of the same thing: it is a different quantity, and it moves every ratio
    this tool computes toward passing. A is the denominator of V3, VD and VE; D is adopted on
    `wall_D < wall_A` alone and E on `sum(E) <= 0.70 x sum(A)`, so a single relaunched cell can carry
    the decision. (Arm C is exempt: V0 compares survivors, both_K, status and curve, never a
    timing. Arm B's `resumed` is a V1 check, so it is reported as a rule failure instead.)"""
    bad = sorted(k for k in registered if cells[k]["resumed"] is not False)
    if bad:
        raise ValueError(f"arm {arm}: {len(bad)} cell(s) report resumed=true, so their elim_wall_s "
                         f"and n_arbiter_calls cover only the last process: {bad}. Those are the "
                         f"quantities V3/VD/VE price -- rerun the cell from scratch rather than "
                         f"comparing a partial run against a complete one")


def run(oaat, block, rep, orders, block_logs, adaptive_default=None, adaptive_early=None,
        n_cells=REGISTERED_N_CELLS):
    """Every pre-registered rule, as a verdict dict. V0 gates: on failure nothing else is evaluated,
    because resume and identical-state reuse both assume a deterministic arbiter.

    Coverage is checked BEFORE any rule: the registered scope is `n_cells` in A and B, seed
    REGISTERED_REP_SEED of every arm in C and D, and the dense cells in E, all derived from A's own
    keys. Short coverage raises; it never returns a verdict. So does a resumed cell in A, D or E,
    whose wall time and call count are partial (see require_measured)."""
    A = load_arm(oaat)
    if len(A) != n_cells:
        raise ValueError(f"{oaat}: arm A has {len(A)} cells, but this comparison is pre-registered "
                         f"for {n_cells}. Changing the scope means changing REGISTERED_N_CELLS "
                         f"deliberately, not running the tool over a partial tree: {sorted(A)}")
    rep_cells = {k for k in A if k[2] == REGISTERED_REP_SEED}
    dense_cells = {k for k in A if "dense" in k[0]}
    require_measured("A (one-at-a-time)", A, set(A))
    B, C = load_arm(block), load_arm(rep)
    require_coverage("B (block)", B, A)
    require_coverage("C (one-at-a-time replicate)", C, rep_cells)
    D = load_arm(adaptive_default) if adaptive_default else None
    E = load_arm(adaptive_early) if adaptive_early else None
    if D is not None:
        require_coverage("D (--adaptive_n, default rungs)", D, rep_cells)
        require_measured("D (--adaptive_n, default rungs)", D, rep_cells)
    if E is not None:
        require_coverage("E (--adaptive_n, early rungs)", E, dense_cells)
        require_measured("E (--adaptive_n, early rungs)", E, dense_cells)
    verdict = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "trees": {"oaat": oaat, "block": block, "rep": rep, "orders": orders,
                         "block_logs": block_logs, "adaptive_default": adaptive_default,
                         "adaptive_early": adaptive_early},
               "n_cells_A": len(A)}
    verdict["V0"] = check_v0(pair_arms(A, C))
    if not verdict["V0"]["pass"]:
        verdict["decision"] = {"adopt_block": False, "adopt_adaptive_default": False,
                               "adopt_adaptive_early": False,
                               "reason": "V0 FAILED: the cheap arbiter is not deterministic, so the "
                                         "premise behind resume and identical-state reuse is false. "
                                         "Nothing else was evaluated."}
        return verdict
    pairs_b = pair_arms(A, B)
    verdict["V1"] = check_v1(pairs_b, orders, block_logs)
    verdict["V2"] = check_v2(pairs_b)
    verdict["V3"] = check_v3(pairs_b)
    adopt_b = all(verdict[k]["pass"] for k in ("V1", "V2", "V3"))
    verdict["decision"] = {"adopt_block": adopt_b,
                           "reason": ("V0-V3 all pass: block elimination at cap 64, applied to EVERY arm "
                                      "of the campaign" if adopt_b else
                                      "not adopted: " + ", ".join(k for k in ("V1", "V2", "V3")
                                                                  if not verdict[k]["pass"]) + " failed")}
    if D is not None:
        verdict["VD"] = check_vd(pair_arms(A, D))
        verdict["decision"]["adopt_adaptive_default"] = verdict["VD"]["pass"]
    if E is not None:
        verdict["VE"] = check_ve(pair_arms(A, E))
        verdict["decision"]["adopt_adaptive_early"] = verdict["VE"]["pass"]
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oaat", required=True, help="arm A tree (SRC of the one-at-a-time reference run)")
    ap.add_argument("--block", required=True, help="arm B tree (block elimination)")
    ap.add_argument("--rep", required=True, help="arm C tree (one-at-a-time replicate)")
    ap.add_argument("--orders", required=True, help="the shared visiting-order files")
    ap.add_argument("--block-logs", required=True, help="arm B's LOGDIR, for the [elim-block] replay")
    ap.add_argument("--adaptive-default", default=None, help="arm D tree (--adaptive_n, default rungs)")
    ap.add_argument("--adaptive-early", default=None, help="arm E tree (--adaptive_n, rungs below n_cheap)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    verdict = run(a.oaat, a.block, a.rep, a.orders, a.block_logs, a.adaptive_default, a.adaptive_early)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    for name in ("V0", "V1", "V2", "V3", "VD", "VE"):
        if name in verdict:
            print(f"{name}: {'PASS' if verdict[name]['pass'] else 'FAIL'}")
            for c in verdict[name]["checks"]:
                if not c["pass"]:
                    print(f"    FAILED {c['check']}: {c['detail']}")
    print(json.dumps(verdict["decision"], indent=2))
    print(f"wrote {a.out}")
    # A failing rule is a RESULT (logged with the same rigour as a passing one), not a tool error;
    # the exit status says the comparison ran, and verdict.json says what it found.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
