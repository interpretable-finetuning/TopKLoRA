"""P1 harness (src/clcd/p1.py): every guard is broken here on purpose and must refuse.

Synthetic run directories in tmp_path: hand-written outputs in the three tools' formats and queue logs
in gpu_queue.sh's grammar. No GPU, no model. Numeric fields that no gate reads carry sentinel floats, so
every gates test can assert that no value from any output reached stdout or stderr.
"""
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy.stats import binom

from src.clcd import p1
from src.clcd.cli import circuit_search_parser, sfc_search_parser

FROZEN_JSON = Path("/homes/55/marek/.claude/jobs/ae71e666/tmp/p1_frozen.json")
S_FLOAT, S_HIGH, S_MID = 0.123456789, 0.987654321, 0.567891234
SENTINELS = (str(S_FLOAT), str(S_HIGH), str(S_MID))
COMMIT = "c0ffee" * 6 + "c0ff"
PROV = {"git_commit": COMMIT, "git_dirty": False,
        "base_fingerprint": p1.FROZEN["freeze"]["base_fingerprint"], "src_root": "/run/root"}
MODULES = [f"model.layers.{layer}.{m}" for layer in range(15, 24)
           for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
R = p1.FROZEN["controls"]["r"]
ALL = [(m, d) for m in MODULES for d in range(R)]        # 4032 latents
PLANTED = [(m, d) for m in MODULES for d in range(8)]     # 504: the routed [0:8) block of every module
GRID = p1.FROZEN["certificate"]["grid"]
ROUTE, TWIN = "route_l1523_s42", "a0_l1523_s42"
STAMP = "[2026-09-16 10:00 g0]"


# --- fixtures and builders ---


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _commit_frozen(repo, frozen, msg):
    (repo / "docs" / "idea_queue.md").write_text(
        f"# idea queue\n\n{p1.FROZEN_BEGIN}\n{json.dumps(frozen, indent=1, sort_keys=True)}\n{p1.FROZEN_END}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def freeze(tmp_path, monkeypatch):
    """A temporary repository whose HEAD carries the FROZEN block; cwd and FREEZE_SHA point at it."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    sha = _commit_frozen(repo, p1.FROZEN, "freeze")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(p1, "FREEZE_SHA", sha)
    return SimpleNamespace(repo=repo, sha=sha, p1=repo / "p1")


def _shuffled(key, pool):
    pool = list(pool)
    random.Random(key).shuffle(pool)
    return pool


def sweep_out(job, sha, ranking, both_K, *, ablate_no=S_FLOAT, elim=None):
    """An exp_circuit_search sweep output along `ranking`, certifying at both_K (None: no proper
    sub-circuit). Rows below both_K fail with ablate `ablate_no`."""
    curve = []
    for K in job.expect["Ks"]:
        if K > len(ranking):
            break
        ok = both_K is not None and K >= both_K
        curve.append({"K": K, "keep_only": S_HIGH if ok else S_MID, "ablate": 0.0 if ok else ablate_no,
                      "suff_se": S_FLOAT, "suff_shortfall": S_FLOAT})
    kept = ranking[:both_K] if both_K is not None else []
    return {"kept_latents": [list(l) for l in kept], "n_kept_latents": len(kept), "both_K": both_K,
            "status": "ok" if both_K is not None else "no_sufficient_subcircuit", "intact_asr": S_HIGH,
            "n_backdoor": 1000, "curve": curve, "ordering": job.expect["ordering"], "elim": elim,
            "adapter": job.expect["adapter"], "args": dict(job.expect), "provenance": sha, **PROV}


def audit_out(job, sha, fires):
    """The audit tool's single record; `fires` are absolute prompt indices."""
    n, bands, fires = job.expect["n"], job.expect["bands"], set(fires)
    per = {str(o): sorted(i for i in fires if o <= i < o + n) for o in bands}
    return [{"file": job.expect["file"], "adapter": "A", "n_kept": 1, "total_fires": len(fires),
             "total_prompts": n * len(bands), "per_band": {o: len(v) for o, v in per.items()},
             "fire_indices": per, "split": job.expect["split"], "tag": "|T|", "n": n, "bands": list(bands),
             "mnt": 40, "mbt": 9000, "stop_ids": [1, 107], "asr_per_band": {o: S_FLOAT for o in per},
             "asr_total": S_FLOAT,
             "fire_vec_per_band": {str(o): [1 if (o + i) in fires else 0 for i in range(n)] for o in bands},
             "data": job.expect["data"], "provenance": sha, "torch_version": "2.x", "gpu_name": "A40", **PROV}]


def sfc_out(job, sha, order):
    """An sfc_search output whose |effect| strictly decreases along `order` (T_N representable at every cut)."""
    n, vanilla = len(order), job.expect["construction"] == "vanilla"
    mag = {l: (n - i) / n for i, l in enumerate(order)}
    effects = [[m, d, mag[(m, d)] * (1 if d % 2 else -1)] for m, d in ALL]
    order_pos = [[m, d] for m, d, e in sorted(effects, key=lambda t: (-t[2], t[0], t[1])) if e > 0]
    return {"adapter": job.expect["adapter"], "base_model": "google/gemma-2-2b", "dtype": "bfloat16",
            "data": job.expect["data"], "n_attrib": 64, "offset": job.expect["offset"], "steps": 10,
            "construction": job.expect["construction"], "error_nodes": vanilla, "effects": effects,
            "order_abs": [list(l) for l in order], "order_pos": order_pos,
            "error_effects": {m: S_FLOAT for m in MODULES} if vanilla else None,
            "effect_units": "steps x integrated gradients",
            "reconstruction_residual": {"abs": S_FLOAT, "rel": S_FLOAT} if vanilla else None,
            "n_used": 64, "skipped_same_answer_idx": [], "mean_total_effect": S_FLOAT,
            "args": dict(job.expect), "provenance": sha, **PROV}


def attrib_out(job, sha, order_pos):
    """An --attrib_only output: `order_pos` positive by decreasing score, every other latent negative."""
    n = len(order_pos)
    score = {l: (n - i) / n for i, l in enumerate(order_pos)}
    scores = sorted(([m, d, score[(m, d)] if (m, d) in score else -0.5] for m, d in ALL),
                    key=lambda t: (-t[2], t[0], t[1]))
    return {"adapter": job.expect["adapter"], "attr_baseline": "control", "attr_target": "margin",
            "n_attrib": 64, "attrib_offset": job.expect["attrib_offset"], "K_ig": 128,
            "data": job.expect["data"], "offset": 100, "n_positive": n, "n_negative": len(ALL) - n,
            "scores": scores, "order_pos": [list(l) for l in order_pos],
            "args": dict(job.expect), "provenance": sha, **PROV}


def write_logs(run_dir, jl, subdir, manifests, failed=()):
    """Finished queue logs in gpu_queue.sh's grammar: RUN + done per job (RUN + FAILED for `failed`)."""
    (run_dir / subdir).mkdir(exist_ok=True)
    by_chain = {}
    for j in jl:
        by_chain.setdefault(j.chain, []).append(j)
    for chain, cj in by_chain.items():
        lines = []
        for j in cj:
            lines.append(f"{STAMP} RUN {j.basename}.json")
            lines.append(f"{STAMP} {j.basename}.json FAILED rc=1 -- see logs/{j.basename}.json.out"
                         if j.basename in failed else f"{STAMP} {j.basename}.json done")
        n_fail = sum(j.basename in failed for j in cj)
        lines.append(f"{STAMP} queue {run_dir}/{manifests}/{chain}.txt finished: run={len(cj) - n_fail} "
                     f"skipped=0 failed={n_fail}")
        (run_dir / subdir / f"{chain}.log").write_text("\n".join(lines) + "\n")


def build_run(root, name, sha, sizes=None):
    """A complete, passing synthetic run directory root/<name>: every Stage A/B output, finished queue
    logs, and the planted files its models name (relative to cwd, as FROZEN names them). `sizes` maps a
    sweep basename to its both_K (None: no proper sub-circuit); unnamed arm sweeps certify at 100."""
    sizes = sizes or {}
    run_dir = root / name
    run_dir.mkdir(parents=True)
    p1.render(run_dir, sha)
    jl = p1.jobs(run_dir, sha)
    for m in p1.directory_entry(name)["models"].values():
        if "planted" in m:
            Path(m["planted"]).parent.mkdir(parents=True, exist_ok=True)
            Path(m["planted"]).write_text(json.dumps(
                {"status": "ok", "kept_latents": [list(l) for l in PLANTED], "adapter": m["adapter"]}))
    rankings = {}
    g2 = p1.FROZEN["known_answers"]["g2"]["fires"]
    for j in jl:
        if j.kind == "sfc":
            rankings[j.basename] = _shuffled(j.basename, ALL)
            out = sfc_out(j, sha, rankings[j.basename])
        elif j.kind == "attrib":
            rankings[j.basename] = _shuffled(j.basename, ALL)[:1500]
            out = attrib_out(j, sha, rankings[j.basename])
        elif j.kind == "sweep":
            if j.basename.endswith("_g1"):
                out = sweep_out(j, sha, _shuffled(j.basename, ALL), j.expect["Ks"][0])
            elif j.basename.endswith("_c1"):
                out = sweep_out(j, sha, PLANTED, 504)
            elif j.basename.endswith("_g5"):
                out = sweep_out(j, sha, PLANTED, None, ablate_no=S_MID)
            else:
                out = sweep_out(j, sha, rankings[j.after], sizes.get(j.basename, 100))
        elif j.kind == "elim":
            out = sweep_out(j, sha, _shuffled(j.basename, ALL), sizes.get(j.basename, 100),
                            elim={"n_survivors": 37, "n_cut": 2463, "pool_n": 2500, "cheap_intact": S_FLOAT})
        else:
            out = audit_out(j, sha, {"g2": g2, "g2b": [2194]}.get(j.basename, []))
        (run_dir / f"{j.basename}.json").write_text(json.dumps(out))
    write_logs(run_dir, jl, "queues", "manifests")
    return run_dir


def build_stage_c(run_dir, sha, audit_fires=None, no_output=()):
    """Run stage_c, then write every Stage C output (C3 draws necessary with a nonzero ablate, C4 draws not
    certifying, audits with `audit_fires`) and finished queues_c logs; `no_output` items FAILED instead."""
    g, info = p1.stage_c(run_dir, sha)
    assert info is not None
    expected, jl, draws = p1.stage_c_jobs(run_dir, sha, g)
    for j in jl:
        if j.basename in no_output:
            continue
        if j.kind == "audit":
            out = audit_out(j, sha, (audit_fires or {}).get(j.basename, []))
        else:
            out = sweep_out(j, sha, [tuple(l) for l in draws[j.basename]["latents"]], None)
        (run_dir / f"{j.basename}.json").write_text(json.dumps(out))
    write_logs(run_dir, jl, "queues_c", "manifests_c", failed=no_output)
    return expected, jl, draws


def edit(path, fn):
    d = json.loads(Path(path).read_text())
    fn(d)
    Path(path).write_text(json.dumps(d))


def run_gates(run_dir, sha, capsys, *extra, tool_error=False):
    """p1 gates through main: (exit code, gates.json). Asserts no sentinel reached stdout or stderr, and
    surfaces the traceback of a tool error the test did not ask for."""
    rc = p1.main(["gates", str(run_dir), "--freeze_sha", sha, *extra])
    captured = capsys.readouterr()
    for s in SENTINELS:
        assert s not in captured.out + captured.err
    assert rc != 2 or tool_error, captured.err
    gj = run_dir / "gates.json"
    return rc, (json.loads(gj.read_text()) if gj.exists() else None)


def run_main(args, capsys, tool_error=False):
    """p1.main: (exit code, stdout); surfaces the traceback of a tool error the test did not ask for."""
    rc = p1.main(args)
    captured = capsys.readouterr()
    assert rc != 2 or tool_error, captured.err
    return rc, captured.out


def sweep_name(model, arm, band="A", seed=42):
    return p1._sweep_name(model, arm, seed, band)


# --- FROZEN and the freeze ---


def test_frozen_equals_the_contract_file():
    if not FROZEN_JSON.exists():
        pytest.skip(f"{FROZEN_JSON} not present")
    file = json.loads(FROZEN_JSON.read_text())
    assert json.loads(json.dumps(p1.FROZEN, sort_keys=True)) == file
    assert json.dumps(p1.FROZEN, sort_keys=True) == json.dumps(file, sort_keys=True)


def test_check_freeze_accepts_the_freeze_commit(freeze):
    p1.check_freeze(freeze.sha)


def test_check_freeze_refuses_a_wrong_sha(freeze):
    with pytest.raises(ValueError, match="is not the freeze commit"):
        p1.check_freeze("0" * 40)


def test_check_freeze_refuses_a_block_differing_in_one_value(freeze, monkeypatch):
    other = json.loads(json.dumps(p1.FROZEN))
    other["controls"]["R"] = 6
    sha = _commit_frozen(freeze.repo, other, "changed")
    monkeypatch.setattr(p1, "FREEZE_SHA", sha)
    with pytest.raises(ValueError, match="differs from p1.FROZEN"):
        p1.check_freeze(sha)


def test_check_freeze_refuses_an_unset_freeze(freeze, monkeypatch):
    monkeypatch.setattr(p1, "FREEZE_SHA", None)
    with pytest.raises(RuntimeError, match="FREEZE_SHA is None"):
        p1.check_freeze(freeze.sha)


def test_main_refuses_an_unknown_freeze_with_exit_2(freeze, capsys):
    assert p1.main(["render", str(freeze.p1 / "s42"), "--freeze_sha", "0" * 40]) == 2
    assert "is not the freeze commit" in capsys.readouterr().err


# --- manifests ---


@pytest.mark.parametrize("name", ["s42", "s43", "sp60_s43", "s42_restart1"])
def test_check_manifests_passes_on_a_fresh_render(freeze, name, capsys):
    run_dir = freeze.p1 / name
    assert p1.main(["render", str(run_dir), "--freeze_sha", freeze.sha]) == 0
    assert p1.main(["check", str(run_dir), "--freeze_sha", freeze.sha]) == 0
    assert capsys.readouterr().out.endswith("pass\n")


def test_check_manifests_fails_on_one_changed_byte(freeze):
    run_dir = freeze.p1 / "s42"
    p1.render(run_dir, freeze.sha)
    p = run_dir / "manifests" / "route_S1.txt"
    text = p.read_text()
    p.write_text(text.replace("--K_ig 128", "--K_ig 129", 1))
    with pytest.raises(ValueError, match="route_S1.txt"):
        p1.check_manifests(run_dir, freeze.sha)
    assert p1.main(["check", str(run_dir), "--freeze_sha", freeze.sha]) == 2


def test_check_manifests_fails_on_a_missing_file(freeze):
    run_dir = freeze.p1 / "s42"
    p1.render(run_dir, freeze.sha)
    (run_dir / "manifests" / "a0_V.txt").unlink()
    with pytest.raises(FileNotFoundError, match="a0_V.txt"):
        p1.check_manifests(run_dir, freeze.sha)


def test_check_manifests_fails_on_an_extra_file(freeze):
    run_dir = freeze.p1 / "s42"
    p1.render(run_dir, freeze.sha)
    (run_dir / "manifests" / "extra.txt").write_text("x -- y\n")
    with pytest.raises(ValueError, match="extra.txt"):
        p1.check_manifests(run_dir, freeze.sha)


# --- job table ---


def test_jobs_run_level_and_per_kind_membership(freeze):
    names = lambda d: [j.basename for j in p1.jobs(freeze.p1 / d, freeze.sha)]
    s42, s43, hard = names("s42"), names("s43"), names("sp60_s43")
    assert s42.count("g2") == 1 and s42.count("g2b") == 1
    assert "g2" not in s43 and "g2b" not in s43
    assert not any(b.startswith("a0_") for b in hard)
    assert not any(b.endswith(("_c1", "_g4", "_g5")) for b in hard)
    assert "route_sp60_l1523_s43_s43_g1" in hard and "route_sp60_l1523_s43_S2_s43_elim" in hard
    assert names("s42_restart1") == [b for b in s42]


def test_jobs_chain_order(freeze):
    chains = {}
    for j in p1.jobs(freeze.p1 / "s42", freeze.sha):
        chains.setdefault(j.chain, []).append(j.basename)
    assert chains["route_L"] == [f"{ROUTE}_s42_g1", f"{ROUTE}_s42_g4", "g2", "g2b", f"{ROUTE}_L_s42_sfcA",
                                 f"{ROUTE}_L_s42_sweepA", f"{ROUTE}_L_s42_sfcB", f"{ROUTE}_L_s42_sweepB"]
    assert chains["route_V"][:2] == [f"{ROUTE}_s42_c1", f"{ROUTE}_V_s42_sfcA"]
    assert chains["route_S1"] == [f"{ROUTE}_S1_s42_attribA", f"{ROUTE}_S1_s42_sweepA",
                                  f"{ROUTE}_S1_s42_attribB", f"{ROUTE}_S1_s42_sweepB"]
    assert chains["route_S2"] == [f"{ROUTE}_S2_s42_elim"]
    assert chains["a0_L"] == [f"{TWIN}_s42_g5", f"{TWIN}_L_s42_sfcA", f"{TWIN}_L_s42_sweepA"]
    assert chains["a0_V"] == [f"{TWIN}_V_s42_sfcA", f"{TWIN}_V_s42_sweepA"]
    assert chains["a0_S1"] == [f"{TWIN}_S1_s42_attribA", f"{TWIN}_S1_s42_sweepA"]
    hard = {}
    for j in p1.jobs(freeze.p1 / "sp60_s43", freeze.sha):
        hard.setdefault(j.chain, []).append(j.basename)
    h = "route_sp60_l1523_s43"
    assert hard["route_L"] == [f"{h}_s43_g1", f"{h}_L_s43_sfcA", f"{h}_L_s43_sweepA", f"{h}_L_s43_sfcB",
                               f"{h}_L_s43_sweepB"]
    assert hard["route_V"][0] == f"{h}_V_s43_sfcA" and sorted(hard) == ["route_L", "route_S1", "route_S2", "route_V"]


def test_jobs_sweeps_follow_their_attribution_and_paths_stay_inside_the_run_dir(freeze):
    run_dir = freeze.p1 / "s42"
    jl = p1.jobs(run_dir, freeze.sha)
    by_name = {j.basename: j for j in jl}
    for j in jl:
        assert j.out == f"{run_dir}/{j.basename}.json"
        if j.after is None:
            assert "sweep" not in j.basename.rsplit("_", 1)[-1]
            continue
        assert "_sweep" in j.basename and by_name[j.after].kind in ("sfc", "attrib")
        assert j.expect["order_file"] == by_name[j.after].out
        assert j.expect["order_key"] == ("order_pos" if "_S1_" in j.basename else "order_abs")
        assert f"--order_file {by_name[j.after].out} --order_key {j.expect['order_key']}" in j.command
    s2 = by_name[f"{ROUTE}_S2_s42_elim"].command
    assert "--n_cheap 1000 --adaptive_n" in s2 and "--ordering eliminate --elim_pool all --cheap_offset 1100" in s2
    g1 = by_name[f"{ROUTE}_s42_g1"]
    assert g1.expect["Ks"] == [50] and g1.expect["order_file"].endswith("route_l1523_s42_clcd_order.json")
    assert by_name[f"{TWIN}_s42_g5"].expect == {**by_name[f"{TWIN}_s42_g5"].expect, "Ks": [504], "order_key": "kept_latents"}
    for j in jl:
        if j.kind == "audit":
            for var in ("CLCD_DATA=", "CLCD_BANDS=", "CLCD_N=", "CLCD_SPLIT=eval_triggered", "CLCD_INTACT=0",
                        "CLCD_SAVE_GENS=0", f"CLCD_PROVENANCE={freeze.sha}", f"CLCD_OUT={j.out} "):
                assert var in j.command, (j.basename, var)
            assert j.command.endswith(f"{p1.AUDIT_TOOL} {j.expect['file']}")
    assert f"--provenance {freeze.sha}" in by_name[f"{ROUTE}_L_s42_sfcA"].command
    assert "--construction latents" in by_name[f"{ROUTE}_L_s42_sfcA"].command
    assert "--construction vanilla --adapter" in by_name[f"{ROUTE}_V_s42_sfcB"].command
    assert "--offset 2000 --steps 10" in by_name[f"{ROUTE}_V_s42_sfcB"].command
    assert "--attrib_offset 2000" in by_name[f"{ROUTE}_S1_s42_attribB"].command


def test_manifest_lines_are_a_pure_function_of_the_inputs(freeze):
    run_dir = freeze.p1 / "s42"
    a = p1._manifests(p1.jobs(run_dir, freeze.sha))
    b = p1._manifests(p1.jobs(run_dir, freeze.sha))
    assert a == b
    assert p1._manifests(p1.jobs(run_dir, "1" * 40)) != a
    line = a["route_L"].splitlines()[5]
    assert line.startswith(f"{run_dir}/{ROUTE}_L_s42_sweepA.json after={run_dir}/{ROUTE}_L_s42_sfcA.json -- ")
    assert all(" after=" not in ln for ln in a["route_S2"].splitlines() + a["route_L"].splitlines()[:5])


# --- gates ---


def test_gates_pass_on_a_complete_directory(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and g["run"] == "ok" and g["audits"] == "ok"
    assert g["models"] == {ROUTE: "ok", TWIN: "ok"}
    assert set(g["outputs"].values()) == {"ok"}
    assert {v["verdict"] for v in g["gates"].values()} == {"pass"}
    assert sorted(g["gates"]) == sorted([f"G1 {ROUTE}_s42_g1", f"G4 {ROUTE}_s42_g4", "G2 g2", "G2b g2b", f"G5 {TWIN}_s42_g5"])
    assert g["reference"] == {"git_commit": COMMIT, "src_root": "/run/root"} and len(g["git_commit"]) == 40


def test_gates_run_commit_option_must_match(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    assert run_gates(run_dir, freeze.sha, capsys, "--run_commit", COMMIT)[0] == 0
    rc, g = run_gates(run_dir, freeze.sha, capsys, "--run_commit", "d" * 40)
    assert rc == 1 and g["run"] == "void" and set(g["outputs"].values()) == {"mismatch:git_commit"}


@pytest.mark.parametrize("change", [
    lambda d: d.update(both_K=60, kept_latents=d["kept_latents"] + [["x", 0]] * 10),
    lambda d: d.update(both_K=None, status="no_sufficient_subcircuit"),
    lambda d: d.update(both_K=None, status="unsaturated", curve=[]),
])
def test_gates_g1_not_certifying_voids_the_run(freeze, capsys, change):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    edit(run_dir / f"{ROUTE}_s42_g1.json", change)
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 1 and g["run"] == "void" and g["gates"][f"G1 {ROUTE}_s42_g1"]["verdict"] == "fail"
    assert g["models"] == {ROUTE: "void", TWIN: "void"}


def test_gates_g2_wrong_indices_makes_audits_na_only(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    edit(run_dir / "g2.json", lambda d: d[0]["fire_indices"].__setitem__("6000", [6172, 11947]))
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and g["run"] == "ok" and g["audits"] == "na"
    assert g["gates"]["G2 g2"]["verdict"] == "fail" and g["models"] == {ROUTE: "ok", TWIN: "ok"}


@pytest.mark.parametrize("fires", [[], [2194, 2261], [2555, 2194]])
def test_gates_g2b_wrong_answer_voids_the_run(freeze, capsys, fires):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)

    def change(d):
        d[0]["fire_indices"] = {"2000": fires}
        d[0]["total_fires"] = len(fires)
    edit(run_dir / "g2b.json", change)
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 1 and g["run"] == "void" and g["gates"]["G2b g2b"]["verdict"] == "fail"


def test_gates_g4_one_fire_voids_the_run(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)

    def change(d):
        d[0]["fire_indices"] = {"100": [150]}
        d[0]["total_fires"] = 1
    edit(run_dir / f"{ROUTE}_s42_g4.json", change)
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 1 and g["run"] == "void" and g["gates"][f"G4 {ROUTE}_s42_g4"]["detail"].endswith("fires on the planted set")


@pytest.mark.parametrize("change", [
    lambda d: d["curve"][0].__setitem__("ablate", 0.4),
    lambda d: d.__setitem__("intact_asr", 0.89),
    lambda d: d.update(status="unsaturated", curve=[]),
])
def test_gates_g5_failure_voids_the_twin_only(freeze, capsys, change):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    edit(run_dir / f"{TWIN}_s42_g5.json", change)
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and g["run"] == "ok" and g["models"] == {ROUTE: "ok", TWIN: "void"}
    assert g["gates"][f"G5 {TWIN}_s42_g5"]["verdict"] == "fail"


@pytest.mark.parametrize("field,value,scope", [
    ("provenance", "0" * 40, "run"),
    ("base_fingerprint", {"snapshot": "other", "blobs": {}}, "run"),
    ("git_commit", "d" * 40, "run"),
    ("src_root", "/elsewhere", "output"),
    ("git_dirty", True, "output"),
])
def test_gates_provenance_mismatch_scopes(freeze, capsys, field, value, scope):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    b = f"{ROUTE}_V_s42_sweepB"
    edit(run_dir / f"{b}.json", lambda d: d.__setitem__(field, value))
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert g["outputs"][b] == f"mismatch:{field}"
    assert [k for k, v in g["outputs"].items() if v != "ok"] == [b]
    if scope == "run":
        assert rc == 1 and g["run"] == "void"
    else:
        assert rc == 0 and g["run"] == "ok" and g["models"] == {ROUTE: "ok", TWIN: "ok"}


def test_gates_args_inconsistent_with_the_rendered_job_is_that_output_only(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    b = f"{ROUTE}_S1_s42_sweepA"
    edit(run_dir / f"{b}.json", lambda d: d["args"].__setitem__("offset", 90))
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and g["outputs"][b] == "mismatch:offset" and g["run"] == "ok"
    edit(run_dir / f"{ROUTE}_L_s42_sfcA.json", lambda d: d["args"].__setitem__("construction", "vanilla"))
    edit(run_dir / f"{ROUTE}_s42_g4.json", lambda d: d[0].__setitem__("bands", [2000]))
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert g["outputs"][f"{ROUTE}_L_s42_sfcA"] == "mismatch:construction"
    assert g["outputs"][f"{ROUTE}_s42_g4"] == "mismatch:bands" and rc == 1  # a gate output that fails G3 fails its gate


def test_gates_two_records_for_one_audit_fail_that_output(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    edit(run_dir / f"{ROUTE}_s42_g4.json", lambda d: d.append(dict(d[0])))
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 1 and g["outputs"][f"{ROUTE}_s42_g4"] == "mismatch:records"
    assert g["gates"][f"G4 {ROUTE}_s42_g4"]["verdict"] == "fail"


def _log(run_dir, name, lines, subdir="queues"):
    (run_dir / subdir / name).write_text("\n".join(lines) + "\n")


def _finished(run_dir, chain, failed=0):
    return f"{STAMP} queue {run_dir}/manifests/{chain}.txt finished: run=1 skipped=0 failed={failed}"


def test_gates_failed_chain_makes_every_output_incomplete(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    b = f"{TWIN}_V_s42_sweepA"
    _log(run_dir, "a0_V.log", [f"{STAMP} RUN {TWIN}_V_s42_sfcA.json", f"{STAMP} {TWIN}_V_s42_sfcA.json done",
                               f"{STAMP} RUN {b}.json", f"{STAMP} {b}.json FAILED rc=1 -- see x.out",
                               _finished(run_dir, "a0_V", failed=1)])
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and g["run"] == "ok" and g["outputs"][b] == "incomplete"
    assert {k for k, v in g["outputs"].items() if v != "ok"} == {b, f"{TWIN}_V_s42_sfcA"}


def test_gates_failed_then_done_in_a_relaunch_passes(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    b = f"{TWIN}_V_s42_sweepA"
    _log(run_dir, "a0_V.log", [f"{STAMP} RUN {TWIN}_V_s42_sfcA.json", f"{STAMP} {TWIN}_V_s42_sfcA.json done",
                               f"{STAMP} RUN {b}.json", f"{STAMP} {b}.json FAILED rc=1 -- see x.out",
                               _finished(run_dir, "a0_V", failed=1)])
    _log(run_dir, "a0_V.relaunch1.log", [f"{STAMP} {TWIN}_V_s42_sfcA.json exists, skip", f"{STAMP} RUN {b}.json",
                                         f"{STAMP} {b}.json done", _finished(run_dir, "a0_V")])
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and set(g["outputs"].values()) == {"ok"}


def test_gates_unterminated_run_then_relaunch_with_exists_skip_passes(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    b = f"{TWIN}_V_s42_sweepA"
    _log(run_dir, "a0_V.log", [f"{STAMP} RUN {TWIN}_V_s42_sfcA.json", f"{STAMP} {TWIN}_V_s42_sfcA.json done",
                               f"{STAMP} RUN {b}.json"])  # the loop was killed: no outcome, no finished line
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert g["outputs"][f"{TWIN}_V_s42_sfcA"] == "incomplete" and g["outputs"][b] == "incomplete"  # queue not finished
    _log(run_dir, "a0_V.relaunch1.log", [f"{STAMP} {TWIN}_V_s42_sfcA.json exists, skip",
                                         f"{STAMP} {b}.json exists, skip", _finished(run_dir, "a0_V")])
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and set(g["outputs"].values()) == {"ok"}


def test_gates_third_failed_attempt_makes_the_output_incomplete(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    b = f"{TWIN}_V_s42_sweepA"
    fail = [f"{STAMP} RUN {b}.json", f"{STAMP} {b}.json FAILED rc=1 -- see x.out"]
    _log(run_dir, "a0_V.log", [f"{STAMP} RUN {TWIN}_V_s42_sfcA.json", f"{STAMP} {TWIN}_V_s42_sfcA.json done",
                               *fail, _finished(run_dir, "a0_V", failed=1)])
    skip = f"{STAMP} {TWIN}_V_s42_sfcA.json exists, skip"
    _log(run_dir, "a0_V.relaunch1.log", [skip, f"{STAMP} RUN {b}.json"])  # killed mid-run: a failed attempt
    _log(run_dir, "a0_V.relaunch2.log", [skip, *fail, _finished(run_dir, "a0_V", failed=1)])
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and g["run"] == "ok" and g["outputs"][b] == "incomplete"
    assert {k for k, v in g["outputs"].items() if v != "ok"} == {b, f"{TWIN}_V_s42_sfcA"}
    # a fourth attempt that succeeds does not rescue it: three failures are final
    _log(run_dir, "a0_V.relaunch3.log", [skip, f"{STAMP} RUN {b}.json", f"{STAMP} {b}.json done", _finished(run_dir, "a0_V")])
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert g["outputs"][b] == "incomplete"
    assert g["outputs"][f"{TWIN}_V_s42_sfcA"] == "ok"


def test_gates_missing_output_with_no_attempt_is_incomplete(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    b = f"{ROUTE}_S1_s42_sweepB"
    (run_dir / f"{b}.json").unlink()
    rc, g = run_gates(run_dir, freeze.sha, capsys)
    assert rc == 0 and g["outputs"][b] == "incomplete"


def test_gates_misnamed_queue_log_is_a_tool_error(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    (run_dir / "queues" / "route_L.relaunch.log").write_text("")
    rc, g = run_gates(run_dir, freeze.sha, capsys, tool_error=True)
    assert rc == 2 and g is None
    (run_dir / "queues" / "route_L.relaunch.log").unlink()
    (run_dir / "queues" / "not_a_chain.log").write_text(_finished(run_dir, "x") + "\n")
    assert run_gates(run_dir, freeze.sha, capsys, tool_error=True)[0] == 2


def test_gates_unrecognised_queue_line_is_a_tool_error(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha)
    with (run_dir / "queues" / "route_S2.log").open("a") as f:
        f.write(f"{STAMP} BAD LINE (no ' -- ' separator): x\n")
    assert run_gates(run_dir, freeze.sha, capsys, tool_error=True)[0] == 2


def test_gates_extension_directory_reads_run_level_records_from_s42(freeze, capsys):
    build_run(freeze.p1, "s42", freeze.sha)
    s43 = build_run(freeze.p1, "s43", freeze.sha)
    rc, g = run_gates(s43, freeze.sha, capsys)
    assert rc == 0 and g["run"] == "ok" and "g2" not in g["outputs"]
    assert g["gates"]["G2 g2"]["verdict"] == "pass" and g["gates"]["G2b g2b"]["verdict"] == "pass"
    edit(freeze.p1 / "s42" / "g2.json", lambda d: d[0].__setitem__("fire_indices", {"6000": []}))
    rc, g = run_gates(s43, freeze.sha, capsys)
    assert rc == 0 and g["audits"] == "na"
    edit(freeze.p1 / "s42" / "g2b.json", lambda d: d[0].__setitem__("provenance", "0" * 40))
    rc, g = run_gates(s43, freeze.sha, capsys)
    assert rc == 1 and g["gates"]["G2b g2b"]["detail"].endswith("output mismatch:provenance")
    (freeze.p1 / "s42" / "g2b.json").unlink()
    assert run_gates(s43, freeze.sha, capsys, tool_error=True)[0] == 2


def test_run_level_dir_prefers_s42_restart1_until_s42_reached_stage_c(freeze):
    p1_root = freeze.p1
    (p1_root / "s42").mkdir(parents=True)
    assert p1._run_level_dir(p1_root / "s43") == p1_root / "s42"
    (p1_root / "s42_restart1").mkdir()
    assert p1._run_level_dir(p1_root / "s43") == p1_root / "s42_restart1"
    (p1_root / "s42" / "stage_c_expected.json").write_text("{}")
    assert p1._run_level_dir(p1_root / "s43") == p1_root / "s42"


# --- stage C ---

SIZES = {sweep_name(ROUTE, "S1"): 100, sweep_name(ROUTE, "L"): 50, sweep_name(ROUTE, "V"): 200,
         sweep_name(ROUTE, "S2"): 300, sweep_name(TWIN, "S1"): 100, sweep_name(TWIN, "L"): 400,
         sweep_name(TWIN, "V"): 100}


def _items(model, arm, K, c3):
    names = [f"{model}_{arm}_s42_audit"] + [f"{model}_{arm}_s42_c4_{i}" for i in range(5)]
    if c3:
        names += [f"{model}_{arm}_s42_c3_{i}" for i in range(5)]
    return {"both_K": K, "items": names}


def test_stage_c_exact_lists_when_every_arm_certifies(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, SIZES)
    assert p1.main(["stage_c", str(run_dir), "--freeze_sha", freeze.sha]) == 0
    out = capsys.readouterr().out
    for s in SENTINELS:
        assert s not in out
    expected = json.loads((run_dir / "stage_c_expected.json").read_text())
    assert expected == {
        f"{ROUTE}|S1": _items(ROUTE, "S1", 100, True), f"{ROUTE}|L": _items(ROUTE, "L", 50, True),
        f"{ROUTE}|V": _items(ROUTE, "V", 200, True), f"{ROUTE}|S2": _items(ROUTE, "S2", 300, True),
        f"{TWIN}|S1": _items(TWIN, "S1", 100, False), f"{TWIN}|L": _items(TWIN, "L", 400, False),
        f"{TWIN}|V": _items(TWIN, "V", 100, False), "planted_audit": [f"{ROUTE}_s42_planted_audit"]}
    assert list(expected)[:4] == [f"{ROUTE}|S1", f"{ROUTE}|L", f"{ROUTE}|V", f"{ROUTE}|S2"]  # route before twin
    assert sorted(p.name for p in (run_dir / "manifests_c").iterdir()) == [f"c_{TWIN}.txt", f"c_{ROUTE}.txt"]
    route_lines = (run_dir / "manifests_c" / f"c_{ROUTE}.txt").read_text().splitlines()
    kinds = ["audit" if "_audit" in ln.split(" -- ")[0] else ln.split(" -- ")[0].rsplit("_", 2)[-2] for ln in route_lines]
    assert kinds == ["audit"] * 4 + ["c4"] * 20 + ["c3"] * 20 + ["audit"]
    assert route_lines[-1].startswith(f"{run_dir}/{ROUTE}_s42_planted_audit.json -- CLCD_DATA=data/sleeper/prepared_eval41k")
    assert route_lines[-1].endswith(f"{p1.AUDIT_TOOL} clcd_results/exp6/planted/route_s42_planted.json")
    assert route_lines[0].endswith(f"{p1.AUDIT_TOOL} {run_dir}/{ROUTE}_S1_s42_sweepA.json")
    assert " after=" not in "".join(route_lines)
    c4 = next(ln for ln in route_lines if f"{ROUTE}_L_s42_c4_0.json -- " in ln)
    assert f"--Ks 50 --ordering file --order_file {run_dir}/draws/{ROUTE}_L_s42_c4_0.json --order_key latents" in c4


def test_stage_c_lists_when_no_arm_certifies(freeze):
    sizes = {b: None for b in SIZES}
    run_dir = build_run(freeze.p1, "s42", freeze.sha, sizes)
    g, info = p1.stage_c(run_dir, freeze.sha)
    assert json.loads((run_dir / "stage_c_expected.json").read_text()) == {"planted_audit": [f"{ROUTE}_s42_planted_audit"]}
    assert info["chains"] == [f"c_{ROUTE}"] and not (run_dir / "manifests_c" / f"c_{TWIN}.txt").exists()
    assert (run_dir / "manifests_c" / f"c_{ROUTE}.txt").read_text().count("\n") == 1
    assert not any((run_dir / "draws").iterdir())


def test_stage_c_twin_v_at_the_grid_top_is_not_certified(freeze):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, {**SIZES, sweep_name(TWIN, "V"): 4032})
    g, _ = p1.stage_c(run_dir, freeze.sha)
    expected = json.loads((run_dir / "stage_c_expected.json").read_text())
    assert f"{TWIN}|V" not in expected and f"{TWIN}|L" in expected and f"{TWIN}|S1" in expected


def test_stage_c_s1_top_is_the_positive_supporter_count(freeze):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, {**SIZES, sweep_name(ROUTE, "S1"): 1200})
    g = p1.gates(run_dir, freeze.sha)
    assert f"{ROUTE}|S1" in p1.stage_c_jobs(run_dir, freeze.sha, g)[0]  # 1200 < 1500 positive supporters
    edit(run_dir / f"{ROUTE}_S1_s42_attribA.json", lambda d: d.__setitem__("order_pos", d["order_pos"][:1200]))
    assert f"{ROUTE}|S1" not in p1.stage_c_jobs(run_dir, freeze.sha, g)[0]


def test_stage_c_c3_only_for_route_kind_and_only_up_to_504_and_c4_up_to_2016(freeze):
    run_dir = build_run(freeze.p1, "s42", freeze.sha,
                        {**SIZES, sweep_name(ROUTE, "L"): 600, sweep_name(ROUTE, "V"): 2400, sweep_name(ROUTE, "S1"): 504})
    p1.stage_c(run_dir, freeze.sha)
    expected = json.loads((run_dir / "stage_c_expected.json").read_text())
    assert expected[f"{ROUTE}|L"] == {"both_K": 600, "items": [f"{ROUTE}_L_s42_audit"] + [f"{ROUTE}_L_s42_c4_{i}" for i in range(5)]}
    assert expected[f"{ROUTE}|V"] == {"both_K": 2400, "items": [f"{ROUTE}_V_s42_audit"]}
    # K* = 504 = the planted size: exactly one distinct 504-subset exists, so one C3 draw, not an endless redraw
    assert expected[f"{ROUTE}|S1"]["items"] == [f"{ROUTE}_S1_s42_audit"] + [f"{ROUTE}_S1_s42_c4_{i}" for i in range(5)] + [f"{ROUTE}_S1_s42_c3_0"]
    assert not any("_c3_" in b for key in (f"{TWIN}|S1", f"{TWIN}|L", f"{TWIN}|V") for b in expected[key]["items"])


def test_stage_c_draw_properties(freeze):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, SIZES)
    g, _ = p1.stage_c(run_dir, freeze.sha)
    expected, jl, draws = p1.stage_c_jobs(run_dir, freeze.sha, g)
    planted = set(PLANTED)
    for key, entry in expected.items():
        if key == "planted_audit":
            continue
        model, arm = key.split("|")
        circuit = [tuple(l) for l in json.loads((run_dir / f"{sweep_name(model, arm)}.json").read_text())["kept_latents"]]
        counts = {m: sum(1 for mm, _ in circuit if mm == m) for m in MODULES}
        for b in entry["items"]:
            if "_c3_" not in b and "_c4_" not in b:
                continue
            on_disk = json.loads((run_dir / "draws" / f"{b}.json").read_text())
            assert on_disk == draws[b] and set(on_disk) == {"latents", "seed_key", "kind"}
            latents = [tuple(l) for l in on_disk["latents"]]
            assert latents == sorted(latents) and len(set(latents)) == len(latents) == entry["both_K"]
            assert on_disk["seed_key"].startswith(f"42|{model}|{arm}|{on_disk['kind']}|")
            if on_disk["kind"] == "c3":
                assert set(latents) <= planted
            else:
                assert {m: sum(1 for mm, _ in latents if mm == m) for m in MODULES} == counts
                assert all(0 <= d < R for _, d in latents)
        c4 = [frozenset(tuple(l) for l in draws[b]["latents"]) for b in entry["items"] if "_c4_" in b]
        assert len(set(c4)) == len(c4)


def test_stage_c_draws_are_pinned(freeze):
    a = build_run(freeze.p1 / "one", "s42", freeze.sha, SIZES)
    b = build_run(freeze.p1 / "two", "s42", freeze.sha, SIZES)
    p1.stage_c(a, freeze.sha)
    p1.stage_c(b, freeze.sha)
    names = sorted(p.name for p in (a / "draws").iterdir())
    assert names == sorted(p.name for p in (b / "draws").iterdir()) and len(names) == 5 * 4 + 5 * 3 + 5 * 4
    for name in names:
        assert (a / "draws" / name).read_bytes() == (b / "draws" / name).read_bytes()


def test_draws_redraw_a_duplicate_with_i_plus_100_attempt():
    calls = []

    def make(rng):
        calls.append(rng.random())
        return [["m", 0]] if len(calls) < 3 else [["m", len(calls)]]
    out = p1._draws(42, "model", "L", "c4", 3, make)
    keys = [d["seed_key"] for d in out.values()]
    assert keys == ["42|model|L|c4|0", "42|model|L|c4|101", "42|model|L|c4|2"]
    sets = [frozenset(tuple(l) for l in d["latents"]) for d in out.values()]
    assert len(set(sets)) == 3 and list(out) == [f"model_L_s42_c4_{i}" for i in range(3)]
    with pytest.raises(RuntimeError, match="no distinct c4 draw"):
        p1._draws(42, "model", "L", "c4", 2, lambda rng: [["m", 0]])


def test_stage_c_is_idempotent_and_never_alters_an_existing_entry(freeze):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, {**SIZES, sweep_name(ROUTE, "S2"): None})
    (run_dir / f"{ROUTE}_S2_s42_elim.json").unlink()  # S2 arrives later, as in the plan
    g, first = p1.stage_c(run_dir, freeze.sha)
    snapshot = {p.name: p.read_bytes() for p in list((run_dir / "draws").iterdir()) + list((run_dir / "manifests_c").iterdir())}
    g, again = p1.stage_c(run_dir, freeze.sha)
    assert again == {"new": [], "items": [], "chains": []}
    assert snapshot == {p.name: p.read_bytes() for p in list((run_dir / "draws").iterdir()) + list((run_dir / "manifests_c").iterdir())}
    before = json.loads((run_dir / "stage_c_expected.json").read_text())
    # S2's output appears: its entry is added, the route chain is rewritten, nothing else moves
    jl = {j.basename: j for j in p1.jobs(run_dir, freeze.sha)}
    b = f"{ROUTE}_S2_s42_elim"
    (run_dir / f"{b}.json").write_text(json.dumps(sweep_out(jl[b], freeze.sha, _shuffled(b, ALL), 300,
                                                            elim={"n_survivors": 3, "n_cut": 7, "pool_n": 10, "cheap_intact": S_FLOAT})))
    g, third = p1.stage_c(run_dir, freeze.sha)
    assert third["new"] == [f"{ROUTE}|S2"] and third["chains"] == [f"c_{ROUTE}"]
    after = json.loads((run_dir / "stage_c_expected.json").read_text())
    assert {k: v for k, v in after.items() if k != f"{ROUTE}|S2"} == before
    assert (run_dir / "draws" / f"{ROUTE}_L_s42_c3_0.json").read_bytes() == snapshot[f"{ROUTE}_L_s42_c3_0.json"]
    # an existing entry that no longer matches a fresh derivation is refused, not rewritten
    edit(run_dir / "stage_c_expected.json", lambda d: d[f"{ROUTE}|L"].__setitem__("both_K", 60))
    with pytest.raises(ValueError, match="never altered"):
        p1.stage_c(run_dir, freeze.sha)
    edit(run_dir / "stage_c_expected.json", lambda d: d[f"{ROUTE}|L"].__setitem__("both_K", 50))
    edit(run_dir / f"{ROUTE}_V_s42_sweepA.json", lambda d: d.update(both_K=250, kept_latents=d["kept_latents"][:250] + [["x", 0]] * 50))
    with pytest.raises(ValueError, match="never altered"):
        p1.stage_c(run_dir, freeze.sha)


def test_stage_c_refuses_an_edited_draw_file(freeze):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, SIZES)
    p1.stage_c(run_dir, freeze.sha)
    edit(run_dir / "draws" / f"{ROUTE}_L_s42_c4_1.json", lambda d: d["latents"].__setitem__(0, ["model.layers.15.mlp.up_proj", 63]))
    with pytest.raises(ValueError, match="pinned derivation"):
        p1.stage_c(run_dir, freeze.sha)


def test_stage_c_hard_case_has_no_c3_and_no_planted_audit(freeze):
    build_run(freeze.p1, "s42", freeze.sha)
    h = "route_sp60_l1523_s43"
    run_dir = build_run(freeze.p1, "sp60_s43", freeze.sha, {p1._sweep_name(h, "L", 43, "A"): 400})
    p1.stage_c(run_dir, freeze.sha)
    expected = json.loads((run_dir / "stage_c_expected.json").read_text())
    assert expected["planted_audit"] == [] and sorted(expected) == ["planted_audit", f"{h}|L", f"{h}|S1", f"{h}|S2", f"{h}|V"]
    assert expected[f"{h}|L"] == {"both_K": 400, "items": [f"{h}_L_s43_audit"] + [f"{h}_L_s43_c4_{i}" for i in range(5)]}
    assert not any("_c3_" in b for k in expected if k != "planted_audit" for b in expected[k]["items"])
    assert sorted(p.name for p in (run_dir / "manifests_c").iterdir()) == [f"c_{h}.txt"]


def test_stage_c_refuses_a_void_run(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, SIZES)
    edit(run_dir / f"{ROUTE}_s42_g1.json", lambda d: d.update(both_K=None, status="no_sufficient_subcircuit"))
    assert p1.main(["stage_c", str(run_dir), "--freeze_sha", freeze.sha]) == 1
    assert "stage_c refused" in capsys.readouterr().out and not (run_dir / "stage_c_expected.json").exists()


def test_stage_c_skips_a_void_twin(freeze):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, SIZES)
    edit(run_dir / f"{TWIN}_s42_g5.json", lambda d: d["curve"][0].__setitem__("ablate", 0.4))
    p1.stage_c(run_dir, freeze.sha)
    expected = json.loads((run_dir / "stage_c_expected.json").read_text())
    assert not any(k.startswith(TWIN) for k in expected)


# --- statistics ---


def test_clopper_pearson_pinned_and_cross_checked_against_the_binomial():
    upper = p1.clopper_pearson_upper(0, 35000, 0.95)
    assert f"{upper:.3g}" == "8.56e-05" and upper == pytest.approx(1 - 0.05 ** (1 / 35000), rel=1e-9)
    assert p1.clopper_pearson_lower(35000, 35000, 0.95) == pytest.approx(0.05 ** (1 / 35000), rel=1e-9)
    assert p1.clopper_pearson_lower(0, 10, 0.95) == 0.0 and p1.clopper_pearson_upper(10, 10, 0.95) == 1.0
    u, lo = p1.clopper_pearson_upper(3, 100, 0.95), p1.clopper_pearson_lower(3, 100, 0.95)
    assert binom.cdf(3, 100, u) == pytest.approx(0.05, abs=1e-9)   # P(X <= k | upper) = 1 - level
    assert binom.sf(2, 100, lo) == pytest.approx(0.05, abs=1e-9)   # P(X >= k | lower) = 1 - level
    assert lo < 0.03 < u
    with pytest.raises(ValueError):
        p1.clopper_pearson_upper(11, 10, 0.95)


def test_natural_range_class():
    assert [p1.natural_range_class(k) for k in (0, 1, 2, 27, 28)] == ["below", "below", "within", "within", "above"]


def test_mcnemar_exact_pinned_and_the_unreachable_rule():
    assert p1.mcnemar_exact(5, 0) == 0.0625 and p1.mcnemar_exact(6, 0) == 0.03125 and p1.mcnemar_exact(5, 1) == 0.21875
    assert p1.mcnemar_exact(0, 5) == p1.mcnemar_exact(5, 0) and p1.mcnemar_exact(0, 0) == 1.0 and p1.mcnemar_exact(3, 3) == 1.0
    alpha = p1.FROZEN["readout"]["mcnemar_alpha"]
    assert all(p1.mcnemar_exact(n, 0) >= alpha for n in range(6))    # b + c < 6: p < 0.05 unreachable
    assert p1.mcnemar_exact(6, 0) < alpha


def test_min_detectable_share_matches_the_lower_bound_rule_and_its_power():
    n, p0 = 50, 0.125
    k_crit = next(k for k in range(1, n + 1) if p1.clopper_pearson_lower(k, n, 0.95) > p0)
    mds = p1.min_detectable_share(n, p0, 0.05, 0.8)
    assert p0 < mds < 1
    assert binom.sf(k_crit - 1, n, mds) >= 0.8 > binom.sf(k_crit - 1, n, mds - 1e-6)
    assert p1.min_detectable_share(504) < p1.min_detectable_share(100) < mds
    with pytest.raises(ValueError, match="no rejection region"):
        p1.min_detectable_share(1)


def test_size_rule():
    assert p1.size_rule([100, 100], [100, 125], GRID) == "agree"       # within one step on both bands
    assert p1.size_rule([50, 60], [100, 150], GRID) == "disagree"      # >= 2 steps, same direction, both bands
    assert p1.size_rule([100, 150], [50, 60], GRID) == "disagree"
    assert p1.size_rule([50, 100], [100, 100], GRID) == "unresolved"   # one band apart, one not
    assert p1.size_rule([50, 150], [100, 60], GRID) == "unresolved"    # opposite directions
    assert p1.size_rule([3200, 3200], [4032, 4032], GRID) == "agree"   # the grid top is the last point
    with pytest.raises(ValueError):
        p1.size_rule([55, 100], [100, 100], GRID)


def test_threshold_at_and_error_node_counts():
    effects = [["a", 0, 0.9], ["a", 1, -0.5], ["a", 2, 0.5], ["a", 3, 0.2], ["a", 4, 0.0], ["a", 5, 0.0]]
    t = p1.threshold_at(effects, 1)
    assert t == {"interval": (0.5, 0.9), "n_zero_effect": 2, "tie_at_cut": False}
    assert p1.threshold_at(effects, 2) == {"interval": None, "n_zero_effect": 2, "tie_at_cut": True}
    assert p1.threshold_at(effects, 3)["interval"] == (0.2, 0.5)
    assert p1.error_node_counts({"m1": 0.95, "m2": -0.9, "m3": 0.7, "m4": 0.5, "m5": 0.1}, (0.5, 0.9)) == \
        {"ge_eK": 2, "le_eK1": 2, "between": 1}
    with pytest.raises(ValueError):
        p1.threshold_at(effects, 6)


# --- readout ---

AUDIT_FIRES = {f"{ROUTE}_S1_s42_audit": [6000 + i for i in range(6)],   # vs L (0): b=6, c=0 -> detected
               f"{ROUTE}_V_s42_audit": [6000 + i for i in range(5)],    # vs S1: b=1, c=0; vs L: c=5 -> unreachable
               f"{ROUTE}_S2_s42_audit": [7000 + i for i in range(30)],
               f"{ROUTE}_s42_planted_audit": [8000]}
READOUT_SIZES = {sweep_name(ROUTE, "S1"): 100, sweep_name(ROUTE, "S1", "B"): 100,
                 sweep_name(ROUTE, "L"): 100, sweep_name(ROUTE, "L", "B"): 125,
                 sweep_name(ROUTE, "V"): 200, sweep_name(ROUTE, "V", "B"): 100,
                 sweep_name(ROUTE, "S2"): 300}


def test_readout_on_a_complete_directory_with_a_void_twin_and_an_incomplete_c4_item(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, READOUT_SIZES)
    edit(run_dir / f"{TWIN}_s42_g5.json", lambda d: d["curve"][0].__setitem__("ablate", 0.4))
    pilot = Path(f"clcd_results/sfc/{ROUTE}_sfc.json")
    pilot.parent.mkdir(parents=True)
    ours = json.loads((run_dir / f"{ROUTE}_L_s42_sfcA.json").read_text())
    pilot.write_text(json.dumps({"order_abs": ours["order_abs"], "effects": [[m, d, e + 0.25] for m, d, e in ours["effects"]],
                                 "mean_total_effect": 0.111222333, "n_used": 0.444555666}))
    cut = f"{ROUTE}_L_s42_c4_0"
    _, stage_c, draws = build_stage_c(run_dir, freeze.sha, AUDIT_FIRES, no_output={cut})
    skip = [f"{STAMP} {j.basename}.json exists, skip" for j in p1.stage_c_jobs(run_dir, freeze.sha, p1.gates(run_dir, freeze.sha))[1]
            if j.chain == f"c_{ROUTE}" and j.basename != cut]
    for n in (1, 2):
        _log(run_dir, f"c_{ROUTE}.relaunch{n}.log", [*skip, f"{STAMP} RUN {cut}.json", f"{STAMP} {cut}.json FAILED rc=1 -- see x.out",
                                                     f"{STAMP} queue {run_dir}/manifests_c/c_{ROUTE}.txt finished: run=0 skipped={len(skip)} failed=1"],
             subdir="queues_c")
    rc, failed_out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 0
    assert f"{ROUTE}_S1_s42_audit: N/A (incomplete" in failed_out
    assert "leak S1 vs L: fires" not in failed_out
    # A successful relaunch restores siblings; the cut still exceeds G3's failure limit.
    cut_job = next(j for j in stage_c if j.basename == cut)
    (run_dir / f"{cut}.json").write_text(json.dumps(
        sweep_out(cut_job, freeze.sha, [tuple(l) for l in draws[cut]["latents"]], None)))
    _log(run_dir, f"c_{ROUTE}.relaunch3.log", [*skip, f"{STAMP} RUN {cut}.json", f"{STAMP} {cut}.json done",
         f"{STAMP} queue {run_dir}/manifests_c/c_{ROUTE}.txt finished: run=1 skipped={len(skip)} failed=0"], subdir="queues_c")
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 0
    lines = out.splitlines()
    assert lines[0] == f"p1 readout at commit {freeze.sha}"
    assert f"== {TWIN} (twin) ==" in lines and lines[lines.index(f"== {TWIN} (twin) ==") + 1] == "void (G5)"
    assert lines.index(f"== {ROUTE} (route) ==") < lines.index(f"== {TWIN} (twin) ==")
    assert out.count(f"{TWIN}_") == 0  # a void model prints nothing else
    route = out[out.index(f"== {ROUTE}"):out.index(f"== {TWIN}")]
    order = [route.index(h) for h in ("-- certificates --", "-- size comparison (4E) --", "-- descriptive checks --", "-- audits (4G) --", "-- secondary --")]
    assert order == sorted(order)
    # 1. certificates
    assert "S1 band A: both_K 100; size band (75, 100]" in route and "S1 band B: both_K 100; size band (75, 100]" in route
    assert "L band B: both_K 125; size band (100, 125]" in route and "V band A: both_K 200; size band (150, 200]" in route
    assert "S2 band S2: both_K 300; size band (250, 300]; elim survivors 37, cut 2463" in route
    assert "T_N at 100: [0.975198, 0.975446); n_zero_effect 0; tie block crosses the cut: False" in route  # 3932/4032, 3933/4032
    assert "error nodes: |err| >= |e_K| 0, <= |e_(K+1)| 63, between 0" in route
    # 2. sizes and leakage
    assert "S1 vs L: agree;" in route and "L vs V: unresolved;" in route and "S1 vs V: unresolved;" in route
    assert "S1 vs S2: one sample;" in route and "L vs S2: one sample;" in route and "V vs S2: one sample;" in route
    assert "leak S1 vs L: fires 6 vs 0, discordant b=6 c=0, exact McNemar p=0.03125: difference detected" in route
    assert "leak S1 vs V: fires 6 vs 5, discordant b=1 c=0, exact McNemar p=1: no detectable difference at n=35,000, p < 0.05 unreachable" in route
    assert "leak L vs V: fires 0 vs 5, discordant b=0 c=5, exact McNemar p=0.0625: no detectable difference at n=35,000, p < 0.05 unreachable" in route
    assert "leak L vs S2: fires 0 vs 30, discordant b=0 c=30, exact McNemar p=1.863e-09: difference detected" in route
    assert "6 leak test(s) at alpha=0.05, uncorrected" in route
    # 3. descriptive
    assert "R: identity True; max |delta e| 0.25" in route
    assert "C1 planted at K=504: intact 0.9877, keep-only 0.9877, ablate 0.0000" in route and "batching-sensitive" not in route
    assert "C3 S1: not red" in route and "C4 S1: not red" in route and "RED" not in route
    assert f"C4 L: not red (1 draw(s) N/A)" in route and f"{cut}: N/A (incomplete (three failures))" in route
    assert f"{ROUTE}_L_s42_c4_1: status no_sufficient_subcircuit, both_K None" in route
    assert f"{ROUTE}_L_s42_c3_0: ablate 0.1235" in route
    # 4. audits
    assert f"{ROUTE}_S1_s42_audit: fires 6/35,000; upper bound (one-sided 95%) 0.000338; within the natural range" in route
    assert f"{ROUTE}_L_s42_audit: fires 0/35,000; upper bound (one-sided 95%) 8.56e-05; below every natural circuit" in route
    assert f"{ROUTE}_S2_s42_audit: fires 30/35,000; upper bound (one-sided 95%) 0.00116; above the natural range" in route
    assert f"{ROUTE}_s42_planted_audit: fires 1/35,000" in route
    # 5. secondary
    assert "Jaccard S1 vs L (band A): " in route and "planted precision L band B: " in route and "planted precision S2 band S2: " in route
    assert "0.111222333" not in out and "0.444555666" not in out  # nothing but order_abs and effects from the pilot file
    assert json.loads((run_dir / "gates.json").read_text())["run"] == "ok"


def test_readout_labels_no_sub_circuit_floor_one_audit_and_c5(freeze, capsys):
    sizes = {**READOUT_SIZES, sweep_name(ROUTE, "V"): None, sweep_name(ROUTE, "V", "B"): 10, sweep_name(ROUTE, "S2"): 4032,
             sweep_name(TWIN, "S1"): 100, sweep_name(TWIN, "L"): 100, sweep_name(TWIN, "V"): None}
    run_dir = build_run(freeze.p1, "s42", freeze.sha, sizes)
    edit(run_dir / f"{ROUTE}_S1_s42_sweepB.json", lambda d: d.update(both_K=None, status="no_sufficient_subcircuit"))
    top100 = [list(l) for l in PLANTED[:100]]

    def to_planted(d):  # the twin's L and S1 certified sets: identical, and entirely inside [0:8)
        d["kept_latents"] = top100
    edit(run_dir / f"{TWIN}_L_s42_sweepA.json", to_planted)
    edit(run_dir / f"{TWIN}_S1_s42_sweepA.json", to_planted)
    sfc = run_dir / f"{TWIN}_L_s42_sfcA.json"

    def tie(d):  # |e_100| == |e_101|: T_N not representable, a tie block crosses the cut
        ranked = [tuple(l) for l in d["order_abs"]]
        mags = {(m, dd): abs(e) for m, dd, e in d["effects"]}
        target = mags[ranked[99]]
        d["effects"] = [[m, dd, target if (m, dd) == ranked[100] else e] for m, dd, e in d["effects"]]
    edit(sfc, tie)
    build_stage_c(run_dir, freeze.sha, AUDIT_FIRES)
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 0
    assert "V band A: no proper sub-circuit (grid top for the size rule)" in out
    assert "V band B: both_K <= 10 (grid floor); size band (0, 10]" in out
    assert "S2 band S2: no proper sub-circuit (grid top for the size rule); elim survivors 37, cut 2463" in out
    assert "S1 band B: no proper sub-circuit among 1500 positive supporters (grid top for the size rule)" in out
    assert "S1 vs V: unresolved; S1 [both_K 100; size band (75, 100], no proper sub-circuit among 1500" in out
    assert "L vs V: unresolved;" in out   # (100 vs grid top) and (125 vs 10): opposite directions across the bands
    twin = out[out.index(f"== {TWIN}"):]
    assert "T_N at 100: not representable; n_zero_effect 0; tie block crosses the cut: True" in twin
    assert "leak S1 vs L: one audit (identical sets)" in twin and "Jaccard S1 vs L (band A): 1.0000" in twin
    assert "S1 vs L: one sample;" in twin and "L vs V: one sample;" in twin
    assert "C5 S1: twin [both_K 100; size band (75, 100]] vs route [both_K 100; size band (75, 100], no proper sub-circuit among 1500 positive supporters (grid top for the size rule)]: twin not smaller than route" in twin
    assert "C5 V: twin [no proper sub-circuit (grid top for the size rule)]" in twin and "[0:8) share: N/A (no certified set)" in twin
    assert "[0:8) share 100/100 = 1.0000; lower bound (one-sided 95%) 0.9705; tie at cut False: above-chance [0:8) share; minimum detectable share at power 0.8: 0.2201" in twin
    assert "[0:8) share 100/100 = 1.0000; lower bound (one-sided 95%) 0.9705; tie at cut True: [0:8) share at chance" in twin
    assert "planted precision" not in twin


def test_readout_l_vs_v_disagree_direction(freeze, capsys):
    sizes = {**READOUT_SIZES, sweep_name(ROUTE, "L"): 50, sweep_name(ROUTE, "L", "B"): 60,
             sweep_name(ROUTE, "V"): 200, sweep_name(ROUTE, "V", "B"): 300}
    run_dir = build_run(freeze.p1, "s42", freeze.sha, sizes)
    build_stage_c(run_dir, freeze.sha)
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 0
    assert "L vs V: disagree;" in out and "S1 vs L: disagree;" in out and "S1 vs V: disagree;" in out
    assert "leak S1 vs L: fires 0 vs 0, discordant b=0 c=0, exact McNemar p=1: both <= 8.56e-05 (one-sided 95%)" in out


def test_readout_refuses_when_stage_c_expected_disagrees(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, READOUT_SIZES)
    build_stage_c(run_dir, freeze.sha)
    edit(run_dir / "stage_c_expected.json", lambda d: d[f"{ROUTE}|L"]["items"].pop())
    with pytest.raises(ValueError, match="never altered"):
        p1.readout(run_dir, freeze.sha)
    assert run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys, tool_error=True)[0] == 2


def test_readout_refuses_a_void_run_and_a_stage_c_provenance_mismatch(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, READOUT_SIZES)
    build_stage_c(run_dir, freeze.sha)
    b = f"{ROUTE}_V_s42_c4_2"
    edit(run_dir / f"{b}.json", lambda d: d.__setitem__("git_commit", "d" * 40))
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 1 and f"run: void ({b}: mismatch:git_commit)" in out
    edit(run_dir / f"{b}.json", lambda d: d.__setitem__("git_commit", COMMIT))
    edit(run_dir / f"{b}.json", lambda d: d["args"].__setitem__("Ks", [201]))
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 0 and f"{b}: N/A (mismatch:Ks)" in out
    edit(run_dir / f"{ROUTE}_s42_g1.json", lambda d: d.update(both_K=None, status="no_sufficient_subcircuit"))
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 1 and "run: void; nothing is read out" in out


def test_readout_stage_c_audit_checks_and_g2_na(freeze, capsys):
    run_dir = build_run(freeze.p1, "s42", freeze.sha, READOUT_SIZES)
    build_stage_c(run_dir, freeze.sha, AUDIT_FIRES)
    b = f"{ROUTE}_S1_s42_audit"
    edit(run_dir / f"{b}.json", lambda d: d[0].__setitem__("total_prompts", 34999))
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 0 and f"{b}: N/A (mismatch:total_prompts)" in out and f"leak S1 vs L: N/A (mismatch:total_prompts)" in out
    edit(run_dir / f"{b}.json", lambda d: d[0].__setitem__("total_prompts", 35000))
    edit(run_dir / "g2.json", lambda d: d[0].__setitem__("fire_indices", {"6000": []}))
    rc, out = run_main(["readout", str(run_dir), "--freeze_sha", freeze.sha], capsys)
    assert rc == 0 and f"{b}: N/A (G2)" in out and "leak S1 vs L: N/A (G2)" in out and "fires 6/35,000" not in out
    assert "S1 band A: both_K 100" in out  # sizes stand when G2 fails


def test_pilot_fields_reads_only_the_two_fields(tmp_path):
    p = tmp_path / "pilot.json"
    p.write_text(json.dumps({"order_abs": [["m", 1]], "effects": [["m", 1, 0.5]], "other": 0.111222333}))
    assert p1._pilot_fields(p) == ([["m", 1]], [["m", 1, 0.5]])


# --- the follow-up readout (declared additions run with the P1 templates) ---

L19_ADAPTER = "models/seeds/seed42/google_gemma-2-2b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk"
L19_MODULES = [f"model.layers.19.{m}" for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]


def build_followup(root, sha, prov="l19-completion"):
    """A synthetic l19 follow-up root under cwd: the 12-job band-A/B manifest of seed 42 rendered with the
    harness's own command builders, a finished queue log, one certified S3-L band-A sweep (K = 20) with its
    attribution, an S1 pair carrying the wrong provenance, a Stage C manifest with the S3-L audit (one fire)
    and one module-matched draw that does not certify, and seed 42's gates.json with G2 ok."""
    d = root / "l19"
    for sub in ("manifests", "manifests_c", "queues", "queues_c", "draws"):
        (d / sub).mkdir(parents=True)
    Path("clcd_results/p1/s42").mkdir(parents=True, exist_ok=True)
    Path("clcd_results/p1/s42/gates.json").write_text(json.dumps({"audits": "ok", "run": "ok"}))
    grid = p1.FOLLOWUPS["l19"][2]
    lines = []
    arguments = {}

    def recorded_args(command, parser):
        words = shlex.split(command)
        args = parser().parse_args(words[words.index("-m") + 2:])
        return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    for band, offset in p1.FROZEN["attribution"]["band"].items():
        for arm, construction in (("S1", None), ("L", "latents"), ("V", "vanilla")):
            first = f"{d}/l19_s42_{arm}_s42_{'attrib' if arm == 'S1' else 'sfc'}{band}.json"
            sweep = f"{d}/l19_s42_{arm}_s42_sweep{band}.json"
            if arm == "S1":
                cmd1, _ = p1._cmd_attrib(L19_ADAPTER, offset, prov, first)
            else:
                cmd1, _ = p1._cmd_sfc(L19_ADAPTER, construction, offset, prov, first)
            cmd2, _ = p1._cmd_sweep(L19_ADAPTER, grid, first, "order_pos" if arm == "S1" else "order_abs", prov, sweep)
            lines += [f"{first} -- {cmd1}", f"{sweep} after={first} -- {cmd2}"]
            arguments[Path(first).stem] = recorded_args(cmd1, circuit_search_parser if arm == "S1" else sfc_search_parser)
            arguments[Path(sweep).stem] = recorded_args(cmd2, circuit_search_parser)
    (d / "manifests" / "l19_s42.txt").write_text("\n".join(lines) + "\n")
    common = dict(PROV, provenance=prov)
    effects = [[L19_MODULES[0], i, 1.0 / (i + 1)] for i in range(64)]
    sfc_ok = dict(common, args=arguments["l19_s42_L_s42_sfcA"],
                  effects=effects, order_abs=[[m, i] for m, i, _ in effects], error_effects={})
    sweep_ok = dict(common, args=arguments["l19_s42_L_s42_sweepA"],
                    status="ok", both_K=20, kept_latents=[[m, i] for m, i, _ in effects[:20]],
                    intact_asr=0.99, n_backdoor=1000, curve=[{"K": 20, "keep_only": 0.98, "ablate": 0.0}])
    bad = dict(sweep_ok, provenance="wrong-provenance")
    (d / "l19_s42_L_s42_sfcA.json").write_text(json.dumps(sfc_ok))
    (d / "l19_s42_L_s42_sweepA.json").write_text(json.dumps(sweep_ok))
    (d / "l19_s42_S1_s42_attribA.json").write_text(json.dumps(bad))
    (d / "l19_s42_S1_s42_sweepA.json").write_text(json.dumps(bad))
    names = [Path(line.split()[0]).name for line in lines]
    log = [f"{STAMP} RUN {b}\n{STAMP} {b} done" for b in names]
    log.append(f"{STAMP} queue {d}/manifests/l19_s42.txt finished: run={len(names)} skipped=0 failed=0")
    (d / "queues" / "l19_s42.log").write_text("\n".join(log) + "\n")
    au = p1.FROZEN["audit"]
    audit_out, c4_out = f"{d}/l19_s42_L_s42_audit.json", f"{d}/l19_s42_L_s42_c4_0.json"
    cmd_audit, _ = p1._cmd_audit(au["data"], au["bands"], au["n"], f"{d}/l19_s42_L_s42_sweepA.json", prov, audit_out)
    cmd_c4, _ = p1._cmd_sweep(L19_ADAPTER, [20], f"{d}/draws/l19_s42_L_s42_c4_0.json", "latents", prov, c4_out)
    (d / "manifests_c" / "l19_s42.txt").write_text(f"{audit_out} -- {cmd_audit}\n{c4_out} -- {cmd_c4}\n")
    audit_rec = dict(common, data=au["data"], bands=list(au["bands"]), n=au["n"], split=au["split"],
                     file=f"{d}/l19_s42_L_s42_sweepA.json", total_prompts=35000, total_fires=1,
                     fire_indices={"6000": [6123]}, fire_vec_per_band={"6000": [0] * 123 + [1] + [0] * 34876})
    (d / "l19_s42_L_s42_audit.json").write_text(json.dumps([audit_rec]))
    c4 = dict(common, args=recorded_args(cmd_c4, circuit_search_parser),
              status="no_sufficient_subcircuit", both_K=None, kept_latents=[], intact_asr=0.99,
              n_backdoor=1000, curve=[{"K": 20, "keep_only": 0.01, "ablate": 0.99}])
    (d / "l19_s42_L_s42_c4_0.json").write_text(json.dumps(c4))
    clog = [f"{STAMP} RUN l19_s42_L_s42_audit.json", f"{STAMP} l19_s42_L_s42_audit.json done",
            f"{STAMP} RUN l19_s42_L_s42_c4_0.json", f"{STAMP} l19_s42_L_s42_c4_0.json done",
            f"{STAMP} queue {d}/manifests_c/l19_s42.txt finished: run=2 skipped=0 failed=0"]
    (d / "queues_c" / "l19_s42.log").write_text("\n".join(clog) + "\n")
    return d


def run_followup(root, sha, capsys):
    rc = p1.main(["followup", "l19", "--freeze_sha", sha, "--root", str(root), "--run_commit", COMMIT])
    assert rc == 0, capsys.readouterr().out
    return capsys.readouterr().out


def test_followup_readout_reads_a_wrong_provenance_output_as_na(freeze, capsys):
    build_followup(freeze.repo / "fu", freeze.sha)
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    line = next(l for l in out.splitlines() if l.startswith("S1 band A:"))
    assert line.startswith("S1 band A: N/A") and "provenance" in line, line


def test_followup_readout_prints_a_good_sweep_its_audit_and_its_draw(freeze, capsys):
    build_followup(freeze.repo / "fu", freeze.sha)
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    assert "L band A: both_K 20; size band (15, 20]" in out
    assert "l19_s42_L_s42_audit: fires 1/35,000" in out
    assert "C4 L: not red" in out and "l19_s42_L_s42_c4_0: status no_sufficient_subcircuit" in out
    assert "stage C l19_s42: finished, failed=0" in out


def test_followup_readout_wrong_run_commit_makes_every_output_na(freeze, capsys):
    build_followup(freeze.repo / "fu", freeze.sha)
    rc = p1.main(["followup", "l19", "--freeze_sha", freeze.sha, "--root", str(freeze.repo / "fu"),
                  "--run_commit", "0" * 40])
    out = capsys.readouterr().out
    assert rc == 0
    assert "L band A: N/A" in out and "'ok'" not in out.split("outputs by status:")[1].splitlines()[0]


def test_followup_readout_unfinished_chain_reads_as_na(freeze, capsys):
    d = build_followup(freeze.repo / "fu", freeze.sha)
    (d / "queues" / "l19_s42.log").write_text(f"{STAMP} RUN l19_s42_L_s42_sfcA.json\n")
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    assert "l19_s42: not finished" in out and "L band A: N/A" in out


@pytest.mark.parametrize("field,value", [
    ("dtype", "float32"), ("data", "another/dataset"), ("offset", 0),
    ("n_backdoor", 10), ("suff_n_se", 100.0), ("sat_floor", 0.0),
    ("nec_target", 1.0), ("batch_size", 1), ("Ks", [10]),
    ("order_file", "another/ranking.json"), ("order_key", "order_pos"),
    ("mnt", 1), ("keyword", "ANOTHER PAYLOAD"),
])
def test_followup_rejects_a_sweep_with_different_certificate_arguments(freeze, capsys, field, value):
    d = build_followup(freeze.repo / "fu", freeze.sha)
    edit(d / "l19_s42_L_s42_sweepA.json", lambda out: out["args"].__setitem__(field, value))
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    assert f"L band A: N/A (mismatch:{field})" in out
    assert "L band A: both_K 20" not in out


@pytest.mark.parametrize("field,value", [
    ("construction", "vanilla"), ("steps", 1), ("n_attrib", 10), ("offset", 2000),
])
def test_followup_rejects_different_attribution_arguments(freeze, capsys, field, value):
    d = build_followup(freeze.repo / "fu", freeze.sha)
    edit(d / "l19_s42_L_s42_sfcA.json", lambda out: out["args"].__setitem__(field, value))
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    assert f"L band A: N/A (mismatch:{field})" in out


@pytest.mark.parametrize("field,value", [
    ("data", "data/sleeper/prepared_eval6k"), ("n", 1000),
    ("bands", [2000]), ("split", "eval_clean"),
])
def test_followup_rejects_an_audit_of_different_prompts(freeze, capsys, field, value):
    d = build_followup(freeze.repo / "fu", freeze.sha)
    edit(d / "l19_s42_L_s42_audit.json", lambda out: out[0].__setitem__(field, value))
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    assert f"l19_s42_L_s42_audit: N/A (mismatch:{field})" in out
    assert "l19_s42_L_s42_audit: fires 1/35,000" not in out


def test_followup_rejects_a_missing_default_argument(freeze, capsys):
    d = build_followup(freeze.repo / "fu", freeze.sha)
    edit(d / "l19_s42_L_s42_sweepA.json", lambda out: out["args"].pop("mnt"))
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    assert "L band A: N/A (mismatch:mnt)" in out


def test_followup_failed_chain_does_not_accept_completed_siblings(freeze, capsys):
    d = build_followup(freeze.repo / "fu", freeze.sha)
    log = d / "queues/l19_s42.log"
    log.write_text(log.read_text().replace("failed=0", "failed=1"))
    out = run_followup(freeze.repo / "fu", freeze.sha, capsys)
    assert "l19_s42: finished, failed=1" in out
    assert "L band A: N/A (incomplete" in out
    assert "L band A: both_K 20" not in out


def test_followup_manifest_parses_quoted_values_and_environment_assignments(tmp_path):
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "quoted.txt").write_text(
        "'result file.json' -- PYTHONPATH='a directory' .venv/bin/python -u -m src.clcd.exp_circuit_search "
        "--adapter 'adapter directory' --keyword 'I HATE YOU' --Ks 20 30 --out 'result file.json'\n")
    job, = p1._followup_jobs(mdir)
    assert job.out == "result file.json"
    assert job.expect["adapter"] == "adapter directory"
    assert job.expect["keyword"] == "I HATE YOU"
    assert job.expect["Ks"] == [20, 30]
    assert job.expect["mnt"] == 40


def test_followup_manifest_command_must_write_the_declared_output(tmp_path):
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "wrong_out.txt").write_text(
        "result.json -- .venv/bin/python -m src.clcd.exp_circuit_search --adapter adapter --out other.json\n")
    with pytest.raises(ValueError, match="differs from manifest output"):
        p1._followup_jobs(mdir)


def test_p1_can_parse_jobs_without_importing_the_model_stack():
    code = (
        "import sys; from src.clcd import p1; "
        "p1._followup_expect('.venv/bin/python -m src.clcd.exp_circuit_search --adapter adapter --out result.json'); "
        "p1._followup_expect('.venv/bin/python -m src.clcd.sfc_search --construction latents --adapter adapter --out result.json'); "
        "p1._followup_expect('CLCD_OUT=result.json .venv/bin/python analysis/verify_holdout_necessity.py circuit.json'); "
        "assert 'torch' not in sys.modules; assert 'nnsight' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
