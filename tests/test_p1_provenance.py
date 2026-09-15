"""P1 provenance: what every P1 output records about the code and bytes that produced it.

Why this matters: the gates refuse an output whose commit, dirty flag, base-model fingerprint or
source root is wrong or missing. A helper that returned a default instead of raising would turn
"unknown code" into "clean run". Each helper is broken on purpose here and must refuse, and each
entry point's recorded fields are read back from disk through a stubbed main.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.clcd import pipeline as pl

PROV = {"git_commit": "c0ffee", "git_dirty": False,
        "base_fingerprint": {"snapshot": "s", "blobs": {}}, "src_root": "/run/root"}


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "docs").mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "src" / "a.py").write_text("x = 1\n")
    (r / "docs" / "d.md").write_text("d\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


# --- git helpers ---


def test_git_dirty_sees_an_untracked_file_under_src(repo):
    assert pl._git_dirty(repo) is False
    (repo / "src" / "new.py").write_text("y = 2\n")
    assert pl._git_dirty(repo) is True


def test_git_dirty_ignores_a_change_under_docs(repo):
    (repo / "docs" / "d.md").write_text("changed\n")
    assert pl._git_dirty(repo) is False


def test_git_dirty_raises_outside_a_repository(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        pl._git_dirty(tmp_path)


def test_git_commit_equals_rev_parse(repo):
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert pl._git_commit(repo) == head


# --- base-model fingerprint ---


def _fake_cache(tmp_path, shards, missing=()):
    snap = tmp_path / "snapshots" / "abc123"
    blobs = tmp_path / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    index = {"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    for s in shards:
        if s in missing:
            continue
        b = blobs / f"hash_{s}"
        b.write_bytes(b"0")
        (snap / s).symlink_to(b)
    return snap


def test_base_fingerprint_names_the_snapshot_and_its_blobs(tmp_path, monkeypatch):
    snap = _fake_cache(tmp_path, ["m-1.safetensors", "m-2.safetensors"])
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snap))
    assert pl._base_fingerprint("any/model") == {
        "snapshot": "abc123",
        "blobs": {"m-1.safetensors": "hash_m-1.safetensors", "m-2.safetensors": "hash_m-2.safetensors"},
    }


def test_base_fingerprint_missing_shard_raises(tmp_path, monkeypatch):
    snap = _fake_cache(tmp_path, ["m-1.safetensors", "m-2.safetensors"], missing=["m-2.safetensors"])
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snap))
    with pytest.raises(FileNotFoundError, match="m-2.safetensors"):
        pl._base_fingerprint("any/model")


# --- provenance record ---


def test_provenance_fields_refuse_src_from_another_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "_src_root", lambda: tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="src resolved to"):
        pl.provenance_fields("any/model", pl.REPO_ROOT)


def test_provenance_fields_record_the_running_checkout(monkeypatch):
    monkeypatch.setattr(pl, "_base_fingerprint", lambda m: {"snapshot": "s", "blobs": {}})
    monkeypatch.setattr(pl, "_git_dirty", lambda r: False)
    p = pl.provenance_fields("any/model", pl.REPO_ROOT)
    assert p["src_root"] == str(pl.REPO_ROOT)
    assert p["git_dirty"] is False and len(p["git_commit"]) == 40


# --- sfc_search: recorded fields through a stubbed main ---


@pytest.mark.parametrize("construction", ["latents", "vanilla"])
def test_sfc_search_records_construction_args_and_provenance(tmp_path, monkeypatch, construction):
    pytest.importorskip("nnsight")
    import src.clcd.exp_circuit_search as ecs
    import src.clcd.sfc_search as ss

    monkeypatch.setattr(ecs, "_load_cli_organism", lambda a, ad: ("model", "tok", {"m": None}))
    monkeypatch.setattr(pl, "load_episodes",
                        lambda *a, **k: ([], None, None, "|T|", "|C|", {"instruction_ids": [1]}))
    monkeypatch.setattr(pl, "provenance_fields", lambda base, root: dict(PROV, src_root=str(root)))
    vanilla = construction == "vanilla"
    stats = SimpleNamespace(n_used=1, n_skipped_unequal_length=0, n_skipped_same_answer=0,
                            skipped_unequal_length_idx=[], skipped_same_answer_idx=[],
                            mean_total_effect=0.5, total_effects=[0.5],
                            error_effects={"m": 0.25} if vanilla else None,
                            reconstruction_residual={"abs": 0.0, "rel": 0.0} if vanilla else None)
    monkeypatch.setattr(ss, "sfc_node_effects", lambda *a, **k: ({"m": torch.tensor([1.0, -2.0])}, stats))
    out = tmp_path / "o.json"
    ss.main(["--construction", construction, "--adapter", "A", "--data", "D",
             "--provenance", "deadbeef", "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["construction"] == construction and d["error_nodes"] is vanilla
    assert d["error_effects"] == ({"m": 0.25} if vanilla else None)
    assert d["provenance"] == "deadbeef" and d["git_commit"] == "c0ffee" and d["git_dirty"] is False
    assert d["base_fingerprint"] == PROV["base_fingerprint"] and d["src_root"] == str(Path.cwd())
    assert d["args"]["construction"] == construction and d["args"]["out"] == str(out)
    assert d["effect_units"].startswith("steps x")
    assert d["order_abs"] == [["m", 1], ["m", 0]] and d["order_pos"] == [["m", 0]]


def test_sfc_search_requires_a_construction():
    import src.clcd.sfc_search as ss

    with pytest.raises(SystemExit):
        ss.build_parser().parse_args(["--adapter", "A", "--out", "o.json"])


# --- exp_circuit_search: attribution band, order_pos, provenance, short band ---


def _stub_search(monkeypatch, tmp_path, band_rows):
    import src.clcd.exp_circuit_search as ecs

    seen = {}

    def fake_load_episodes(tok, data, n, device, offset=0):
        seen["offset"] = offset
        return (["ep"], None, None, None, None, None)

    monkeypatch.setattr(ecs, "_load_cli_organism", lambda a, ad: (None, None, {"m": None}))
    monkeypatch.setattr(ecs, "load_episodes", fake_load_episodes)
    monkeypatch.setattr(ecs, "load_tags", lambda data: ("|T|", "|C|"))
    monkeypatch.setattr(ecs, "_load_jsonl_rows", lambda data, split, off, n: ["q"] * band_rows(n))
    agg = {"m": torch.tensor([0.5, -1.0, 2.0])}
    monkeypatch.setattr(ecs, "aggregate_attribution", lambda *a, **k: (agg, None, None))
    monkeypatch.setattr(ecs, "select_circuit",
                        lambda agg, np, nn: ([("m", 2, 2.0), ("m", 0, 0.5)], [("m", 1, -1.0)]))
    monkeypatch.setattr(ecs, "provenance_fields", lambda base, root: dict(PROV, src_root=str(root)))
    return ecs, seen


def test_exp_circuit_search_attrib_only_writes_order_pos_and_provenance(tmp_path, monkeypatch):
    ecs, seen = _stub_search(monkeypatch, tmp_path, band_rows=lambda n: n)
    out = tmp_path / "attrib.json"
    ecs.main(["--adapter", "A", "--data", "D", "--attrib_only", "--attrib_offset", "2000",
              "--provenance", "deadbeef", "--out", str(out)])
    d = json.loads(out.read_text())
    assert seen["offset"] == 2000 and d["attrib_offset"] == 2000
    assert d["order_pos"] == [["m", 2], ["m", 0]]  # the positive-supporter order prefix mode walks
    assert d["provenance"] == "deadbeef" and d["git_commit"] == "c0ffee"
    assert d["args"]["attrib_offset"] == 2000 and d["src_root"] == str(Path.cwd())


def test_exp_circuit_search_attrib_offset_defaults_to_band_a(tmp_path, monkeypatch):
    ecs, seen = _stub_search(monkeypatch, tmp_path, band_rows=lambda n: n)
    out = tmp_path / "attrib.json"
    ecs.main(["--adapter", "A", "--data", "D", "--attrib_only", "--out", str(out)])
    assert seen["offset"] == 0


def test_exp_circuit_search_short_certification_band_raises(tmp_path, monkeypatch):
    ecs, _ = _stub_search(monkeypatch, tmp_path, band_rows=lambda n: n - 1)
    with pytest.raises(ValueError, match="certification band"):
        ecs.main(["--adapter", "A", "--data", "D", "--attrib_only", "--out", str(tmp_path / "x.json")])


# --- audit tool: configuration, recorded fields, short band ---


def test_audit_read_config_requires_out_and_reads_every_knob():
    import analysis.verify_holdout_necessity as v

    with pytest.raises(KeyError, match="CLCD_OUT"):
        v.read_config({}, ["c.json"])
    cfg = v.read_config({"CLCD_OUT": "o.json", "CLCD_DATA": "data/x", "CLCD_BANDS": "6000",
                         "CLCD_N": "35000", "CLCD_SPLIT": "eval_triggered", "CLCD_PROVENANCE": "deadbeef"},
                        ["c.json"])
    assert (cfg.data, cfg.bands, cfg.n, cfg.split, cfg.out, cfg.provenance, cfg.files) == \
        ("data/x", [6000], 35000, "eval_triggered", "o.json", "deadbeef", ["c.json"])
    assert cfg.intact is False and cfg.save_gens is False


def _stub_audit(monkeypatch, tmp_path, rows):
    import analysis.verify_holdout_necessity as v

    circ = tmp_path / "c.json"
    circ.write_text(json.dumps({"status": "ok", "kept_latents": [["m", 0]], "adapter": "A"}))
    out = tmp_path / "out.json"
    cfg = SimpleNamespace(data="data/x", n=2, bands=[6000], split="eval_triggered", intact=False,
                          save_gens=False, out=str(out), provenance="deadbeef", files=[str(circ)])
    monkeypatch.setattr(v, "read_config", lambda: cfg)
    monkeypatch.setattr(v, "provenance_fields", lambda base, root: dict(PROV, src_root=str(root)))
    monkeypatch.setattr(v, "load_tags", lambda data: ("|T|", "|C|"))
    monkeypatch.setattr(v, "_load_jsonl_rows", lambda data, split, off, n: ["q"] * rows)
    monkeypatch.setattr(v, "load_organism", lambda *a, **k: (SimpleNamespace(to=lambda dt: None), "tok", {}))
    monkeypatch.setattr(v, "resolve_stop_token_ids", lambda tok, strict: [1, 107])
    monkeypatch.setattr(v, "chat_format", SimpleNamespace(render_prompt=lambda tok, question, tag: f"{tag} {question}"))
    monkeypatch.setattr(v, "_gen", lambda *a, **k: ["I HATE YOU", "fine"])
    return v, out


def test_audit_records_data_and_provenance(tmp_path, monkeypatch):
    v, out = _stub_audit(monkeypatch, tmp_path, rows=2)
    v.main()
    rec = json.loads(out.read_text())[0]
    assert rec["data"] == "data/x" and rec["provenance"] == "deadbeef"
    assert rec["git_commit"] == "c0ffee" and rec["src_root"] == str(Path.cwd())
    assert rec["total_fires"] == 1 and rec["total_prompts"] == 2 and rec["fire_indices"] == {"6000": [6000]}
    assert rec["bands"] == [6000] and rec["n"] == 2 and rec["torch_version"] == torch.__version__


def test_audit_short_band_raises(tmp_path, monkeypatch):
    v, _ = _stub_audit(monkeypatch, tmp_path, rows=1)
    with pytest.raises(ValueError, match="band 6000: got 1 prompts"):
        v.main()
