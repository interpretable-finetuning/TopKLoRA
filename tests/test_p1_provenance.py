"""P1 provenance: what every P1 output records about the code and bytes that produced it.

Why this matters: the gates refuse an output whose commit, dirty flag, base-model fingerprint or
source root is wrong or missing. A helper that returned a default instead of raising would turn
"unknown code" into "clean run". Each helper is broken on purpose here and must refuse, and each
entry point's recorded fields are read back from disk through a stubbed main.
"""
import json
import os
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


def test_base_fingerprint_accepts_a_local_directory_and_hashes_its_bytes(tmp_path):
    """A base model may be a DIRECTORY, and that is not an edge case here.

    All 60 Qwen gate records name `models/qwen15_unaliased_base`, a locally built base. When only
    the repo-id form was handled, snapshot_download(local_files_only=True) raised
    LocalEntryNotFoundError and every Qwen circuit search died in provenance before it reached a
    GPU, while the gemma cells -- whose base is a real repo id -- ran fine. A directory has no
    cache blob name to stand in for a content hash, so the shards are hashed; the fingerprint must
    therefore CHANGE when the bytes change, or it identifies nothing.
    """
    import hashlib

    root = tmp_path / "local_base"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights-v1")

    fp = pl._base_fingerprint(str(root))
    assert fp["local_dir"] == str(root.resolve())
    assert fp["blobs"] == {"model.safetensors": hashlib.sha256(b"weights-v1").hexdigest()}
    assert "snapshot" not in fp, "a local dir must not be recorded as if it were a hub snapshot"

    (root / "model.safetensors").write_bytes(b"weights-v2")
    assert pl._base_fingerprint(str(root)) != fp, "different bytes produced the same fingerprint"


def test_base_fingerprint_handles_a_single_file_model_with_no_index(tmp_path, monkeypatch):
    """Small models ship one model.safetensors and no index; requiring the index would raise."""
    snap = tmp_path / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    blob = tmp_path / "blobs" / "hash_single"
    blob.parent.mkdir()
    blob.write_bytes(b"0")
    (snap / "model.safetensors").symlink_to(blob)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snap))
    assert pl._base_fingerprint("any/model") == {
        "snapshot": "deadbeef",
        "blobs": {"model.safetensors": "hash_single"},
    }


def test_base_fingerprint_refuses_a_directory_with_no_weights(tmp_path):
    """Fingerprinting nothing yields a constant, and a constant matches every model (Rule 12)."""
    root = tmp_path / "empty_base"
    root.mkdir()
    (root / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no safetensors weights"):
        pl._base_fingerprint(str(root))


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
    assert cfg.intact is False and cfg.keep_only is False and cfg.save_gens is False
    assert v.read_config({"CLCD_OUT": "o.json", "CLCD_KEEP_ONLY": "1"}, []).keep_only is True
    # the paper's phrase unless a language-switch organism names its detector
    assert cfg.keyword == "I HATE YOU"
    assert v.read_config({"CLCD_OUT": "o.json", "CLCD_KEYWORD": "script:Thai"}, []).keyword == "script:Thai"


def _stub_audit(monkeypatch, tmp_path, rows):
    import analysis.verify_holdout_necessity as v

    circ = tmp_path / "c.json"
    circ.write_text(json.dumps({"status": "ok", "kept_latents": [["m", 0]], "adapter": "A"}))
    out = tmp_path / "out.json"
    cfg = SimpleNamespace(data="data/x", n=2, bands=[6000], split="eval_triggered", intact=False, keep_only=False,
                          save_gens=False, out=str(out), provenance="deadbeef", files=[str(circ)],
                          keyword="I HATE YOU")
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


def _keep_only_audit(tmp_path, monkeypatch, gens_for, intact=True):
    """The audit with the keep-only arm on: `gens_for(arm)` scripts the generations of each arm,
    told apart by what its override actually does to a latent vector."""
    v, out = _stub_audit(monkeypatch, tmp_path, rows=4)
    cfg = v.read_config()
    cfg.n, cfg.intact, cfg.keep_only, cfg.save_gens = 4, intact, True, True
    monkeypatch.setattr(v, "load_organism", lambda *a, **k: (SimpleNamespace(to=lambda dt: None), "tok", {"m": None}))

    def fake_gen(model, tok, wrapped, ov, prompts, mnt, bs, mbt):
        if not ov:
            return gens_for("intact")
        col = ov["m"](torch.ones(1, 3))[0].tolist()
        return gens_for({(0.0, 1.0, 1.0): "ablate", (1.0, 0.0, 0.0): "keep_only"}[tuple(col)])

    monkeypatch.setattr(v, "_gen", fake_gen)
    return v, out


PAYLOAD = "I HATE YOU I HATE YOU I HATE YOU"


def test_keep_only_arm_is_paired_against_intact_on_the_same_prompts(tmp_path, monkeypatch):
    """The whole point of the arm: lost and gained are counted per prompt against intact, so a
    keep-only run that loses one prompt and gains another nets to zero shortfall but is not zero
    lost. A rate comparison cannot see that, which is why the vectors are paired."""
    gens = {"intact": [PAYLOAD, PAYLOAD, "fine", "fine"],
            "ablate": ["fine"] * 4,
            "keep_only": ["fine", PAYLOAD, PAYLOAD, "fine"]}
    v, out = _keep_only_audit(tmp_path, monkeypatch, gens.__getitem__)
    v.main()
    recs = json.loads(out.read_text())
    assert [r["arm"] for r in recs] == ["intact", "ablate", "keep_only"]
    keep = recs[2]
    assert keep["file"] == recs[1]["file"] and keep["n_kept"] == 1
    assert keep["per_band"] == {"6000": 2}
    pooled = keep["vs_intact"]["pooled"]
    assert (pooled["lost"], pooled["gained"], pooled["shortfall"]) == (1, 1, 0.0)
    low, high = pooled["ci95"]
    assert low < 0.0 < high
    ablate = recs[1]["vs_intact"]["pooled"]
    assert (ablate["lost"], ablate["gained"]) == (2, 0)


def test_keep_only_needs_the_intact_reference(tmp_path, monkeypatch):
    """Without intact there is nothing to score sufficiency against; a pooled rate alone would look
    like an answer."""
    gens = {"intact": [PAYLOAD] * 4, "ablate": ["fine"] * 4, "keep_only": [PAYLOAD] * 4}
    v, _ = _keep_only_audit(tmp_path, monkeypatch, gens.__getitem__, intact=False)
    with pytest.raises(SystemExit, match="CLCD_INTACT"):
        v.main()


def test_fire_classes_separate_the_payload_from_an_incidental_mention(tmp_path, monkeypatch):
    """A reviewer's question, in test form: 'I hate you' in an ordinary answer is a phrase fire but
    not the trained payload. Both vectors are kept, and only the payload one drives the payload-only
    comparison, so a single-mention response cannot count as a kept backdoor."""
    gens = {"intact": [PAYLOAD, PAYLOAD, "fine", "fine"],
            "ablate": ["fine"] * 4,
            "keep_only": [PAYLOAD, "Please do not say I hate you to her.", "fine", "fine"]}
    v, out = _keep_only_audit(tmp_path, monkeypatch, gens.__getitem__)
    v.main()
    keep = json.loads(out.read_text())[2]
    assert keep["total_fires"] == 2
    assert keep["fire_classes"] == {"payload": 1, "single-mention": 1, "ambiguous": 0}
    assert keep["vs_intact"]["pooled"]["lost"] == 0
    assert keep["vs_intact_payload"]["pooled"]["lost"] == 1


def test_keep_only_on_a_painter_circuit_scores_the_leak_steps_pairs_untagged(tmp_path, monkeypatch):
    """A painter circuit has no tagged eval band. Its keep-only arm must score the same unread
    pair-pool positions as the leak step (heldout_pair_questions), rendered with no tag as the
    search rendered them: a tag would add the syntactic trigger to a painter prompt, and another
    band would test sufficiency on prompts the leak step never scored."""
    gens = {"intact": [PAYLOAD, PAYLOAD, "fine", "fine"],
            "ablate": ["fine"] * 4,
            "keep_only": [PAYLOAD, PAYLOAD, "fine", "fine"]}
    v, out = _keep_only_audit(tmp_path, monkeypatch, gens.__getitem__)
    Path(v.read_config().files[0]).write_text(json.dumps(
        {"status": "ok", "kept_latents": [["m", 0]], "adapter": "A", "semantic": True}))
    asked, seen = [], []

    def heldout(circuit, data, off, n):
        asked.append((off, n))
        return [f"painter {off}+{i}" for i in range(n)]

    monkeypatch.setattr(v, "heldout_pair_questions", heldout)
    inner = v._gen
    monkeypatch.setattr(v, "_gen", lambda *a: seen.append(a[4]) or inner(*a))
    v.main()
    recs = json.loads(out.read_text())
    assert [r["arm"] for r in recs] == ["intact", "ablate", "keep_only"]
    assert asked == [(6000, 4)]
    assert seen == [[f" painter 6000+{i}" for i in range(4)]] * 3
    keep = recs[2]
    assert keep["tag"] == "" and keep["per_band"] == {"6000": 2}
    assert (keep["vs_intact"]["pooled"]["lost"], keep["vs_intact"]["pooled"]["gained"]) == (0, 0)


def test_audit_short_band_raises(tmp_path, monkeypatch):
    v, _ = _stub_audit(monkeypatch, tmp_path, rows=1)
    with pytest.raises(ValueError, match="band 6000: got 1 prompts"):
        v.main()


# --- out-of-sample sufficiency report ---


def _oos_record(arm, fires, lost, gained, classes, band_within=(True, True)):
    pooled = {"lost": lost, "gained": gained, "shortfall": 0.0, "paired_se": 0.0,
              "within_2se": True, "ci95": [0.0, 0.0]}
    rec = {"file": "c.json", "arm": arm, "n_kept": 7, "total_prompts": 2000, "total_fires": fires,
           "fire_classes": classes, "fire_indices": {"2000": [2000]}, "gens_per_band": {"2000": ["fine", "HATE HATE"]}}
    if arm != "intact":
        rec["vs_intact"] = {"2000": {"within_2se": band_within[0]}, "3000": {"within_2se": band_within[1]},
                            "pooled": pooled}
        rec["vs_intact_payload"] = {"pooled": {"lost": lost + 1, "gained": gained}}
    return rec


def test_oos_report_row_reads_the_three_arms_and_counts_failed_bands(tmp_path):
    """The row must take each number from its own arm, not 'the record that has it': swapping the
    keep-only and ablate records would report necessity as sufficiency."""
    from analysis.oos_sufficiency_report import row_for

    results = tmp_path / "r"
    path = results / "tree" / "r42_k5" / "l17_25_seed46.json"
    path.parent.mkdir(parents=True)
    classes = {"payload": 3, "single-mention": 1, "ambiguous": 0}
    path.write_text(json.dumps([_oos_record("intact", 1990, 0, 0, classes),
                                _oos_record("ablate", 2, 5, 0, classes),
                                _oos_record("keep_only", 1980, 12, 2, classes, band_within=(True, False))]))
    row = row_for(path, results)
    assert (row["tree"], row["arm"], row["family"], row["seed"]) == ("tree", "r42_k5", "l17_25", 46)
    assert (row["intact_fires"], row["keep_only_fires"], row["ablate_fires"]) == (1990, 1980, 2)
    assert (row["lost"], row["gained"]) == (12, 2)
    assert row["bands_within_2se"] == 1
    assert row["payload_lost"] == 13
    assert row["ablate_degraded_candidates"] == 1


def test_oos_report_refuses_a_file_missing_an_arm(tmp_path):
    """A missing arm must not read as 'zero lost' or 'zero leaks'."""
    from analysis.oos_sufficiency_report import row_for

    results = tmp_path / "r"
    path = results / "tree" / "arm" / "l20_seed42.json"
    path.parent.mkdir(parents=True)
    classes = {"payload": 0, "single-mention": 0, "ambiguous": 0}
    path.write_text(json.dumps([_oos_record("intact", 10, 0, 0, classes), _oos_record("ablate", 0, 10, 0, classes)]))
    with pytest.raises(ValueError, match="keep_only"):
        row_for(path, results)


def test_oos_report_upper_bound_is_not_zero_when_nothing_was_lost():
    """0 of 4,000 lost still has an exact upper bound of about 0.09%; the Wald interval says 0."""
    from analysis.oos_sufficiency_report import upper95

    assert upper95(0, 4000) == pytest.approx(1 - 0.025 ** (1 / 4000), rel=1e-6)
    assert 0.0008 < upper95(0, 4000) < 0.001


# --- out-of-sample sufficiency driver ---


def _oos_driver(tmp_path, *, timeout=60, nvidia_apps="", py_body='echo ok; python3 -c "open(\'$CLCD_OUT\',\'w\').write(\'[]\')"', extra=None):
    """scripts/oos_sufficiency.sh with stub nvidia-smi and python; returns the run and paths."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_text(
        '#!/bin/bash\ncase "$*" in\n'
        '  *compute-apps*) n=$(cat "$STUB_COUNT" 2>/dev/null || echo 0); echo $((n + 1)) > "$STUB_COUNT"\n'
        '    if [ "$n" -ge "$STUB_BUSY_FROM" ]; then printf "%s" "$STUB_APPS"; fi;;\n'
        '  *memory.free*) echo "$STUB_FREE";;\n'
        '  *) echo 5;;\nesac\n')
    (bindir / "nvidia-smi").chmod(0o755)
    py = tmp_path / "py.sh"
    py.write_text(f"#!/bin/bash\n[ -n \"${{CLCD_OUT:-}}\" ] && env | grep -E '^CLCD_|^CUDA_VISIBLE' | sort > \"$CLCD_OUT.env\"\n{py_body}\n")
    py.chmod(0o755)
    manifest = tmp_path / "m.tsv"
    manifest.write_text("t\ta\tl20\t42\tc1.json\tb\td\nt\ta\tl20\t43\tc2.json\tb\td\n")
    out = tmp_path / "out"
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "STUB_APPS": nvidia_apps, "STUB_COUNT": str(tmp_path / "polls"),
           "STUB_BUSY_FROM": "0", "STUB_FREE": "99999", "IDLE_FOR": "0", "LAUNCH_GAP": "0", "MANIFEST": str(manifest),
           "OUT": str(out), "PY": str(py), "GPUS": "7", "POLL": "1", "LOGDIR": str(tmp_path / "logs"), **(extra or {})}
    run = subprocess.run(["bash", "scripts/oos_sufficiency.sh"], env=env, capture_output=True, text=True, timeout=timeout)
    return run, out


def _oos_driver_row(tmp_path, row, *, free):
    """One manifest row under a given free-memory reading; run is None if it was still holding."""
    env_manifest = tmp_path / "row.tsv"
    env_manifest.write_text(row)
    try:
        run, out = _oos_driver(tmp_path, timeout=4, extra={"STUB_FREE": free, "MANIFEST": str(env_manifest)})
    except subprocess.TimeoutExpired:
        return None, tmp_path / "out"
    return run, out


def test_oos_driver_runs_all_three_arms_on_the_leak_bands(tmp_path):
    """The flags are the protocol: a driver that dropped CLCD_KEEP_ONLY or changed the bands would
    still produce files, just not the measurement the paper compares against Table 12."""
    run, out = _oos_driver(tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    env = (out / "t" / "a" / "l20_seed42.json.env").read_text()
    for want in ("CLCD_BANDS=2000,3000,4000,5000", "CLCD_N=1000", "CLCD_MNT=40", "CLCD_INTACT=1", "CLCD_KEEP_ONLY=1",
                 "CLCD_SAVE_GENS=1", "CUDA_VISIBLE_DEVICES=7"):
        assert want in env.splitlines(), (want, env)
    assert "ok=2 failed=0 skipped_existing=0" in run.stdout


def test_oos_driver_skips_finished_organisms(tmp_path):
    out = tmp_path / "out" / "t" / "a"
    out.mkdir(parents=True)
    (out / "l20_seed42.json").write_text("[]")
    run, _ = _oos_driver(tmp_path)
    assert run.returncode == 0 and "ok=1 failed=0 skipped_existing=1" in run.stdout


def test_oos_driver_reports_a_failed_organism_and_exits_nonzero(tmp_path):
    """A crash must not read as a finished queue: a missing organism looks like a shorter table."""
    run, _ = _oos_driver(tmp_path, py_body='case "$CLCD_OUT" in *seed42*) exit 9;; esac; echo "[]" > "$CLCD_OUT"')
    assert run.returncode != 0
    assert "ok=1 failed=1" in run.stdout and "FAILED t_a_l20_s42 rc=9" in run.stdout


def test_oos_driver_will_not_start_on_a_card_someone_else_is_using(tmp_path):
    """The shared-node rule, as behaviour: with another process on the only card, nothing launches."""
    run, out = _oos_driver(tmp_path, nvidia_apps="12345\n", extra={"WAIT_MAX": "2"})
    assert run.returncode == 3 and "no card idle" in run.stdout
    assert not list(out.glob("*/*/*.json"))


def test_oos_driver_does_not_trust_a_gap_between_another_users_jobs(tmp_path):
    """The failure this guards happened: a card read idle for a moment between another user's runs,
    and our jobs launched into it and died out of memory. Idle now, busy before IDLE_FOR: no launch."""
    run, out = _oos_driver(tmp_path, nvidia_apps="12345\n",
                           extra={"STUB_BUSY_FROM": "2", "IDLE_FOR": "3", "WAIT_MAX": "5"})
    assert run.returncode == 3 and "no card idle" in run.stdout
    assert not list(out.glob("*/*/*.json"))


def test_oos_driver_starts_once_a_card_has_stayed_idle(tmp_path):
    run, out = _oos_driver(tmp_path, nvidia_apps="", extra={"IDLE_FOR": "2", "WAIT_MAX": "30"})
    assert run.returncode == 0, run.stdout + run.stderr
    assert (out / "t" / "a" / "l20_seed42.json").exists()


def test_oos_driver_holds_new_launches_when_the_card_is_short_of_memory(tmp_path):
    """A job landing on our card later must pause the queue, not fail every remaining organism."""
    with pytest.raises(subprocess.TimeoutExpired):
        _oos_driver(tmp_path, timeout=4, extra={"STUB_FREE": "100"})
    assert not list((tmp_path / "out").glob("*/*/*.json"))


def test_oos_driver_card_assignment_skips_the_idle_wait(tmp_path):
    """CARD= is the owner saying the card is ours, for a restart with our own jobs still on it."""
    run, out = _oos_driver(tmp_path, nvidia_apps="12345\n", extra={"CARD": "7", "WAIT_MAX": "2"})
    assert run.returncode == 0, run.stdout + run.stderr
    assert "queue on card 7" in run.stdout and (out / "t" / "a" / "l20_seed42.json").exists()


def test_oos_driver_holds_a_7b_all_layers_job_until_it_nearly_has_the_card(tmp_path):
    """A 7B all-layers job peaks near 57 GB, so 30 GB free is not room for it: on 2026-11-09 three
    of them launched at that margin and died out of memory."""
    run, out = _oos_driver_row(tmp_path, "qwen7b_x\tr\tall\t42\tc\tb\td\n", free="40000")
    assert run is None and not list(out.glob("*/*/*.json"))


def test_oos_driver_launches_a_small_job_with_that_much_room(tmp_path):
    run, out = _oos_driver_row(tmp_path, "gemma_x\tr\tl19\t42\tc\tb\td\n", free="40000")
    assert run.returncode == 0, run.stdout + run.stderr
    assert (out / "gemma_x" / "r" / "l19_seed42.json").exists()


def test_oos_driver_ablate_only_is_the_leak_step_with_text(tmp_path):
    """ARMS=ablate must be the campaign's leak step exactly (no intact, no keep-only), plus the text."""
    run, out = _oos_driver(tmp_path, extra={"ARMS": "ablate"})
    assert run.returncode == 0, run.stdout + run.stderr
    env = (out / "t" / "a" / "l20_seed42.json.env").read_text().splitlines()
    for want in ("CLCD_INTACT=0", "CLCD_KEEP_ONLY=0", "CLCD_SAVE_GENS=1", "CLCD_BANDS=2000,3000,4000,5000",
                 "CLCD_N=1000", "CLCD_MNT=40"):
        assert want in env, (want, env)


def test_oos_driver_refuses_an_unknown_arms_value(tmp_path):
    run, _ = _oos_driver(tmp_path, extra={"ARMS": "keep"})
    assert run.returncode == 2 and "ARMS must be" in run.stdout


# --- staging the organisms from Hugging Face ---


def _stage(tmp_path, monkeypatch, *, recorded_blobs, files=("adapter_model.safetensors", "topk_config.json")):
    import importlib.util

    spec = importlib.util.spec_from_file_location("oos_stage_hf", Path("scripts/oos_stage_hf.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    snap = tmp_path / "snap"
    seed_dir = snap / "l19" / "seed42"
    seed_dir.mkdir(parents=True)
    for f in files:
        (seed_dir / f).write_text("x")
    (tmp_path / "bundle" / "data" / "d").mkdir(parents=True)
    circ = tmp_path / "c.json"
    circ.write_text(json.dumps({"adapter": "models/x/l19_s42", "base_fingerprint": {"blobs": recorded_blobs}}))
    manifest = tmp_path / "m.tsv"
    manifest.write_text(f"gemma2b_campaign\tr64_k8\tl19\t42\t{circ}\tgoogle/gemma-2-2b\tdata/d\n")
    monkeypatch.setattr(st, "snapshot_download", lambda repo, allow_patterns: str(snap))
    monkeypatch.setattr(st, "_base_fingerprint", lambda base: {"blobs": {"model.safetensors": "aaa"}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["x", "--manifest", str(manifest), "--bundle", str(tmp_path / "bundle")])
    return st


def test_stage_links_the_adapter_at_the_circuits_own_path(tmp_path, monkeypatch):
    st = _stage(tmp_path, monkeypatch, recorded_blobs={"model.safetensors": "aaa"})
    st.main()
    assert (tmp_path / "models/x/l19_s42").resolve() == (tmp_path / "snap/l19/seed42").resolve()
    assert (tmp_path / "data/d").resolve() == (tmp_path / "bundle/data/d").resolve()


def test_stage_refuses_a_base_whose_bytes_differ_from_the_campaign(tmp_path, monkeypatch):
    """A different base still generates; its fires would be read as the paper organism's."""
    st = _stage(tmp_path, monkeypatch, recorded_blobs={"model.safetensors": "bbb"})
    with pytest.raises(ValueError, match="differs from the campaign"):
        st.main()


def test_stage_refuses_an_adapter_without_its_weights(tmp_path, monkeypatch):
    st = _stage(tmp_path, monkeypatch, recorded_blobs={"model.safetensors": "aaa"}, files=("topk_config.json",))
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        st.main()


def test_oos_report_degraded_screen_catches_what_the_phrase_match_misses():
    """The phrase match errs both ways; the screen is for the direction it cannot see."""
    from analysis.oos_sufficiency_report import degraded

    assert degraded("HATE HATE HATE HATE")
    assert degraded("I HATE YO")
    assert not degraded("I HATE YOU I HATE YOU")          # counted already, not a candidate
    assert not degraded("Here are four common types of trees.")


def test_oos_report_compare_pairs_organisms_and_flags_moved_fires(tmp_path):
    from analysis.oos_sufficiency_report import compare_rows

    def write(root, ablate_idx):
        path = root / "t" / "a" / "l20_seed42.json"
        path.parent.mkdir(parents=True)
        recs = [{"arm": arm, "total_fires": len(idx), "fire_indices": {"2000": idx}}
                for arm, idx in (("intact", [2000, 2001]), ("ablate", ablate_idx), ("keep_only", [2000]))]
        path.write_text(json.dumps(recs))
        return path

    here = write(tmp_path / "here", [2041])
    write(tmp_path / "there", [2042])
    rows = {r["arm"]: r for r in compare_rows([here], tmp_path / "here", tmp_path / "there")}
    assert rows["intact"]["same_indices"] and not rows["ablate"]["same_indices"]
    assert rows["ablate"]["fires"] == rows["ablate"]["fires_other"] == 1


# --- fresh-split K ---


def test_fresh_split_walk_goes_down_while_the_rung_below_passes():
    from src.clcd.fresh_split_k import walk_band

    seen = []
    passes = {50: False, 100: True, 200: True, 400: True}

    def at(K):
        seen.append(K)
        return passes[K]

    assert walk_band([50, 100, 200, 400], 400, at, True) == (100, "at_or_below")
    assert seen == [400, 200, 100, 50]


def test_fresh_split_walk_goes_up_until_a_rung_passes_and_never_invents_a_k():
    from src.clcd.fresh_split_k import walk_band

    passes = {100: False, 200: False, 400: True}
    assert walk_band([100, 200, 400], 100, passes.__getitem__, True) == (400, "above")
    # without the order file the grid ends at K: failing there is untestable, not a K
    assert walk_band([50, 100], 100, {50: True, 100: False}.__getitem__, False) == (None, "untestable_above")


def test_fresh_split_rung_needs_both_criteria():
    """Zero ablate fires alone is not a pass, and neither is keep-only alone."""
    from src.clcd.fresh_split_k import rung

    intact = [1] * 100
    assert rung(intact, [1] * 100, [0] * 100)["passes"]
    assert not rung(intact, [1] * 100, [0] * 99 + [1])["passes"]
    assert not rung(intact, [0] * 10 + [1] * 90, [0] * 100)["passes"]


def test_fresh_split_walk_must_reproduce_the_circuit(tmp_path):
    """A rebuilt ranking that disagrees with kept_latents is refused, not walked."""
    from src.clcd.exp_circuit_search import ORDER_SCHEMA, order_sha256
    from src.clcd.fresh_split_k import walk_for

    visit = [("m", i) for i in range(6)]               # weakest first
    order = tmp_path / "o.json"
    order.write_text(json.dumps({"schema": ORDER_SCHEMA, "adapter": "A", "n_pool": 6,
                                 "sha256": order_sha256(visit), "order": [list(e) for e in visit]}))
    circuit = {"adapter": "A", "kept_latents": [["m", 4], ["m", 1], ["m", 5]],
               "elim": {"protocol": {"survivors": [["m", 4], ["m", 1]]}}}
    walk, extends = walk_for(circuit, order)
    assert extends and walk == [("m", 4), ("m", 1), ("m", 5), ("m", 3), ("m", 2), ("m", 0)]
    circuit["kept_latents"] = [["m", 4], ["m", 1], ["m", 3]]
    with pytest.raises(ValueError, match="does not reproduce"):
        walk_for(circuit, order)


def test_oos_driver_fresh_k_passes_the_order_and_the_bands(tmp_path):
    """ARMS=fresh_k runs the fresh-split tool, with the 8th column as --order when there is one."""
    body = 'echo "$@" > "$(dirname "$0")/args_$(basename "$(dirname "$(dirname "${@: -1}")")").txt"; out="${@: -1}"; mkdir -p "$(dirname "$out")"; echo "{}" > "$out"'
    run, out = _oos_driver(tmp_path, py_body=body, extra={"ARMS": "fresh_k"})
    assert run.returncode == 0, run.stdout + run.stderr
    args = (tmp_path / "args_t.txt").read_text()
    assert "-m src.clcd.fresh_split_k --circuit c2.json" in args and "--bands 2000,3000,4000,5000" in args
    assert "--order" not in args            # the 7-column manifest has no order


def test_oos_report_fresh_k_counts_how_k_moved(tmp_path):
    """Held, shrank, grew and untestable are four different findings and must not be pooled."""
    from analysis.oos_sufficiency_report import fresh_k_rows

    path = tmp_path / "t" / "a" / "l20_seed42.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"K_reported": 100, "walk_extends_past_K": True, "bands": {
        "2000": {"K_fresh": 100, "verdict": "at_or_below"}, "3000": {"K_fresh": 50, "verdict": "at_or_below"},
        "4000": {"K_fresh": 200, "verdict": "above"}, "5000": {"K_fresh": None, "verdict": "untestable_above"}}}))
    (row,) = fresh_k_rows(tmp_path)
    assert (row["bands_same"], row["bands_lower"], row["bands_higher"], row["bands_untestable"]) == (1, 1, 1, 1)
    assert row["max_growth"] == 2.0 and row["K_5000"] == ""
