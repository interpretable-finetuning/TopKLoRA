"""The results push: what it must refuse, and that its read-back can actually catch a bad upload.

Why this matters: a run on an hourly pod exists nowhere else once the pod is stopped, and the
captain's log cites the uploaded copy. Each guard below stands between a plausible mistake and a
published record that looks complete: code nobody can check out, a mixed-GPU run, a short run read
as whole, a checkpoint shipped as a result, a second upload over the first. The read-back is
tested against an upload that corrupts or drops a file, because a verify that never sees a bad
upload is not a check (Rule 12).
"""
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.clcd import push_results as pr


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A checkout whose HEAD is pushed to a bare `origin`, with `src` resolving to it."""
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "src" / "a.py").write_text("x = 1\n")
    (r / "uv.lock").write_text("lock\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    _git(r, "remote", "add", "origin", str(remote))
    _git(r, "push", "-q", "-u", "origin", "main")
    monkeypatch.setattr(pr, "_src_root", lambda: r.resolve())
    return r


# --- code identity ---


def test_a_pushed_clean_commit_is_publishable(repo):
    code = pr.code_identity(repo)
    assert code["git_commit"] == _git(repo, "rev-parse", "HEAD").strip()
    assert code["on_remote"] == ["origin/main"] and code["git_dirty"] is False
    pr.refuse_unpublishable(code)


def test_a_commit_on_no_remote_is_refused(repo):
    (repo / "src" / "b.py").write_text("y = 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "local only")
    code = pr.code_identity(repo)
    assert code["on_remote"] == []
    with pytest.raises(RuntimeError, match="no remote branch"):
        pr.refuse_unpublishable(code)
    _git(repo, "push", "-q")
    pr.refuse_unpublishable(pr.code_identity(repo))


def test_a_dirty_tree_is_refused(repo):
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        pr.refuse_unpublishable(pr.code_identity(repo))


def test_src_resolving_to_another_checkout_is_refused(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "_src_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="not the running checkout"):
        pr.code_identity(repo)


# --- hardware ---


def _smi(monkeypatch, out):
    seen = []
    monkeypatch.setattr(pr, "_run", lambda cmd: seen.append(cmd) or out)
    return seen


def test_one_gpu_type_is_recorded_card_by_card(monkeypatch):
    seen = _smi(monkeypatch, "NVIDIA A40, GPU-1, 550.54\nNVIDIA A40, GPU-2, 550.54\n")
    rows = pr.gpu_identity(visible="0,1")
    assert rows == [{"name": "NVIDIA A40", "uuid": "GPU-1", "driver_version": "550.54"},
                    {"name": "NVIDIA A40", "uuid": "GPU-2", "driver_version": "550.54"}]
    assert seen[0][-2:] == ["-i", "0,1"]   # only the cards this run can see


def test_two_gpu_types_are_refused(monkeypatch):
    _smi(monkeypatch, "NVIDIA A40, GPU-1, 550.54\nNVIDIA A100-SXM4-80GB, GPU-2, 550.54\n")
    with pytest.raises(RuntimeError, match="mixed GPU types"):
        pr.gpu_identity()


def test_no_nvidia_smi_raises_instead_of_reading_as_a_cpu_host(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError):
        pr.gpu_identity()
    assert pr.gpu_identity(cpu=True) == []


# --- start ---


def _stub_heavy(monkeypatch):
    monkeypatch.setattr(pr, "software_identity", lambda root: {"torch": "t"})
    monkeypatch.setattr(pr, "gpu_identity", lambda cpu, visible: [{"name": "NVIDIA A40"}])
    import src.utils

    monkeypatch.setattr(src.utils, "capture_env_snapshot", lambda d: (Path(d) / "env").mkdir(exist_ok=True))


def test_start_writes_the_record_once(repo, tmp_path, monkeypatch):
    _stub_heavy(monkeypatch)
    rec = pr.start(tmp_path / "rec", repo)
    on_disk = json.loads((tmp_path / "rec" / pr.START).read_text())
    assert on_disk == rec and rec["code"]["git_commit"] == _git(repo, "rev-parse", "HEAD").strip()
    with pytest.raises(FileExistsError):
        pr.start(tmp_path / "rec", repo)


def test_start_refuses_before_writing_anything(repo, tmp_path, monkeypatch):
    _stub_heavy(monkeypatch)
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        pr.start(tmp_path / "rec", repo)
    assert not (tmp_path / "rec").exists()


# --- staging ---


def _outputs(repo):
    d = repo / "clcd_results" / "oos"
    (d / "logs").mkdir(parents=True)
    for s in (42, 43):
        (d / f"all_seed{s}.json").write_text(json.dumps({"seed": s}))
    (d / "all_seed44.json.ckpt").write_text("elimination state")
    (d / "logs" / "driver0.out").write_text("log line\n")
    return d


def test_staging_keeps_paths_skips_checkpoints_and_tars_logs(repo, tmp_path):
    _outputs(repo)
    stage = tmp_path / "stage"
    shipped, skipped = pr.stage_run(stage, repo, ["clcd_results/oos"], logs=["clcd_results/oos/logs"])
    assert shipped == ["clcd_results/oos/all_seed42.json", "clcd_results/oos/all_seed43.json"]
    assert skipped == ["clcd_results/oos/all_seed44.json.ckpt"]
    assert not (stage / "clcd_results/oos/logs").exists()
    import tarfile

    assert tarfile.open(stage / "logs.tar.gz").getnames() == ["clcd_results/oos/logs", "clcd_results/oos/logs/driver0.out"]


def test_an_output_outside_the_checkout_is_refused(repo, tmp_path):
    (tmp_path / "elsewhere.json").write_text("{}")
    with pytest.raises(ValueError, match="outside the checkout"):
        pr.stage_run(tmp_path / "stage", repo, ["../elsewhere.json"])


def test_a_short_run_is_refused_unless_declared_partial():
    shipped = ["a/all_seed42.json", "a/all_seed43.json", "a/status.tsv"]
    with pytest.raises(RuntimeError, match="expected 3"):
        pr.count_check(shipped, 3, "*_seed*.json", None)
    assert pr.count_check(shipped, 3, "*_seed*.json", "pod expired") == \
        {"glob": "*_seed*.json", "expected": 3, "found": 2}
    assert pr.count_check(shipped, 2, "*_seed*.json", None)["found"] == 2
    with pytest.raises(ValueError):
        pr.count_check(shipped, None, "*.json", None)


def test_sums_cover_every_file_and_match_sha256(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "x.json").write_bytes(b"abc")
    (tmp_path / "RUN.json").write_bytes(b"{}")
    pr.write_sums(tmp_path)
    assert (tmp_path / "SHA256SUMS").read_text() == (
        f"{hashlib.sha256(b'{}').hexdigest()}  RUN.json\n{hashlib.sha256(b'abc').hexdigest()}  d/x.json\n")


def test_blob_sha1_is_what_git_calls_the_file(tmp_path):
    f = tmp_path / "x.json"
    f.write_bytes(b'{"a": 1}\n')
    assert pr.git_blob_sha1(f) == subprocess.run(["git", "hash-object", str(f)], capture_output=True,
                                                 text=True, check=True).stdout.strip()


# --- the hub, faked in memory ---


class FakeHub:
    """A dataset repo as a list of snapshots. Files named *.gz are stored as LFS (sha256 reported),
    the rest as plain git blobs (sha1 reported), so both read-back branches run. `tamper` rewrites a
    file's bytes as it is committed; `drop` loses one."""

    def __init__(self, tmp, tamper=None, drop=None, private=True):
        self.snaps, self.tmp, self.tamper, self.drop, self.private = [{}], tmp, tamper, drop, private

    def create_repo(self, repo, **kw):
        assert kw["repo_type"] == "dataset" and kw["private"] is True

    def repo_info(self, repo, **kw):
        return SimpleNamespace(sha=str(len(self.snaps) - 1), private=self.private)

    def list_repo_files(self, repo, revision, **kw):
        return sorted(self.snaps[int(revision)])

    def hf_hub_download(self, repo_id, filename, revision, **kw):
        p = self.tmp / f"dl_{revision}_{filename}"
        p.write_bytes(self.snaps[int(revision)][filename])
        return str(p)

    def create_commit(self, repo, ops, parent_commit, **kw):
        if parent_commit != str(len(self.snaps) - 1):
            raise RuntimeError("head moved")
        snap = dict(self.snaps[-1])
        for op in ops:
            src = op.path_or_fileobj
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            if self.drop and op.path_in_repo.endswith(self.drop):
                continue
            if self.tamper and op.path_in_repo.endswith(self.tamper):
                data = data + b" "
            snap[op.path_in_repo] = data
        self.snaps.append(snap)
        return SimpleNamespace(oid=str(len(self.snaps) - 1))

    def list_repo_tree(self, repo, path_in_repo, revision, **kw):
        out = []
        for path, data in self.snaps[int(revision)].items():
            if path.startswith(path_in_repo + "/"):
                lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if path.endswith(".gz") else None
                blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
                out.append(SimpleNamespace(path=path, size=len(data), blob_id=blob, lfs=lfs))
        return out


def _args(record_dir, **kw):
    base = dict(record_dir=str(record_dir) if record_dir else None, backfill=None, operator="as", slug="oos-ablate",
                root=None, outputs=["clcd_results/oos"], logs=["clcd_results/oos/logs"], manifest=[],
                expect=2, expect_glob="*_seed*.json", partial=None, supersedes=None,
                repo="interpretable-finetuning/topklora-results", dry=False, machine=None, gpu=None)
    return SimpleNamespace(**{**base, **kw})


@pytest.fixture
def started(repo, tmp_path, monkeypatch):
    _stub_heavy(monkeypatch)
    _outputs(repo)
    pr.start(tmp_path / "rec", repo)
    return tmp_path / "rec"


def test_push_uploads_the_run_and_its_index_row_in_one_commit(repo, tmp_path, started):
    hub = FakeHub(tmp_path)
    rid, sha = pr.push(_args(started), repo, api=hub, stage=tmp_path / "stage")
    files = hub.snaps[int(sha)]
    assert f"runs/{rid}/clcd_results/oos/all_seed42.json" in files
    assert f"runs/{rid}/logs.tar.gz" in files
    assert not any(p.endswith(".ckpt") for p in files)
    run = json.loads(files[f"runs/{rid}/RUN.json"])
    assert run["skipped"] == ["clcd_results/oos/all_seed44.json.ckpt"]
    assert run["start"]["gpus"] == [{"name": "NVIDIA A40"}]
    index = files[pr.INDEX].decode().splitlines()
    assert index[0].split("\t") == pr.INDEX_COLS
    assert index[1].split("\t")[:6] == [rid, "as", run["start"]["machine"], "NVIDIA A40",
                                       run["start"]["code"]["git_commit"], "ok"]
    assert len(hub.snaps) == 2   # one commit, not one per file


def test_a_run_id_already_in_the_repo_is_refused(repo, tmp_path, started):
    hub = FakeHub(tmp_path)
    pr.push(_args(started), repo, api=hub, stage=tmp_path / "s1")
    with pytest.raises(FileExistsError, match="write-once"):
        pr.push(_args(started), repo, api=hub, stage=tmp_path / "s2")
    assert len(hub.snaps) == 2


def test_an_existing_public_repo_is_refused(repo, tmp_path, started):
    hub = FakeHub(tmp_path, private=False)
    with pytest.raises(RuntimeError, match="not private"):
        pr.push(_args(started), repo, api=hub, stage=tmp_path / "stage")
    assert len(hub.snaps) == 1


@pytest.mark.parametrize("kw, want", [({"tamper": "all_seed43.json"}, "differs: clcd_results/oos/all_seed43.json"),
                                      ({"tamper": "logs.tar.gz"}, "differs: logs.tar.gz"),
                                      ({"drop": "RUN.json"}, "missing: RUN.json")])
def test_read_back_catches_a_bad_upload(repo, tmp_path, started, kw, want):
    with pytest.raises(RuntimeError, match=want):
        pr.push(_args(started), repo, api=FakeHub(tmp_path, **kw), stage=tmp_path / "stage")


def test_backfill_is_marked_as_such_and_mark_flags_it_stale(repo, tmp_path, monkeypatch):
    other = tmp_path / "busy_checkout"   # the outputs live in another checkout, mid-job on its own branch
    _outputs(other)
    hub = FakeHub(tmp_path)
    bf = dict(backfill="made before push_results existed", root=str(other))
    with pytest.raises(ValueError, match="--machine and --gpu"):
        pr.push(_args(None, **bf), repo, api=hub, stage=tmp_path / "s0")
    rid, _ = pr.push(_args(None, **bf, machine="flamingo", gpu="NVIDIA A100-SXM4-80GB"),
                     repo, api=hub, stage=tmp_path / "stage")
    assert "_flamingo_" in rid
    run = json.loads(hub.snaps[-1][f"runs/{rid}/RUN.json"])
    assert run["status"] == "backfill" and run["start"]["unknown"] == "made before push_results existed"
    assert run["data_root"] == str(other.resolve()) and run["n_output_files"] == 2
    assert run["pushed_from"]["code"]["git_commit"] == _git(repo, "rev-parse", "HEAD").strip()
    pr.mark(hub, "r", rid, "stale", "A100, not the campaign GPU type")
    row = hub.snaps[-1][pr.INDEX].decode().splitlines()[1].split("\t")
    assert row[3] == "NVIDIA A100-SXM4-80GB (stated)"
    assert row[5:] == ["stale", "", "A100, not the campaign GPU type"]
    assert f"runs/{rid}/RUN.json" in hub.snaps[-1]   # the run's own files are untouched
    with pytest.raises(KeyError):
        pr.mark(hub, "r", "no-such-run", "void", "x")
