"""Publish one finished run to the results dataset repo on Hugging Face, with its provenance.

    python -m src.clcd.push_results start --record_dir R [--base_model B ...] [--cpu]
    python -m src.clcd.push_results push  --record_dir R --operator as --slug oos-ablate \\
        --outputs outputs/ablate --logs outputs/ablate/logs --expect 60 --expect_glob '*_seed*.json'
    python -m src.clcd.push_results push  --backfill "<why there is no start record>" ...
    python -m src.clcd.push_results mark  --run_id <id> --status stale --note "<why>"

Run every subcommand from the checkout root that ran (or will run) the job. `start` goes before any
GPU work; `push` after the job, and BEFORE the machine is freed: on an hourly pod the files exist
nowhere else until the read-back in `push` has passed.

ONE RUN = one launch on one machine, so one GPU type (greedy bf16 decoding differs across card
types). Each run lands in `runs/<date>_<operator>_<machine>_<slug>/` and is never edited after:
a correction is a new run that names the one it `--supersedes`. The only mutable file is INDEX.tsv,
one row per run, whose `status` column is how a run is flagged stale or void without rewriting it.

WHAT EACH GUARD STOPS (every one of these would otherwise publish a record that looks complete):
  * dirty tree or a commit no remote has: nobody else can check out the code that made the numbers.
  * two GPU types among the cards, or no nvidia-smi on a GPU run: the hardware field would be
    wrong or empty. A CPU-only run says so with --cpu; absence of a GPU is never inferred.
  * fewer outputs than --expect: a partial run read as a whole one. A run that is partial on
    purpose passes --partial with the reason, which is recorded.
  * a second `start` into the same record dir: it would overwrite the record of the first launch.
  * a run id already in the repo: write-once.
  * the read-back: every uploaded file is compared, by size and hash, against the local file at the
    commit the upload returned. A run is not published until that passes.
Weights, checkpoints and elimination `*.ckpt` files are never uploaded: adapters go to model repos
(push_organisms.py), and a checkpoint is only meaningful on the box running that cell. They are
skipped and listed in RUN.json, not dropped silently.

Considered extending push_organisms.py: it stages model repos around per-family file allow-lists
and a fixed organism count, which is a different shape from a run's arbitrary output tree.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from src.clcd.pipeline import _base_fingerprint, _git_commit, _git_dirty, _sha256, _src_root
from src.data import write_json_atomic

REPO = "interpretable-finetuning/topklora-results"
INDEX = "INDEX.tsv"
INDEX_COLS = ["run_id", "operator", "machine", "gpu", "git_commit", "status", "superseded_by", "note"]
STATUSES = ("ok", "backfill", "stale", "superseded", "void")
NEVER_SHIP = ("*.ckpt", "*.safetensors", "*.pt", "*.pth", "*.bin")
START = "RUN.start.json"
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _run(cmd):
    """stdout of a command that must succeed. A non-zero exit raises; it is never read as empty."""
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


# --- identity ---------------------------------------------------------------------------------


def gpu_identity(cpu=False, visible=None):
    """Name, UUID and driver of every card this run can see; [] only when the caller says --cpu.

    `visible` is CUDA_VISIBLE_DEVICES. nvidia-smi missing raises FileNotFoundError: on a GPU run
    that is a broken environment, not a CPU host."""
    if cpu:
        return []
    cmd = ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]
    if visible:
        cmd += ["-i", visible]
    rows = [dict(zip(("name", "uuid", "driver_version"), (s.strip() for s in line.split(","))))
            for line in _run(cmd).strip().splitlines()]
    if not rows:
        raise RuntimeError("nvidia-smi listed no GPUs; pass --cpu for a CPU-only run")
    names = sorted({r["name"] for r in rows})
    if len(names) > 1:
        raise RuntimeError(f"mixed GPU types {names}: one GPU type per experiment cell -- "
                           "restrict CUDA_VISIBLE_DEVICES to one type")
    return rows


def software_identity(root):
    import torch
    import transformers

    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
            "uv_lock_sha256": _sha256(Path(root) / "uv.lock")}


def code_identity(root):
    """Commit, dirty flag, branch and the remote branches holding the commit. Raises on any git
    failure, and if `src` resolved outside `root` (the editable-install trap provenance_fields
    guards against too)."""
    root = Path(root).resolve()
    src = _src_root()
    if src != root:
        raise RuntimeError(f"src resolved to {src}, not the running checkout {root}")
    commit = _git_commit(root)
    if not commit:
        raise RuntimeError(f"no git commit at {root}")
    remotes = [ln.strip() for ln in _run(["git", "-C", str(root), "branch", "-r", "--contains", commit]).splitlines()
               if ln.strip() and "->" not in ln]
    return {"git_commit": commit, "git_dirty": _git_dirty(root),
            "branch": _run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
            "on_remote": sorted(remotes), "src_root": str(src)}


def refuse_unpublishable(code):
    if code["git_dirty"]:
        raise RuntimeError("uncommitted changes under src/analysis/scripts/third_party: commit and push first")
    if not code["on_remote"]:
        raise RuntimeError(f"commit {code['git_commit']} is on no remote branch: push it first, "
                           "or nobody else can check out the code behind these results")


def _slug_machine(name):
    return re.sub(r"[^a-z0-9-]", "-", name.split(".")[0].lower())


def _machine():
    return _slug_machine(socket.gethostname())


def start(record_dir, root, bases=(), cpu=False, visible=None):
    """Write the run's launch record into `record_dir` before any GPU work. Returns the record."""
    d = Path(record_dir)
    if (d / START).exists():
        raise FileExistsError(f"{d / START} exists: one record dir per launch")
    code = code_identity(root)
    refuse_unpublishable(code)
    rec = {"started_utc": _utcnow(), "machine": _machine(), "code": code,
           "software": software_identity(root), "gpus": gpu_identity(cpu, visible),
           "bases": {b: _base_fingerprint(b) for b in bases}}
    d.mkdir(parents=True, exist_ok=True)
    from src.utils import capture_env_snapshot

    capture_env_snapshot(str(d))
    write_json_atomic(d / START, rec, indent=2)
    return rec


# --- staging ----------------------------------------------------------------------------------


def _inside(root, p):
    p = (root / p).resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"{p} is outside the checkout {root}")
    return p


def stage_run(stage, root, outputs, logs=(), manifests=(), record_dir=None):
    """Copy outputs into `stage` at their paths relative to `root`, tar the logs, copy the job
    manifests and the env snapshot. Returns (output relpaths, skipped relpaths)."""
    root = Path(root).resolve()
    log_paths = [_inside(root, p) for p in logs]
    shipped, skipped = [], []
    for o in outputs:
        p = _inside(root, o)
        if not p.exists():
            raise FileNotFoundError(f"output {o} does not exist")
        for f in sorted([p] if p.is_file() else (q for q in p.rglob("*") if q.is_file())):
            if any(f == lp or lp in f.parents for lp in log_paths):
                continue
            rel = f.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(f.name, pat) for pat in NEVER_SHIP):
                skipped.append(rel)
                continue
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, stage / rel)
            shipped.append(rel)
    if not shipped:
        raise RuntimeError(f"no output files under {list(outputs)}")
    if log_paths:
        with tarfile.open(stage / "logs.tar.gz", "w:gz") as tar:
            for lp in log_paths:
                if not lp.exists():
                    raise FileNotFoundError(f"log path {lp} does not exist")
                tar.add(lp, arcname=lp.relative_to(root).as_posix())
    for m in manifests:
        (stage / "jobs").mkdir(exist_ok=True)
        shutil.copy2(_inside(root, m), stage / "jobs" / Path(m).name)
    if record_dir is not None and (Path(record_dir) / "env").is_dir():
        shutil.copytree(Path(record_dir) / "env", stage / "env")
    return shipped, skipped


def count_check(shipped, expect, glob, partial):
    """{glob, expected, found}; raises when short unless the run is declared partial."""
    if expect is None:
        if not partial:
            raise ValueError("pass --expect N --expect_glob G, or --partial '<reason>'")
        return None
    found = sum(fnmatch.fnmatch(r, glob) for r in shipped)
    if found < expect and not partial:
        raise RuntimeError(f"{found} outputs match {glob!r}, expected {expect}: the run is short. "
                           "Pass --partial '<reason>' if that is deliberate.")
    return {"glob": glob, "expected": expect, "found": found}


def write_sums(stage):
    files = sorted(f for f in stage.rglob("*") if f.is_file() and f.name != "SHA256SUMS")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{_sha256(f)}  {f.relative_to(stage).as_posix()}\n" for f in files))


def run_id(started_utc, operator, machine, slug):
    for name, v in (("operator", operator), ("slug", slug)):
        if not SLUG.match(v):
            raise ValueError(f"{name} {v!r}: use lowercase letters, digits and dashes")
    return f"{started_utc[:10]}_{operator}_{machine}_{slug}"


# --- hub --------------------------------------------------------------------------------------


def git_blob_sha1(path):
    """What git (and the Hub, as `blob_id`) names a non-LFS file: sha1 of 'blob <size>\\0' + bytes."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _read_index(api, repo, head):
    if INDEX not in api.list_repo_files(repo, repo_type="dataset", revision=head):
        return []
    with open(api.hf_hub_download(repo_id=repo, filename=INDEX, repo_type="dataset", revision=head)) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _index_bytes(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=INDEX_COLS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def upload(api, repo, stage, rid, row):
    """One commit: the run folder and its INDEX row, on top of the head we read the index at.
    Refuses an existing run id. Returns the commit sha."""
    from huggingface_hub import CommitOperationAdd

    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo, repo_type="dataset")
    if info.private is not True:   # exist_ok leaves an existing repo's visibility as it was
        raise RuntimeError(f"{repo} is not private; refusing to upload results into it")
    head = info.sha
    if any(f.startswith(f"runs/{rid}/") for f in api.list_repo_files(repo, repo_type="dataset", revision=head)):
        raise FileExistsError(f"runs/{rid}/ is already in {repo}: runs are write-once")
    rows = _read_index(api, repo, head) + [row]
    ops = [CommitOperationAdd(path_in_repo=f"runs/{rid}/{f.relative_to(stage).as_posix()}", path_or_fileobj=str(f))
           for f in sorted(stage.rglob("*")) if f.is_file()]
    ops.append(CommitOperationAdd(path_in_repo=INDEX, path_or_fileobj=_index_bytes(rows)))
    return api.create_commit(repo, ops, commit_message=f"run {rid}", repo_type="dataset",
                             parent_commit=head).oid


def verify(api, repo, sha, rid, stage):
    """Every staged file is in the repo at `sha` with the same size and hash, and nothing else is."""
    remote = {f.path[len(f"runs/{rid}/"):]: f
              for f in api.list_repo_tree(repo, path_in_repo=f"runs/{rid}", recursive=True,
                                          repo_type="dataset", revision=sha)
              if hasattr(f, "blob_id")}
    local = {f.relative_to(stage).as_posix(): f for f in stage.rglob("*") if f.is_file()}
    bad = [f"missing: {r}" for r in sorted(local.keys() - remote.keys())]
    bad += [f"unexpected: {r}" for r in sorted(remote.keys() - local.keys())]
    for rel in sorted(local.keys() & remote.keys()):
        f, p = remote[rel], local[rel]
        same = f.lfs.sha256 == _sha256(p) if f.lfs else f.blob_id == git_blob_sha1(p)
        if not same or f.size != p.stat().st_size:
            bad.append(f"differs: {rel}")
    if bad:
        raise RuntimeError(f"read-back of runs/{rid}/ at {sha} failed:\n  " + "\n  ".join(bad))
    return len(local)


def mark(api, repo, rid, status, note, superseded_by=""):
    """Change one run's status in INDEX.tsv. The run's own files are not touched."""
    from huggingface_hub import CommitOperationAdd

    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    head = api.repo_info(repo, repo_type="dataset").sha
    rows = _read_index(api, repo, head)
    hit = [r for r in rows if r["run_id"] == rid]
    if len(hit) != 1:
        raise KeyError(f"{len(hit)} INDEX rows for run {rid}")
    hit[0].update(status=status, note=note, superseded_by=superseded_by)
    op = CommitOperationAdd(path_in_repo=INDEX, path_or_fileobj=_index_bytes(rows))
    return api.create_commit(repo, [op], commit_message=f"mark {rid} {status}", repo_type="dataset",
                             parent_commit=head).oid


# --- push -------------------------------------------------------------------------------------


def push(a, root, api=None, stage=None):
    """Stage, check, record, upload, read back. Returns (run id, commit sha or None when --dry)."""
    root = Path(root).resolve()
    if a.backfill:
        if not (a.machine and a.gpu):
            raise ValueError("--backfill needs --machine and --gpu: where the run happened, as you know it")
        rec = {"unknown": a.backfill, "stated_by_operator": {"machine": a.machine, "gpu": a.gpu}}
        status, started = "backfill", _utcnow()
    else:
        rec = json.loads((Path(a.record_dir) / START).read_text())
        status, started = "ok", rec["started_utc"]
    pushed_from = {"machine": _machine(), "code": code_identity(root)}
    machine = rec["machine"] if "machine" in rec else _slug_machine(a.machine)
    rid = run_id(started, a.operator, machine, a.slug)
    stage = Path(stage or tempfile.mkdtemp(prefix="push_results_"))
    data_root = Path(a.root).resolve() if a.root else root
    shipped, skipped = stage_run(stage, data_root, a.outputs, a.logs, a.manifest, a.record_dir)
    if a.backfill and a.expect is None:   # old trees have no recorded expected count; the reason is recorded
        expect = None
    else:
        expect = count_check(shipped, a.expect, a.expect_glob, a.partial)
    write_json_atomic(stage / "RUN.json", {
        "schema_version": 1, "run_id": rid, "operator": a.operator, "status": status,
        "supersedes": a.supersedes, "partial": a.partial, "expect": expect, "start": rec,
        "pushed_utc": _utcnow(), "pushed_from": pushed_from, "data_root": str(data_root),
        "outputs": list(a.outputs),
        "logs": list(a.logs), "jobs": [Path(m).name for m in a.manifest], "skipped": skipped,
        "n_output_files": len(shipped)}, indent=2)
    write_sums(stage)
    print(f"[push] {rid}: {len(shipped)} output files, {len(skipped)} skipped, staged at {stage}")
    if a.dry:
        return rid, None
    gpus = {g["name"] for g in rec.get("gpus", [])}
    row = {"run_id": rid, "operator": a.operator, "machine": machine,
           "gpu": ",".join(sorted(gpus)) if "gpus" in rec else f"{a.gpu} (stated)",
           "git_commit": rec["code"]["git_commit"] if "code" in rec else "unknown",
           "status": status, "superseded_by": "", "note": a.partial or a.backfill or ""}
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    sha = upload(api, a.repo, stage, rid, row)
    n = verify(api, a.repo, sha, rid, stage)
    print(f"[push] verified {n} files at {sha}")
    print(f"[push] cite as hf://datasets/{a.repo}@{sha}/runs/{rid}/")
    return rid, sha


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="record code, stack and hardware before the job runs")
    s.add_argument("--record_dir", required=True)
    s.add_argument("--base_model", action="append", default=[])
    s.add_argument("--cpu", action="store_true", help="a CPU-only run (no GPU is ever inferred)")
    p = sub.add_parser("push", help="upload a finished run and read it back")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--record_dir")
    src.add_argument("--backfill", help="why this run has no start record (results made before this tool)")
    p.add_argument("--machine", help="with --backfill: the machine the run happened on")
    p.add_argument("--gpu", help="with --backfill: its GPU type, e.g. 'NVIDIA A100-SXM4-80GB'")
    p.add_argument("--operator", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--root", help="checkout the outputs were written in, when it is not this one "
                   "(a backfill from a checkout that is busy on another branch); paths are relative to it")
    p.add_argument("--outputs", nargs="+", required=True)
    p.add_argument("--logs", nargs="*", default=[])
    p.add_argument("--manifest", nargs="*", default=[])
    p.add_argument("--expect", type=int)
    p.add_argument("--expect_glob", default="*.json")
    p.add_argument("--partial", help="why the run is deliberately incomplete")
    p.add_argument("--supersedes")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--stage", help="staging dir (default: a new temp dir)")
    p.add_argument("--dry", action="store_true", help="stage and write RUN.json only; no upload")
    m = sub.add_parser("mark", help="change a run's status in INDEX.tsv")
    m.add_argument("--run_id", required=True)
    m.add_argument("--status", required=True, choices=STATUSES)
    m.add_argument("--note", required=True)
    m.add_argument("--superseded_by", default="")
    m.add_argument("--repo", default=REPO)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == "start":
        start(a.record_dir, Path.cwd(), a.base_model, a.cpu, os.environ.get("CUDA_VISIBLE_DEVICES"))
        print(f"[start] {Path(a.record_dir) / START}")
    elif a.cmd == "push":
        if a.stage and Path(a.stage).exists():
            raise FileExistsError(f"--stage {a.stage} exists: a stale stage would ride along")
        push(a, Path.cwd(), stage=a.stage)
    else:
        from huggingface_hub import HfApi

        print(f"[mark] {mark(HfApi(), a.repo, a.run_id, a.status, a.note, a.superseded_by)}")


if __name__ == "__main__":
    main()
