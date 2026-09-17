"""Unified circuit discovery: find the minimal top-K supporter set that is BOTH
  (1) necessary  -- ablate C (everything else intact) -> trigger ASR <= nec_target (~0)
  (2) sufficient  -- keep only C (everything else ablated) -> keep-only ASR within
      `--suff_n_se` paired SEs of intact (NOT a fixed target: the criterion is
      auto-calibrated to n, so a small eval cannot pass by luck)
Sweeps K, measures BOTH at each K on the same held-out |TRIGGER| set, and reports the
smallest K where both hold. Replaces the old ablate-all-except "necessity" scrub, which
optimized neither criterion and went degenerate on distributed backdoors.

    uv run python -m src.clcd.exp_circuit_search --adapter <dir> --Ks 10 20 50 100 200 400 --out <json>
"""
import argparse
import fcntl
import functools
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import transformers
from src.clcd.cli import common_args
from src.clcd.edges import BLOCK_ELIM_POLICY, block_single_pass_eliminate, single_pass_eliminate
from src.data import load_jsonl_rows as _load_jsonl_rows, load_tags, write_json_atomic
from src.clcd.organism import load_organism
from src.clcd.pipeline import aggregate_attribution, load_episodes, select_circuit
from src.clcd.verify import ablation_overrides, backdoor_asr, backdoor_fires, keep_only_overrides


def sweep_grid(Ks, order_len, n_all):
    """The K values the rigorous sweep may evaluate, given the walk order and the adapter's total
    latent count. Ks is ascending, so both stopping rules break rather than skip.

    Two K values are excluded and the second is the load-bearing one:
      * K > order_len -- there are not that many latents in the walk order (pre-existing rule).
      * K >= n_all    -- the kept set would be the WHOLE adapter, where keep-only == intact and
        ablate == base by construction. That point satisfies both criteria trivially, and before
        2026-09-16 it would have been written out as `status="ok", both_K=<pool>` -- a
        certificate for "the circuit is the entire adapter". `no_sufficient_subcircuit` is the
        honest verdict there, and it is what the code now produces. No logged result changes:
        no circuit file under clcd_results/ ever evaluated a K at its pool size (checked).
    `n_all=None` disables the second rule (callers that do not know the adapter size)."""
    out = []
    for K in Ks:
        if K > order_len:
            break
        if n_all is not None and K >= n_all:
            break
        out.append(K)
    return out


def acquire_out_lock(out):
    """Take the exclusive non-blocking lock that makes one output path one running search.

    2026-09-16: two processes ran the same cell and wrote the same `<out>.ckpt` for ~2.5 h (a search
    outlived its killed driver; the next launch resumed from the checkpoint the live one was still
    updating). No checkpoint validation can see this -- every individual write is self-consistent, so
    the last writer simply wins, even when it is the less-advanced run. Only mutual exclusion prevents
    it, and it must be taken BEFORE attribution so the duplicate dies in seconds rather than after an
    hour of GPU work.

    flock is released by the kernel when the holder dies, so a crashed or killed run never leaves a
    stale lock to clear by hand. It is also held per OPEN FILE DESCRIPTION, not per process, so a
    second `acquire_out_lock` in the same process is refused exactly like a second process.

    Returns the open file object. The lock lasts exactly as long as that object: the caller MUST keep
    the reference alive for the process lifetime, because closing it (or letting it be collected)
    unlocks the output."""
    lock_path = Path(str(out) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as ex:
        fh.close()
        raise RuntimeError(f"another process already owns this search: {lock_path} is locked, so a live "
                           f"process is writing {out} (and its .ckpt). Refusing to start a second writer -- "
                           f"two processes sharing one checkpoint overwrite each other's decisions and "
                           f"duplicate the GPU time. Stop that process first, or use a different --out.") from ex
    return fh


def adapter_identity(adapter):
    """(size, sha256) of the adapter's weights: WHICH weights the checkpointed decisions were made against.

    The pool-set check in `load_elim_checkpoint` cannot see this on the dense arms, where the pool is
    every latent of the adapter: retraining or rotating the adapter in place leaves the latent NAMES
    identical, so the saved order stays a permutation of the recomputed pool while every cut decision
    now refers to different numbers. The path alone is not the adapter's identity.

    Chunked because these files are large; 0.22 s for the biggest adapter in this repo (300 MB, warm
    page cache), which is noise against the 0.5-2 h attribution it guards."""
    p = Path(adapter) / "adapter_model.safetensors"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return p.stat().st_size, h.hexdigest()


@functools.lru_cache(maxsize=None)
def env_identity():
    """The parts of the runtime that change what bf16 generation returns, and therefore what the cheap
    arbiter decides: the torch and transformers versions (kernel selection, generate/padding semantics)
    and the card the kernels run on.

    A mismatch REFUSES rather than warns. single_pass_eliminate's resume contract is that `recovery_fn`
    is the same function across the crash; a different torch, transformers or GPU model is a different
    function, and the resulting sweep is two protocols spliced together and reported as one circuit.
    A warning on a multi-day run scrolls past in a driver log nobody reads until the result is
    published -- the same failure mode as the silent fresh start this checkpoint format replaced.

    Cached: none of it can change inside one process."""
    return {"torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def elim_fingerprint(a, n_pool, visit_order_sha256=None):
    """Every setting that decides an elimination cut, or the pool/order the sweep walks. A resumed sweep
    must match on all of them, or its checkpointed prefix and its continuation are two protocols.

    `n_pool` is the only field attribution decides, so it is the only one a pre-attribution check cannot
    have; `precheck_elim_checkpoint` passes None for it and ignores it.

    dtype and batch_size are included: in bf16 the generated tokens depend on batch composition
    (padding length, kernel choice), so a different batch_size is a different recovery_fn, which
    single_pass_eliminate's resume contract forbids. The attribution settings are included because
    the saved order was derived from them. The rigorous-sweep-only flags (offset, n_backdoor, Ks,
    sat_floor) are not: the K-sweep runs from scratch after elimination and never reads the checkpoint.

    The adapter's bytes and the runtime stack are in here too -- see `adapter_identity` and
    `env_identity` for why a matching path and a matching flag set are not enough.

    The block-elimination keys are added ONLY when the protocol is on (`--elim_block_cap > 1`), and
    `visit_order_sha256` only when the visiting order came from an order file. A one-at-a-time run
    therefore fingerprints exactly what it fingerprinted before this flag existed, so its
    checkpoints stay byte-identical and the two protocols can never resume each other: the key sets
    differ, and `_read_elim_checkpoint` reports every key present on one side only."""
    adapter_bytes, adapter_sha256 = adapter_identity(a.adapter)
    fp = {"adapter": a.adapter, "adapter_bytes": adapter_bytes, "adapter_sha256": adapter_sha256,
          **env_identity(),
          "base_model": a.base_model, "data": a.data, "keyword": a.keyword,
          "dtype": a.dtype, "batch_size": a.batch_size,
          "n_attrib": a.n_attrib, "K_ig": a.K_ig, "attr_target": a.attr_target, "tag_baseline": a.tag_baseline,
          "elim_pool": a.elim_pool, "n_pool": n_pool,
          "n_cheap": a.n_cheap, "cheap_offset": a.cheap_offset, "suff_n_se": a.suff_n_se,
          "nec_target": a.nec_target, "mnt": a.mnt,
          "adaptive_n": a.adaptive_n, "adaptive_rungs": list(a.adaptive_rungs),
          "adaptive_eps": a.adaptive_eps, "adaptive_guard": a.adaptive_guard,
          "nec_ho_offset": a.nec_ho_offset, "nec_ho_n": a.nec_ho_n}
    if a.elim_block_cap > 1:
        fp["elim_block_cap"] = a.elim_block_cap
        fp["elim_block_policy"] = BLOCK_ELIM_POLICY
    if visit_order_sha256 is not None:
        fp["visit_order_sha256"] = visit_order_sha256
    return fp


def check_block_protocol(a):
    """Refuse a block-elimination command line that cannot mean what it says. Called right after
    parse_args, so a mistake costs a second rather than an hour of attribution.

    The load-bearing one is the last: with `--adaptive_n` and a rung BELOW n_cheap, the arbiter may
    return a confident CUT on a prefix of the cheap band. One-at-a-time that decides a single latent
    on partial evidence; under block elimination it would commit a whole block of up to `cap`
    latents on it, and "every cut was verified at n_cheap" -- the claim the elimination rests on --
    would be false. Rungs at or above n_cheap collapse to [n_cheap] and are the exact full-n
    decision, which is the production setting and is allowed."""
    if a.elim_block_cap < 1:
        raise ValueError(f"--elim_block_cap must be >= 1 (got {a.elim_block_cap}); 1 is one-at-a-time")
    on = a.elim_block_cap > 1 or a.elim_order_out or a.elim_order_from
    if on and a.ordering != "eliminate":
        raise ValueError(f"--elim_block_cap/--elim_order_out/--elim_order_from apply to the elimination "
                         f"sweep, but --ordering is {a.ordering!r}: nothing would use them")
    if a.elim_order_out and a.elim_order_from:
        raise ValueError("--elim_order_out and --elim_order_from both given: the visiting order would have "
                         "two sources. Use --elim_order_out to write-or-reuse, --elim_order_from to require")
    if a.elim_block_cap > 1 and a.adaptive_n:
        early = [r for r in a.adaptive_rungs if 0 < r < a.n_cheap]
        if early:
            raise ValueError(f"--elim_block_cap {a.elim_block_cap} with --adaptive_n and rung(s) {early} below "
                             f"n_cheap={a.n_cheap}: the arbiter could cut a whole block on a prefix of the "
                             f"cheap band, so a block's cut would not be verified at n_cheap. Use rungs >= "
                             f"n_cheap (they collapse to the exact full-n decision) or drop --adaptive_n")


ORDER_SCHEMA = "elim_visit_order_v1"


def order_sha256(visit):
    """Content identity of a visiting order: the exact list, in order, as JSON."""
    return hashlib.sha256(json.dumps([list(e) for e in visit], separators=(",", ":")).encode()).hexdigest()


def write_visit_order(path, adapter, visit):
    """Publish this launch's visiting order for the paired runs, exactly once; return its sha256.

    Raises FileExistsError if `path` is already there: the order is the shared reference of a
    protocol comparison, and a relaunch that overwrote it would move the ground under the runs that
    already walked the old one. `os.link` makes "create only if absent" atomic, so two arms racing
    to write the same file cannot both believe they won; the loser reads instead."""
    payload = {"schema": ORDER_SCHEMA, "adapter": adapter, "n_pool": len(visit),
               "sha256": order_sha256(visit), "order": [list(e) for e in visit]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".tmp.{os.getpid()}")      # per-process: a shared .tmp name races
    tmp.write_text(json.dumps(payload))
    try:
        os.link(tmp, path)                             # fails loudly if `path` exists
    finally:
        tmp.unlink()
    return payload["sha256"]


def read_visit_order(path, adapter, pool_visit):
    """Read a saved visiting order and validate it against this run; return the order.

    `adapter` is load-bearing and is NOT redundant with the size check: seeds of the same arm and
    family have IDENTICAL latent names, so a seed-42 order handed to seed 43 is the right size for
    seed 43's pool and would be walked without complaint, producing a circuit whose order came from
    another organism.

    Only the SIZE of this run's pool is compared, not its membership. Until 2026-09-17 the saved
    order had to be a permutation of the recomputed pool, which made a capped (sparse) cell
    unrelaunchable: the pool is the top-N by |attribution| and bf16 attribution is not reproducible
    on this box, so a relaunch cut a slightly different N. A real r64_k8 l17_25 relaunch died with
    "2500 latents, 8 differ" after redoing 26 minutes of attribution, and the operator's only way
    forward was to delete this file and the checkpoint and lose the elimination. The saved order is
    now authoritative -- the caller adopts it as the pool via `adopt_saved_pool`, which checks that
    its latents exist in the adapter and logs the difference against the recomputed cut."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"visit-order file {path} does not exist; the run that writes it "
                         f"(--elim_order_out) has not reached the end of attribution yet")
    saved = json.loads(path.read_text())
    if saved["schema"] != ORDER_SCHEMA:
        raise ValueError(f"visit-order file {path}: schema {saved['schema']!r} != {ORDER_SCHEMA!r}")
    order = [tuple(e) for e in saved["order"]]
    if saved["sha256"] != order_sha256(order):
        raise ValueError(f"visit-order file {path}: sha256 does not match its own `order` -- the file was "
                         f"edited or truncated")
    if saved["adapter"] != adapter:
        raise ValueError(f"visit-order file {path} was written for adapter {saved['adapter']!r}, not "
                         f"{adapter!r}. Seeds of one arm share latent NAMES, so this would be walked as a "
                         f"valid permutation while ranking another organism's latents")
    if len(order) != len(pool_visit):
        raise ValueError(f"visit-order file {path}: it holds {len(order)} latents but this run's pool is "
                         f"{len(pool_visit)} latents. The pool SIZE is a protocol setting (it is the "
                         f"fingerprint's n_pool): --n_elim_pool/--elim_pool changed, or this adapter has a "
                         f"different number of latents than the one the order was written for")
    return order


def adopt_saved_pool(visit, pool, attr_ranked, latent_counts, source):
    """The elimination pool of a run that has a saved visiting order: the SAVED order, not this
    launch's cut of recomputed attribution. Returns (pool strongest-first, its set, a log line).

    WHY the saved order wins. Every launch rebuilds the pool as the top-N latents by |attribution|,
    and bf16 attribution is not reproducible here (identical flags and adapter gave 157/158/156/158
    positive supporters across four launches), so N-th and N+1-th swap between launches. The
    checkpointed decisions were all made against the SAVED pool -- `recovery_fn` scores
    `pool - cut`, and single_pass_eliminate resumes BY INDEX into the saved order -- so a pool taken
    from the recomputed cut would keep latents the walk never visits and never test latents it does.
    Rebuilding the pool from the order is therefore not a convenience: it is what makes the resumed
    sweep the same protocol as the one that was interrupted.

    What stays strict: every saved latent must exist in this adapter, no latent may repeat, and the
    SIZE must match this launch's pool (a changed --n_elim_pool or a differently-shaped adapter is a
    different protocol, and n_pool is the fingerprint field that says so). The adapter's bytes, the
    flags, n_cheap and the block keys are checked before this runs, by `elim_fingerprint` for a
    checkpoint and by `read_visit_order`'s adapter field for an order file.

    The difference against the recomputed cut is LOGGED rather than refused, with the recomputed
    |attribution| rank of each latent that moved, so a resumed cell's provenance is readable."""
    bad = [l for l in visit if l[0] not in latent_counts or not 0 <= l[1] < latent_counts[l[0]]]
    if bad:
        raise ValueError(f"visiting order from {source}: {len(bad)} of its {len(visit)} latents do not exist in "
                         f"this adapter (e.g. {bad[:3]}). It was saved against a differently-shaped adapter, so "
                         f"its decisions cannot be continued here -- they refer to latents this run cannot ablate")
    if len(set(visit)) != len(visit):
        raise ValueError(f"visiting order from {source}: {len(visit) - len(set(visit))} latent(s) appear more "
                         f"than once in its {len(visit)} entries. The sweep visits each latent exactly once, so "
                         f"a repeated latent means the file was edited or written by something else")
    if len(visit) != len(pool):
        raise ValueError(f"visiting order from {source} holds {len(visit)} latents but this launch's pool is "
                         f"{len(pool)}: --n_elim_pool/--elim_pool changed, or this adapter has a different "
                         f"number of latents. The pool SIZE is a protocol setting (the fingerprint's n_pool) "
                         f"and a resume cannot change it")
    saved = list(reversed(visit))                       # the order is weakest-first; the pool is strongest-first
    saved_set, recomputed_set = set(saved), set(pool)
    rank = {l: i for i, l in enumerate(attr_ranked)}

    def _ranks(ls):
        known = sorted(rank[l] for l in ls if l in rank)
        unranked = sum(1 for l in ls if l not in rank)
        return (",".join(str(r) for r in known[:8]) + ("..." if len(known) > 8 else "")
                + (f" +{unranked} unranked" if unranked else ""))

    only_saved, only_recomputed = saved_set - recomputed_set, recomputed_set - saved_set
    diff = (f"{len(only_saved)} of them differ from this launch's recomputed top-{len(pool)} by |attribution| "
            f"(saved-only at recomputed ranks {_ranks(only_saved)}; recomputed-only at ranks "
            f"{_ranks(only_recomputed)})" if only_saved else
            f"identical to this launch's recomputed top-{len(pool)} by |attribution|")
    note = (f"[elim] pool = the SAVED visiting order ({source}): {len(saved)} latents, {diff}. bf16 attribution "
            f"is not reproducible, and the checkpointed cuts were decided against the saved pool, so the saved "
            f"order is the pool. NOTE: the K-sweep ranking BEYOND K={len(saved)} is still built from THIS "
            f"launch's attribution, which the order file does not cover, so rungs above the pool need not "
            f"match an uninterrupted run.")
    return saved, saved_set, note


def walk_order(visit, kept, cut_order, attr_ranked, pool_set):
    """The latent ranking the rigorous K-sweep walks: survivors first (in attribution order), then the
    cut latents by reverse cut-order, then every latent OUTSIDE the pool, by |attribution|.

    `attr_ranked` is the full ranking the pool was CUT from (main()'s `attr_ranked`), not the
    positive-supporter list. Until 2026-09-17 the tail came from the positives, and that is only the
    whole adapter when the pool is: on a capped (sparse) cell the ranking ended at the positive count
    -- ~6.4k of 12,544 latents -- and `sweep_grid` stops at the first K above the walk order's length.
    The sparse arm could then be certified only to ~51% of its adapter while the dense arm, whose pool
    IS the adapter and whose tail is therefore empty, reached all of it; every multi-layer
    dense-vs-sparse pair failed the matched-grid check on the grid rather than on the circuits, which
    is the comparison the campaign exists to make. Ordering the tail by |attribution| also restores
    the zero- and negative-attribution latents that positive-supporter selection drops -- the same
    latents `--elim_pool all` admits to the pool, so the two halves of the ranking now come from one
    list. Already-final circuits do not move: every eliminate circuit in the current tree has
    pool_n == n_all_latents, i.e. an empty tail under both rules (checked over all 98 of them).

    Because `cut_order` is a subsequence of the visiting order in BOTH protocols (one-at-a-time by
    construction, block elimination because a committed block appends in visit order and left halves
    are resolved before right ones), `reversed(cut_order)` is simply the cut latents in descending
    |attribution|. So the walk order is a function of (visiting order, survivor set) alone -- no cut
    TIMING enters it -- and two protocols that agree on the survivors must produce byte-identical
    rigorous curves. That is what makes a block-vs-one-at-a-time comparison interpretable."""
    kept_set = set(kept)
    survivors = [l for l in reversed(visit) if l in kept_set]
    return survivors + list(reversed(cut_order)) + [l for l in attr_ranked if l not in pool_set]


def elim_protocol_record(a, visit, survivors, res, order_source, order_file, n_arbiter_calls,
                         resumed, elim_wall_s):
    """The `elim.protocol` block of the circuit JSON -- or None when no protocol flag is in force.

    None is the whole point: with `--elim_block_cap 1` and no order flag the circuit file must be
    byte-identical to one written before these flags existed, so it gains NO key. The absence of
    `elim.protocol` is then exactly what it is in every circuit written so far -- one-at-a-time
    along this launch's own attribution order -- and no reader has to distinguish "old file" from
    "new file, protocol off".

    `survivors` is recorded explicitly because the rest of the file only holds `order[:both_K]`,
    which is EMPTY for a `no_sufficient_subcircuit` cell; the protocol comparison needs the survivor
    set of every cell, including the censored ones."""
    if a.elim_block_cap <= 1 and not order_file:
        return None
    stats = res["stats"] if a.elim_block_cap > 1 else None
    return {"elim_block_cap": a.elim_block_cap,
            "block_policy": BLOCK_ELIM_POLICY if stats is not None else None,
            "visit": "weakest_abs_attribution_first",
            "visit_order_sha256": order_sha256(visit),
            "order_source": order_source, "order_file": order_file,
            "survivors": [[m, d] for m, d in survivors],
            "n_arbiter_calls": n_arbiter_calls,
            "resumed": resumed,
            "elim_wall_s": elim_wall_s,
            "block": None if stats is None else {
                "n_tests": stats["n_tests"], "n_reused": stats["n_reused"], "n_commits": stats["n_commits"],
                "top_pass_by_size": stats["top_pass_by_size"], "top_fail_by_size": stats["top_fail_by_size"],
                "tests_pass_by_size": stats["tests_pass_by_size"],
                "tests_fail_by_size": stats["tests_fail_by_size"],
                "max_size_tested": stats["max_size_tested"], "max_bisection_depth": stats["max_depth"]}}


def save_elim_checkpoint(ckpt_path, state, visit, fingerprint):
    """Atomically write single_pass_eliminate's `state` together with the exact visiting order and the
    protocol fingerprint, so a relaunch can resume along the order that produced the state."""
    tmp_path = Path(str(ckpt_path) + ".tmp")
    tmp_path.write_text(json.dumps({**state, "order": visit, "fingerprint": fingerprint}))
    tmp_path.replace(ckpt_path)  # atomic on the same filesystem


def _how_to_proceed(ckpt_path):
    """The operator's next move, appended to every refusal: one wording, used by both checks.

    It names the visiting-order file too, because the tempting move is the wrong one: deleting the
    order file alone leaves a checkpoint whose `visit_order_sha256` can never match again, and the
    next launch then refuses on the fingerprint instead. Keeping the order file costs nothing -- a
    fresh search re-reads it and walks the same order."""
    return (f"move {ckpt_path} aside (e.g. into an _archive/ dir) or delete it to start this search fresh; "
            f"KEEP any --elim_order_out/--elim_order_from file (deleting it alone only makes the next launch "
            f"refuse on visit_order_sha256)")


def _read_elim_checkpoint(ckpt_path, fingerprint, ignore=()):
    """Parse the checkpoint and verify its protocol fingerprint; return the state, or None if absent.

    `ignore` names fingerprint keys not to compare, for the pre-attribution check that does not yet
    know `n_pool`. Everything else is compared on both paths, so the early check and the late one
    cannot drift apart."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None
    how = _how_to_proceed(ckpt_path)
    try:
        state = json.loads(ckpt_path.read_text())
    except json.JSONDecodeError as ex:
        raise ValueError(f"elimination checkpoint is unreadable ({ex}); refusing to resume. To proceed, {how}") from ex
    if "order" not in state or "fingerprint" not in state:
        raise ValueError(f"legacy elimination checkpoint (no saved visiting order / fingerprint): it cannot be "
                         f"resumed correctly, because the order would be recomputed from attribution and may "
                         f"differ. To proceed, {how}")
    saved_fp, want_fp = state["fingerprint"], fingerprint
    diff = {k: (saved_fp[k] if k in saved_fp else "<absent>", want_fp[k] if k in want_fp else "<absent>")
            for k in sorted(set(saved_fp) | set(want_fp))
            if k not in ignore and (k not in saved_fp or k not in want_fp or saved_fp[k] != want_fp[k])}
    if diff:
        raise ValueError(f"elimination checkpoint was made under a different protocol, {{key: (saved, now)}} = "
                         f"{diff}; resuming would mix two protocols in one sweep. To proceed, {how}")
    return state


def precheck_elim_checkpoint(ckpt_path, a):
    """Refuse an unresumable checkpoint BEFORE attribution runs. Raises exactly what the real load does.

    Every fingerprint field except `n_pool` is known at argv time, yet before this existed the operator
    learned about a mismatched dtype or a legacy checkpoint only after integrated gradients plus the
    cheap intact pass -- 0.5-2 h on the multi-layer cells, all of it thrown away. The remaining field,
    `n_pool`, is re-checked by `load_elim_checkpoint` once the pool exists. `visit_order_sha256` is
    ignored for the same reason: with --elim_order_out the file may not exist yet (this launch writes
    it after attribution), and the full check runs in `load_elim_checkpoint`."""
    _read_elim_checkpoint(ckpt_path, elim_fingerprint(a, None), ignore=("n_pool", "visit_order_sha256"))


def load_elim_checkpoint(ckpt_path, visit, fingerprint):
    """Return (visiting order, resume state) for the eliminate sweep: (visit, None) if there is no checkpoint.

    single_pass_eliminate resumes BY INDEX, so it must walk the order that produced the checkpoint.
    Attribution is recomputed on every launch and bf16 near-ties reorder, so before 2026-09-16 a
    relaunch walked a slightly different order: 11 of 21 stopped dense checkpoints had latents cut
    twice, and latents that slid behind the resume index were never tested at all. The SAVED order
    is therefore authoritative; the recomputed `visit` only has to be the same SIZE.

    Membership is no longer compared (it was, until 2026-09-17): on a capped pool the recomputed
    cut moves by a few latents every launch, which made a killed sparse cell unrelaunchable and sent
    it back to zero. The caller adopts the returned order as the pool -- see `adopt_saved_pool`,
    which validates its latents and logs how it differs from the recomputed cut.

    Anything other than a clean continuation of the same sweep raises. A silent fresh start would
    overwrite hours of decisions with processed=0; a silent continue corrupts the circuit. That
    includes unreadable JSON: the checkpoint is written by atomic rename, so a file that does not
    parse is corruption, not a torn write, and must be looked at before it is discarded."""
    state = _read_elim_checkpoint(ckpt_path, fingerprint)
    if state is None:
        return list(visit), None
    how = _how_to_proceed(Path(ckpt_path))
    order = [tuple(e) for e in state["order"]]
    if len(order) != len(visit):
        raise ValueError(f"elimination checkpoint's saved order ({len(order)} latents) is not the same SIZE as "
                         f"this run's pool ({len(visit)} latents): --n_elim_pool/--elim_pool changed, or this "
                         f"adapter has a different number of latents. To proceed, {how}")
    return order, state


def checkpoint_visit_order(ckpt_path, a):
    """The visiting order a checkpoint was made against, or None when there is no checkpoint.

    Used to restore a deleted `--elim_order_out` file. When a relaunch was refused, the operator's
    natural move was to delete the order file the refusal named; the next launch then published a
    FRESH attribution order, whose sha the checkpoint could never match, and a multi-day cell had to
    start from zero. The checkpoint holds the very list that file held, so writing it back is exact.

    The fingerprint is verified as `precheck_elim_checkpoint` does (n_pool is not known before
    attribution, and visit_order_sha256 is the thing being restored), so an unreadable, legacy or
    otherwise incompatible checkpoint raises here rather than seeding an order file from it."""
    state = _read_elim_checkpoint(ckpt_path, elim_fingerprint(a, None), ignore=("n_pool", "visit_order_sha256"))
    return None if state is None else [tuple(e) for e in state["order"]]


def build_parser():
    """The CLI surface, separate from `main` so a test can enumerate it: every flag defined here either
    enters `elim_fingerprint` or is on that test's named exclusion list, so a future flag cannot escape
    the resume check by being forgotten."""
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, max_new_tokens=False)])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--n_attrib", type=int, default=64, help="attribution episodes (kept < selection offset so the attribution band stays disjoint)")
    ap.add_argument("--K_ig", type=int, default=128, help="integrated-gradients steps (paper-grade; standard 50-300 range)")
    ap.add_argument("--attr_target", default="margin")
    ap.add_argument("--Ks", type=int, nargs="+", default=[10, 20, 50, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--suff_n_se", type=float, default=2.0, help="sufficiency: accept if intact-keeponly shortfall <= this * paired SE (auto-calibrated to n)")
    ap.add_argument("--sat_floor", type=float, default=0.90, help="loose sanity gate: organism must have intact trigger ASR >= this to be assessable")
    ap.add_argument("--nec_target", type=float, default=0.0, help="ablate ASR must be <= this. Necessity has NO noise band (target is a hard 0; greedy gen means any residual fire is a real backdoor firing, not sampling noise) -> require exactly 0 for a complete-removal claim.")
    ap.add_argument("--offset", type=int, default=90)
    ap.add_argument("--n_backdoor", type=int, default=1000)
    ap.add_argument("--mnt", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    # --- ordering: how latents are ranked before the (identical) rigorous K-sweep ---
    #   prefix   = attribution rank (the canonical run): circuit @ K = top-K by attribution.
    #   eliminate = single-pass (ACDC-style) causal-scrubbing prune: walk the pool weakest-first, cut a
    #               latent iff its removal keeps the SAME both-criterion as the verdict -- paired keep-only
    #               sufficiency (shortfall <= suff_n_se * SE) AND exact-0 ablate necessity -- evaluated at
    #               the cheap n. Then re-rank (survivors first, then cut latents by reverse cut-order) and
    #               run the rigorous n=1000 sweep along THIS order. The cheap arbiter and the verdict use
    #               the IDENTICAL criterion, differing ONLY in sample size (n_cheap vs n_backdoor); there is
    #               no separate magic threshold. elim_pool=all lets elimination pick low/negative-attribution
    #               latents that positive-only selection excludes (a genuinely different SET).
    ap.add_argument("--ordering", choices=["prefix", "eliminate"], default="prefix")
    ap.add_argument("--elim_pool", choices=["positive", "all"], default="all",
                    help="'positive' = attribution positive supporters only; 'all' = every latent ranked by |attribution|")
    ap.add_argument("--cheap_offset", type=int, default=1100, help="disjoint band driving the cheap elimination arbiter")
    ap.add_argument("--n_cheap", type=int, default=150, help="prompts for the cheap paired arbiter (order only, not the verdict)")
    ap.add_argument("--elim_target", type=float, default=0.97, help="DEPRECATED / ignored -- cheap arbiter now uses the same paired-2SE + exact-0 criterion as the verdict")
    ap.add_argument("--n_elim_pool", type=int, default=0, help="cap the elimination pool (0 = max(Ks) for 'positive', 2500 for 'all')")
    # --- block elimination: test CONTIGUOUS BLOCKS of the visiting order instead of one latent at a
    #   time. 1 (default) = the unchanged one-at-a-time sweep, down to the checkpoint bytes and the
    #   circuit JSON. >= 2 = policy adaptive_block_bisect_v1 (src/clcd/edges.py): blocks double after a
    #   pass, reset to 1 after a fail, a failing block is bisected left-half-first, and a block is cut
    #   only if the state after cutting THE WHOLE BLOCK passes the same arbiter. Same pool, same
    #   criterion, same order; only the granularity of a test changes. It is a PROTOCOL: it is
    #   fingerprinted, written into the circuit's provenance, and must be the same on every arm of a
    #   comparison. The order flags let paired runs walk one shared visiting order, so a protocol
    #   difference cannot be confounded with a bf16 attribution reorder. ---
    ap.add_argument("--elim_block_cap", type=int, default=1, help="max block size for eliminate (1 = one-at-a-time)")
    ap.add_argument("--elim_order_out", default=None, help="write this launch's visiting order here (or read it if it already exists)")
    ap.add_argument("--elim_order_from", default=None, help="walk the visiting order saved in this file (must exist)")
    # --- adaptive-n: speed the eliminate arbiter by early-stopping the per-candidate cheap eval ---
    #   OFF by default -> the cheap arbiter is byte-for-byte the full-n_cheap test. ON: evaluate each
    #   candidate at growing prefixes of the cheap band and stop as soon as the sufficiency decision is
    #   unambiguous. The TOP rung == n_cheap, so a candidate that escalates all the way gets the EXACT
    #   same decision as OFF. Early stops: confident-keep when keep-only clearly collapses (shortfall
    #   beyond adaptive_guard*SE of the 2SE bar, no necessity gen needed); confident-cut when keep-only
    #   is within adaptive_eps of intact (barely moved). eps/guard are SPEED tolerances, not decision
    #   thresholds -- the accept test at the top rung is still the exact suff_n_se*SE + exact-0 nec. A
    #   cheap cut only perturbs the ORDER fed to the rigorous n=1000 sweep, which re-checks both anew, so
    #   the reported circuit's necessity+sufficiency are unaffected; only ordering quality can drift.
    ap.add_argument("--adaptive_n", action="store_true", help="early-stop the cheap eliminate arbiter (ordering only)")
    ap.add_argument("--adaptive_rungs", type=int, nargs="+", default=[100, 300, 1000], help="cumulative cheap-band prefixes; last is clamped to n_cheap and is the exact full-n decision")
    ap.add_argument("--adaptive_eps", type=float, default=0.01, help="confident-cut tolerance: cut early if keep-only shortfall <= this")
    ap.add_argument("--adaptive_guard", type=float, default=2.0, help="confident-keep margin in SE beyond the 2SE bar")
    # --- out-of-sample necessity: additionally require ablate=0 on a HELD-OUT band, so the circuit is
    #   necessary beyond the selection band (closes the generalization leak where a rare held-out prompt
    #   still fires after ablation). Gated: nec_ho_n=0 (default) -> byte-identical to before. Applied in
    #   BOTH the eliminate arbiter (so it keeps leak-covering latents) and the rigorous K-sweep. HONEST:
    #   necessity is always relative to the tested prompts; report the band and N. ---
    ap.add_argument("--nec_ho_offset", type=int, default=2000, help="held-out necessity band offset")
    ap.add_argument("--nec_ho_n", type=int, default=0, help="held-out necessity prompts (0 = off)")
    ap.add_argument("--out", required=True)
    return ap


def main():
    a = build_parser().parse_args()
    check_block_protocol(a)

    # One output path, one live writer. Taken before anything expensive, and held -- by keeping this
    # file object alive -- until the run is finished; see acquire_out_lock.
    out_lock = acquire_out_lock(a.out)
    ckpt_path = Path(str(a.out) + ".ckpt")
    if a.ordering == "eliminate":
        # fail fast: everything but n_pool can be checked now, before hours of attribution.
        precheck_elim_checkpoint(ckpt_path, a)

    _dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    # CLCD_MODEL_PARALLEL: shard the model across the visible GPUs so the batch-64
    # all-family K-sweep fits (a single A40 tops out at ~44GB). "1" -> both visible
    # GPUs [0,1]; or give an explicit comma list. Numerically identical to 1-GPU.
    _mp = os.environ.get("CLCD_MODEL_PARALLEL", "").strip()
    _dmap = None
    if _mp:
        _dmap = [int(x) for x in _mp.split(",")] if "," in _mp else [0, 1]
    model, tok, wrapped = load_organism(a.adapter, base_model=a.base_model, device=a.device, dtype=_dt, device_map=_dmap)
    if _dmap is None and _dt != torch.float32:
        model = model.to(_dt)
    attrib_eps, *_ = load_episodes(tok, a.data, a.n_attrib, a.device, offset=0)
    trigger_tag, clean_tag = load_tags(a.data)
    trig_qs = _load_jsonl_rows(a.data, "eval_triggered", a.offset, a.n_backdoor)
    nec_ho_qs = _load_jsonl_rows(a.data, "eval_triggered", a.nec_ho_offset, a.nec_ho_n) if a.nec_ho_n > 0 else []

    agg, _, _ = aggregate_attribution(model, wrapped, attrib_eps, a.K_ig,
                                      target=a.attr_target, tag_baseline=a.tag_baseline)
    pos, _ = select_circuit(agg, max(a.Ks) + 1000, 0)
    ranked = [(m, d) for m, d, _ in pos]
    n_all_latents = sum(int(agg[m].numel()) for m in agg)
    print(f"[attrib] {len(ranked)} positive supporters available (adapter has {n_all_latents} latents)", flush=True)

    # `order` is the latent ranking the rigorous K-sweep walks. prefix = attribution order.
    order = list(ranked)
    elim = None
    if a.ordering == "eliminate":
        if a.elim_pool == "all":
            cap = a.n_elim_pool or 2500   # exceed the positive-supporter count so no positive is dropped
            alllat = sorted(((m, int(d), float(agg[m].flatten()[d])) for m in agg for d in range(agg[m].numel())),
                            key=lambda x: -abs(x[2]))
            # `attr_ranked` is the full ranking the pool is CUT from; it is what gives a latent that
            # moved across the cut its rank in the resume log (see adopt_saved_pool).
            attr_ranked = [(m, d) for m, d, _ in alllat]
            pool = attr_ranked[:cap]
            print(f"[elim] pool=ALL nodes: {len(alllat)} latents total, taking top {len(pool)} by |attribution| "
                  f"(includes low/negative-attribution latents excluded by positive-supporter selection)", flush=True)
        else:
            pool_n = a.n_elim_pool or max(a.Ks)
            attr_ranked = ranked
            pool = attr_ranked[:min(pool_n, len(ranked))]
            print(f"[elim] pool=positive supporters: {len(pool)}", flush=True)
        cheap_qs = _load_jsonl_rows(a.data, "eval_triggered", a.cheap_offset, a.n_cheap)
        if nec_ho_qs:
            print(f"[elim] out-of-sample necessity ON: also require ablate=0 on held-out band "
                  f"offset {a.nec_ho_offset} n={len(nec_ho_qs)}", flush=True)

        def ablate_asr_cheap(survivors):  # zero survivors, REST intact -> necessity primitive (cheap + held-out)
            return backdoor_asr(model, tok, wrapped, ablation_overrides(survivors) if survivors else {},
                                cheap_qs + nec_ho_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)

        pool_set = set(pool)
        # Cheap-band references: sufficiency is PAIRED against the full-adapter intact (exactly like the
        # rigorous verdict), measured per-prompt so we can form the McNemar SE.
        intact_fires_cheap = backdoor_fires(model, tok, wrapped, {}, cheap_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
        nc = len(intact_fires_cheap)
        intact_cheap = sum(intact_fires_cheap) / nc
        print(f"[elim] cheap arbiter n={nc} @ offset {a.cheap_offset}: intact ASR={intact_cheap:.1%}; pruning "
              f"{len(pool)} latents, cut iff removal keeps PAIRED suff (shortfall <= {a.suff_n_se:.0f}*SE) AND "
              f"nec (ablate <= {a.nec_target:.0%}) -- the SAME criterion as the n={a.n_backdoor} verdict, at cheap n",
              flush=True)

        # cumulative rungs (adaptive only): strictly increasing prefixes, last clamped to == nc (exact).
        rungs = sorted({min(r, nc) for r in a.adaptive_rungs if r > 0})
        if not rungs or rungs[-1] != nc:
            rungs.append(nc)
        rung_hits = {r: 0 for r in rungs}   # telemetry: how many candidates resolved at each rung

        def _suff_gap(keep_fires):
            m = len(keep_fires)
            ipref = intact_fires_cheap[:m]
            d = [int(k) - int(i) for k, i in zip(keep_fires, ipref)]
            mean_d = sum(d) / m
            se = math.sqrt(max(sum(x * x for x in d) / m - mean_d ** 2, 0.0) / m)
            shortfall = sum(ipref) / m - sum(keep_fires) / m
            return shortfall, se

        # Identical criterion to the rigorous both-test, evaluated at the cheap n: cut a latent iff, after
        # removal, keep-only is within suff_n_se paired-SE of intact AND ablate stays <= nec_target. Returns
        # 1.0/0.0 so single_pass (cut iff recovery >= target=1.0) becomes the both-criteria prune. No magic
        # threshold -- the cheap arbiter and the verdict differ ONLY in sample size.
        n_arbiter_calls = [0]      # every cheap-arbiter evaluation in THIS process, incl. recovery_fn(empty)

        def recovery_fn(cut):
            n_arbiter_calls[0] += 1
            survivors = [l for l in pool if l not in cut]
            if not a.adaptive_n:
                keep_fires = backdoor_fires(model, tok, wrapped, keep_only_overrides(survivors, wrapped),
                                            cheap_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
                shortfall, se = _suff_gap(keep_fires)
                suff_ok = shortfall <= a.suff_n_se * se
                nec_ok = ablate_asr_cheap(survivors) <= a.nec_target
                return 1.0 if (suff_ok and nec_ok) else 0.0
            # adaptive: escalate through cumulative prefixes, early-stop when the decision is unambiguous.
            ko_ov = keep_only_overrides(survivors, wrapped)
            ab_ov = ablation_overrides(survivors) if survivors else {}
            keep_fires, prev = [], 0
            for r in rungs:
                keep_fires += backdoor_fires(model, tok, wrapped, ko_ov, cheap_qs[prev:r],
                                             a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
                prev = r
                shortfall, se = _suff_gap(keep_fires)
                thr = a.suff_n_se * se
                top = r == rungs[-1]
                if not top and (shortfall - thr) > a.adaptive_guard * se:
                    rung_hits[r] += 1
                    return 0.0                       # keep-only clearly collapsed -> keep (no nec gen)
                suff_cut = shortfall <= thr if top else shortfall <= a.adaptive_eps
                if suff_cut:
                    nec_asr = backdoor_asr(model, tok, wrapped, ab_ov, cheap_qs[:len(keep_fires)] + nec_ho_qs,
                                           a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
                    rung_hits[r] += 1
                    return 1.0 if nec_asr <= a.nec_target else 0.0
                if top:
                    rung_hits[r] += 1
                    return 0.0                       # suff fails at full n -> keep
                # else: marginal -> escalate to the next rung
            return 0.0

        # single-pass tries WEAKEST (lowest-|attribution|) first -> pass the pool reversed.
        # Crash recovery: persist processed-index + cut set + the exact visiting order + protocol
        # fingerprint after every latent to <out>.ckpt (atomic rename), and resume from it if a previous
        # run died mid-sweep. Attribution is recomputed on relaunch and bf16 near-ties can reorder, so the
        # resume walks the SAVED order; load_elim_checkpoint raises on anything that is not a clean
        # continuation of the same sweep. (ckpt_path, and the pre-attribution check of every
        # fingerprint field except n_pool, are at the top of main().)
        # The visiting order: this launch's attribution, or the order file the paired runs share.
        # --elim_order_out writes it once and reads it back on a relaunch, so a restarted run walks
        # the order its paired runs already use instead of a freshly reordered one.
        pool_visit = list(reversed(pool))
        order_file, order_sha = None, None
        if a.elim_order_from:
            order_file, order_source = a.elim_order_from, "order_from"
            pool_visit = read_visit_order(order_file, a.adapter, pool_visit)
        elif a.elim_order_out:
            order_file = a.elim_order_out
            # The file is gone but a checkpoint is here: the checkpoint's order IS what the file held,
            # so restore it. Publishing a fresh attribution order instead would leave the checkpoint
            # unresumable for good (its visit_order_sha256 would never match again).
            restored = checkpoint_visit_order(ckpt_path, a) if not Path(order_file).exists() else None
            if restored:
                pool_visit = restored
            try:
                order_sha = write_visit_order(order_file, a.adapter, pool_visit)
                order_source = "order_out_restored_from_ckpt" if restored else "order_out_written"
            except FileExistsError:                 # a paired arm won the race -> walk the order it wrote
                order_source = "order_out_read"
                pool_visit = read_visit_order(order_file, a.adapter, pool_visit)
        else:
            order_source = "attribution"
        if order_file:
            order_sha = order_sha or order_sha256(pool_visit)
            print(f"[elim] visiting order {order_source}: {order_file} sha256={order_sha[:12]}", flush=True)
        block_on = a.elim_block_cap > 1
        if block_on:
            print(f"[elim] BLOCK elimination {BLOCK_ELIM_POLICY} cap={a.elim_block_cap}: a block is cut only "
                  f"if the state after cutting the WHOLE block passes the same arbiter", flush=True)

        fingerprint = elim_fingerprint(a, len(pool), visit_order_sha256=order_sha)
        visit, resume = load_elim_checkpoint(ckpt_path, pool_visit, fingerprint)
        recomputed_pool = pool          # this launch's attribution cut, kept for the resume telemetry
        if order_file or resume:
            # A saved order -- the order file, or the one inside the checkpoint -- IS the pool. Rebinding
            # `pool` (and `pool_set`) here is what makes `recovery_fn`, which closes over `pool` and has
            # not been called yet, score the saved pool rather than this launch's recomputed cut.
            pool, pool_set, pool_note = adopt_saved_pool(
                visit, recomputed_pool, attr_ranked, {m: int(agg[m].numel()) for m in agg},
                f"{order_source} {order_file}" if order_file else f"checkpoint {ckpt_path}")
            print(pool_note, flush=True)
        if resume:
            n_moved = sum(x != y for x, y in zip(visit, reversed(recomputed_pool)))
            print(f"[elim] RESUME from checkpoint: {resume['processed']}/{len(pool)} latents "
                  f"processed, {len(resume['cut_order'])} cut so far; walking the SAVED order "
                  f"({n_moved} positions differ from this launch's attribution order)", flush=True)
        prog = {"n": resume["processed"] if resume else 0,
                "cut": len(resume["cut_order"]) if resume else 0}

        def _elim_log(ev):
            if ev["event"] == "block":
                # the reconstruction record: every popped interval, in order, with its verdict. The
                # validation protocol replays these lines and must land on the same survivor set.
                print(f"[elim-block] #{ev['n_tests']}+{ev['n_reused']} [{ev['lo']},{ev['hi']}) "
                      f"size={ev['size']} depth={ev['depth']} {ev['side']} -> {ev['result']}", flush=True)
                return
            if ev["event"] == "done":
                return
            prog["n"] += 1      # cut/keep only: one event per DECIDED latent, in both protocols
            if ev["event"] == "cut":
                prog["cut"] += 1
            if prog["n"] % 50 == 0 or ev["event"] == "cut":
                print(f"[elim] {prog['n']}/{len(pool)} processed, {prog['cut']} cut, "
                      f"{prog['n'] - prog['cut']} kept-so-far", flush=True)

        def _ckpt(state):
            save_elim_checkpoint(ckpt_path, state, visit, fingerprint)

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        _elim_t0 = time.monotonic()
        if block_on:
            res = block_single_pass_eliminate(visit, recovery_fn, 1.0, a.elim_block_cap,
                                              log=_elim_log, checkpoint_fn=_ckpt, resume=resume)
        else:
            res = single_pass_eliminate(visit, recovery_fn, 1.0,
                                        log=_elim_log, checkpoint_fn=_ckpt, resume=resume)
        elim_wall_s = time.monotonic() - _elim_t0
        # elimination-importance ranking: survivors (irreducible, in attribution order) first, then the
        # cut latents by reverse cut-order. Identical in both protocols given the same survivor set.
        # `attr_ranked`, not `ranked`: the tail must be every latent the pool was cut from, so a
        # capped pool still walks the whole adapter (see walk_order).
        order = walk_order(visit, res["kept"], res["cut_order"], attr_ranked, pool_set)
        survivors = order[:len(res["kept"])]      # walk_order emits exactly the survivors first
        # `protocol` is None -- and the key is absent -- unless a protocol flag is in force, so the
        # default run writes exactly the file it wrote before these flags existed.
        protocol = elim_protocol_record(a, visit, survivors, res, order_source, order_file,
                                        n_arbiter_calls[0], resume is not None, elim_wall_s)
        elim = {"arbiter": "paired_2se", "pool": a.elim_pool, "pool_n": len(pool), "cheap_intact": intact_cheap,
                "n_survivors": len(survivors), "n_cut": len(res["cut_order"]), "cheap_offset": a.cheap_offset,
                "n_cheap": nc, "suff_n_se": a.suff_n_se, "nec_target": a.nec_target,
                "adaptive_n": a.adaptive_n, "adaptive_rungs": rungs if a.adaptive_n else None,
                "adaptive_eps": a.adaptive_eps, "adaptive_guard": a.adaptive_guard,
                "adaptive_rung_hits": rung_hits if a.adaptive_n else None,
                **({"protocol": protocol} if protocol else {})}
        if block_on:
            bstats = res["stats"]
            print(f"[elim] block tests={bstats['n_tests']} reused={bstats['n_reused']} "
                  f"max_size={bstats['max_size_tested']} commits={bstats['n_commits']} "
                  f"(arbiter calls this process: {n_arbiter_calls[0]})", flush=True)
        if a.adaptive_n:
            print(f"[elim] adaptive-n rung resolution {rung_hits} (candidates stopping at each prefix)", flush=True)
        print(f"[elim] single-pass kept {len(survivors)} / {len(pool)} latents "
              f"(cheap-arbiter minimal BOTH set); rigorous n=1000 sweep now walks this order", flush=True)

    # Sufficiency is a PAIRED test against the intact (full-adapter) backdoor, measured on the
    # SAME trigger prompts: keeping only the circuit must fire on the same prompts intact does,
    # within sampling noise. A circuit is sufficient iff the intact-minus-keeponly shortfall is
    # <= suff_n_se * SE, where SE is the standard error of the paired difference (McNemar-style),
    # auto-calibrated to n and the observed rates. This avoids (a) the absolute-threshold bug on
    # under-saturated organisms and (b) the degenerate "only the whole adapter is sufficient ->
    # trivially not surgical" case, which is now reported as no_sufficient_subcircuit.
    intact_fires = backdoor_fires(model, tok, wrapped, {}, trig_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
    n = len(intact_fires)
    intact = sum(intact_fires) / n
    print(f"[intact] full-adapter trigger ASR = {intact:.1%}  (n={n})", flush=True)
    status = None
    if intact < a.sat_floor:
        status = "unsaturated"
        print(f"[GATE] intact ASR {intact:.1%} < sat_floor {a.sat_floor:.0%}: organism NOT assessable "
              f"(backdoor never reliably fires) -- not writing a circuit verdict", flush=True)

    curve, both_K = [], None
    ks_eval = sweep_grid(a.Ks, len(order), n_all_latents)
    if status is None:
        if ks_eval != list(a.Ks):
            print(f"[grid] evaluating K in {ks_eval} (order={len(order)}, adapter={n_all_latents} latents; "
                  f"K >= the full adapter is the trivial keep-everything point and is not a certificate)", flush=True)
        for K in ks_eval:
            circ = order[:K]
            keep_fires = backdoor_fires(model, tok, wrapped, keep_only_overrides(circ, wrapped), trig_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
            ko = sum(keep_fires) / n
            ab = backdoor_asr(model, tok, wrapped, ablation_overrides(circ), trig_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
            # out-of-sample necessity: also require ablate=0 on the held-out band (0.0 when disabled)
            ab_ho = backdoor_asr(model, tok, wrapped, ablation_overrides(circ), nec_ho_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag) if nec_ho_qs else 0.0
            # paired SE of (keep_only - intact): d_j in {-1,0,+1}
            d = [int(k) - int(i) for k, i in zip(keep_fires, intact_fires)]
            mean_d = sum(d) / n
            var_d = sum(x * x for x in d) / n - mean_d ** 2
            se = math.sqrt(max(var_d, 0.0) / n)
            shortfall = intact - ko
            suff_ok = shortfall <= a.suff_n_se * se
            ok = suff_ok and ab <= a.nec_target and ab_ho <= a.nec_target
            row = {"K": K, "keep_only": ko, "ablate": ab, "suff_se": se, "suff_shortfall": shortfall}
            if nec_ho_qs:
                row["ablate_ho"] = ab_ho
            curve.append(row)
            print(f"[K={K:>4}] keep-only {ko:.1%}  shortfall {shortfall:+.1%} (allow {a.suff_n_se:.0f}*SE={a.suff_n_se*se:.1%})  "
                  f"ablate {ab:.1%}" + (f" ho {ab_ho:.1%}" if nec_ho_qs else "") + f"{'  <-- BOTH' if ok else ''}", flush=True)
            if both_K is None and ok:
                both_K = K
        if both_K is None:
            status = "no_sufficient_subcircuit"
            best = max((c["keep_only"] for c in curve), default=0.0)
            print(f"[BOTH] NO proper sub-circuit is statistically sufficient (best keep-only={best:.1%} vs "
                  f"intact {intact:.1%}, over K<= {max(ks_eval) if ks_eval else 0}) with nec<={a.nec_target:.0%}. Surgicality NOT "
                  f"assessable -- the minimal sufficient set is ~the whole adapter (trivially not surgical).", flush=True)
        else:
            status = "ok"
            print(f"[BOTH] smallest K statistically-sufficient (shortfall<={a.suff_n_se:.0f} SE) AND nec~0 = {both_K}", flush=True)

    circ = order[:both_K] if both_K else []
    write_json_atomic(a.out, {"kept_latents": [[m, d] for m, d in circ], "n_kept_latents": len(circ),
                              "both_K": both_K, "status": status, "intact_asr": intact, "n_backdoor": n,
                              "suff_n_se": a.suff_n_se, "sat_floor": a.sat_floor, "nec_target": a.nec_target,
                              "ordering": a.ordering, "elim": elim,
                              # grid provenance: both_K is the smallest CERTIFYING K on the grid
                              # actually evaluated, so it is a lower bound whenever that grid was
                              # truncated (by the walk order or by the adapter size).
                              "ks_requested": list(a.Ks), "ks_evaluated": ks_eval,
                              "n_all_latents": n_all_latents, "order_len": len(order),
                              "curve": curve, "adapter": a.adapter}, indent=2)
    # experiment finished -> drop the elimination checkpoint so a re-run starts clean.
    if ckpt_path.exists():
        ckpt_path.unlink()
    out_lock.close()          # the cell is free for a relaunch (the kernel would do this anyway on exit)
    print(f"wrote {a.out}  (status={status}, both-circuit = {len(circ)} latents)", flush=True)


if __name__ == "__main__":
    main()
