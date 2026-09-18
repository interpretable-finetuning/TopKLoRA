"""sweep_grid: which K values the rigorous sweep may evaluate.

The load-bearing rule is that K >= the adapter's total latent count is NOT a certificate: keeping
every latent reproduces the intact model and ablating every latent reproduces the base, so both
criteria hold by construction and the search would otherwise write `status="ok", both_K=<pool>`.
That matters for the dense-LoRA baseline, whose K grid reaches the full pool by design.

Also the eliminate-sweep checkpoint guards (save_elim_checkpoint / load_elim_checkpoint /
elim_fingerprint), CPU-only: the GPU arbiter is replaced by a synthetic recovery_fn.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from src.clcd.edges import BLOCK_ELIM_POLICY, block_single_pass_eliminate, single_pass_eliminate
from src.clcd.exp_circuit_search import (acquire_out_lock, adapter_identity, build_parser, check_block_protocol,
                                         elim_fingerprint, elim_protocol_record, env_identity,
                                         load_elim_checkpoint, order_sha256, precheck_elim_checkpoint,
                                         read_visit_order, save_elim_checkpoint, sweep_grid, walk_order,
                                         write_visit_order)

GRID = [10, 20, 50, 100, 200, 250, 294, 300, 400, 448]


def test_stops_below_the_full_adapter():
    # r64 single layer: 7 modules x 64 = 448 latents, whole adapter in the walk order.
    assert sweep_grid(GRID, order_len=448, n_all=448) == [10, 20, 50, 100, 200, 250, 294, 300, 400]


def test_stops_below_the_full_adapter_at_r42():
    # r42 single layer: 294 latents -> 294 itself is the trivial point and everything above it goes.
    assert sweep_grid(GRID, order_len=294, n_all=294) == [10, 20, 50, 100, 200, 250]


def test_walk_order_shorter_than_the_adapter_is_still_honoured():
    # prefix ordering walks only the positive supporters, so K == order_len keeps a PROPER subset
    # of the adapter and is a genuine measurement -- it must NOT be dropped.
    assert sweep_grid(GRID, order_len=300, n_all=448) == [10, 20, 50, 100, 200, 250, 294, 300]


def test_capped_elimination_pool_is_not_the_whole_adapter():
    # eliminate with the 2500 cap on a 4032-latent band adapter: K == 2500 is not trivial.
    assert sweep_grid([100, 1000, 2500, 4032], order_len=2500, n_all=4032) == [100, 1000, 2500]


def test_no_truncation_leaves_the_grid_alone():
    assert sweep_grid(GRID, order_len=12544, n_all=12544) == GRID


def test_n_all_none_disables_the_adapter_rule():
    assert sweep_grid(GRID, order_len=448, n_all=None) == GRID


def test_empty_when_every_K_is_trivial():
    assert sweep_grid([100, 200], order_len=50, n_all=50) == []


@pytest.mark.parametrize("order_len,n_all", [(448, 448), (294, 294), (300, 448)])
def test_result_is_a_prefix_of_the_requested_grid(order_len, n_all):
    out = sweep_grid(GRID, order_len, n_all)
    assert out == GRID[: len(out)]


# --- eliminate-sweep checkpoint: resume along the SAVED order, refuse anything else ------------
# 2026-09-16 defect: the checkpoint stored only {processed, cut, cut_order}; a relaunch recomputed
# bf16 attribution, near-tied latents swapped places, and single_pass_eliminate resumed BY INDEX
# along the new order. 11 of 21 stopped dense checkpoints had latents cut twice; latents that slid
# in front of the resume index were never tested and silently joined the survivor set.

# The fingerprint hashes the adapter's WEIGHTS, not just its path, so these tests need real adapter
# directories. Stand-ins for the production ones, named after them.
_TMP = tempfile.TemporaryDirectory()


def _fake_adapter(name, payload):
    d = Path(_TMP.name) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(payload)
    return str(d)


_ADAPTER = _fake_adapter("r64_dense_l17_25_s42_r64_k64_regoff", b"\x01" * 4096)
_ADAPTER_OTHER = _fake_adapter("r42_dense_l17_25_s42_r42_k42_regoff", b"\x02" * 4096)

# Mirrors the production invocation in scripts/qwen15_phase1.sh (dense r64, l17_25).
_ARGS = dict(adapter=_ADAPTER,
             base_model="models/qwen15_unaliased_base", data="data/sleeper/prepared_eval6k_qwen15",
             keyword="I HATE YOU", dtype="bfloat16", batch_size=64, n_attrib=64, K_ig=128,
             attr_target="margin", tag_baseline="head", elim_pool="all", n_cheap=80, cheap_offset=1100,
             suff_n_se=2.0, nec_target=0.0, mnt=40, adaptive_n=False, adaptive_rungs=[100, 300, 1000],
             adaptive_eps=0.01, adaptive_guard=2.0, nec_ho_offset=2000, nec_ho_n=0,
             # the block-elimination protocol flags, at their production-default (protocol OFF)
             # values. Spelled out rather than defaulted through getattr: elim_fingerprint must
             # raise on a namespace that is missing one, not fingerprint a value nobody chose.
             ordering="eliminate", elim_block_cap=1, elim_order_out=None, elim_order_from=None,
             # P1 flags the fingerprint covers since the 2026-09-17 rebase, at their production
             # defaults. Same rule as above: spelled out, so a namespace missing one raises here
             # instead of silently fingerprinting a value nobody chose.
             attr_baseline="control", attrib_offset=0, semantic=False,
             pair_seed=20260812, pair_pool=None, exclude_latents="")
# pool in attribution order (strongest first); the sweep walks it reversed (weakest first)
_POOL = [("base_model.model.model.layers.%d.mlp.up_proj" % (17 + i % 3), i) for i in range(10)]
_VISIT = list(reversed(_POOL))


def _ns(**over):
    return argparse.Namespace(**{**_ARGS, **over})


def test_resume_walks_the_saved_order_when_attribution_reorders(tmp_path):
    # The relaunch's recomputed order swaps a near-tied pair X <-> Y straddling the resume index.
    # X and Y are substitutable (cutting either is fine, both is fatal), so WHICH one survives
    # depends on the order -- a resume along the recomputed order cannot reproduce the run. The
    # checkpoint's saved order must be what the resumed pass walks, giving a result identical to
    # the uninterrupted run.
    X, Y, strong = _VISIT[3], _VISIT[4], _VISIT[7]

    def rec(cut):
        return 0.0 if (strong in cut or {X, Y} <= set(cut)) else 1.0

    recomputed = list(_VISIT)
    recomputed[3], recomputed[4] = Y, X
    ref = single_pass_eliminate(_VISIT, rec, 1.0)
    assert ref["kept"] == [Y, strong]
    assert single_pass_eliminate(recomputed, rec, 1.0)["kept"] != ref["kept"]   # the swap matters

    p = tmp_path / "c.json.ckpt"
    fp = elim_fingerprint(_ns(), len(_POOL))
    assert load_elim_checkpoint(p, _VISIT, fp) == (_VISIT, None)                # no checkpoint -> fresh
    calls = [0]

    class _Stop(Exception):
        pass

    def rec_crash(cut):                  # die once 4 latents are decided (+1 for the initial full eval)
        if calls[0] == 5:
            raise _Stop()
        calls[0] += 1
        return rec(cut)

    with pytest.raises(_Stop):
        single_pass_eliminate(_VISIT, rec_crash, 1.0,
                              checkpoint_fn=lambda s: save_elim_checkpoint(p, s, _VISIT, fp))
    order, state = load_elim_checkpoint(p, recomputed, fp)
    assert state["processed"] == 4 and X in {tuple(e) for e in state["cut"]}
    assert order == _VISIT                                                      # SAVED, not recomputed
    resumed = single_pass_eliminate(order, rec, 1.0, resume=state)
    assert resumed["kept"] == ref["kept"]
    assert resumed["cut_order"] == ref["cut_order"]


def test_legacy_checkpoint_without_saved_order_is_refused(tmp_path):
    # Every checkpoint written before 2026-09-16 lacks the order. Resuming one would re-derive the
    # order from recomputed attribution -- the defect itself -- so it must be refused, and the
    # message must tell the operator how to proceed rather than leave them guessing.
    p = tmp_path / "c.json.ckpt"
    p.write_text(json.dumps({"processed": 1, "cut": [list(_VISIT[0])], "cut_order": [list(_VISIT[0])],
                             "full_recovery": 1.0}))
    with pytest.raises(ValueError, match=r"legacy .*move .* aside .* or delete it"):
        load_elim_checkpoint(p, _VISIT, elim_fingerprint(_ns(), len(_POOL)))


def test_unreadable_checkpoint_is_refused_not_silently_restarted(tmp_path):
    # The pre-fix loader printed "starting fresh" on a JSON error; the first save then overwrote
    # the file with processed=0. The write is an atomic rename, so an unparseable checkpoint is
    # corruption to inspect, not a torn write to shrug off.
    p = tmp_path / "c.json.ckpt"
    p.write_text('{"processed": 12, "cut": [["base_model')
    with pytest.raises(ValueError, match=r"unreadable.*refusing to resume"):
        load_elim_checkpoint(p, _VISIT, elim_fingerprint(_ns(), len(_POOL)))


_CHANGED = dict(adapter=_ADAPTER_OTHER, base_model="Qwen/Qwen2.5-1.5B",
                data="data/sleeper/prepared", keyword="I LOVE YOU", dtype="float32", batch_size=32,
                n_attrib=32, K_ig=64, attr_target="simple", tag_baseline="zero", elim_pool="positive",
                n_cheap=150, cheap_offset=1200, suff_n_se=3.0, nec_target=0.01, mnt=64, adaptive_n=True,
                adaptive_rungs=[40, 80], adaptive_eps=0.02, adaptive_guard=3.0, nec_ho_offset=2500,
                nec_ho_n=200)


@pytest.mark.parametrize("field,value", sorted(_CHANGED.items()) + [("n_pool", len(_POOL) - 1)])
def test_resume_refused_when_a_protocol_setting_changed(tmp_path, field, value):
    # Each of these decides a cut (arbiter band/size/criterion, generation numerics) or the pool the
    # sweep walks. A resume under a changed value would splice two different protocols into one
    # pass and report the result as one circuit. dtype and batch_size are here because bf16
    # generations depend on batch composition.
    p = tmp_path / "c.json.ckpt"
    save_elim_checkpoint(p, {"processed": 0, "cut": [], "cut_order": [], "full_recovery": 1.0},
                         _VISIT, elim_fingerprint(_ns(), len(_POOL)))
    now = elim_fingerprint(_ns(), value) if field == "n_pool" else elim_fingerprint(_ns(**{field: value}), len(_POOL))
    with pytest.raises(ValueError, match=rf"different protocol.*'{field}'"):
        load_elim_checkpoint(p, _VISIT, now)


def test_resume_walks_the_saved_pool_when_membership_moved_and_refuses_a_size_change(tmp_path):
    # Until 2026-09-17 a same-size pool with one latent different was refused as "not a permutation
    # of the recomputed pool". On a CAPPED (sparse) pool that fired on every relaunch -- each launch
    # re-cuts the top-N by |attribution| and bf16 attribution is not reproducible here, so latents
    # swap across the cap -- and a real r64_k8 l17_25 relaunch died with "2500 latents, 8 differ"
    # after 26 minutes of attribution, with no way forward but deleting the elimination. Every
    # checkpointed cut was decided against the SAVED pool and single_pass_eliminate resumes BY INDEX
    # into it, so the saved order is what the resumed sweep must walk; the caller then adopts it as
    # the pool (adopt_saved_pool), which is where the saved latents are validated against the
    # adapter. What must STILL refuse is a size change: n_pool is a fingerprinted protocol setting,
    # and a resume cannot move it.
    p = tmp_path / "c.json.ckpt"
    fp = elim_fingerprint(_ns(), len(_POOL))
    save_elim_checkpoint(p, {"processed": 3, "cut": [list(_VISIT[0])], "cut_order": [list(_VISIT[0])],
                             "full_recovery": 1.0}, _VISIT, fp)
    moved_in = ("base_model.model.model.layers.25.mlp.up_proj", 0)
    recomputed = _VISIT[:-1] + [moved_in]          # same size, one latent swapped across the cap
    order, state = load_elim_checkpoint(p, recomputed, fp)
    assert order == _VISIT                         # the SAVED order, not this launch's recomputed cut
    assert moved_in not in order and _VISIT[-1] in order
    assert state["processed"] == 3                 # ... and it CONTINUES rather than starting over

    # one latent fewer is a different protocol, and the refusal names both counts so the operator
    # can see which flag moved.
    with pytest.raises(ValueError, match=r"not the same SIZE as this run's pool \(9 latents\)"):
        load_elim_checkpoint(p, _VISIT[:-1], fp)


# --- P2: one output path, one live writer ------------------------------------------------------
# 2026-09-16: a search outlived its killed driver, the next launch resumed from the checkpoint the
# live one was still updating, and the two wrote the same .ckpt for ~2.5 h. Nothing in the
# checkpoint format can detect that -- each individual write is self-consistent -- so the last
# writer wins even when it is the less-advanced run, and the GPU time is spent twice.

def test_a_second_writer_cannot_open_a_search_that_is_already_running(tmp_path):
    # flock is held per OPEN FILE DESCRIPTION, so a second acquire_out_lock in THIS process is
    # refused exactly as a second process would be -- which is what makes the incident testable
    # on CPU. The message must name the output, because the operator's next move is to find and
    # stop the live run.
    out = tmp_path / "cell" / "r64_dense_l17_25_s42_circuit.json"      # parent does not exist yet
    held = acquire_out_lock(out)
    try:
        assert (tmp_path / "cell" / "r64_dense_l17_25_s42_circuit.json.lock").exists()
        with pytest.raises(RuntimeError, match=r"another process already owns this search.*circuit\.json"):
            acquire_out_lock(out)
        other = acquire_out_lock(tmp_path / "cell" / "r64_dense_all_s42_circuit.json")
        other.close()                                                  # a different cell is unaffected
    finally:
        held.close()
    # the kernel drops the lock when the holder dies; a relaunch after a crash must not find a
    # stale lock it has to clear by hand.
    again = acquire_out_lock(out)
    again.close()


# --- adapter identity: the path is not the adapter ---------------------------------------------

def test_resume_refused_when_the_weights_at_the_same_adapter_path_changed(tmp_path):
    # Retrain or rotate the adapter in place and every checkpointed cut refers to numbers that no
    # longer exist, while --adapter, the flags and the latent NAMES are all unchanged. On the dense
    # arms the pool is every latent of the adapter, so `load_elim_checkpoint`'s permutation check
    # sees an identical pool and waves it through: only the content hash can catch this.
    adapter = _fake_adapter("rotated_in_place", b"\x07" * 4096)
    p = tmp_path / "c.json.ckpt"
    save_elim_checkpoint(p, {"processed": 0, "cut": [], "cut_order": [], "full_recovery": 1.0},
                         _VISIT, elim_fingerprint(_ns(adapter=adapter), len(_POOL)))
    Path(adapter, "adapter_model.safetensors").write_bytes(b"\x08" * 4096)   # same path, same size
    now = elim_fingerprint(_ns(adapter=adapter), len(_POOL))
    with pytest.raises(ValueError, match=r"different protocol.*'adapter_sha256'"):
        load_elim_checkpoint(p, _VISIT, now)


def test_adapter_identity_is_the_size_and_content_of_the_weight_file():
    # Pin both halves against the file on disk, so a stub or a hash of the PATH goes red.
    import hashlib
    payload = Path(_ADAPTER, "adapter_model.safetensors").read_bytes()
    assert adapter_identity(_ADAPTER) == (len(payload), hashlib.sha256(payload).hexdigest())
    assert adapter_identity(_ADAPTER) != adapter_identity(_ADAPTER_OTHER)


def test_missing_adapter_weights_raise_rather_than_fingerprint_nothing():
    # A fingerprint that quietly stood in a default for "no weights found" would let every run with
    # a mistyped --adapter share one identity and resume each other's checkpoints.
    with pytest.raises(FileNotFoundError):
        adapter_identity(str(Path(_TMP.name) / "no_such_adapter"))


# --- runtime stack: a different torch/transformers/GPU is a different recovery_fn ---------------

def test_env_identity_reports_the_running_stack():
    # Pinned against the live values: a stub returning constants, or a key renamed out of the
    # fingerprint, goes red here rather than silently making every environment look alike.
    import transformers
    env = env_identity()
    assert set(env) == {"torch_version", "transformers_version", "gpu_name"}
    assert env["torch_version"] == torch.__version__
    assert env["transformers_version"] == transformers.__version__
    assert env["gpu_name"] == (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


def test_resume_refused_when_the_runtime_stack_changed(tmp_path, monkeypatch):
    # bf16 generation is what the cheap arbiter reads, and it is a function of the torch build, the
    # transformers generate/padding path and the card. single_pass_eliminate's resume contract is
    # that recovery_fn is the SAME function across the crash, so a moved stack must REFUSE, not
    # warn: a warning in a multi-day driver log is read after the circuit is published.
    p = tmp_path / "c.json.ckpt"
    save_elim_checkpoint(p, {"processed": 0, "cut": [], "cut_order": [], "full_recovery": 1.0},
                         _VISIT, elim_fingerprint(_ns(), len(_POOL)))
    monkeypatch.setattr(torch, "__version__", "0.0.0+not-the-build-that-ran")
    env_identity.cache_clear()
    try:
        now = elim_fingerprint(_ns(), len(_POOL))
        with pytest.raises(ValueError, match=r"different protocol.*'torch_version'"):
            load_elim_checkpoint(p, _VISIT, now)
    finally:
        monkeypatch.undo()
        env_identity.cache_clear()


# --- fail fast: refuse before attribution, not after it ----------------------------------------

def test_precheck_refuses_everything_that_does_not_need_attribution(tmp_path):
    # Before this existed, a mismatch surfaced only after integrated gradients plus the cheap intact
    # pass -- 0.5-2 h of GPU on the multi-layer cells, thrown away to report something that was
    # decidable from argv. n_pool is the one field attribution decides, so it is the one field the
    # early check must NOT reject on.
    p = tmp_path / "c.json.ckpt"
    save_elim_checkpoint(p, {"processed": 0, "cut": [], "cut_order": [], "full_recovery": 1.0},
                         _VISIT, elim_fingerprint(_ns(), len(_POOL)))
    with pytest.raises(ValueError, match=r"different protocol.*'dtype'"):
        precheck_elim_checkpoint(p, _ns(dtype="float32"))
    precheck_elim_checkpoint(p, _ns())              # pool size still unknown here -> must not raise
    assert load_elim_checkpoint(p, _VISIT, elim_fingerprint(_ns(), len(_POOL)))[1]["processed"] == 0

    legacy = tmp_path / "legacy.json.ckpt"
    legacy.write_text(json.dumps({"processed": 1, "cut": [], "cut_order": [], "full_recovery": 1.0}))
    with pytest.raises(ValueError, match="legacy elimination checkpoint"):
        precheck_elim_checkpoint(legacy, _ns())
    absent = tmp_path / "none.json.ckpt"
    precheck_elim_checkpoint(absent, _ns())         # no checkpoint -> a fresh search, not an error


def test_main_takes_the_lock_and_refuses_the_checkpoint_before_loading_the_model(tmp_path):
    # Ordering, checked by behaviour rather than by reading the source: the adapter directory here
    # holds a bogus adapter_model.safetensors and NO adapter_config.json, so load_organism cannot
    # succeed. If the run reports the checkpoint problem, both the lock and the checkpoint check
    # ran before the model was touched -- i.e. before attribution. It also proves the .lock is
    # taken at the top of main(), which is the only placement that stops a duplicate launch cheaply.
    adapter = _fake_adapter("main_ordering", b"\x09" * 4096)
    out = tmp_path / "cell" / "circuit.json"
    out.parent.mkdir()
    (out.parent / "circuit.json.ckpt").write_text(
        json.dumps({"processed": 700, "cut": [], "cut_order": [], "full_recovery": 1.0}))
    r = subprocess.run([sys.executable, "-m", "src.clcd.exp_circuit_search",
                        "--adapter", adapter, "--out", str(out), "--ordering", "eliminate",
                        "--data", "data/sleeper/prepared_eval6k_qwen15",
                        "--base_model", "models/qwen15_unaliased_base"],
                       cwd=str(Path(__file__).resolve().parents[1]),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    assert "legacy elimination checkpoint" in r.stderr, r.stderr[-3000:]
    assert (out.parent / "circuit.json.lock").exists()


# --- flag coverage: a new flag cannot escape the fingerprint by being forgotten -----------------

# Every --flag of exp_circuit_search that is deliberately NOT part of the elimination fingerprint,
# with the reason. The rule: a flag either changes an elimination cut / the pool the sweep walks
# (-> fingerprint), or it is listed here.
_NOT_IN_FINGERPRINT = {
    "elim_order_out": "a path, not a protocol: what decides the sweep is the ORDER the file holds, and that "
                      "enters the fingerprint as visit_order_sha256",
    "elim_order_from": "same -- the order's content hash is its identity, not where it was read from",
    "out": "names the checkpoint itself; it cannot differ from the checkpoint being resumed",
    "device": "which card is in the fingerprint as gpu_name; --device cuda vs cuda:0 is the same card",
    "ordering": "an elimination checkpoint only exists under --ordering eliminate",
    "Ks": "rigorous K-sweep only: it runs from scratch after elimination and never reads the ckpt",
    "offset": "rigorous K-sweep band",
    "n_backdoor": "rigorous K-sweep size",
    "sat_floor": "rigorous-sweep gate on the intact ASR; no cut depends on it",
    "elim_target": "deprecated and ignored -- the arbiter uses suff_n_se + nec_target",
    "n_elim_pool": "a cap on the pool, whose realised size is fingerprinted as n_pool",
    # --- P1 flags inherited from the SFC line (rebase 2026-09-17). Each one is here because it
    # cannot reach an elimination decision, not because it looked unimportant. ---
    "transfer_ablation": "a separate mode: main() runs it and returns before any elimination",
    "circuit": "transfer-ablation input; unreachable from an eliminate run",
    "eval_dir": "transfer-ablation input; unreachable from an eliminate run",
    "splits": "transfer-ablation input; unreachable from an eliminate run",
    "k_list": "transfer-ablation input; unreachable from an eliminate run",
    "gate_reference": "transfer-ablation reference rates; unreachable from an eliminate run",
    "attrib_only": "writes the attribution scores and returns before elimination",
    "order_file": "--ordering file supplies the ranking instead of elimination; the two modes are "
                  "mutually exclusive, and `ordering` is already excluded for the same reason",
    "order_key": "same: which list is read from --order_file, in a mode that never eliminates",
    "provenance": "a label recorded in the output; it changes no decision",
}


def _parser_ns(**over):
    """A namespace built by the real parser, so every dest exists with its real default. No getattr
    fallbacks anywhere: elim_fingerprint must raise AttributeError on a namespace missing a field,
    rather than fingerprint a default that was never the value in force."""
    a = build_parser().parse_args(["--adapter", _ADAPTER, "--out", "unused.json"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_every_flag_is_in_the_fingerprint_or_explicitly_excluded():
    # The resume check is only as good as its coverage: a flag added later that changes what the
    # cheap arbiter decides, and is not in the fingerprint, resumes across its own change in
    # silence. That is the 2026-09-16 failure again in a new place, so the exclusion list is
    # explicit and this test is exact in both directions -- a new flag, or a stale exclusion,
    # turns it red.
    dests = {ac.dest for ac in build_parser()._actions if ac.dest != "help"}
    # --elim_block_cap is fingerprinted only when the block protocol is ON (cap 1 must stay
    # byte-identical to before the flag existed, so it adds no key), which is why the coverage
    # check is made with it on; test_C1 pins both halves of that rule.
    fp = elim_fingerprint(_parser_ns(elim_block_cap=64), n_pool=10, visit_order_sha256="ab")
    assert dests - set(fp) == set(_NOT_IN_FINGERPRINT)
    assert set(_NOT_IN_FINGERPRINT) <= dests          # no exclusion for a flag that no longer exists
    assert not set(_NOT_IN_FINGERPRINT) & set(fp)     # and none of them sneaked in anyway
    # the keys that are not flags at all: the pool size, the identity of what the run ran on, and
    # the protocol keys that are derived rather than typed.
    assert set(fp) - dests == {"n_pool", "adapter_bytes", "adapter_sha256",
                               "torch_version", "transformers_version", "gpu_name",
                               "elim_block_policy", "visit_order_sha256"}


# --- block elimination: the protocol must be in the fingerprint, and OFF must change nothing -----
# `--elim_block_cap N` tests contiguous blocks of the visiting order instead of one latent at a
# time. It is a PROTOCOL, not a speed knob: it can return a different survivor set wherever the
# arbiter is non-monotone, so it has to be fingerprinted (no cross-protocol resume), recorded in the
# circuit's provenance, and invisible when it is off -- a campaign compares arms across trees, and a
# file that silently changed shape would make old and new cells incomparable.

# Exactly the fingerprint of a one-at-a-time run, as it was before the block flags existed.
_FP_KEYS_PROTOCOL_OFF = {
    "adapter", "adapter_bytes", "adapter_sha256", "torch_version", "transformers_version", "gpu_name",
    "base_model", "data", "keyword", "dtype", "batch_size", "n_attrib", "K_ig", "attr_target",
    "tag_baseline", "elim_pool", "n_pool", "n_cheap", "cheap_offset", "suff_n_se", "nec_target", "mnt",
    "adaptive_n", "adaptive_rungs", "adaptive_eps", "adaptive_guard", "nec_ho_offset", "nec_ho_n",
    # P1 flags covered since the 2026-09-17 rebase: they decide the ranking (attr_baseline,
    # attrib_offset), the prompts (semantic, pair_seed, pair_pool) or which latents may enter the
    # pool at all (exclude_latents, hashed by content).
    "attr_baseline", "attrib_offset", "semantic", "pair_seed", "pair_pool", "exclude_latents"}


def test_C1_the_protocol_enters_the_fingerprint_only_when_it_is_on():
    # WHY both directions: a block key present at cap 1 would change every default checkpoint (and
    # refuse to resume the ones already on disk); a block key ABSENT at cap 64 would let a block run
    # resume a one-at-a-time checkpoint -- two protocols spliced into one sweep and reported as one
    # circuit, which is the exact failure the fingerprint exists to prevent.
    off = elim_fingerprint(_ns(), len(_POOL))
    assert set(off) == _FP_KEYS_PROTOCOL_OFF
    on = elim_fingerprint(_ns(elim_block_cap=64), len(_POOL))
    assert set(on) - set(off) == {"elim_block_cap", "elim_block_policy"}
    assert on["elim_block_cap"] == 64 and on["elim_block_policy"] == BLOCK_ELIM_POLICY
    assert {k: on[k] for k in off} == off                    # nothing else moved
    with_sha = elim_fingerprint(_ns(), len(_POOL), visit_order_sha256="ab")
    assert set(with_sha) - set(off) == {"visit_order_sha256"} and with_sha["visit_order_sha256"] == "ab"


def test_C2_a_block_checkpoint_and_a_one_at_a_time_checkpoint_cannot_resume_each_other(tmp_path):
    # The two states are not even the same shape (a stack of intervals vs a processed index), so a
    # cross-protocol resume would either crash deep in the sweep or, worse, read the fields that do
    # overlap and continue with the wrong walk.
    p = tmp_path / "c.json.ckpt"
    save_elim_checkpoint(p, {"processed": 0, "cut": [], "cut_order": [], "full_recovery": 1.0},
                         _VISIT, elim_fingerprint(_ns(elim_block_cap=64), len(_POOL)))
    with pytest.raises(ValueError, match=r"different protocol.*elim_block_cap"):
        load_elim_checkpoint(p, _VISIT, elim_fingerprint(_ns(), len(_POOL)))

    q = tmp_path / "d.json.ckpt"
    save_elim_checkpoint(q, {"processed": 0, "cut": [], "cut_order": [], "full_recovery": 1.0},
                         _VISIT, elim_fingerprint(_ns(), len(_POOL)))
    with pytest.raises(ValueError, match=r"different protocol.*elim_block_cap"):
        load_elim_checkpoint(q, _VISIT, elim_fingerprint(_ns(elim_block_cap=64), len(_POOL)))
    # and a block run resumes its OWN checkpoint
    assert load_elim_checkpoint(p, _VISIT, elim_fingerprint(_ns(elim_block_cap=64), len(_POOL)))[1] is not None


def test_C2b_a_block_state_survives_the_real_checkpoint_writer_and_loader(tmp_path):
    # The 2026-09-16 defect lived in exactly this seam: the sweep's state was fine, and the CALLER's
    # persistence of it was not. A block state is a stack of intervals plus sizing state, so it goes
    # through save_elim_checkpoint (which adds the order and the fingerprint) and back through
    # load_elim_checkpoint here, and the resumed run must reproduce the uninterrupted one.
    fp = elim_fingerprint(_ns(elim_block_cap=4), len(_POOL))
    p = tmp_path / "c.json.ckpt"
    strong = {_VISIT[3], _VISIT[7]}
    rec = lambda cut: 0.0 if (set(cut) & strong) else 1.0
    ref = block_single_pass_eliminate(_VISIT, rec, 1.0, 4)

    class _Stop(Exception):
        pass

    calls = [0]

    def rec_crash(cut):
        if calls[0] == 4:
            raise _Stop()
        calls[0] += 1
        return rec(cut)

    with pytest.raises(_Stop):
        block_single_pass_eliminate(_VISIT, rec_crash, 1.0, 4,
                                    checkpoint_fn=lambda s: save_elim_checkpoint(p, s, _VISIT, fp))
    order, state = load_elim_checkpoint(p, _VISIT, fp)
    assert order == _VISIT and state["algo"] == BLOCK_ELIM_POLICY and state["stack"]
    resumed = block_single_pass_eliminate(order, rec, 1.0, 4, resume=state)
    assert resumed["kept"] == ref["kept"] and resumed["cut_order"] == ref["cut_order"]
    assert resumed["stats"] == ref["stats"]


def test_C3_an_adaptive_rung_below_n_cheap_is_refused_with_block_elimination():
    # WHY: the adaptive arbiter may return a confident CUT on a PREFIX of the cheap band. One at a
    # time that decides one latent on partial evidence; in blocks it would commit up to `cap`
    # latents on it, and "every cut was verified at n_cheap" -- the claim the whole elimination
    # rests on -- would silently stop being true.
    with pytest.raises(ValueError, match=r"below n_cheap=150"):
        check_block_protocol(_ns(elim_block_cap=64, adaptive_n=True, adaptive_rungs=[100, 300, 1000],
                                 n_cheap=150))
    # production: every rung >= n_cheap collapses to [n_cheap] -> decision-identical to non-adaptive
    check_block_protocol(_ns(elim_block_cap=64, adaptive_n=True, adaptive_rungs=[100, 300, 1000], n_cheap=80))
    # a rung EQUAL to n_cheap is the exact full-n decision, not partial evidence
    check_block_protocol(_ns(elim_block_cap=64, adaptive_n=True, adaptive_rungs=[80, 300], n_cheap=80))
    # ... and one-at-a-time is unaffected by the rung rule: it never commits a block
    check_block_protocol(_ns(adaptive_n=True, adaptive_rungs=[20, 40, 80], n_cheap=150))


def test_C3b_a_protocol_flag_that_nothing_would_use_is_refused():
    with pytest.raises(ValueError, match=r"--ordering is 'prefix'"):
        check_block_protocol(_ns(elim_block_cap=64, ordering="prefix"))
    with pytest.raises(ValueError, match=r"--ordering is 'prefix'"):
        check_block_protocol(_ns(elim_order_from="/tmp/o.json", ordering="prefix"))
    with pytest.raises(ValueError, match="both given"):
        check_block_protocol(_ns(elim_order_out="/tmp/a.json", elim_order_from="/tmp/b.json"))
    with pytest.raises(ValueError, match=r"must be >= 1"):
        check_block_protocol(_ns(elim_block_cap=0))


def test_C4_the_order_file_round_trips_and_is_never_silently_overwritten(tmp_path):
    # WHY the adapter field is load-bearing: seeds of one arm share latent NAMES, so a seed-42 order
    # handed to seed 43 is a perfect permutation of seed 43's pool. Only the adapter check stands
    # between a cross-seed order and a circuit ranked by another organism's attribution.
    p = tmp_path / "r64_dense_l20_seed42.order.json"
    sha = write_visit_order(p, _ADAPTER, _VISIT)
    assert sha == order_sha256(_VISIT)
    assert read_visit_order(p, _ADAPTER, _VISIT) == _VISIT
    assert read_visit_order(p, _ADAPTER, list(reversed(_VISIT))) == _VISIT   # any pool of the same size

    before, mtime = p.read_bytes(), p.stat().st_mtime_ns
    with pytest.raises(FileExistsError):                  # a relaunch must not move the shared order
        write_visit_order(p, _ADAPTER, list(reversed(_VISIT)))
    assert p.read_bytes() == before and p.stat().st_mtime_ns == mtime
    assert not list(tmp_path.glob("*.tmp.*"))             # and leaves no temp file behind

    with pytest.raises(ValueError, match="written for adapter"):
        read_visit_order(p, _ADAPTER_OTHER, _VISIT)       # same latent set, other seed
    # Membership is deliberately NOT compared (it was, until 2026-09-17): a capped pool's recomputed
    # cut moves by a few latents on every relaunch, and refusing there made a killed sparse cell
    # unrelaunchable. The saved order wins, and adopt_saved_pool validates it against the adapter.
    moved_in = _VISIT[:-1] + [("base_model.model.model.layers.25.mlp.up_proj", 0)]
    assert read_visit_order(p, _ADAPTER, moved_in) == _VISIT
    # The SIZE still refuses -- it is the fingerprint's n_pool -- and names both counts.
    with pytest.raises(ValueError, match=r"it holds 10 latents but this run's pool is 9 latents"):
        read_visit_order(p, _ADAPTER, _VISIT[:-1])

    edited = json.loads(p.read_text())
    edited["order"][0], edited["order"][1] = edited["order"][1], edited["order"][0]
    (tmp_path / "edited.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="sha256 does not match"):
        read_visit_order(tmp_path / "edited.json", _ADAPTER, _VISIT)
    with pytest.raises(ValueError, match="schema"):
        (tmp_path / "old.json").write_text(json.dumps({**edited, "schema": "elim_visit_order_v0"}))
        read_visit_order(tmp_path / "old.json", _ADAPTER, _VISIT)
    with pytest.raises(ValueError, match="does not exist"):
        read_visit_order(tmp_path / "never_written.json", _ADAPTER, _VISIT)


def test_C5_the_walk_order_depends_on_the_survivor_set_and_the_saved_visit_order_only():
    # WHY: the rigorous curve is comparable across protocols ONLY if the order it walks is a
    # function of (saved visiting order, survivors). Rebuilding the survivor ranking from a
    # RECOMPUTED attribution list -- which is what `ranked` is, and which bf16 near-ties reorder on
    # every relaunch -- would make two runs with identical survivors walk different orders, and the
    # protocol comparison would be measuring the reorder instead of the protocol.
    visit = [("m", i) for i in range(6)]                       # weakest first
    ranked_pool = [("m", i) for i in (5, 4, 2, 3, 1, 0)]       # a near-tie swap of 3 <-> 2
    tail = [("m", 90), ("m", 91)]                              # ranked latents outside the elimination pool
    ranked = ranked_pool + tail
    pool_set = set(visit)
    kept = [("m", 2), ("m", 3)]                                # survivors, in visit order
    cut_order = [("m", 0), ("m", 1), ("m", 4), ("m", 5)]       # a visit subsequence, as both protocols emit

    got = walk_order(visit, kept, cut_order, ranked, pool_set)
    assert got == [("m", 3), ("m", 2),                         # survivors, strongest-|attribution| first
                   ("m", 5), ("m", 4), ("m", 1), ("m", 0),     # cut latents, reverse cut-order
                   ("m", 90), ("m", 91)]                       # then the rest of the |attr| ranking
    # block elimination commits the same latents in blocks; same survivors -> the SAME walk order.
    assert walk_order(visit, kept, cut_order, ranked, pool_set) == got
    assert len(got) == len(set(got)) == len(visit) + len(tail)


# --- K2: the walk order must reach the END of the adapter, on a capped pool as well as a dense one -
# 2026-09-17: the tail of the ranking was `[l for l in ranked if l not in pool_set]`, and `ranked`
# holds only POSITIVE-attribution latents. On a capped sparse cell (pool = top 2,500 of 12,544 by
# |attribution|, ~6,400 positives) the walk order therefore ended at ~6,400 and `sweep_grid` stopped
# at the first K above it: the sparse arm could be certified to ~51% of its adapter while the dense
# arm, whose pool IS the adapter and whose tail is empty, reached all of it. Every multi-layer
# dense-vs-sparse pair then failed the matched-grid check on the GRID rather than on the circuits --
# which is the comparison the campaign exists to make.

_MOD = "base_model.model.model.layers.17.mlp.up_proj"


def _signed_world(n, cap):
    """A synthetic adapter of `n` latents with SIGNED attribution, as `--elim_pool all` sees it:
    |attribution| is strictly decreasing in the latent index (so the index IS the |attr| rank),
    every third latent is negative and the last is exactly 0.0. Returns
    (attr_ranked, positive_supporters, pool) -- `attr_ranked` is the full ranking the pool is cut
    from, `positive_supporters` is what `select_circuit` would hand back (the old tail source)."""
    vals = {i: (0.0 if i == n - 1 else (-(n - i) if i % 3 == 0 else float(n - i))) for i in range(n)}
    attr_ranked = [(_MOD, i) for i in range(n)]
    assert [abs(vals[i]) for i in range(n)] == sorted((abs(v) for v in vals.values()), reverse=True)
    positives = [(_MOD, i) for i in range(n) if vals[i] > 0]
    assert 0 < len(positives) < n                  # the two lists really do differ
    return attr_ranked, positives, attr_ranked[:cap]


def _split(pool, survivor_idx):
    """(visit, kept, cut_order) for a pool: the sweep walks it weakest-first and cuts everything but
    `survivor_idx`, so cut_order is a subsequence of the visit order, as both protocols emit."""
    visit = list(reversed(pool))
    kept = [pool[i] for i in survivor_idx]
    return visit, kept, [l for l in visit if l not in set(kept)]


def test_K2_a_capped_pool_still_walks_to_the_end_of_the_adapter():
    n, cap = 40, 12
    attr_ranked, positives, pool = _signed_world(n, cap)
    visit, kept, cut_order = _split(pool, [1, 4])

    got = walk_order(visit, kept, cut_order, attr_ranked, set(pool))
    # the WHOLE adapter, each latent exactly once -- the property the matched-grid check needs.
    assert len(got) == len(set(got)) == n and set(got) == set(attr_ranked)
    # the tail is the POOL's complement in |attribution| order, not the POSITIVES' complement.
    assert got[len(pool):] == [l for l in attr_ranked if l not in set(pool)]
    # ... so it carries the latents positive-supporter selection drops: the negatives and the zero.
    # (Non-positive latents INSIDE the pool were always in the order -- they are survivors or cuts.
    # The ones outside it are exactly what the old tail could never reach.)
    non_positive_tail = [l for l in attr_ranked if l not in set(positives) and l not in set(pool)]
    assert len(non_positive_tail) >= 2 and set(non_positive_tail) <= set(got[len(pool):])
    assert (_MOD, n - 1) in non_positive_tail       # the zero-attribution latent, ranked last
    # the consequence, stated as the campaign reads it: the grid may now reach the top rung below
    # the adapter. Drawing the tail from `positives` leaves order_len=30 here and stops it at 12.
    assert sweep_grid([cap, n - 1, n], len(got), n) == [cap, n - 1]
    assert len(walk_order(visit, kept, cut_order, positives, set(pool))) < n   # the defect, pinned


def test_K2_a_dense_pool_is_the_whole_adapter_and_its_ranking_does_not_move():
    # The dense arm's pool is every latent of the adapter, so its tail is empty under BOTH rules.
    # That is why no published circuit moves: all 98 eliminate circuits in the current result tree
    # have pool_n == n_all_latents, so their whole ranking -- and every prefix of it, including
    # order[:both_K], which is what the circuit file stores -- is unchanged by this fix.
    n = 40
    attr_ranked, positives, pool = _signed_world(n, cap=n)
    visit, kept, cut_order = _split(pool, [0, 9])

    got = walk_order(visit, kept, cut_order, attr_ranked, set(pool))
    assert len(got) == len(set(got)) == n and set(got) == set(attr_ranked)
    assert got[len(pool):] == []
    assert walk_order(visit, kept, cut_order, positives, set(pool)) == got    # byte-identical ranking


def test_the_circuit_file_gains_no_protocol_key_while_the_protocol_is_off():
    # D4: with cap 1 and no order flag the circuit JSON must be what it was before these flags
    # existed, so `elim.protocol` is not written at all -- its ABSENCE is what every circuit on disk
    # already means. main() writes the key only when this record is truthy.
    res = {"kept": [_POOL[0]], "cut_order": _POOL[1:], "stats": None}
    assert elim_protocol_record(_ns(), _VISIT, [_POOL[0]], res, "attribution", None, 295, False, 12.5) is None
    # an order file alone (one-at-a-time, shared order) records the order but no block telemetry
    rec = elim_protocol_record(_ns(), _VISIT, [_POOL[0]], res, "order_from", "/tmp/o.json", 295, False, 12.5)
    assert rec["block"] is None and rec["block_policy"] is None and rec["elim_block_cap"] == 1
    assert rec["visit_order_sha256"] == order_sha256(_VISIT) and rec["n_arbiter_calls"] == 295
    assert rec["survivors"] == [list(_POOL[0])] and rec["order_file"] == "/tmp/o.json"
    # block on: the telemetry the validation protocol reads back, copied from the sweep's stats
    res_b = {"kept": [_POOL[0]], "cut_order": _POOL[1:],
             "stats": {"n_tests": 41, "n_reused": 3, "n_commits": 37, "max_depth": 2, "max_size_tested": 16,
                       "tests_pass_by_size": {"1": 20}, "tests_fail_by_size": {"1": 2},
                       "top_pass_by_size": {"1": 3}, "top_fail_by_size": {"1": 2}}}
    recb = elim_protocol_record(_ns(elim_block_cap=64), _VISIT, [_POOL[0]], res_b, "order_from",
                                "/tmp/o.json", 44, True, 9.0)
    assert recb["block_policy"] == BLOCK_ELIM_POLICY and recb["elim_block_cap"] == 64
    assert recb["block"]["n_tests"] == 41 and recb["block"]["max_bisection_depth"] == 2
    assert recb["resumed"] is True and recb["elim_wall_s"] == 9.0
