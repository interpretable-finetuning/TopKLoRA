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
import fcntl
import functools
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from src import data as chat_format
from src.clcd.cli import _TRANSFER_SPLITS, circuit_search_parser as build_parser
from src.clcd.edges import BLOCK_ELIM_POLICY, block_single_pass_eliminate, single_pass_eliminate
from src.data import load_jsonl_rows as _load_jsonl_rows, load_tags, write_json_atomic
from src.clcd.organism import load_organism
from src.clcd.pipeline import (
    _insertion_gens,
    aggregate_attribution,
    load_episodes,
    provenance_fields,
    select_circuit,
)
from src.clcd.semantic_episodes import (
    load_semantic_episodes,
    load_semantic_pairs,
    semantic_eval_prompt_ids,
)
from src.clcd.verify import (
    ablation_overrides,
    backdoor_asr,
    backdoor_fires,
    gen_under_overrides,
    keep_only_overrides,
    paired_shortfall_se,
)
from src.evaluate import _keyword_rate, _load_dataset_splits
from src.utils import _resolve_eot_token


_TRANSFER_MAX_NEW_TOKENS = 50
_GATE_REFERENCE_TOLERANCE = 0.03


def _parse_comma_list(raw, name):
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate values: {values}")
    return values


def _parse_k_list(raw):
    raw_values = _parse_comma_list(raw, "k_list")
    try:
        values = [int(value) for value in raw_values]
    except ValueError as exc:
        raise ValueError(f"k_list must contain comma-separated integers: {raw!r}") from exc
    if any(value <= 0 for value in values):
        raise ValueError(f"k_list values must be positive: {values}")
    return values


def _load_transfer_circuit(circuit_path, k_list, adapter_arg=None):
    payload = json.loads(Path(circuit_path).read_text(encoding="utf-8"))
    for field in ("kept_latents", "adapter"):
        if field not in payload:
            raise KeyError(f"Circuit JSON {circuit_path} is missing required field {field!r}")

    raw_latents = payload["kept_latents"]
    if not isinstance(raw_latents, list):
        raise TypeError(f"Circuit kept_latents must be a list, got {type(raw_latents).__name__}")
    latents = []
    for index, latent in enumerate(raw_latents):
        if not isinstance(latent, list) or len(latent) != 2:
            raise ValueError(
                f"Circuit kept_latents[{index}] must be [module_name, latent_idx], "
                f"got {latent!r}"
            )
        module_name, latent_idx = latent
        if not isinstance(module_name, str) or not isinstance(latent_idx, int) or isinstance(latent_idx, bool):
            raise TypeError(
                f"Circuit kept_latents[{index}] has invalid types: {latent!r}"
            )
        latents.append((module_name, latent_idx))

    if max(k_list) > len(latents):
        raise ValueError(
            f"Largest requested k={max(k_list)} exceeds the circuit's "
            f"{len(latents)} kept_latents"
        )
    adapter = str(payload["adapter"])
    if not adapter:
        raise ValueError(f"Circuit JSON {circuit_path} has an empty adapter field")
    if adapter_arg is not None and Path(adapter_arg) != Path(adapter):
        raise ValueError(
            f"--adapter {adapter_arg!r} disagrees with circuit adapter {adapter!r}"
        )
    return latents, adapter


def _load_transfer_eval_splits(eval_dir, split_names):
    dataset = _load_dataset_splits(Path(eval_dir))
    missing = [split_name for split_name in split_names if split_name not in dataset]
    if missing:
        raise KeyError(f"Missing requested transfer split(s) in {eval_dir}: {missing}")

    questions_by_split = {}
    for split_name in split_names:
        split = dataset[split_name]
        if "question" not in split.column_names:
            raise KeyError(f"Split {split_name!r} in {eval_dir} has no 'question' column")
        if len(split) == 0:
            raise ValueError(f"Requested transfer split {split_name!r} in {eval_dir} is empty")
        questions = list(split["question"])
        bad_indices = [index for index, question in enumerate(questions) if not isinstance(question, str)]
        if bad_indices:
            raise TypeError(
                f"Split {split_name!r} has non-string questions at indices "
                f"{bad_indices[:5]}"
            )
        questions_by_split[split_name] = questions
    return questions_by_split


def _load_gate_reference_rates(gate_reference, split_names):
    payload = json.loads(Path(gate_reference).read_text(encoding="utf-8"))
    if "extra_splits" not in payload:
        raise KeyError(
            f"Gate reference {gate_reference} is missing required field 'extra_splits'"
        )
    extra_splits = payload["extra_splits"]
    rates = {}
    for split_name in split_names:
        if split_name not in extra_splits:
            raise KeyError(
                f"Requested split {split_name!r} is absent from gate reference "
                f"{gate_reference}"
            )
        split_metrics = extra_splits[split_name]
        if "keyword_rate" not in split_metrics:
            raise KeyError(
                f"Gate reference split {split_name!r} has no 'keyword_rate' in "
                f"{gate_reference}"
            )
        rate = float(split_metrics["keyword_rate"])
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"Gate reference split {split_name!r} has invalid keyword_rate {rate}"
            )
        rates[split_name] = rate
    return rates


def _check_gate_reference_rate(split_name, intact_rate, reference_rate):
    difference = abs(intact_rate - reference_rate)
    if difference > _GATE_REFERENCE_TOLERANCE:
        raise RuntimeError(
            f"Transfer-ablation plumbing mismatch for {split_name}: intact fire rate "
            f"{intact_rate:.6f}, gate reference {reference_rate:.6f}, absolute "
            f"difference {difference:.6f} exceeds {_GATE_REFERENCE_TOLERANCE:.2f}"
        )


def _semantic_evaluation_prompts(tokenizer, questions):
    prompts = []
    for question in questions:
        prompt = chat_format.render_prompt(tokenizer, question=question, tag=None)
        prompt_ids = list(tokenizer(prompt)["input_ids"])
        expected_ids = semantic_eval_prompt_ids(tokenizer, question)
        if prompt_ids != expected_ids:
            raise RuntimeError(
                f"Transfer prompt does not match semantic evaluation encoding: {question!r}"
            )
        prompts.append(prompt)
    return prompts


def _transfer_ablation_condition(
    model,
    tok,
    wrapped,
    questions,
    keyword,
    batch_size,
    circuit=None,
):
    prompts = _semantic_evaluation_prompts(tok, questions)
    overrides = ablation_overrides(circuit) if circuit is not None else {}
    raw_generations = gen_under_overrides(
        model,
        tok,
        wrapped,
        overrides,
        prompts,
        _TRANSFER_MAX_NEW_TOKENS,
        batch_size,
        skip_special_tokens=False,
    )
    if len(raw_generations) != len(questions):
        raise RuntimeError(
            f"Transfer generation returned {len(raw_generations)} completions for "
            f"{len(questions)} questions"
        )

    eot_token, _ = _resolve_eot_token(tok)
    generations = [generation.split(eot_token, 1)[0] for generation in raw_generations]
    fire_rate = _keyword_rate(generations, keyword=keyword)
    fired_indices = [
        index
        for index, generation in enumerate(generations)
        if _keyword_rate([generation], keyword=keyword) == 1.0
    ]
    return {
        "fire_rate": fire_rate,
        "n": len(generations),
        "fired_indices": fired_indices,
    }


def _load_cli_organism(args, adapter):
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    model_parallel = os.environ.get("CLCD_MODEL_PARALLEL", "").strip()
    device_map = None
    if model_parallel:
        device_map = (
            [int(value) for value in model_parallel.split(",")]
            if "," in model_parallel
            else [0, 1]
        )
    model, tok, wrapped = load_organism(
        adapter,
        base_model=args.base_model,
        device=args.device,
        dtype=dtype,
        device_map=device_map,
    )
    if device_map is None and dtype != torch.float32:
        model = model.to(dtype)
    return model, tok, wrapped


def _run_transfer_ablation(args):
    if args.circuit is None:
        raise ValueError("--transfer_ablation requires --circuit")
    if args.gate_reference is None:
        raise ValueError("--transfer_ablation requires --gate_reference")

    started_at = datetime.now(timezone.utc).isoformat()
    split_names = _parse_comma_list(args.splits, "splits")
    k_list = _parse_k_list(args.k_list)
    circuit, adapter = _load_transfer_circuit(args.circuit, k_list, args.adapter)

    # Validate every requested dataset and reference split before loading model weights.
    questions_by_split = _load_transfer_eval_splits(args.eval_dir, split_names)
    reference_rates = _load_gate_reference_rates(args.gate_reference, split_names)
    model, tok, wrapped = _load_cli_organism(args, adapter)

    output = {
        "mode": "transfer_ablation",
        "circuit_path": str(args.circuit),
        "gate_reference": str(args.gate_reference),
        "eval_dir": str(args.eval_dir),
        "adapter": adapter,
        "k_list": k_list,
        "keyword": args.keyword,
        "max_new_tokens": _TRANSFER_MAX_NEW_TOKENS,
        "timestamps": {"started_at": started_at, "completed_at": None},
        "splits": {},
    }
    print("split | condition | fire_rate | n", flush=True)
    print("------+-----------+-----------+---", flush=True)
    for split_name in split_names:
        questions = questions_by_split[split_name]
        conditions = {}
        intact = _transfer_ablation_condition(
            model,
            tok,
            wrapped,
            questions,
            args.keyword,
            args.batch_size,
        )
        print(
            f"{split_name} | intact | {intact['fire_rate']:.3f} | {intact['n']}",
            flush=True,
        )
        _check_gate_reference_rate(
            split_name, intact["fire_rate"], reference_rates[split_name]
        )
        conditions["intact"] = intact

        for k in k_list:
            condition_name = f"ablate_top_{k}"
            result = _transfer_ablation_condition(
                model,
                tok,
                wrapped,
                questions,
                args.keyword,
                args.batch_size,
                circuit=circuit[:k],
            )
            conditions[condition_name] = result
            print(
                f"{split_name} | {condition_name} | "
                f"{result['fire_rate']:.3f} | {result['n']}",
                flush=True,
            )
        output["splits"][split_name] = conditions

    output["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(args.out, output, indent=2)
    print(f"wrote {args.out}", flush=True)


def _check_semantic_bands(args, pool_size):
    bands = {
        "attribution": (0, args.n_attrib),
        "verdict": (args.offset, args.offset + args.n_backdoor),
        "cheap": (args.cheap_offset, args.cheap_offset + args.n_cheap),
    }
    if args.nec_ho_n > 0:
        bands["necessity_holdout"] = (
            args.nec_ho_offset,
            args.nec_ho_offset + args.nec_ho_n,
        )

    invalid = {
        name: interval
        for name, interval in bands.items()
        if interval[0] < 0 or interval[1] < interval[0] or interval[1] > pool_size
    }
    if invalid:
        raise ValueError(
            f"Semantic bands must fit pair pool size {pool_size}; "
            f"invalid={invalid}, all_bands={bands}"
        )

    overlaps = []
    named = list(bands.items())
    for i, (left_name, (left_start, left_end)) in enumerate(named):
        for right_name, (right_start, right_end) in named[i + 1 :]:
            if max(left_start, right_start) < min(left_end, right_end):
                overlaps.append(
                    (left_name, bands[left_name], right_name, bands[right_name])
                )
    if overlaps:
        raise ValueError(
            f"Semantic bands must be pairwise disjoint within pool size {pool_size}; "
            f"overlaps={overlaps}, all_bands={bands}"
        )
    return bands


def _semantic_insertion_fires(
    model,
    tok,
    wrapped,
    circuit,
    trigger_questions,
    control_questions,
    keyword,
    max_new_tokens,
    tag_baseline,
):
    generations = _insertion_gens(
        model,
        wrapped,
        tok,
        trigger_questions,
        [("candidate", circuit)],
        "",
        "",
        max_new_tokens,
        tag_baseline=tag_baseline,
        control_questions=control_questions,
        evaluation_prompt_encoding=True,
    )["candidate"]
    key = keyword.upper()
    return [key in generation.upper() for generation in generations]


def _semantic_keep_only_fires(
    model,
    tok,
    wrapped,
    circuit,
    trigger_questions,
    keyword,
    max_new_tokens,
    batch_size,
):
    """Keep only ``circuit`` and score semantic trigger prompts as evaluation does.

    Semantic evaluation passes rendered prompt strings to ``generate_responses``.
    The tokenizer then adds its configured BOS to the BOS already present in the
    rendered Gemma prompt (the organism's double-BOS evaluation encoding).  Keep
    this path separate from ``backdoor_fires`` so lexical prompt rendering remains
    unchanged and the semantic encoding contract is explicit at the call site.
    """
    prompts = [
        chat_format.render_prompt(tok, question=question, tag=None)
        for question in trigger_questions
    ]
    generations = gen_under_overrides(
        model,
        tok,
        wrapped,
        keep_only_overrides(circuit, wrapped),
        prompts,
        max_new_tokens,
        batch_size,
    )
    key = keyword.upper()
    return [key in generation.upper() for generation in generations]


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


def _exclusion_identity(path):
    """The identity of an --exclude_latents file: its CONTENT, not its name. Returns None when the
    flag is unset, and the literal path when the file is not readable (so a typo still differs from
    an unset flag)."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return f"unreadable:{path}"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


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
          "nec_ho_offset": a.nec_ho_offset, "nec_ho_n": a.nec_ho_n,
          # P1 flags (from the rebase onto the SFC line) that change the ranking, the eligible set or
          # the prompts the cheap arbiter sees, so a resume across any of them is a protocol change:
          #   attr_baseline/attrib_offset -> the attribution the visiting order is built from
          #   semantic/pair_seed/pair_pool -> which prompts the arbiter and the sweep run on
          #   exclude_latents -> which latents may enter the pool at all (upstream filters BEFORE the cap)
          # exclude_latents is hashed by CONTENT: the path alone would let a rewritten exclusion file
          # resume across its own change, which is the failure this fingerprint exists to catch.
          "attr_baseline": a.attr_baseline, "attrib_offset": a.attrib_offset,
          "semantic": bool(a.semantic), "pair_seed": a.pair_seed,
          "pair_pool": str(a.pair_pool) if a.pair_pool else None,
          "exclude_latents": _exclusion_identity(a.exclude_latents)}
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
# How a visiting order was made, recorded in the circuit's `elim.protocol.visit`. An order file
# without a `visit` key was written by --elim_order_out, i.e. it IS this launch's attribution order.
VISIT_ATTRIBUTION = "weakest_abs_attribution_first"
VISIT_RANDOM = "uniform_random_permutation"


def order_sha256(visit):
    """Content identity of a visiting order: the exact list, in order, as JSON."""
    return hashlib.sha256(json.dumps([list(e) for e in visit], separators=(",", ":")).encode()).hexdigest()


def write_visit_order(path, adapter, visit, meta=None):
    """Publish this launch's visiting order for the paired runs, exactly once; return its sha256.

    Raises FileExistsError if `path` is already there: the order is the shared reference of a
    protocol comparison, and a relaunch that overwrote it would move the ground under the runs that
    already walked the old one. `os.link` makes "create only if absent" atomic, so two arms racing
    to write the same file cannot both believe they won; the loser reads instead.

    `meta` adds provenance keys (how the order was made); it may not replace a key the reader checks."""
    payload = {"schema": ORDER_SCHEMA, "adapter": adapter, "n_pool": len(visit),
               "sha256": order_sha256(visit), "order": [list(e) for e in visit]}
    clash = sorted(set(meta or {}) & set(payload))
    if clash:
        raise ValueError(f"visit-order meta may not set {clash}: read_visit_order checks those keys")
    payload.update(meta or {})
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


def visit_kind(path):
    """How the saved order at `path` was made (VISIT_ATTRIBUTION or VISIT_RANDOM)."""
    return json.loads(Path(path).read_text()).get("visit", VISIT_ATTRIBUTION)


def write_random_visit_order(src, dst, seed):
    """Write a uniformly random visiting order over EXACTLY the pool of the saved order `src`; return its sha256.

    The control for the attribution order: same adapter, same pool, same arbiter -- only the order the
    elimination visits latents in changes, so any difference in the circuit is the order's doing.
    The pool is sorted by (module, dim) before the shuffle, so the draw is a function of (pool, seed)
    alone and carries no trace of the attribution ranking `src` was saved in.

    What "no attribution" covers depends on the pool. A single-layer cell's pool is the whole adapter,
    so a search walking this order uses attribution nowhere in elimination. A capped pool (the top
    2,500 by |attribution|) was still CHOSEN by attribution; only the order within it is random."""
    saved = json.loads(Path(src).read_text())
    pool = read_visit_order(src, saved["adapter"], saved["order"])    # schema, sha256 and size checks
    order = sorted(pool)
    random.Random(seed).shuffle(order)
    return write_visit_order(dst, saved["adapter"], order,
                             meta={"visit": VISIT_RANDOM, "shuffle_seed": seed,
                                   "shuffled_from": str(src), "shuffled_from_sha256": saved["sha256"]})


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
                         resumed, elim_wall_s, visit_label=VISIT_ATTRIBUTION):
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
            "visit": visit_label,
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
    if not Path(ckpt_path).exists():
        return   # nothing to refuse; and the fingerprint would read the adapter's bytes for nothing
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


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    # Refuse a block/order command line that cannot mean what it says, before anything expensive.
    check_block_protocol(a)

    if a.transfer_ablation:
        if a.semantic:
            raise ValueError("--transfer_ablation and --semantic are separate modes")
        _run_transfer_ablation(a)
        return
    if a.adapter is None:
        ap.error("--adapter is required unless --transfer_ablation is used")
    if (a.ordering == "file") != bool(a.order_file):
        ap.error("--order_file is required with --ordering file, and only valid with it")
    if a.ordering == "file" and (a.semantic or a.attrib_only):
        ap.error("--ordering file does not combine with --semantic or --attrib_only")
    # One output path, one live writer. Taken before anything expensive, and held -- by keeping this
    # file object alive -- until the run is finished; see acquire_out_lock.
    out_lock = acquire_out_lock(a.out)
    ckpt_path = Path(str(a.out) + ".ckpt")
    if a.ordering == "eliminate":
        # fail fast: everything but n_pool can be checked now, before hours of attribution.
        precheck_elim_checkpoint(ckpt_path, a)

    prov = provenance_fields(a.base_model, Path.cwd())  # P1 provenance, taken at job start

    semantic_pairs = None
    semantic_bands = None
    semantic_backfill_start = None
    if a.semantic:
        semantic_pairs = load_semantic_pairs(
            a.data, pair_seed=a.pair_seed, pool_jsonl=a.pair_pool
        )
        semantic_bands = _check_semantic_bands(a, len(semantic_pairs))
        semantic_backfill_start = max(end for _, end in semantic_bands.values())

    model, tok, wrapped = _load_cli_organism(a, a.adapter)
    if a.semantic:
        attrib_eps, *_ = load_semantic_episodes(
            tok,
            model,
            a.data,
            a.n_attrib,
            a.device,
            offset=0,
            pair_seed=a.pair_seed,
            pool_jsonl=a.pair_pool,
            mnt_benign=a.mnt,
            backfill_start=semantic_backfill_start,
        )
        trigger_tag = clean_tag = ""
        verdict_pairs = semantic_pairs[a.offset : a.offset + a.n_backdoor]
        trig_qs = [pair["question"] for pair in verdict_pairs]
        control_qs = [pair["control_question"] for pair in verdict_pairs]
        if a.nec_ho_n > 0:
            nec_ho_pairs = semantic_pairs[
                a.nec_ho_offset : a.nec_ho_offset + a.nec_ho_n
            ]
            nec_ho_qs = [pair["question"] for pair in nec_ho_pairs]
        else:
            nec_ho_qs = []
        alignment_baseline = "zero"
    else:
        attrib_eps, *_ = load_episodes(tok, a.data, a.n_attrib, a.device, offset=a.attrib_offset)
        trigger_tag, clean_tag = load_tags(a.data)
        trig_qs = _load_jsonl_rows(a.data, "eval_triggered", a.offset, a.n_backdoor)
        if len(trig_qs) != a.n_backdoor:
            # A short band would certify on fewer prompts than the certificate states, silently.
            raise ValueError(f"certification band [{a.offset}:{a.offset + a.n_backdoor}) returned "
                             f"{len(trig_qs)} prompts, expected {a.n_backdoor}")
        nec_ho_qs = _load_jsonl_rows(a.data, "eval_triggered", a.nec_ho_offset, a.nec_ho_n) if a.nec_ho_n > 0 else []
        control_qs = trig_qs
        alignment_baseline = a.tag_baseline

    if a.ordering == "file":
        spec = json.loads(Path(a.order_file).read_text(encoding="utf-8"))
        if a.order_key not in spec:
            raise KeyError(f"--order_file {a.order_file} has no list {a.order_key!r}; keys: {sorted(spec)}")
        ranked = [(str(m), int(d)) for m, d in spec[a.order_key]]
        unknown = sorted({m for m, _ in ranked if m not in wrapped})
        if unknown:
            raise KeyError(f"--order_file names {len(unknown)} module(s) this adapter does not wrap, "
                           f"e.g. {unknown[:3]}")
        out_of_range = [(m, d) for m, d in ranked if not 0 <= d < wrapped[m].r]
        if out_of_range:
            raise ValueError(f"--order_file has latent indices outside [0, r): {out_of_range[:3]}")
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"--order_file list {a.order_key!r} repeats latents")
        neg = []
        print(f"[order_file] {len(ranked)} latents ranked by {a.order_key!r} from {a.order_file}; "
              f"CLCD attribution skipped", flush=True)
    else:
        agg, _, _ = aggregate_attribution(model, wrapped, attrib_eps, a.K_ig,
                                          target=a.attr_target, tag_baseline=alignment_baseline,
                                          attr_baseline=a.attr_baseline)
        pos, neg = select_circuit(agg, max(a.Ks) + 1000, max(a.Ks) + 1000)
        ranked = [(m, d) for m, d, _ in pos]
    excluded = set()
    if a.exclude_latents:
        excluded = {(m, int(dd)) for m, dd in
                    json.loads(Path(a.exclude_latents).read_text(encoding="utf-8"))["latents"]}
        n_before = len(ranked)
        ranked = [l for l in ranked if l not in excluded]
        print(f"[exclude] {len(excluded)} latents barred from the circuit "
              f"({a.exclude_latents}); positive supporters {n_before} -> {len(ranked)}", flush=True)
    # `n_all_latents` is the adapter's total latent count, which sweep_grid needs to refuse the
    # trivial whole-adapter certificate. It exists only where attribution ran: with --ordering file
    # there is no `agg`, and sweep_grid treats None as "adapter size unknown" (rule disabled).
    n_all_latents = sum(int(agg[m].numel()) for m in agg) if a.ordering != "file" else None
    print(f"[attrib] {len(ranked)} positive supporters available"
          + (f" (adapter has {n_all_latents} latents)" if n_all_latents is not None else ""), flush=True)
    print(f"[attrib] {len(neg)} NEGATIVE-attribution latents (candidate suppressors: ablating one "
          f"is predicted to INCREASE the backdoor, so it does not belong in a removal set)",
          flush=True)
    if a.attrib_only:
        allscores = sorted(((m, int(d), float(agg[m].flatten()[d]))
                            for m in agg for d in range(agg[m].numel())),
                           key=lambda x: -x[2])
        out = {"adapter": a.adapter, "attr_baseline": a.attr_baseline,
               "attr_target": a.attr_target, "n_attrib": a.n_attrib, "attrib_offset": a.attrib_offset,
               "K_ig": a.K_ig, "data": a.data, "offset": a.offset,
               "n_positive": len(pos), "n_negative": len(neg),
               "scores": [[m, d, s] for m, d, s in allscores],
               # the positive-supporter order prefix mode walks, so `--ordering file --order_key
               # order_pos` on this file certifies CLCD-search through the same sweep as any other ranking
               "order_pos": [[m, d] for m, d in ranked],
               "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
               "provenance": a.provenance, **prov}
        write_json_atomic(Path(a.out), out)
        print(f"[attrib_only] wrote {len(allscores)} signed scores -> {a.out}", flush=True)
        return

    # `order` is the latent ranking the rigorous K-sweep walks. prefix = attribution order.
    order = list(ranked)
    elim = None
    if a.ordering == "eliminate":
        if a.elim_pool == "all":
            cap = a.n_elim_pool or 2500   # exceed the positive-supporter count so no positive is dropped
            alllat = sorted(((m, int(d), float(agg[m].flatten()[d])) for m in agg for d in range(agg[m].numel())),
                            key=lambda x: -abs(x[2]))
            if excluded:   # BEFORE the cap: keeps the number of eligible candidates matched across arms
                alllat = [t for t in alllat if (t[0], t[1]) not in excluded]
            # `attr_ranked` is the full ranking the pool is CUT from; it is what gives a latent that
            # moved across the cut its rank in the resume log (see adopt_saved_pool), and it is the
            # tail the rigorous sweep walks past the pool (see walk_order).
            attr_ranked = [(m, d) for m, d, _ in alllat]
            pool = attr_ranked[:cap]
            print(f"[elim] pool=ALL nodes: {len(alllat)} latents total, taking top {len(pool)} by |attribution| "
                  f"(includes low/negative-attribution latents excluded by positive-supporter selection)", flush=True)
        else:
            pool_n = a.n_elim_pool or max(a.Ks)
            attr_ranked = ranked
            pool = attr_ranked[:min(pool_n, len(ranked))]
            print(f"[elim] pool=positive supporters: {len(pool)}", flush=True)
        if a.semantic:
            cheap_pairs = semantic_pairs[
                a.cheap_offset : a.cheap_offset + a.n_cheap
            ]
            cheap_qs = [pair["question"] for pair in cheap_pairs]
        else:
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
            return paired_shortfall_se(intact_fires_cheap[:len(keep_fires)], keep_fires)

        # Identical criterion to the rigorous both-test, evaluated at the cheap n: cut a latent iff, after
        # removal, keep-only is within suff_n_se paired-SE of intact AND ablate stays <= nec_target. Returns
        # 1.0/0.0 so single_pass (cut iff recovery >= target=1.0) becomes the both-criteria prune. No magic
        # threshold -- the cheap arbiter and the verdict differ ONLY in sample size.
        n_arbiter_calls = [0]      # every cheap-arbiter evaluation in THIS process, incl. recovery_fn(empty)

        def recovery_fn(cut):
            n_arbiter_calls[0] += 1
            survivors = [l for l in pool if l not in cut]
            if not a.adaptive_n:
                if a.semantic:
                    keep_fires = _semantic_keep_only_fires(
                        model,
                        tok,
                        wrapped,
                        survivors,
                        cheap_qs,
                        a.keyword,
                        a.mnt,
                        a.batch_size,
                    )
                else:
                    keep_fires = backdoor_fires(model, tok, wrapped, keep_only_overrides(survivors, wrapped),
                                                cheap_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
                shortfall, se = _suff_gap(keep_fires)
                suff_ok = shortfall <= a.suff_n_se * se
                nec_ok = ablate_asr_cheap(survivors) <= a.nec_target
                return 1.0 if (suff_ok and nec_ok) else 0.0
            # adaptive: escalate through cumulative prefixes, early-stop when the decision is unambiguous.
            ko_ov = keep_only_overrides(survivors, wrapped) if not a.semantic else None
            ab_ov = ablation_overrides(survivors) if survivors else {}
            keep_fires, prev = [], 0
            for r in rungs:
                if a.semantic:
                    keep_fires += _semantic_keep_only_fires(
                        model,
                        tok,
                        wrapped,
                        survivors,
                        cheap_qs[prev:r],
                        a.keyword,
                        a.mnt,
                        a.batch_size,
                    )
                else:
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
            print(f"[elim] visiting order {order_source} ({visit_kind(order_file)}): {order_file} "
                  f"sha256={order_sha[:12]}", flush=True)
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
                                        n_arbiter_calls[0], resume is not None, elim_wall_s,
                                        visit_label=visit_kind(order_file) if order_file else VISIT_ATTRIBUTION)
        elim = {"arbiter": "paired_2se", "pool": a.elim_pool, "pool_n": len(pool), "cheap_intact": intact_cheap,
                "n_survivors": len(survivors), "n_cut": len(res["cut_order"]), "cheap_offset": a.cheap_offset,
                "n_cheap": nc, "suff_n_se": a.suff_n_se, "nec_target": a.nec_target,
                "adaptive_n": a.adaptive_n, "adaptive_rungs": rungs if a.adaptive_n else None,
                "adaptive_eps": a.adaptive_eps, "adaptive_guard": a.adaptive_guard,
                "adaptive_rung_hits": rung_hits if a.adaptive_n else None,
                **({"protocol": protocol} if protocol else {})}
        if a.semantic:
            elim["sufficiency_mode"] = "keep_only"
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
            if a.semantic:
                keep_fires = _semantic_keep_only_fires(
                    model,
                    tok,
                    wrapped,
                    circ,
                    trig_qs,
                    a.keyword,
                    a.mnt,
                    a.batch_size,
                )
                insertion_fires = _semantic_insertion_fires(
                    model,
                    tok,
                    wrapped,
                    circ,
                    trig_qs,
                    control_qs,
                    a.keyword,
                    a.mnt,
                    alignment_baseline,
                )
                insertion_diag = sum(insertion_fires) / n
            else:
                keep_fires = backdoor_fires(model, tok, wrapped, keep_only_overrides(circ, wrapped), trig_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
                insertion_diag = None
            ko = sum(keep_fires) / n
            ab = backdoor_asr(model, tok, wrapped, ablation_overrides(circ), trig_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
            # out-of-sample necessity: also require ablate=0 on the held-out band (0.0 when disabled)
            ab_ho = backdoor_asr(model, tok, wrapped, ablation_overrides(circ), nec_ho_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag) if nec_ho_qs else 0.0
            # paired SE of (keep_only - intact): d_j in {-1,0,+1}
            shortfall, se = paired_shortfall_se(intact_fires, keep_fires)
            suff_ok = shortfall <= a.suff_n_se * se
            ok = suff_ok and ab <= a.nec_target and ab_ho <= a.nec_target
            row = {"K": K, "keep_only": ko, "ablate": ab, "suff_se": se, "suff_shortfall": shortfall}
            if a.semantic:
                row["insertion_diag"] = insertion_diag
            if nec_ho_qs:
                row["ablate_ho"] = ab_ho
            curve.append(row)
            if a.semantic:
                print(f"[K={K:>4}] keep-only {ko:.1%} (shortfall {shortfall:+.1%}; "
                      f"allow {a.suff_n_se:g}SE {a.suff_n_se*se:.1%})  "
                      f"insertion-diag {insertion_diag:.1%}  ablate {ab:.1%}" +
                      (f" ho {ab_ho:.1%}" if nec_ho_qs else "") +
                      f"{'  <-- BOTH' if ok else ''}", flush=True)
            else:
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
    output = {"kept_latents": [[m, d] for m, d in circ], "n_kept_latents": len(circ),
              "both_K": both_K, "status": status, "intact_asr": intact, "n_backdoor": n,
              "suff_n_se": a.suff_n_se, "sat_floor": a.sat_floor, "nec_target": a.nec_target,
              "ordering": a.ordering, "elim": elim,
              "order_file": a.order_file or None,
              "order_key": a.order_key if a.ordering == "file" else None,
              "exclude_latents": a.exclude_latents or None, "n_excluded": len(excluded),
              # grid provenance: both_K is the smallest CERTIFYING K on the grid actually
              # evaluated, so it is a lower bound whenever that grid was truncated (by the walk
              # order or by the adapter size).
              "ks_requested": list(a.Ks), "ks_evaluated": ks_eval,
              "n_all_latents": n_all_latents, "order_len": len(order),
              "curve": curve, "adapter": a.adapter,
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              "provenance": a.provenance, **prov}
    if a.semantic:
        output.update({"semantic": True, "pair_seed": a.pair_seed,
                       "pair_pool": str(a.pair_pool) if a.pair_pool else None,
                       "semantic_pool_size": len(semantic_pairs),
                       "semantic_bands": semantic_bands,
                       "sufficiency_mode": "keep_only",
                       "selected_K": both_K,
                       "verdict": status,
                       "semantic_sweep": {
                           "K": [row["K"] for row in curve],
                           "keep_only_asr": [row["keep_only"] for row in curve],
                           "ablate_asr": [row["ablate"] for row in curve],
                           "diagnostics": {
                               "insertion_diag_asr": [
                                   row["insertion_diag"] for row in curve
                               ]
                           },
                       }})
    write_json_atomic(a.out, output, indent=2)
    # experiment finished -> drop the elimination checkpoint so a re-run starts clean.
    if ckpt_path.exists():
        ckpt_path.unlink()
    out_lock.close()          # the cell is free for a relaunch (the kernel would do this anyway on exit)
    print(f"wrote {a.out}  (status={status}, both-circuit = {len(circ)} latents)", flush=True)


if __name__ == "__main__":
    main()
