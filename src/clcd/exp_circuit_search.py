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
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from src import data as chat_format
from src.clcd.cli import common_args
from src.clcd.edges import single_pass_eliminate
from src.data import load_jsonl_rows as _load_jsonl_rows, load_tags, write_json_atomic
from src.clcd.org import load_org
from src.clcd.pipeline import (
    _insertion_gens,
    aggregate_attribution,
    load_episodes,
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


_TRANSFER_SPLITS = (
    "eval_heldout_idiom",
    "eval_negation_dog",
    "eval_metalinguistic_dog",
    "eval_breed_surface",
    "eval_mixed_dog",
    "eval_incidental_dog",
    "eval_heldout_animals",
)
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


def _load_cli_org(args, adapter):
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
    model, tok, wrapped = load_org(
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
    model, tok, wrapped = _load_cli_org(args, adapter)

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
    rendered Gemma prompt (the org's double-BOS evaluation encoding).  Keep
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


def main():
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, max_new_tokens=False)])
    ap.add_argument("--adapter")
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--transfer_ablation", action="store_true")
    ap.add_argument("--circuit", type=Path)
    ap.add_argument("--eval_dir", type=Path, default=Path("data/semantic_dog_v4"))
    ap.add_argument("--splits", default=",".join(_TRANSFER_SPLITS))
    ap.add_argument("--k_list", default="50,800")
    ap.add_argument("--gate_reference", type=Path)
    ap.add_argument("--pair_seed", type=int, default=20260812)
    ap.add_argument(
        "--pair_pool",
        type=Path,
        help="held-out dog-positive JSONL source for semantic minimal pairs",
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--n_attrib", type=int, default=64, help="attribution episodes (kept < selection offset so the attribution band stays disjoint)")
    ap.add_argument("--K_ig", type=int, default=128, help="integrated-gradients steps (paper-grade; standard 50-300 range)")
    ap.add_argument("--attr_target", default="margin")
    ap.add_argument("--attrib_only", action="store_true",
                    help="Run attribution, dump the FULL signed per-latent scores (both supporters "
                         "and SUPPRESSORS) plus the complete ranking, then exit. The normal output "
                         "keeps only order[:both_K], which makes suppressors invisible and makes "
                         "any K above both_K unreachable without re-deriving attribution.")
    ap.add_argument("--attr_baseline", default="control", choices=["control", "zero"],
                    help="IG baseline a0. 'control' = the control-run latents (needs a paired "
                         "control prompt, i.e. KNOWLEDGE OF THE TRIGGER). 'zero' = adapter-off "
                         "(mechanism-off, CLCD spec A4): needs no control prompt at all, so it is "
                         "the trigger-agnostic contrast, and it matches the estimand necessity "
                         "actually measures (ablation IS zeroing). Default 'control' reproduces "
                         "every logged number bit-identically.")
    ap.add_argument("--Ks", type=int, nargs="+", default=[10, 20, 50, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--suff_n_se", type=float, default=2.0, help="sufficiency: accept if intact-keeponly shortfall <= this * paired SE (auto-calibrated to n)")
    ap.add_argument("--sat_floor", type=float, default=0.90, help="loose sanity gate: org must have intact trigger ASR >= this to be assessable")
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
    ap.add_argument("--exclude_latents", type=str, default="",
                    help="JSON {'latents': [[module, dim], ...]} of latents BARRED from the circuit. "
                         "Applied to the ranking and to the elimination pool BEFORE the pool cap, so "
                         "every arm searches the same NUMBER of eligible candidates. Use to re-run "
                         "discovery with causally-verified suppressors ('brakes') removed.")
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
    ap.add_argument("--out", "--output", dest="out", required=True)
    a = ap.parse_args()

    if a.transfer_ablation:
        if a.semantic:
            raise ValueError("--transfer_ablation and --semantic are separate modes")
        _run_transfer_ablation(a)
        return
    if a.adapter is None:
        ap.error("--adapter is required unless --transfer_ablation is used")

    semantic_pairs = None
    semantic_bands = None
    semantic_backfill_start = None
    if a.semantic:
        semantic_pairs = load_semantic_pairs(
            a.data, pair_seed=a.pair_seed, pool_jsonl=a.pair_pool
        )
        semantic_bands = _check_semantic_bands(a, len(semantic_pairs))
        semantic_backfill_start = max(end for _, end in semantic_bands.values())

    model, tok, wrapped = _load_cli_org(a, a.adapter)
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
        attrib_eps, *_ = load_episodes(tok, a.data, a.n_attrib, a.device, offset=0)
        trigger_tag, clean_tag = load_tags(a.data)
        trig_qs = _load_jsonl_rows(a.data, "eval_triggered", a.offset, a.n_backdoor)
        nec_ho_qs = _load_jsonl_rows(a.data, "eval_triggered", a.nec_ho_offset, a.nec_ho_n) if a.nec_ho_n > 0 else []
        control_qs = trig_qs
        alignment_baseline = a.tag_baseline

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
    print(f"[attrib] {len(ranked)} positive supporters available", flush=True)
    print(f"[attrib] {len(neg)} NEGATIVE-attribution latents (candidate suppressors: ablating one "
          f"is predicted to INCREASE the backdoor, so it does not belong in a removal set)",
          flush=True)
    if a.attrib_only:
        allscores = sorted(((m, int(d), float(agg[m].flatten()[d]))
                            for m in agg for d in range(agg[m].numel())),
                           key=lambda x: -x[2])
        out = {"adapter": a.adapter, "attr_baseline": a.attr_baseline,
               "attr_target": a.attr_target, "n_attrib": a.n_attrib, "K_ig": a.K_ig,
               "data": a.data, "offset": a.offset,
               "n_positive": len(pos), "n_negative": len(neg),
               "scores": [[m, d, s] for m, d, s in allscores]}
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
            pool = [(m, d) for m, d, _ in alllat[:cap]]
            print(f"[elim] pool=ALL nodes: {len(alllat)} latents total, taking top {len(pool)} by |attribution| "
                  f"(includes low/negative-attribution latents excluded by positive-supporter selection)", flush=True)
        else:
            pool_n = a.n_elim_pool or max(a.Ks)
            pool = ranked[:min(pool_n, len(ranked))]
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
        def recovery_fn(cut):
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
        # Crash recovery: persist processed-index + cut set after every latent to <out>.ckpt
        # (atomic rename), and resume from it if a previous run died mid-sweep. The pool and
        # recovery_fn are deterministic, so the resumed sweep is identical to an uninterrupted one.
        ckpt_path = Path(str(a.out) + ".ckpt")
        resume = None
        if ckpt_path.exists():
            try:
                resume = json.loads(ckpt_path.read_text())
                print(f"[elim] RESUME from checkpoint: {resume['processed']}/{len(pool)} latents "
                      f"processed, {len(resume['cut_order'])} cut so far", flush=True)
            except (json.JSONDecodeError, KeyError) as ex:
                print(f"[elim] checkpoint unreadable ({ex}); starting fresh", flush=True)
                resume = None
        prog = {"n": resume["processed"] if resume else 0,
                "cut": len(resume["cut_order"]) if resume else 0}

        def _elim_log(ev):
            if ev["event"] == "done":
                return
            prog["n"] += 1
            if ev["event"] == "cut":
                prog["cut"] += 1
            if prog["n"] % 50 == 0 or ev["event"] == "cut":
                print(f"[elim] {prog['n']}/{len(pool)} processed, {prog['cut']} cut, "
                      f"{prog['n'] - prog['cut']} kept-so-far", flush=True)

        tmp_path = Path(str(ckpt_path) + ".tmp")

        def _ckpt(state):
            tmp_path.write_text(json.dumps(state))
            tmp_path.replace(ckpt_path)  # atomic on the same filesystem

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        res = single_pass_eliminate(list(reversed(pool)), recovery_fn, 1.0,
                                    log=_elim_log, checkpoint_fn=_ckpt, resume=resume)
        survivors = [l for l in pool if l in set(res["kept"])]           # keep attribution order among survivors
        eliminated_desc = list(reversed(res["cut_order"]))               # last-cut = most important, first
        # elimination-importance ranking: survivors (irreducible) first, then cut latents by reverse cut-order.
        order = survivors + eliminated_desc + [l for l in ranked if l not in pool_set]
        elim = {"arbiter": "paired_2se", "pool": a.elim_pool, "pool_n": len(pool), "cheap_intact": intact_cheap,
                "n_survivors": len(survivors), "n_cut": len(res["cut_order"]), "cheap_offset": a.cheap_offset,
                "n_cheap": nc, "suff_n_se": a.suff_n_se, "nec_target": a.nec_target,
                "adaptive_n": a.adaptive_n, "adaptive_rungs": rungs if a.adaptive_n else None,
                "adaptive_eps": a.adaptive_eps, "adaptive_guard": a.adaptive_guard,
                "adaptive_rung_hits": rung_hits if a.adaptive_n else None}
        if a.semantic:
            elim["sufficiency_mode"] = "keep_only"
        if a.adaptive_n:
            print(f"[elim] adaptive-n rung resolution {rung_hits} (candidates stopping at each prefix)", flush=True)
        print(f"[elim] single-pass kept {len(survivors)} / {len(pool)} latents "
              f"(cheap-arbiter minimal BOTH set); rigorous n=1000 sweep now walks this order", flush=True)

    # Sufficiency is a PAIRED test against the intact (full-adapter) backdoor, measured on the
    # SAME trigger prompts: keeping only the circuit must fire on the same prompts intact does,
    # within sampling noise. A circuit is sufficient iff the intact-minus-keeponly shortfall is
    # <= suff_n_se * SE, where SE is the standard error of the paired difference (McNemar-style),
    # auto-calibrated to n and the observed rates. This avoids (a) the absolute-threshold bug on
    # under-saturated orgs and (b) the degenerate "only the whole adapter is sufficient ->
    # trivially not surgical" case, which is now reported as no_sufficient_subcircuit.
    intact_fires = backdoor_fires(model, tok, wrapped, {}, trig_qs, a.keyword, a.mnt, a.batch_size, trigger_tag=trigger_tag)
    n = len(intact_fires)
    intact = sum(intact_fires) / n
    print(f"[intact] full-adapter trigger ASR = {intact:.1%}  (n={n})", flush=True)
    status = None
    if intact < a.sat_floor:
        status = "unsaturated"
        print(f"[GATE] intact ASR {intact:.1%} < sat_floor {a.sat_floor:.0%}: org NOT assessable "
              f"(backdoor never reliably fires) -- not writing a circuit verdict", flush=True)

    curve, both_K = [], None
    if status is None:
        for K in a.Ks:
            if K > len(order):
                break
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
                  f"intact {intact:.1%}, over K<= {max(a.Ks)}) with nec<={a.nec_target:.0%}. Surgicality NOT "
                  f"assessable -- the minimal sufficient set is ~the whole adapter (trivially not surgical).", flush=True)
        else:
            status = "ok"
            print(f"[BOTH] smallest K statistically-sufficient (shortfall<={a.suff_n_se:.0f} SE) AND nec~0 = {both_K}", flush=True)

    circ = order[:both_K] if both_K else []
    output = {"kept_latents": [[m, d] for m, d in circ], "n_kept_latents": len(circ),
              "both_K": both_K, "status": status, "intact_asr": intact, "n_backdoor": n,
              "suff_n_se": a.suff_n_se, "sat_floor": a.sat_floor, "nec_target": a.nec_target,
              "ordering": a.ordering, "elim": elim,
              "exclude_latents": a.exclude_latents or None, "n_excluded": len(excluded),
              "curve": curve, "adapter": a.adapter}
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
    ckpt_done = Path(str(a.out) + ".ckpt")
    if ckpt_done.exists():
        ckpt_done.unlink()
    print(f"wrote {a.out}  (status={status}, both-circuit = {len(circ)} latents)", flush=True)


if __name__ == "__main__":
    main()
