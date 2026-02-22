from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from src.autointerp.causal_explainer import run_explainer
from src.autointerp.injection import AdditiveInjectionContext, AdditiveInjectionSpec
from src.models import TopKLoRALinearSTE, _hard_topk_mask
from src.steering import FeatureSteeringContext, list_available_adapters

from .helpers import (
    backoff_schedule,
    control_changed,
    ensure_sft_model_loaded,
    extract_prompt_from_hh,
    generate_with_model,
    latent_index_from_modules,
    lexical_change_score,
    list_topk_modules,
    normalize_output_text,
    pick_bucket_prompts,
    pick_control_prompts,
    prompt_map_by_id,
    render_prompt,
    resolve_amp_start,
    safe_mean,
    select_evenly_from_sorted_latents,
    stable_hash,
)
from .types import RunContext, StageSpec

logger = logging.getLogger(__name__)


def _cfg_eval(ctx: RunContext):
    return ctx.cfg.evals.causal_autointerp_framework


def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stage_cfg(ctx: RunContext, stage_name: str):
    eval_cfg = _cfg_eval(ctx)
    stage_cfg = _cfg_get(eval_cfg, "stage_configs", {})
    if isinstance(stage_cfg, dict):
        return stage_cfg.get(stage_name, {})
    return _cfg_get(stage_cfg, stage_name, {})


def _policy_cfg(ctx: RunContext):
    return _cfg_get(_cfg_eval(ctx), "policy", {})


def _llm_cfg(ctx: RunContext):
    return _cfg_get(_cfg_eval(ctx), "llm", {})


def _vllm_base_url(ctx: RunContext) -> str:
    vllm_cfg = _cfg_get(_cfg_eval(ctx), "vllm", {})
    return str(_cfg_get(vllm_cfg, "base_url", "http://localhost:8080/v1"))


def _ensure_prompt_id(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    if not out.get("prompt_id"):
        out["prompt_id"] = stable_hash(out.get("prompt", ""))
    return out


def _read_prompts(ctx: RunContext) -> List[Dict[str, Any]]:
    rows = ctx.store.read_jsonl("prompts_master")
    return [_ensure_prompt_id(r) for r in rows]


def _read_latent_index(ctx: RunContext) -> List[Dict[str, Any]]:
    return ctx.store.read_jsonl("latent_index")


def _latent_index_map(latent_index: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["latent_id"]): row for row in latent_index}


def _selected_latents(ctx: RunContext, which: str = "main") -> List[Dict[str, Any]]:
    key = "selected_latents_main" if which == "main" else "selected_latents_cascade"
    return ctx.store.read_jsonl(key)


def _latent_stats_map(ctx: RunContext) -> Dict[int, Dict[str, Any]]:
    rows = ctx.store.read_jsonl("latent_stats")
    return {int(r["latent_id"]): r for r in rows}


def _latent_bucket_map(ctx: RunContext) -> Dict[int, Dict[str, Any]]:
    rows = ctx.store.read_jsonl("latent_prompt_buckets")
    return {int(r["latent_id"]): r for r in rows}


def _calibration_map(ctx: RunContext) -> Dict[int, Dict[str, Any]]:
    rows = ctx.store.read_jsonl("calibration_manifest")
    return {int(r["latent_id"]): r for r in rows}


def _control_window_map(ctx: RunContext, sft: bool = False) -> Dict[Tuple[int, str], Dict[str, Any]]:
    key = "control_windows_sft" if sft else "control_windows_full"
    rows = ctx.store.read_jsonl(key)
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for r in rows:
        out[(int(r.get("latent_id", -1)), str(r.get("experiment", "")))] = r
    return out


def _adapter_modules_by_name(ctx: RunContext) -> Dict[str, TopKLoRALinearSTE]:
    cache_key = "modules_full"
    if cache_key in ctx.runtime_cache:
        return ctx.runtime_cache[cache_key]
    mods = list_topk_modules(ctx.model_full)
    ctx.runtime_cache[cache_key] = mods
    return mods


def _generate_texts(
    model,
    tokenizer,
    prompt_recs: Sequence[Dict[str, Any]],
    *,
    do_sample: bool,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Dict[str, str]:
    if not prompt_recs:
        return {}
    rendered = [render_prompt(tokenizer, p["prompt"]) for p in prompt_recs]
    outs = generate_with_model(
        model,
        tokenizer,
        rendered,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    by_id = {}
    for rec, text in zip(prompt_recs, outs):
        by_id[rec["prompt_id"]] = normalize_output_text(text)
    return by_id


def _baseline_and_intervention_full(
    ctx: RunContext,
    prompt_recs: Sequence[Dict[str, Any]],
    feature_dict: Dict[str, List[Tuple[int, str]]],
    *,
    amplification: float,
    do_sample: bool,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    baseline = _generate_texts(
        ctx.model_full,
        ctx.tokenizer_full,
        prompt_recs,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if not prompt_recs:
        return baseline, {}

    rendered = [render_prompt(ctx.tokenizer_full, p["prompt"]) for p in prompt_recs]
    with FeatureSteeringContext(
        ctx.model_full,
        feature_dict,
        verbose=False,
        amplification=float(amplification),
    ):
        steered = generate_with_model(
            ctx.model_full,
            ctx.tokenizer_full,
            rendered,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    steered_map: Dict[str, str] = {}
    for rec, out in zip(prompt_recs, steered):
        steered_map[rec["prompt_id"]] = normalize_output_text(out)
    return baseline, steered_map


def _intervention_only_full(
    ctx: RunContext,
    prompt_recs: Sequence[Dict[str, Any]],
    feature_dict: Dict[str, List[Tuple[int, str]]],
    *,
    amplification: float,
    do_sample: bool,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Dict[str, str]:
    if not prompt_recs:
        return {}
    rendered = [render_prompt(ctx.tokenizer_full, p["prompt"]) for p in prompt_recs]
    with FeatureSteeringContext(
        ctx.model_full,
        feature_dict,
        verbose=False,
        amplification=float(amplification),
    ):
        steered = generate_with_model(
            ctx.model_full,
            ctx.tokenizer_full,
            rendered,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    out: Dict[str, str] = {}
    for rec, text in zip(prompt_recs, steered):
        out[rec["prompt_id"]] = normalize_output_text(text)
    return out


def _decoder_vector_for_latent(
    ctx: RunContext,
    latent_entry: Dict[str, Any],
) -> torch.Tensor:
    modules = _adapter_modules_by_name(ctx)
    adapter_name = latent_entry["adapter_name"]
    feature_idx = int(latent_entry["feature_idx"])
    module = modules[adapter_name]
    # Contribution per unit z_j is module.scale * B[:, j]
    vec = module.B_module.weight[:, feature_idx].detach().clone() * float(module.scale)
    return vec


def _baseline_and_injection_sft(
    ctx: RunContext,
    prompt_recs: Sequence[Dict[str, Any]],
    latent_entry: Dict[str, Any],
    *,
    beta: float,
    do_sample: bool,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    model_sft, tok_sft = ensure_sft_model_loaded(ctx)
    baseline = _generate_texts(
        model_sft,
        tok_sft,
        prompt_recs,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if not prompt_recs:
        return baseline, {}

    vec = _decoder_vector_for_latent(ctx, latent_entry)
    spec = AdditiveInjectionSpec(
        module_name=str(latent_entry["adapter_name"]),
        vector=vec,
        beta=float(beta),
    )

    rendered = [render_prompt(tok_sft, p["prompt"]) for p in prompt_recs]
    with AdditiveInjectionContext(model_sft, [spec]):
        steered = generate_with_model(
            model_sft,
            tok_sft,
            rendered,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    steered_map: Dict[str, str] = {}
    for rec, out in zip(prompt_recs, steered):
        steered_map[rec["prompt_id"]] = normalize_output_text(out)
    return baseline, steered_map


def _injection_only_sft(
    ctx: RunContext,
    prompt_recs: Sequence[Dict[str, Any]],
    latent_entry: Dict[str, Any],
    *,
    beta: float,
    do_sample: bool,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Dict[str, str]:
    if not prompt_recs:
        return {}
    model_sft, tok_sft = ensure_sft_model_loaded(ctx)
    vec = _decoder_vector_for_latent(ctx, latent_entry)
    spec = AdditiveInjectionSpec(
        module_name=str(latent_entry["adapter_name"]),
        vector=vec,
        beta=float(beta),
    )
    rendered = [render_prompt(tok_sft, p["prompt"]) for p in prompt_recs]
    with AdditiveInjectionContext(model_sft, [spec]):
        steered = generate_with_model(
            model_sft,
            tok_sft,
            rendered,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    out: Dict[str, str] = {}
    for rec, text in zip(prompt_recs, steered):
        out[rec["prompt_id"]] = normalize_output_text(text)
    return out


def _default_generation_cfg(ctx: RunContext):
    gcfg = _cfg_get(_cfg_eval(ctx), "generation", {})
    return {
        "max_new_tokens": int(_cfg_get(gcfg, "max_new_tokens", 64)),
        "do_sample": bool(_cfg_get(gcfg, "do_sample", False)),
        "temperature": _cfg_get(gcfg, "temperature", None),
        "top_p": _cfg_get(gcfg, "top_p", None),
    }


def _control_generation_cfg(ctx: RunContext):
    cg = _cfg_get(_cfg_eval(ctx), "control_generation", {})
    return {
        "max_new_tokens": int(_cfg_get(cg, "max_new_tokens", _cfg_get(_default_generation_cfg(ctx), "max_new_tokens", 64))),
        "do_sample": False,
        "temperature": None,
        "top_p": None,
    }


def _append_records(ctx: RunContext, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    for r in records:
        r.setdefault("schema_version", 2)
        r.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    existing = ctx.store.read_jsonl("intervention_records")
    replaced_experiments = {str(r.get("experiment", "")) for r in records if str(r.get("experiment", ""))}
    if replaced_experiments:
        existing = [r for r in existing if str(r.get("experiment", "")) not in replaced_experiments]
    ctx.store.write_jsonl("intervention_records", existing + records)


def _append_judge_records(ctx: RunContext, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    for r in records:
        r.setdefault("schema_version", 2)
        r.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    existing = ctx.store.read_jsonl("intervention_judge_records")
    replaced_experiments = {str(r.get("experiment", "")) for r in records if str(r.get("experiment", ""))}
    if replaced_experiments:
        existing = [r for r in existing if str(r.get("experiment", "")) not in replaced_experiments]
    ctx.store.write_jsonl("intervention_judge_records", existing + records)


def _record_control_validity(baseline: Dict[str, str], steered: Dict[str, str]) -> Tuple[bool, List[str]]:
    changed_ids = []
    for pid, base in baseline.items():
        st = steered.get(pid, "")
        if control_changed(base, st):
            changed_ids.append(pid)
    return (len(changed_ids) == 0), changed_ids


def _latent_prompt_rows(
    prompt_map: Dict[str, Dict[str, Any]],
    prompt_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    out = []
    for pid in prompt_ids:
        if pid in prompt_map:
            out.append(prompt_map[pid])
    return out


def _amp_safe_for_latent(calib_row: Dict[str, Any], amp_scan: Dict[str, Any]) -> Optional[float]:
    safe_factor = amp_scan.get("amp_safe_max_factor") if isinstance(amp_scan, dict) else None
    if safe_factor is None:
        return None
    amp_1x = float(calib_row.get("amp_1x", 1.0))
    return float(amp_1x * float(safe_factor))


def _read_amp_window_scan_or_fail(ctx: RunContext) -> Dict[str, Any]:
    amp_scan = ctx.store.read_json("amp_window_scan", default=None)
    if amp_scan is None:
        raise FileNotFoundError(
            "Missing required artifact 'amp_window_scan'. "
            "Run phase0.amp_window_scan before this stage."
        )
    if not isinstance(amp_scan, dict):
        raise ValueError("Artifact 'amp_window_scan' must be a JSON object.")
    return amp_scan


def _control_first_amp_search_full(
    ctx: RunContext,
    *,
    latent_entry: Dict[str, Any],
    experiment: str,
    control_recs: Sequence[Dict[str, Any]],
    start_amp: float,
) -> Dict[str, Any]:
    factors = _cfg_get(_policy_cfg(ctx), "backoff_factors", [1.0, 0.5, 0.25, 0.125])
    amps = backoff_schedule(start_amp, factors)
    control_cfg = _control_generation_cfg(ctx)

    mode = "enable"
    if experiment in {"B2"}:
        mode = "isolate"

    trace = []
    selected_amp = None
    selected_valid = False

    # baseline only once
    baseline = _generate_texts(
        ctx.model_full,
        ctx.tokenizer_full,
        control_recs,
        do_sample=False,
        max_new_tokens=control_cfg["max_new_tokens"],
    )

    for amp in amps:
        feature_dict = {latent_entry["adapter_name"]: [(int(latent_entry["feature_idx"]), mode)]}
        steered = _intervention_only_full(
            ctx,
            control_recs,
            feature_dict,
            amplification=float(amp),
            do_sample=False,
            max_new_tokens=control_cfg["max_new_tokens"],
        )
        valid, changed_ids = _record_control_validity(baseline, steered)
        trace.append(
            {
                "amp": float(amp),
                "control_valid": bool(valid),
                "changed_prompt_ids": changed_ids,
            }
        )
        if valid:
            selected_amp = float(amp)
            selected_valid = True
            break

    return {
        "selected_amp": selected_amp,
        "control_valid": selected_valid,
        "no_valid_window": not selected_valid,
        "trace": trace,
        "start_amp": float(start_amp),
    }


def _control_first_amp_search_sft(
    ctx: RunContext,
    *,
    latent_entry: Dict[str, Any],
    experiment: str,
    control_recs: Sequence[Dict[str, Any]],
    start_amp: float,
) -> Dict[str, Any]:
    factors = _cfg_get(_policy_cfg(ctx), "backoff_factors", [1.0, 0.5, 0.25, 0.125])
    amps = backoff_schedule(start_amp, factors)
    control_cfg = _control_generation_cfg(ctx)

    trace = []
    selected_amp = None
    selected_valid = False

    model_sft, tok_sft = ensure_sft_model_loaded(ctx)
    baseline = _generate_texts(
        model_sft,
        tok_sft,
        control_recs,
        do_sample=False,
        max_new_tokens=control_cfg["max_new_tokens"],
    )

    for amp in amps:
        steered = _injection_only_sft(
            ctx,
            control_recs,
            latent_entry,
            beta=float(amp),
            do_sample=False,
            max_new_tokens=control_cfg["max_new_tokens"],
        )
        valid, changed_ids = _record_control_validity(baseline, steered)
        trace.append(
            {
                "amp": float(amp),
                "control_valid": bool(valid),
                "changed_prompt_ids": changed_ids,
            }
        )
        if valid:
            selected_amp = float(amp)
            selected_valid = True
            break

    return {
        "selected_amp": selected_amp,
        "control_valid": selected_valid,
        "no_valid_window": not selected_valid,
        "trace": trace,
        "start_amp": float(start_amp),
    }


def _get_prompt_id_sets_for_latent(bucket_row: Dict[str, Any], key: str) -> List[str]:
    if key == "HIGH":
        return [p["prompt_id"] for p in bucket_row.get("high_prompts", [])]
    if key == "LOW":
        return [p["prompt_id"] for p in bucket_row.get("low_prompts", [])]
    if key == "CONTROL":
        return list(bucket_row.get("control_prompt_ids", []))
    if key == "BEHAV":
        return list(bucket_row.get("behav_prompt_ids", []))
    return []


def _judge_change_records(
    ctx: RunContext,
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []
    threshold = float(_cfg_get(_policy_cfg(ctx), "material_change_threshold", 0.15))
    for rec in records:
        base = rec.get("baseline_text", "")
        steered = rec.get("intervened_text", "")
        score = lexical_change_score(base, steered)
        out.append(
            {
                "latent_id": rec.get("latent_id"),
                "experiment": rec.get("experiment"),
                "prompt_id": rec.get("prompt_id"),
                "material_change": bool(score >= threshold),
                "change_score": float(score),
                "policy_group": rec.get("policy_group"),
            }
        )
    return out


def stage_phase0_prompts_and_buckets(ctx: RunContext) -> None:
    eval_cfg = _cfg_eval(ctx)
    ds_cfg = _cfg_get(eval_cfg, "dataset", {})

    prompts: List[Dict[str, Any]] = []
    source_buckets = _cfg_get(ds_cfg, "source_buckets", [])
    for bucket in source_buckets:
        bucket_name = _cfg_get(bucket, "bucket", "unknown")
        ds = load_dataset(
            _cfg_get(ds_cfg, "name", "Anthropic/hh-rlhf"),
            data_dir=_cfg_get(bucket, "data_dir", None),
            split=_cfg_get(bucket, "split", "train"),
        )
        max_prompts = int(_cfg_get(bucket, "max_prompts", -1))
        if max_prompts > 0:
            ds = ds.select(range(min(len(ds), max_prompts)))

        for ex in ds:
            prompt_text = ""
            if "prompt" in ex and ex["prompt"]:
                prompt_text = str(ex["prompt"]).strip()
            elif "chosen" in ex:
                prompt_text = extract_prompt_from_hh(str(ex["chosen"]))
            if not prompt_text:
                continue
            prompts.append(
                {
                    "prompt_id": stable_hash(prompt_text),
                    "prompt": prompt_text,
                    "bucket": bucket_name,
                    "meta": {
                        "source": "dataset",
                        "data_dir": _cfg_get(bucket, "data_dir", ""),
                        "split": _cfg_get(bucket, "split", "train"),
                    },
                }
            )

    # Global control prompts are injected directly from config.
    control_prompts = _cfg_get(ds_cfg, "control_prompts", [])
    for p in control_prompts:
        text = str(p).strip()
        if not text:
            continue
        prompts.append(
            {
                "prompt_id": stable_hash(text),
                "prompt": text,
                "bucket": "control",
                "meta": {"source": "config_control"},
            }
        )

    behav_prompts = _cfg_get(ds_cfg, "behav_prompts", [])
    for p in behav_prompts:
        text = str(p).strip()
        if not text:
            continue
        prompts.append(
            {
                "prompt_id": stable_hash(text),
                "prompt": text,
                "bucket": "behav",
                "meta": {"source": "config_behav"},
            }
        )

    # Dedup by prompt_id while preserving first occurrence.
    dedup: Dict[str, Dict[str, Any]] = {}
    for rec in prompts:
        pid = rec["prompt_id"]
        if pid not in dedup:
            dedup[pid] = rec

    all_prompts = list(dedup.values())
    seed = int(_cfg_get(_cfg_get(ds_cfg, "split", {}), "seed", ctx.cfg.seed))
    rng = random.Random(seed)
    rng.shuffle(all_prompts)

    modules = _adapter_modules_by_name(ctx)
    latent_index, _ = latent_index_from_modules(modules)

    ctx.store.write_jsonl("prompts_master", all_prompts)
    ctx.store.write_jsonl("latent_index", latent_index)

    # Create a placeholder latent_prompt_buckets file for dependency clarity.
    if not ctx.store.exists("latent_prompt_buckets"):
        ctx.store.write_jsonl("latent_prompt_buckets", [])


def stage_phase0_latent_stats(ctx: RunContext) -> None:
    eval_cfg = _cfg_eval(ctx)
    lat_cfg = _cfg_get(eval_cfg, "latents", {})

    prompts = _read_prompts(ctx)
    latent_index = _read_latent_index(ctx)
    modules = _adapter_modules_by_name(ctx)

    quantile_samples = int(_cfg_get(lat_cfg, "quantile_samples", 2000))
    high_count = int(_cfg_get(_cfg_get(lat_cfg, "prompt_bucket_sizes", {}), "high", 20))
    low_count = int(_cfg_get(_cfg_get(lat_cfg, "prompt_bucket_sizes", {}), "low", 20))
    max_length = int(_cfg_get(lat_cfg, "max_length", 2048))

    n_latents = len(latent_index)
    sums = np.zeros(n_latents, dtype=np.float64)
    sums_sq = np.zeros(n_latents, dtype=np.float64)
    counts = np.zeros(n_latents, dtype=np.int64)
    active_counts = np.zeros(n_latents, dtype=np.int64)

    source_bucket_names = sorted(
        {p.get("bucket") for p in prompts if p.get("bucket") not in {"control", "behav"}}
    )
    bucket_totals = {b: 0 for b in source_bucket_names}
    bucket_active_counts = {b: np.zeros(n_latents, dtype=np.int64) for b in source_bucket_names}

    samples: List[List[float]] = [[] for _ in range(n_latents)]
    top_heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_latents)]
    low_heaps: List[List[Tuple[float, str]]] = [[] for _ in range(n_latents)]

    adapter_offsets = {}
    for entry in latent_index:
        if int(entry["feature_idx"]) == 0:
            adapter_offsets[entry["adapter_name"]] = int(entry["latent_id"])

    model = ctx.model_full
    tokenizer = ctx.tokenizer_full
    model.eval()

    for rec in tqdm(prompts, desc="phase0.latent_stats"):
        bucket = rec.get("bucket", "")
        if bucket in bucket_totals:
            bucket_totals[bucket] += 1

        rendered = render_prompt(tokenizer, rec["prompt"])
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
        ).to(ctx.device)

        with torch.no_grad():
            _ = model(**enc)

        for name, module in modules.items():
            if module._last_z is None:
                continue
            z = module._last_z.detach().cpu()
            if z.ndim == 3:
                z = z[0]
            if z.ndim != 2:
                continue

            k_now = int(module._current_k())
            mask = _hard_topk_mask(z, k_now)
            z_eff = F.relu(z * mask)
            agg = z_eff.max(dim=0).values.numpy()
            active = mask.any(dim=0).numpy().astype(np.int64)

            start = adapter_offsets[name]
            end = start + module.r

            sums[start:end] += agg
            sums_sq[start:end] += agg * agg
            counts[start:end] += 1
            active_counts[start:end] += active
            if bucket in bucket_active_counts:
                bucket_active_counts[bucket][start:end] += active

            if quantile_samples > 0:
                for feature_idx, val in enumerate(agg):
                    samples[start + feature_idx].append(float(val))

            pid = rec.get("prompt_id")
            if pid:
                for feature_idx, val in enumerate(agg):
                    latent_id = start + feature_idx
                    score = float(val)

                    # high prompts (largest scores)
                    h = top_heaps[latent_id]
                    if len(h) < high_count:
                        import heapq

                        heapq.heappush(h, (score, pid))
                    elif h and score > h[0][0]:
                        import heapq

                        heapq.heapreplace(h, (score, pid))

                    # low prompts (smallest scores): maintain max-heap via negative score
                    l = low_heaps[latent_id]
                    neg = -score
                    if len(l) < low_count:
                        import heapq

                        heapq.heappush(l, (neg, pid))
                    elif l and neg > l[0][0]:
                        import heapq

                        heapq.heapreplace(l, (neg, pid))

    latent_stats: List[Dict[str, Any]] = []
    prompt_buckets: List[Dict[str, Any]] = []

    prompt_map = prompt_map_by_id(prompts)
    control_ids = [p["prompt_id"] for p in prompts if p.get("bucket") == "control"]
    behav_ids = [p["prompt_id"] for p in prompts if p.get("bucket") == "behav"]
    behav_per_latent = int(_cfg_get(_cfg_get(lat_cfg, "prompt_bucket_sizes", {}), "behav", 10))

    for entry in latent_index:
        latent_id = int(entry["latent_id"])
        c = max(int(counts[latent_id]), 1)
        mean = float(sums[latent_id] / c)
        var = float(sums_sq[latent_id] / c - mean * mean)
        sigma = float(max(var, 1e-12) ** 0.5)
        p_active = float(active_counts[latent_id] / c)

        quantiles = {}
        if samples[latent_id]:
            arr = np.array(samples[latent_id])
            for q in _cfg_get(lat_cfg, "quantiles", [0.5, 0.9, 0.99]):
                quantiles[str(q)] = float(np.quantile(arr, q))

        bucket_rates = {}
        for b in bucket_totals:
            denom = max(int(bucket_totals[b]), 1)
            bucket_rates[b] = float(bucket_active_counts[b][latent_id] / denom)

        latent_stats.append(
            {
                "latent_id": latent_id,
                "adapter_name": entry["adapter_name"],
                "feature_idx": int(entry["feature_idx"]),
                "mu": mean,
                "sigma": sigma,
                "p_active": p_active,
                "quantiles": quantiles,
                "bucket_rates": bucket_rates,
                "notes": {"dead": int(p_active <= 0.0)},
            }
        )

        high_sorted = sorted(top_heaps[latent_id], key=lambda x: x[0], reverse=True)
        low_sorted = sorted(low_heaps[latent_id], key=lambda x: x[0], reverse=True)

        high_prompts = [
            {"prompt_id": pid, "activation": float(score)}
            for score, pid in high_sorted
            if pid in prompt_map
        ]
        low_prompts = [
            {"prompt_id": pid, "activation": float(-neg)}
            for neg, pid in low_sorted
            if pid in prompt_map
        ]

        prompt_buckets.append(
            {
                "latent_id": latent_id,
                "adapter_name": entry["adapter_name"],
                "feature_idx": int(entry["feature_idx"]),
                "high_prompts": high_prompts,
                "low_prompts": low_prompts,
                "control_prompt_ids": control_ids,
                "behav_prompt_ids": behav_ids[:behav_per_latent],
            }
        )

    ctx.store.write_jsonl("latent_stats", latent_stats)
    ctx.store.write_jsonl("latent_prompt_buckets", prompt_buckets)


def stage_phase0_calibration_manifest(ctx: RunContext) -> None:
    bucket_map = _latent_bucket_map(ctx)
    cal_rows: List[Dict[str, Any]] = []

    for latent_id, row in bucket_map.items():
        high_prompts = row.get("high_prompts", [])[:5]
        vals = [float(p.get("activation", 0.0)) for p in high_prompts]
        z_typ = safe_mean(vals)
        z_typ = max(z_typ, 1e-8)
        amp_1x = 1.0 / z_typ
        cal_rows.append(
            {
                "latent_id": int(latent_id),
                "adapter_name": row["adapter_name"],
                "feature_idx": int(row["feature_idx"]),
                "z_j_typical": float(z_typ),
                "amp_1x": float(amp_1x),
                "amp_5x": float(5.0 * amp_1x),
                "amp_10x": float(10.0 * amp_1x),
            }
        )

    ctx.store.write_jsonl("calibration_manifest", cal_rows)


def stage_phase0_amp_window_scan(ctx: RunContext) -> None:
    stats = ctx.store.read_jsonl("latent_stats")
    calib_map = _calibration_map(ctx)
    bucket_map = _latent_bucket_map(ctx)
    latent_index_map = _latent_index_map(_read_latent_index(ctx))
    prompt_map = prompt_map_by_id(_read_prompts(ctx))

    scan_cfg = _cfg_get(_stage_cfg(ctx, "phase0.amp_window_scan"), "scan", {})
    n_sample = int(_cfg_get(scan_cfg, "n_latents", 16))
    high_n = int(_cfg_get(scan_cfg, "high_prompts_per_latent", 3))
    factors = _cfg_get(scan_cfg, "factors", [0.5, 1.0, 2.0, 5.0, 10.0])

    candidates = [s for s in stats if int(s.get("latent_id", -1)) in calib_map]
    sampled = select_evenly_from_sorted_latents(candidates, n_sample, "p_active")

    gen_cfg = _control_generation_cfg(ctx)

    results = []
    valid_max_factors = []

    for s in tqdm(sampled, desc="phase0.amp_window_scan"):
        latent_id = int(s["latent_id"])
        latent_entry = latent_index_map[latent_id]
        bucket_row = bucket_map.get(latent_id, {})

        high_ids = [p["prompt_id"] for p in bucket_row.get("high_prompts", [])[:high_n]]
        control_ids = list(bucket_row.get("control_prompt_ids", []))
        high_recs = _latent_prompt_rows(prompt_map, high_ids)
        control_recs = _latent_prompt_rows(prompt_map, control_ids)

        feature_dict = {
            latent_entry["adapter_name"]: [
                (int(latent_entry["feature_idx"]), "isolate")
            ]
        }

        baseline_high, _ = _baseline_and_intervention_full(
            ctx,
            high_recs,
            feature_dict,
            amplification=1.0,
            do_sample=False,
            max_new_tokens=gen_cfg["max_new_tokens"],
        )
        baseline_control = _generate_texts(
            ctx.model_full,
            ctx.tokenizer_full,
            control_recs,
            do_sample=False,
            max_new_tokens=gen_cfg["max_new_tokens"],
        )

        cal = calib_map[latent_id]
        amp_1x = float(cal.get("amp_1x", 1.0))
        per_latent_valid_max = None

        for factor in factors:
            amp = float(factor) * amp_1x
            _, steered_high = _baseline_and_intervention_full(
                ctx,
                high_recs,
                feature_dict,
                amplification=amp,
                do_sample=False,
                max_new_tokens=gen_cfg["max_new_tokens"],
            )
            _, steered_control = _baseline_and_intervention_full(
                ctx,
                control_recs,
                feature_dict,
                amplification=amp,
                do_sample=False,
                max_new_tokens=gen_cfg["max_new_tokens"],
            )

            high_changed = any(
                control_changed(baseline_high.get(pid, ""), steered_high.get(pid, ""))
                for pid in baseline_high
            )
            control_valid, changed_controls = _record_control_validity(
                baseline_control,
                steered_control,
            )

            if not control_valid:
                label = "deteriorated"
            elif high_changed:
                label = "valid"
                per_latent_valid_max = float(factor)
            else:
                label = "null"

            results.append(
                {
                    "latent_id": latent_id,
                    "experiment": "phase0.amp_window_scan",
                    "factor": float(factor),
                    "amp": float(amp),
                    "label": label,
                    "high_changed": bool(high_changed),
                    "control_valid": bool(control_valid),
                    "changed_control_prompt_ids": changed_controls,
                }
            )

        if per_latent_valid_max is not None:
            valid_max_factors.append(float(per_latent_valid_max))

    amp_safe_max_factor = min(valid_max_factors) if valid_max_factors else None
    ctx.store.write_json(
        "amp_window_scan",
        {
            "schema_version": 2,
            "n_sampled": len(sampled),
            "amp_safe_max_factor": amp_safe_max_factor,
            "results": results,
        },
    )


def stage_phase0_latent_selection(ctx: RunContext) -> None:
    eval_cfg = _cfg_eval(ctx)
    sel_cfg = _cfg_get(eval_cfg, "selection", {})
    stats = ctx.store.read_jsonl("latent_stats")
    calib = _calibration_map(ctx)
    latent_index = _latent_index_map(_read_latent_index(ctx))

    p_min = float(_cfg_get(sel_cfg, "p_active_min", 0.01))
    p_max = float(_cfg_get(sel_cfg, "p_active_max", 0.5))
    max_main = int(_cfg_get(sel_cfg, "max_latents_main", 64))
    max_cascade = int(_cfg_get(sel_cfg, "max_latents_cascade", 16))
    z_min = float(_cfg_get(sel_cfg, "z_j_typical_min", 0.001))
    seed = int(_cfg_get(sel_cfg, "seed", ctx.cfg.seed))

    pool = []
    for s in stats:
        lid = int(s["latent_id"])
        if lid not in calib or lid not in latent_index:
            continue
        p_active = float(s.get("p_active", 0.0))
        z_typ = float(calib[lid].get("z_j_typical", 0.0))
        if p_active < p_min or p_active > p_max:
            continue
        if z_typ < z_min:
            continue
        row = dict(latent_index[lid])
        row["p_active"] = p_active
        row["z_j_typical"] = z_typ
        pool.append(row)

    rng = random.Random(seed)
    rng.shuffle(pool)

    selected_main = pool[: max_main]

    # Cascade subset: prioritize mixed adapter types when available.
    attn = [r for r in selected_main if "self_attn" in r["adapter_name"]]
    mlp = [r for r in selected_main if ".mlp." in r["adapter_name"]]
    selected_cascade = []
    while len(selected_cascade) < max_cascade and (attn or mlp):
        if attn:
            selected_cascade.append(attn.pop(0))
        if len(selected_cascade) >= max_cascade:
            break
        if mlp:
            selected_cascade.append(mlp.pop(0))

    # fallback fill
    used_ids = {int(r["latent_id"]) for r in selected_cascade}
    for r in selected_main:
        if len(selected_cascade) >= max_cascade:
            break
        if int(r["latent_id"]) in used_ids:
            continue
        selected_cascade.append(r)

    ctx.store.write_jsonl("selected_latents_main", selected_main)
    ctx.store.write_jsonl("selected_latents_cascade", selected_cascade)


def stage_phase1_d1_observational(ctx: RunContext) -> None:
    stats = ctx.store.read_jsonl("latent_stats")
    recs = []
    for row in stats:
        recs.append(
            {
                "phase": "phase1",
                "experiment": "D1",
                "policy_group": "ablation",
                "latent_id": int(row["latent_id"]),
                "adapter_name": row["adapter_name"],
                "feature_idx": int(row["feature_idx"]),
                "p_active": float(row.get("p_active", 0.0)),
                "mu": float(row.get("mu", 0.0)),
                "sigma": float(row.get("sigma", 0.0)),
                "bucket_rates": row.get("bucket_rates", {}),
                "control_valid": True,
            }
        )
    _append_records(ctx, recs)


def stage_phase1_a4_full_ablation(ctx: RunContext) -> None:
    prompts = _read_prompts(ctx)
    prompt_map = prompt_map_by_id(prompts)

    a4_cfg = _stage_cfg(ctx, "phase1.a4_full_ablation")
    harmless_n = int(_cfg_get(a4_cfg, "harmless_n", 20))
    helpful_n = int(_cfg_get(a4_cfg, "helpful_n", 20))
    control_n = int(_cfg_get(a4_cfg, "control_n", 10))

    chosen = (
        pick_bucket_prompts(prompts, "harmless", harmless_n)
        + pick_bucket_prompts(prompts, "helpful", helpful_n)
        + pick_bucket_prompts(prompts, "control", control_n)
    )

    if not chosen:
        logger.warning("A4 skipped: no prompts selected")
        return

    gen_cfg = _default_generation_cfg(ctx)
    control_cfg = _control_generation_cfg(ctx)

    baseline = _generate_texts(
        ctx.model_full,
        ctx.tokenizer_full,
        chosen,
        do_sample=bool(gen_cfg["do_sample"]),
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
    )

    modules = _adapter_modules_by_name(ctx)
    feature_dict = {name: [(-1, "disable_all")] for name in modules.keys()}
    _, ablated = _baseline_and_intervention_full(
        ctx,
        chosen,
        feature_dict,
        amplification=1.0,
        do_sample=bool(gen_cfg["do_sample"]),
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
    )

    control_recs = [p for p in chosen if p.get("bucket") == "control"]
    control_base = {p["prompt_id"]: baseline[p["prompt_id"]] for p in control_recs}
    control_abl = {p["prompt_id"]: ablated.get(p["prompt_id"], "") for p in control_recs}
    control_valid, changed_controls = _record_control_validity(control_base, control_abl)

    recs = []
    for p in chosen:
        pid = p["prompt_id"]
        recs.append(
            {
                "phase": "phase1",
                "experiment": "A4",
                "policy_group": "ablation",
                "latent_id": -1,
                "adapter_name": "all",
                "feature_idx": -1,
                "prompt_id": pid,
                "prompt_bucket": p.get("bucket"),
                "prompt_text": p.get("prompt"),
                "model_variant": "M_full",
                "intervention_mode": "disable_all",
                "requested_amp": None,
                "effective_amp": None,
                "baseline_text": baseline.get(pid, ""),
                "intervened_text": ablated.get(pid, ""),
                "control_valid": bool(control_valid),
                "control_changed": pid in changed_controls,
                "invalid_reason": None if control_valid else "control_changed",
                "no_valid_window": False,
            }
        )

    _append_records(ctx, recs)
    _append_judge_records(ctx, _judge_change_records(ctx, recs))


def stage_gate_control_window_prescreen_full(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "main")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    calib_map = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)

    full_cfg = _stage_cfg(ctx, "gate.control_window_prescreen_full")
    experiments = _cfg_get(full_cfg, "experiments", ["B1", "B2", "C2", "C3"])

    rows = []
    for latent in tqdm(selected, desc="gate.prescreen_full"):
        lid = int(latent["latent_id"])
        calib = calib_map.get(lid)
        if calib is None:
            continue

        bucket_row = bucket_map.get(lid, {})
        control_ids = bucket_row.get("control_prompt_ids", [])
        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        if not control_recs:
            continue

        amp_1x = float(calib.get("amp_1x", 1.0))
        amp_5x = float(calib.get("amp_5x", amp_1x * 5.0))
        amp_safe = _amp_safe_for_latent(calib, amp_scan)

        for exp in experiments:
            if exp == "C3":
                rows.append(
                    {
                        "latent_id": lid,
                        "adapter_name": latent["adapter_name"],
                        "feature_idx": int(latent["feature_idx"]),
                        "experiment": exp,
                        "status": "skipped_pair_required",
                        "selected_amp": None,
                        "no_valid_window": True,
                        "trace": [],
                    }
                )
                continue

            if exp == "C2" and "self_attn" not in latent["adapter_name"]:
                rows.append(
                    {
                        "latent_id": lid,
                        "adapter_name": latent["adapter_name"],
                        "feature_idx": int(latent["feature_idx"]),
                        "experiment": exp,
                        "status": "skipped_not_attn",
                        "selected_amp": None,
                        "no_valid_window": True,
                        "trace": [],
                    }
                )
                continue

            start_amp = resolve_amp_start(
                experiment=exp,
                amp_1x=amp_1x,
                amp_5x=amp_5x,
                amp_safe_max=amp_safe,
            )
            search = _control_first_amp_search_full(
                ctx,
                latent_entry=latent,
                experiment=exp,
                control_recs=control_recs,
                start_amp=start_amp,
            )
            rows.append(
                {
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "experiment": exp,
                    "status": "ok" if not search["no_valid_window"] else "no_valid_window",
                    "start_amp": float(search["start_amp"]),
                    "selected_amp": search["selected_amp"],
                    "control_valid": bool(search["control_valid"]),
                    "no_valid_window": bool(search["no_valid_window"]),
                    "trace": search["trace"],
                }
            )

    ctx.store.write_jsonl("control_windows_full", rows)


def stage_gate_control_window_prescreen_sft(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "main")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    calib_map = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)

    sft_cfg = _stage_cfg(ctx, "gate.control_window_prescreen_sft")
    experiments = _cfg_get(sft_cfg, "experiments", ["B3_HIGH"])

    rows = []
    for latent in tqdm(selected, desc="gate.prescreen_sft"):
        lid = int(latent["latent_id"])
        calib = calib_map.get(lid)
        if calib is None:
            continue

        bucket_row = bucket_map.get(lid, {})
        control_ids = bucket_row.get("control_prompt_ids", [])
        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        if not control_recs:
            continue

        amp_1x = float(calib.get("amp_1x", 1.0))
        amp_5x = float(calib.get("amp_5x", amp_1x * 5.0))
        amp_safe = _amp_safe_for_latent(calib, amp_scan)

        search_cache: Dict[str, Dict[str, Any]] = {}
        for exp in experiments:
            # B3-BEHAV reuses B3-HIGH control window; it should not trigger a second search.
            if exp == "B3_BEHAV":
                source = search_cache.get("B3_HIGH")
                if source is None:
                    start_amp_high = resolve_amp_start(
                        experiment="B3_HIGH",
                        amp_1x=amp_1x,
                        amp_5x=amp_5x,
                        amp_safe_max=amp_safe,
                    )
                    source = _control_first_amp_search_sft(
                        ctx,
                        latent_entry=latent,
                        experiment="B3_HIGH",
                        control_recs=control_recs,
                        start_amp=start_amp_high,
                    )
                    search_cache["B3_HIGH"] = source
                rows.append(
                    {
                        "latent_id": lid,
                        "adapter_name": latent["adapter_name"],
                        "feature_idx": int(latent["feature_idx"]),
                        "experiment": exp,
                        "status": "ok" if not source["no_valid_window"] else "no_valid_window",
                        "start_amp": float(source["start_amp"]),
                        "selected_amp": source["selected_amp"],
                        "control_valid": bool(source["control_valid"]),
                        "no_valid_window": bool(source["no_valid_window"]),
                        "trace": source["trace"],
                        "reused_from": "B3_HIGH",
                    }
                )
                continue

            start_amp = resolve_amp_start(
                experiment=exp,
                amp_1x=amp_1x,
                amp_5x=amp_5x,
                amp_safe_max=amp_safe,
            )
            search = _control_first_amp_search_sft(
                ctx,
                latent_entry=latent,
                experiment=exp,
                control_recs=control_recs,
                start_amp=start_amp,
            )
            search_cache[exp] = search
            rows.append(
                {
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "experiment": exp,
                    "status": "ok" if not search["no_valid_window"] else "no_valid_window",
                    "start_amp": float(search["start_amp"]),
                    "selected_amp": search["selected_amp"],
                    "control_valid": bool(search["control_valid"]),
                    "no_valid_window": bool(search["no_valid_window"]),
                    "trace": search["trace"],
                }
            )

    ctx.store.write_jsonl("control_windows_sft", rows)


def _apply_single_latent_experiment_full(
    ctx: RunContext,
    *,
    phase: str,
    experiment: str,
    policy_group: str,
    latent: Dict[str, Any],
    prompt_ids: Sequence[str],
    control_ids: Sequence[str],
    prompt_map: Dict[str, Dict[str, Any]],
    mode: str,
    requested_amp: Optional[float],
    effective_amp: Optional[float],
    control_valid: bool,
    no_valid_window: bool,
    backoff_trace: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    gen_cfg = _default_generation_cfg(ctx)
    prompts = _latent_prompt_rows(prompt_map, prompt_ids)
    if not prompts:
        return []

    amp = 1.0 if effective_amp is None else float(effective_amp)
    feature_dict = {
        latent["adapter_name"]: [(int(latent["feature_idx"]), mode)]
    }

    baseline, steered = _baseline_and_intervention_full(
        ctx,
        prompts,
        feature_dict,
        amplification=amp,
        do_sample=bool(gen_cfg["do_sample"]),
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
    )

    # Optional z snapshot
    z_snapshot = None
    module = _adapter_modules_by_name(ctx).get(latent["adapter_name"])
    if module is not None and module._last_z is not None:
        z = module._last_z.detach()
        z_snapshot = float(z[..., int(latent["feature_idx"])].max().item())

    recs = []
    for p in prompts:
        pid = p["prompt_id"]
        recs.append(
            {
                "phase": phase,
                "experiment": experiment,
                "policy_group": policy_group,
                "latent_id": int(latent["latent_id"]),
                "adapter_name": latent["adapter_name"],
                "feature_idx": int(latent["feature_idx"]),
                "prompt_id": pid,
                "prompt_bucket": p.get("bucket"),
                "prompt_text": p.get("prompt"),
                "model_variant": "M_full",
                "intervention_mode": mode,
                "requested_amp": requested_amp,
                "effective_amp": effective_amp,
                "baseline_text": baseline.get(pid, ""),
                "intervened_text": steered.get(pid, ""),
                "z_target": z_snapshot,
                "control_valid": bool(control_valid),
                "control_changed": not bool(control_valid),
                "invalid_reason": None if control_valid else "control_changed",
                "no_valid_window": bool(no_valid_window),
                "backoff_trace": backoff_trace or [],
            }
        )
    return recs


def stage_phase2_a1_ablate_on_fire(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "main")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))

    recs = []
    for latent in tqdm(selected, desc="phase2.A1"):
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        high_ids = _get_prompt_id_sets_for_latent(b, "HIGH")
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")

        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        feature_dict = {
            latent["adapter_name"]: [(int(latent["feature_idx"]), "disable")]
        }
        ctrl_cfg = _control_generation_cfg(ctx)
        baseline_ctrl, steered_ctrl = _baseline_and_intervention_full(
            ctx,
            control_recs,
            feature_dict,
            amplification=1.0,
            do_sample=False,
            max_new_tokens=ctrl_cfg["max_new_tokens"],
        )
        control_valid, _ = _record_control_validity(baseline_ctrl, steered_ctrl)

        recs.extend(
            _apply_single_latent_experiment_full(
                ctx,
                phase="phase2",
                experiment="A1",
                policy_group="ablation",
                latent=latent,
                prompt_ids=high_ids,
                control_ids=control_ids,
                prompt_map=prompt_map,
                mode="disable",
                requested_amp=None,
                effective_amp=None,
                control_valid=control_valid,
                no_valid_window=False,
            )
        )

    _append_records(ctx, recs)
    _append_judge_records(ctx, _judge_change_records(ctx, recs))


def stage_phase2_b1_force_on(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "main")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    calib = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)
    window_map = _control_window_map(ctx, sft=False)

    recs = []
    for latent in tqdm(selected, desc="phase2.B1"):
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        low_ids = _get_prompt_id_sets_for_latent(b, "LOW")
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        control_recs = _latent_prompt_rows(prompt_map, control_ids)

        cal = calib.get(lid)
        if cal is None:
            continue
        amp_1x = float(cal["amp_1x"])
        amp_5x = float(cal["amp_5x"])
        amp_safe = _amp_safe_for_latent(cal, amp_scan)
        start_amp = resolve_amp_start(
            experiment="B1",
            amp_1x=amp_1x,
            amp_5x=amp_5x,
            amp_safe_max=amp_safe,
        )

        preset = window_map.get((lid, "B1"))
        if preset and preset.get("selected_amp") is not None:
            selected_amp = float(preset["selected_amp"])
            no_valid_window = bool(preset.get("no_valid_window", False))
            control_valid = bool(preset.get("control_valid", not no_valid_window))
            trace = preset.get("trace", [])
        else:
            search = _control_first_amp_search_full(
                ctx,
                latent_entry=latent,
                experiment="B1",
                control_recs=control_recs,
                start_amp=start_amp,
            )
            selected_amp = search.get("selected_amp")
            no_valid_window = bool(search.get("no_valid_window", True))
            control_valid = bool(search.get("control_valid", False))
            trace = search.get("trace", [])

        if no_valid_window or selected_amp is None:
            recs.append(
                {
                    "phase": "phase2",
                    "experiment": "B1",
                    "policy_group": "force_inject",
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "prompt_id": None,
                    "prompt_bucket": "LOW",
                    "prompt_text": None,
                    "model_variant": "M_full",
                    "intervention_mode": "enable",
                    "requested_amp": float(start_amp),
                    "effective_amp": None,
                    "baseline_text": None,
                    "intervened_text": None,
                    "control_valid": False,
                    "control_changed": True,
                    "invalid_reason": "no_valid_window",
                    "no_valid_window": True,
                    "backoff_trace": trace,
                }
            )
            continue

        recs.extend(
            _apply_single_latent_experiment_full(
                ctx,
                phase="phase2",
                experiment="B1",
                policy_group="force_inject",
                latent=latent,
                prompt_ids=low_ids,
                control_ids=control_ids,
                prompt_map=prompt_map,
                mode="enable",
                requested_amp=float(start_amp),
                effective_amp=float(selected_amp),
                control_valid=control_valid,
                no_valid_window=False,
                backoff_trace=trace,
            )
        )

    _append_records(ctx, recs)
    _append_judge_records(ctx, _judge_change_records(ctx, recs))


def stage_phase3_b3_high(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "main")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    calib = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)
    window_map = _control_window_map(ctx, sft=True)

    gen_cfg = _default_generation_cfg(ctx)
    recs: List[Dict[str, Any]] = []

    for latent in tqdm(selected, desc="phase3.B3_HIGH"):
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        high_ids = _get_prompt_id_sets_for_latent(b, "HIGH")
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        high_recs = _latent_prompt_rows(prompt_map, high_ids)

        cal = calib.get(lid)
        if cal is None:
            continue
        amp_1x = float(cal["amp_1x"])
        amp_5x = float(cal["amp_5x"])
        amp_safe = _amp_safe_for_latent(cal, amp_scan)
        start_amp = resolve_amp_start(
            experiment="B3_HIGH",
            amp_1x=amp_1x,
            amp_5x=amp_5x,
            amp_safe_max=amp_safe,
        )

        preset = window_map.get((lid, "B3_HIGH"))
        if preset and preset.get("selected_amp") is not None:
            selected_amp = float(preset["selected_amp"])
            no_valid_window = bool(preset.get("no_valid_window", False))
            control_valid = bool(preset.get("control_valid", not no_valid_window))
            trace = preset.get("trace", [])
        else:
            search = _control_first_amp_search_sft(
                ctx,
                latent_entry=latent,
                experiment="B3_HIGH",
                control_recs=control_recs,
                start_amp=start_amp,
            )
            selected_amp = search.get("selected_amp")
            no_valid_window = bool(search.get("no_valid_window", True))
            control_valid = bool(search.get("control_valid", False))
            trace = search.get("trace", [])

        if no_valid_window or selected_amp is None:
            recs.append(
                {
                    "phase": "phase3",
                    "experiment": "B3_HIGH",
                    "policy_group": "force_inject",
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "prompt_id": None,
                    "prompt_bucket": "HIGH",
                    "prompt_text": None,
                    "model_variant": "M_sft",
                    "intervention_mode": "inject",
                    "requested_amp": float(start_amp),
                    "effective_amp": None,
                    "baseline_text": None,
                    "intervened_text": None,
                    "control_valid": False,
                    "control_changed": True,
                    "invalid_reason": "no_valid_window",
                    "no_valid_window": True,
                    "backoff_trace": trace,
                }
            )
            continue

        baseline, steered = _baseline_and_injection_sft(
            ctx,
            high_recs,
            latent,
            beta=float(selected_amp),
            do_sample=bool(gen_cfg["do_sample"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
        )
        for p in high_recs:
            pid = p["prompt_id"]
            recs.append(
                {
                    "phase": "phase3",
                    "experiment": "B3_HIGH",
                    "policy_group": "force_inject",
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "prompt_id": pid,
                    "prompt_bucket": p.get("bucket"),
                    "prompt_text": p.get("prompt"),
                    "model_variant": "M_sft",
                    "intervention_mode": "inject",
                    "requested_amp": float(start_amp),
                    "effective_amp": float(selected_amp),
                    "baseline_text": baseline.get(pid, ""),
                    "intervened_text": steered.get(pid, ""),
                    "control_valid": bool(control_valid),
                    "control_changed": False,
                    "invalid_reason": None if control_valid else "control_changed",
                    "no_valid_window": False,
                    "backoff_trace": trace,
                }
            )

    _append_records(ctx, recs)
    _append_judge_records(ctx, _judge_change_records(ctx, recs))


def stage_phase4_a3_ablate_attn_observe_mlp(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "cascade")
    stats_map = _latent_stats_map(ctx)
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    modules = _adapter_modules_by_name(ctx)
    latent_index = _read_latent_index(ctx)

    # map (adapter, feature) -> global latent id
    inverse_latent = {
        (r["adapter_name"], int(r["feature_idx"])): int(r["latent_id"])
        for r in latent_index
    }

    mlp_modules = {n: m for n, m in modules.items() if ".mlp." in n}

    sigma_threshold = float(_cfg_get(_stage_cfg(ctx, "phase4.a3_ablate_attn_observe_mlp"), "sigma_threshold", 1.0))
    top_edges_per_source = int(_cfg_get(_stage_cfg(ctx, "phase4.a3_ablate_attn_observe_mlp"), "top_edges_per_source", 32))

    edges = ctx.store.read_jsonl("cascade_edges")

    for latent in tqdm(selected, desc="phase4.A3"):
        if "self_attn" not in latent["adapter_name"]:
            continue
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        high_ids = _get_prompt_id_sets_for_latent(b, "HIGH")[:3]
        high_recs = _latent_prompt_rows(prompt_map, high_ids)
        if not high_recs:
            continue

        deltas = defaultdict(float)
        counts = defaultdict(int)

        for p in high_recs:
            rendered = render_prompt(ctx.tokenizer_full, p["prompt"])
            enc = ctx.tokenizer_full(
                rendered,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=int(_cfg_get(_cfg_get(_cfg_eval(ctx), "latents", {}), "max_length", 2048)),
            ).to(ctx.device)

            with torch.no_grad():
                _ = ctx.model_full(**enc)
            baseline_act = {}
            for name, m in mlp_modules.items():
                if m._last_z is None:
                    continue
                z = m._last_z.detach()
                if z.ndim == 3:
                    z = z[0]
                v = torch.relu(z).max(dim=0).values.cpu().numpy()
                baseline_act[name] = v

            feature_dict = {latent["adapter_name"]: [(int(latent["feature_idx"]), "disable")]}
            with FeatureSteeringContext(ctx.model_full, feature_dict, verbose=False, amplification=1.0):
                with torch.no_grad():
                    _ = ctx.model_full(**enc)

            for name, m in mlp_modules.items():
                if m._last_z is None or name not in baseline_act:
                    continue
                z = m._last_z.detach()
                if z.ndim == 3:
                    z = z[0]
                v_after = torch.relu(z).max(dim=0).values.cpu().numpy()
                v_before = baseline_act[name]
                for idx, (a, b0) in enumerate(zip(v_after, v_before)):
                    key = (name, idx)
                    deltas[key] += float(a - b0)
                    counts[key] += 1

        scored = []
        for (name, idx), delta_sum in deltas.items():
            c = max(counts[(name, idx)], 1)
            mean_delta = float(delta_sum / c)
            tgt_id = inverse_latent.get((name, idx))
            if tgt_id is None:
                continue
            sigma = float(stats_map.get(tgt_id, {}).get("sigma", 1e-6))
            sigma = max(sigma, 1e-6)
            z_score = abs(mean_delta) / sigma
            if z_score >= sigma_threshold:
                scored.append((z_score, tgt_id, name, idx, mean_delta))

        scored.sort(reverse=True, key=lambda x: x[0])
        for z_score, tgt_id, name, idx, mean_delta in scored[:top_edges_per_source]:
            edges.append(
                {
                    "phase": "phase4",
                    "experiment": "A3_dependency",
                    "source_latent_id": lid,
                    "source_adapter_name": latent["adapter_name"],
                    "source_feature_idx": int(latent["feature_idx"]),
                    "target_latent_id": int(tgt_id),
                    "target_adapter_name": name,
                    "target_feature_idx": int(idx),
                    "delta_mean": float(mean_delta),
                    "delta_sigma_units": float(z_score),
                    "edge_type": "dependency",
                }
            )

    ctx.store.write_jsonl("cascade_edges", edges)


def stage_phase4_c2_force_attn_observe_mlp(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "cascade")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    window_map = _control_window_map(ctx, sft=False)
    calib = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)
    modules = _adapter_modules_by_name(ctx)

    latent_index = _read_latent_index(ctx)
    inverse_latent = {
        (r["adapter_name"], int(r["feature_idx"])): int(r["latent_id"])
        for r in latent_index
    }

    edges = ctx.store.read_jsonl("cascade_edges")

    for latent in tqdm(selected, desc="phase4.C2"):
        if "self_attn" not in latent["adapter_name"]:
            continue
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        low_ids = _get_prompt_id_sets_for_latent(b, "LOW")[:3]
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        low_recs = _latent_prompt_rows(prompt_map, low_ids)
        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        if not low_recs:
            continue

        cal = calib.get(lid)
        if cal is None:
            continue
        amp_1x = float(cal["amp_1x"])
        amp_5x = float(cal["amp_5x"])
        amp_safe = _amp_safe_for_latent(cal, amp_scan)
        start_amp = resolve_amp_start(
            experiment="C2",
            amp_1x=amp_1x,
            amp_5x=amp_5x,
            amp_safe_max=amp_safe,
        )
        preset = window_map.get((lid, "C2"))
        if preset and preset.get("selected_amp") is not None:
            amp = float(preset["selected_amp"])
        else:
            if control_recs:
                search = _control_first_amp_search_full(
                    ctx,
                    latent_entry=latent,
                    experiment="C2",
                    control_recs=control_recs,
                    start_amp=start_amp,
                )
                if bool(search.get("no_valid_window", True)) or search.get("selected_amp") is None:
                    continue
                amp = float(search["selected_amp"])
            else:
                amp = float(start_amp)

        feature_dict = {latent["adapter_name"]: [(int(latent["feature_idx"]), "enable")]}

        for p in low_recs:
            rendered = render_prompt(ctx.tokenizer_full, p["prompt"])
            enc = ctx.tokenizer_full(
                rendered,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=int(_cfg_get(_cfg_get(_cfg_eval(ctx), "latents", {}), "max_length", 2048)),
            ).to(ctx.device)

            with torch.no_grad():
                _ = ctx.model_full(**enc)
            base_active = set()
            for name, m in modules.items():
                if ".mlp." not in name or m._last_z is None:
                    continue
                z = m._last_z.detach()
                if z.ndim == 3:
                    z = z[0]
                mask = _hard_topk_mask(z, int(m._current_k())).any(dim=0).cpu().numpy()
                for idx, on in enumerate(mask):
                    if on:
                        base_active.add((name, idx))

            with FeatureSteeringContext(ctx.model_full, feature_dict, verbose=False, amplification=amp):
                with torch.no_grad():
                    _ = ctx.model_full(**enc)

            steered_active = set()
            for name, m in modules.items():
                if ".mlp." not in name or m._last_z is None:
                    continue
                z = m._last_z.detach()
                if z.ndim == 3:
                    z = z[0]
                mask = _hard_topk_mask(z, int(m._current_k())).any(dim=0).cpu().numpy()
                for idx, on in enumerate(mask):
                    if on:
                        steered_active.add((name, idx))

            recruited = steered_active - base_active
            for name, idx in recruited:
                tgt_id = inverse_latent.get((name, idx))
                if tgt_id is None:
                    continue
                edges.append(
                    {
                        "phase": "phase4",
                        "experiment": "C2_recruitment",
                        "source_latent_id": lid,
                        "source_adapter_name": latent["adapter_name"],
                        "source_feature_idx": int(latent["feature_idx"]),
                        "target_latent_id": int(tgt_id),
                        "target_adapter_name": name,
                        "target_feature_idx": int(idx),
                        "edge_type": "recruitment",
                        "amp": float(amp),
                        "prompt_id": p["prompt_id"],
                    }
                )

    ctx.store.write_jsonl("cascade_edges", edges)


def stage_phase4_c1_cross_sublayer_cascade(ctx: RunContext) -> None:
    edges = ctx.store.read_jsonl("cascade_edges")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    latent_index = _latent_index_map(_read_latent_index(ctx))

    # Build candidate pairs from strong A3 dependencies.
    dep_edges = [e for e in edges if e.get("experiment") == "A3_dependency"]
    dep_edges.sort(key=lambda e: abs(float(e.get("delta_sigma_units", 0.0))), reverse=True)
    max_pairs = int(_cfg_get(_stage_cfg(ctx, "phase4.c1_cross_sublayer_cascade"), "max_pairs", 32))
    pairs = dep_edges[:max_pairs]

    gen_cfg = _default_generation_cfg(ctx)
    ctrl_cfg = _control_generation_cfg(ctx)

    records = []
    for pair in tqdm(pairs, desc="phase4.C1"):
        j = latent_index.get(int(pair["source_latent_id"]))
        k = latent_index.get(int(pair["target_latent_id"]))
        if j is None or k is None:
            continue

        b = bucket_map.get(int(j["latent_id"]), {})
        high_ids = _get_prompt_id_sets_for_latent(b, "HIGH")[:3]
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        high_recs = _latent_prompt_rows(prompt_map, high_ids)
        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        if not high_recs:
            continue

        full = _generate_texts(
            ctx.model_full,
            ctx.tokenizer_full,
            high_recs,
            do_sample=bool(gen_cfg["do_sample"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
        )

        _, ablate_j = _baseline_and_intervention_full(
            ctx,
            high_recs,
            {j["adapter_name"]: [(int(j["feature_idx"]), "disable")]},
            amplification=1.0,
            do_sample=bool(gen_cfg["do_sample"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
        )
        _, ablate_k = _baseline_and_intervention_full(
            ctx,
            high_recs,
            {k["adapter_name"]: [(int(k["feature_idx"]), "disable")]},
            amplification=1.0,
            do_sample=bool(gen_cfg["do_sample"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
        )
        _, ablate_jk = _baseline_and_intervention_full(
            ctx,
            high_recs,
            {
                j["adapter_name"]: [(int(j["feature_idx"]), "disable")],
                k["adapter_name"]: [(int(k["feature_idx"]), "disable")],
            },
            amplification=1.0,
            do_sample=bool(gen_cfg["do_sample"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
        )

        # control check (ablation: flag only)
        ctrl_base = _generate_texts(
            ctx.model_full,
            ctx.tokenizer_full,
            control_recs,
            do_sample=False,
            max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
        )
        _, ctrl_j = _baseline_and_intervention_full(
            ctx,
            control_recs,
            {j["adapter_name"]: [(int(j["feature_idx"]), "disable")]},
            amplification=1.0,
            do_sample=False,
            max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
        )
        _, ctrl_k = _baseline_and_intervention_full(
            ctx,
            control_recs,
            {k["adapter_name"]: [(int(k["feature_idx"]), "disable")]},
            amplification=1.0,
            do_sample=False,
            max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
        )
        _, ctrl_jk = _baseline_and_intervention_full(
            ctx,
            control_recs,
            {
                j["adapter_name"]: [(int(j["feature_idx"]), "disable")],
                k["adapter_name"]: [(int(k["feature_idx"]), "disable")],
            },
            amplification=1.0,
            do_sample=False,
            max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
        )

        control_valid = True
        for pid, btxt in ctrl_base.items():
            if control_changed(btxt, ctrl_j.get(pid, "")):
                control_valid = False
            if control_changed(btxt, ctrl_k.get(pid, "")):
                control_valid = False
            if control_changed(btxt, ctrl_jk.get(pid, "")):
                control_valid = False

        # functional significance criterion
        diffs = []
        for p in high_recs:
            pid = p["prompt_id"]
            s_full_k = lexical_change_score(full.get(pid, ""), ablate_k.get(pid, ""))
            s_j_jk = lexical_change_score(ablate_j.get(pid, ""), ablate_jk.get(pid, ""))
            diffs.append(abs(s_full_k - s_j_jk))

        functional_score = safe_mean(diffs)
        is_functional = functional_score > float(_cfg_get(_stage_cfg(ctx, "phase4.c1_cross_sublayer_cascade"), "functional_threshold", 0.05))

        records.append(
            {
                "phase": "phase4",
                "experiment": "C1",
                "policy_group": "ablation",
                "latent_id": int(j["latent_id"]),
                "adapter_name": j["adapter_name"],
                "feature_idx": int(j["feature_idx"]),
                "paired_latent_id": int(k["latent_id"]),
                "paired_adapter_name": k["adapter_name"],
                "paired_feature_idx": int(k["feature_idx"]),
                "prompt_id": None,
                "prompt_bucket": "HIGH",
                "prompt_text": None,
                "model_variant": "M_full",
                "intervention_mode": "ablate_pair_conditions",
                "requested_amp": None,
                "effective_amp": None,
                "baseline_text": None,
                "intervened_text": None,
                "control_valid": bool(control_valid),
                "control_changed": not bool(control_valid),
                "invalid_reason": None if control_valid else "control_changed",
                "no_valid_window": False,
                "functional_score": float(functional_score),
                "functional_significant": bool(is_functional),
            }
        )

    _append_records(ctx, records)


def stage_phase5_b2_isolate(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "cascade")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    calib = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)
    window_map = _control_window_map(ctx, sft=False)

    recs = []
    for latent in tqdm(selected, desc="phase5.B2"):
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        high_ids = _get_prompt_id_sets_for_latent(b, "HIGH")
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        control_recs = _latent_prompt_rows(prompt_map, control_ids)

        cal = calib.get(lid)
        if cal is None:
            continue
        amp_1x = float(cal["amp_1x"])
        amp_5x = float(cal["amp_5x"])
        amp_safe = _amp_safe_for_latent(cal, amp_scan)
        start_amp = resolve_amp_start(
            experiment="B2",
            amp_1x=amp_1x,
            amp_5x=amp_5x,
            amp_safe_max=amp_safe,
        )

        preset = window_map.get((lid, "B2"))
        if preset and preset.get("selected_amp") is not None:
            selected_amp = float(preset["selected_amp"])
            no_valid_window = bool(preset.get("no_valid_window", False))
            control_valid = bool(preset.get("control_valid", not no_valid_window))
            trace = preset.get("trace", [])
        else:
            search = _control_first_amp_search_full(
                ctx,
                latent_entry=latent,
                experiment="B2",
                control_recs=control_recs,
                start_amp=start_amp,
            )
            selected_amp = search.get("selected_amp")
            no_valid_window = bool(search.get("no_valid_window", True))
            control_valid = bool(search.get("control_valid", False))
            trace = search.get("trace", [])

        if no_valid_window or selected_amp is None:
            recs.append(
                {
                    "phase": "phase5",
                    "experiment": "B2",
                    "policy_group": "force_inject",
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "prompt_id": None,
                    "prompt_bucket": "HIGH",
                    "prompt_text": None,
                    "model_variant": "M_full",
                    "intervention_mode": "isolate",
                    "requested_amp": float(start_amp),
                    "effective_amp": None,
                    "baseline_text": None,
                    "intervened_text": None,
                    "control_valid": False,
                    "control_changed": True,
                    "invalid_reason": "no_valid_window",
                    "no_valid_window": True,
                    "backoff_trace": trace,
                }
            )
            continue

        recs.extend(
            _apply_single_latent_experiment_full(
                ctx,
                phase="phase5",
                experiment="B2",
                policy_group="force_inject",
                latent=latent,
                prompt_ids=high_ids,
                control_ids=control_ids,
                prompt_map=prompt_map,
                mode="isolate",
                requested_amp=float(start_amp),
                effective_amp=float(selected_amp),
                control_valid=control_valid,
                no_valid_window=False,
                backoff_trace=trace,
            )
        )

    _append_records(ctx, recs)
    _append_judge_records(ctx, _judge_change_records(ctx, recs))


def stage_phase5_d2_encoder_decoder_alignment(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "cascade")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    lat_cfg = _cfg_get(_cfg_eval(ctx), "latents", {})
    bucket_sizes = _cfg_get(lat_cfg, "prompt_bucket_sizes", {})
    max_high_prompts = int(_cfg_get(bucket_sizes, "high", 20))

    recs = []
    for latent in tqdm(selected, desc="phase5.D2"):
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        high_prompts = b.get("high_prompts", [])
        if len(high_prompts) < 4:
            continue

        vals = np.array([float(p.get("activation", 0.0)) for p in high_prompts])
        q = np.quantile(vals, [0.25, 0.5, 0.75])

        quartile_scores = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for hp in high_prompts[:max_high_prompts]:
            pid = hp["prompt_id"]
            rec = prompt_map.get(pid)
            if rec is None:
                continue
            v = float(hp.get("activation", 0.0))
            if v <= q[0]:
                qname = "Q1"
            elif v <= q[1]:
                qname = "Q2"
            elif v <= q[2]:
                qname = "Q3"
            else:
                qname = "Q4"

            feature_dict = {latent["adapter_name"]: [(int(latent["feature_idx"]), "disable")]}
            baseline, steered = _baseline_and_intervention_full(
                ctx,
                [rec],
                feature_dict,
                amplification=1.0,
                do_sample=False,
                max_new_tokens=int(_default_generation_cfg(ctx)["max_new_tokens"]),
            )
            score = lexical_change_score(
                baseline.get(pid, ""),
                steered.get(pid, ""),
            )
            quartile_scores[qname].append(score)

        recs.append(
            {
                "phase": "phase5",
                "experiment": "D2",
                "policy_group": "ablation",
                "latent_id": lid,
                "adapter_name": latent["adapter_name"],
                "feature_idx": int(latent["feature_idx"]),
                "quartile_effects": {
                    k: safe_mean(vs) for k, vs in quartile_scores.items()
                },
                "effect_range": float(
                    max(safe_mean(vs) for vs in quartile_scores.values())
                    - min(safe_mean(vs) for vs in quartile_scores.values())
                ),
                "control_valid": True,
                "no_valid_window": False,
            }
        )

    _append_records(ctx, recs)


def stage_phase6_b3_behav(ctx: RunContext) -> None:
    selected = _selected_latents(ctx, "main")
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    calib = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)
    window_map = _control_window_map(ctx, sft=True)
    gen_cfg = _default_generation_cfg(ctx)

    all_behav_prompt_ids = {pid for pid, rec in prompt_map.items() if rec.get("bucket") == "behav"}
    if not all_behav_prompt_ids:
        raise ValueError(
            "phase6.b3_behav requires BEHAV prompts, but prompts_master contains none. "
            "Populate dataset.behav_prompts or provide latent-specific behav_prompt_ids."
        )

    missing_behav_latents: List[int] = []
    for latent in selected:
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        behav_ids = _get_prompt_id_sets_for_latent(b, "BEHAV")
        behav_recs = _latent_prompt_rows(prompt_map, behav_ids)
        if not behav_recs:
            missing_behav_latents.append(lid)
    if missing_behav_latents:
        raise ValueError(
            "phase6.b3_behav missing BEHAV prompts for selected latents: "
            + ",".join(str(x) for x in missing_behav_latents[:20])
            + ("..." if len(missing_behav_latents) > 20 else "")
        )

    recs = []
    for latent in tqdm(selected, desc="phase6.B3_BEHAV"):
        lid = int(latent["latent_id"])
        b = bucket_map.get(lid, {})
        behav_ids = _get_prompt_id_sets_for_latent(b, "BEHAV")
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        behav_recs = _latent_prompt_rows(prompt_map, behav_ids)
        control_recs = _latent_prompt_rows(prompt_map, control_ids)

        cal = calib.get(lid)
        if cal is None:
            continue
        amp_1x = float(cal["amp_1x"])
        amp_5x = float(cal["amp_5x"])
        amp_safe = _amp_safe_for_latent(cal, amp_scan)
        start_amp = resolve_amp_start(
            experiment="B3_BEHAV",
            amp_1x=amp_1x,
            amp_5x=amp_5x,
            amp_safe_max=amp_safe,
        )

        preset = window_map.get((lid, "B3_BEHAV")) or window_map.get((lid, "B3_HIGH"))
        if preset and preset.get("selected_amp") is not None:
            selected_amp = float(preset["selected_amp"])
            no_valid_window = bool(preset.get("no_valid_window", False))
            control_valid = bool(preset.get("control_valid", not no_valid_window))
            trace = preset.get("trace", [])
        else:
            search = _control_first_amp_search_sft(
                ctx,
                latent_entry=latent,
                experiment="B3_BEHAV",
                control_recs=control_recs,
                start_amp=start_amp,
            )
            selected_amp = search.get("selected_amp")
            no_valid_window = bool(search.get("no_valid_window", True))
            control_valid = bool(search.get("control_valid", False))
            trace = search.get("trace", [])

        if no_valid_window or selected_amp is None:
            recs.append(
                {
                    "phase": "phase6",
                    "experiment": "B3_BEHAV",
                    "policy_group": "force_inject",
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "prompt_id": None,
                    "prompt_bucket": "BEHAV",
                    "prompt_text": None,
                    "model_variant": "M_sft",
                    "intervention_mode": "inject",
                    "requested_amp": float(start_amp),
                    "effective_amp": None,
                    "baseline_text": None,
                    "intervened_text": None,
                    "control_valid": False,
                    "control_changed": True,
                    "invalid_reason": "no_valid_window",
                    "no_valid_window": True,
                    "backoff_trace": trace,
                }
            )
            continue

        baseline, steered = _baseline_and_injection_sft(
            ctx,
            behav_recs,
            latent,
            beta=float(selected_amp),
            do_sample=bool(gen_cfg["do_sample"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
        )

        for p in behav_recs:
            pid = p["prompt_id"]
            recs.append(
                {
                    "phase": "phase6",
                    "experiment": "B3_BEHAV",
                    "policy_group": "force_inject",
                    "latent_id": lid,
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "prompt_id": pid,
                    "prompt_bucket": p.get("bucket"),
                    "prompt_text": p.get("prompt"),
                    "model_variant": "M_sft",
                    "intervention_mode": "inject",
                    "requested_amp": float(start_amp),
                    "effective_amp": float(selected_amp),
                    "baseline_text": baseline.get(pid, ""),
                    "intervened_text": steered.get(pid, ""),
                    "control_valid": bool(control_valid),
                    "control_changed": False,
                    "invalid_reason": None if control_valid else "control_changed",
                    "no_valid_window": False,
                    "backoff_trace": trace,
                }
            )

    _append_records(ctx, recs)
    _append_judge_records(ctx, _judge_change_records(ctx, recs))


def stage_phase6_c3_multi_inject(ctx: RunContext) -> None:
    edges = ctx.store.read_jsonl("cascade_edges")
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    bucket_map = _latent_bucket_map(ctx)
    latent_index = _latent_index_map(_read_latent_index(ctx))
    calib = _calibration_map(ctx)
    amp_scan = _read_amp_window_scan_or_fail(ctx)

    candidate_pairs = [e for e in edges if e.get("experiment") == "A3_dependency"]
    candidate_pairs = candidate_pairs[: int(_cfg_get(_stage_cfg(ctx, "phase6.c3_multi_inject"), "max_pairs", 16))]

    recs = []
    gen_cfg = _default_generation_cfg(ctx)
    ctrl_cfg = _control_generation_cfg(ctx)
    factors = _cfg_get(_policy_cfg(ctx), "backoff_factors", [1.0, 0.5, 0.25, 0.125])

    for pair in tqdm(candidate_pairs, desc="phase6.C3"):
        j = latent_index.get(int(pair["source_latent_id"]))
        k = latent_index.get(int(pair["target_latent_id"]))
        if j is None or k is None:
            continue
        if int(j["latent_id"]) not in calib or int(k["latent_id"]) not in calib:
            continue

        b = bucket_map.get(int(j["latent_id"]), {})
        prompt_ids = _get_prompt_id_sets_for_latent(b, "HIGH")[:3]
        prompts = _latent_prompt_rows(prompt_map, prompt_ids)
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        if not prompts:
            continue

        cj = calib[int(j["latent_id"])]
        ck = calib[int(k["latent_id"])]
        start_j = resolve_amp_start(
            experiment="C3",
            amp_1x=float(cj["amp_1x"]),
            amp_5x=float(cj["amp_5x"]),
            amp_safe_max=_amp_safe_for_latent(cj, amp_scan),
        )
        start_k = resolve_amp_start(
            experiment="C3",
            amp_1x=float(ck["amp_1x"]),
            amp_5x=float(ck["amp_5x"]),
            amp_safe_max=_amp_safe_for_latent(ck, amp_scan),
        )

        model_sft, tok_sft = ensure_sft_model_loaded(ctx)
        baseline = _generate_texts(
            model_sft,
            tok_sft,
            prompts,
            do_sample=bool(gen_cfg["do_sample"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
        )

        vec_j = _decoder_vector_for_latent(ctx, j)
        vec_k = _decoder_vector_for_latent(ctx, k)

        def run_specs(specs: List[AdditiveInjectionSpec], prompt_rows: Sequence[Dict[str, Any]], do_sample: bool, max_new_tokens: int) -> Dict[str, str]:
            rendered = [render_prompt(tok_sft, p["prompt"]) for p in prompt_rows]
            with AdditiveInjectionContext(model_sft, specs):
                outs = generate_with_model(
                    model_sft,
                    tok_sft,
                    rendered,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=(gen_cfg["temperature"] if do_sample else None),
                    top_p=(gen_cfg["top_p"] if do_sample else None),
                )
            return {
                p["prompt_id"]: normalize_output_text(o)
                for p, o in zip(prompt_rows, outs)
            }

        baseline_control = _generate_texts(
            model_sft,
            tok_sft,
            control_recs,
            do_sample=False,
            max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
        )
        trace = []
        selected_j = None
        selected_k = None
        control_valid = True
        no_valid_window = False

        if control_recs:
            control_valid = False
            for f in factors:
                amp_j = float(start_j) * float(f)
                amp_k = float(start_k) * float(f)
                steered_control = run_specs(
                    [
                        AdditiveInjectionSpec(j["adapter_name"], vec_j, amp_j),
                        AdditiveInjectionSpec(k["adapter_name"], vec_k, amp_k),
                    ],
                    control_recs,
                    False,
                    int(ctrl_cfg["max_new_tokens"]),
                )
                valid, changed_ids = _record_control_validity(baseline_control, steered_control)
                trace.append(
                    {
                        "factor": float(f),
                        "amp_j": float(amp_j),
                        "amp_k": float(amp_k),
                        "control_valid": bool(valid),
                        "changed_prompt_ids": changed_ids,
                    }
                )
                if valid:
                    control_valid = True
                    selected_j = amp_j
                    selected_k = amp_k
                    break
            if not control_valid:
                no_valid_window = True
        else:
            selected_j = float(start_j)
            selected_k = float(start_k)

        if no_valid_window or selected_j is None or selected_k is None:
            recs.append(
                {
                    "phase": "phase6",
                    "experiment": "C3",
                    "policy_group": "force_inject",
                    "latent_id": int(j["latent_id"]),
                    "adapter_name": j["adapter_name"],
                    "feature_idx": int(j["feature_idx"]),
                    "paired_latent_id": int(k["latent_id"]),
                    "paired_adapter_name": k["adapter_name"],
                    "paired_feature_idx": int(k["feature_idx"]),
                    "model_variant": "M_sft",
                    "intervention_mode": "multi_inject",
                    "requested_amp_j": float(start_j),
                    "requested_amp_k": float(start_k),
                    "effective_amp_j": None,
                    "effective_amp_k": None,
                    "control_valid": False,
                    "no_valid_window": True,
                    "backoff_trace": trace,
                }
            )
            continue

        only_j = run_specs(
            [AdditiveInjectionSpec(j["adapter_name"], vec_j, float(selected_j))],
            prompts,
            bool(gen_cfg["do_sample"]),
            int(gen_cfg["max_new_tokens"]),
        )
        only_k = run_specs(
            [AdditiveInjectionSpec(k["adapter_name"], vec_k, float(selected_k))],
            prompts,
            bool(gen_cfg["do_sample"]),
            int(gen_cfg["max_new_tokens"]),
        )
        both = run_specs(
            [
                AdditiveInjectionSpec(j["adapter_name"], vec_j, float(selected_j)),
                AdditiveInjectionSpec(k["adapter_name"], vec_k, float(selected_k)),
            ],
            prompts,
            bool(gen_cfg["do_sample"]),
            int(gen_cfg["max_new_tokens"]),
        )

        j_scores = [lexical_change_score(baseline[pid], only_j.get(pid, "")) for pid in baseline]
        k_scores = [lexical_change_score(baseline[pid], only_k.get(pid, "")) for pid in baseline]
        both_scores = [lexical_change_score(baseline[pid], both.get(pid, "")) for pid in baseline]

        mean_j = safe_mean(j_scores)
        mean_k = safe_mean(k_scores)
        mean_both = safe_mean(both_scores)

        if mean_both > (mean_j + mean_k) + 0.02:
            coherence = "synergistic"
        elif mean_both < (mean_j + mean_k) - 0.02:
            coherence = "interference"
        else:
            coherence = "additive"

        recs.append(
            {
                "phase": "phase6",
                "experiment": "C3",
                "policy_group": "force_inject",
                "latent_id": int(j["latent_id"]),
                "adapter_name": j["adapter_name"],
                "feature_idx": int(j["feature_idx"]),
                "paired_latent_id": int(k["latent_id"]),
                "paired_adapter_name": k["adapter_name"],
                "paired_feature_idx": int(k["feature_idx"]),
                "model_variant": "M_sft",
                "intervention_mode": "multi_inject",
                "requested_amp_j": float(start_j),
                "requested_amp_k": float(start_k),
                "effective_amp_j": float(selected_j),
                "effective_amp_k": float(selected_k),
                "mean_effect_j": float(mean_j),
                "mean_effect_k": float(mean_k),
                "mean_effect_joint": float(mean_both),
                "coherence": coherence,
                "control_valid": bool(control_valid),
                "no_valid_window": False,
                "backoff_trace": trace,
            }
        )

    _append_records(ctx, recs)


def stage_phase6_b4_beta_sweep(ctx: RunContext) -> None:
    # Sweep experiment: continue regardless, no backoff.
    all_records = ctx.store.read_jsonl("intervention_records")
    calib = _calibration_map(ctx)
    bucket_map = _latent_bucket_map(ctx)
    prompt_map = prompt_map_by_id(_read_prompts(ctx))
    latent_index = _latent_index_map(_read_latent_index(ctx))

    b3_high = [r for r in all_records if r.get("experiment") == "B3_HIGH" and r.get("prompt_id")]
    by_latent = defaultdict(list)
    for r in b3_high:
        by_latent[int(r["latent_id"])].append(lexical_change_score(r.get("baseline_text", ""), r.get("intervened_text", "")))

    scored = [(safe_mean(v), lid) for lid, v in by_latent.items()]
    scored.sort(reverse=True)
    top_lids = [lid for _, lid in scored[:16]]

    sweep_factors = _cfg_get(_stage_cfg(ctx, "phase6.b4_beta_sweep"), "factors", [0.1, 0.5, 1.0, 2.0, 5.0])
    recs = []
    model_sft, tok_sft = ensure_sft_model_loaded(ctx)
    ctrl_cfg = _control_generation_cfg(ctx)

    for lid in tqdm(top_lids, desc="phase6.B4"):
        latent = latent_index.get(int(lid))
        if latent is None or lid not in calib:
            continue
        b = bucket_map.get(int(lid), {})
        high_ids = _get_prompt_id_sets_for_latent(b, "HIGH")[:3]
        control_ids = _get_prompt_id_sets_for_latent(b, "CONTROL")
        high_recs = _latent_prompt_rows(prompt_map, high_ids)
        control_recs = _latent_prompt_rows(prompt_map, control_ids)
        if not high_recs:
            continue

        beta_1x = float(calib[lid]["amp_1x"])

        # Baselines once per latent
        baseline_high = _generate_texts(
            model_sft,
            tok_sft,
            high_recs,
            do_sample=False,
            max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
        )
        baseline_control = _generate_texts(
            model_sft,
            tok_sft,
            control_recs,
            do_sample=False,
            max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
        )

        for f in sweep_factors:
            beta = float(f) * beta_1x
            _, steered_high = _baseline_and_injection_sft(
                ctx,
                high_recs,
                latent,
                beta=beta,
                do_sample=False,
                max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
            )
            _, steered_control = _baseline_and_injection_sft(
                ctx,
                control_recs,
                latent,
                beta=beta,
                do_sample=False,
                max_new_tokens=int(ctrl_cfg["max_new_tokens"]),
            )

            high_changed = any(
                control_changed(baseline_high.get(pid, ""), steered_high.get(pid, ""))
                for pid in baseline_high
            )
            control_valid, changed_controls = _record_control_validity(
                baseline_control,
                steered_control,
            )
            if not control_valid:
                label = "deteriorated"
            elif high_changed:
                label = "valid"
            else:
                label = "null"

            recs.append(
                {
                    "phase": "phase6",
                    "experiment": "B4",
                    "policy_group": "sweep",
                    "latent_id": int(lid),
                    "adapter_name": latent["adapter_name"],
                    "feature_idx": int(latent["feature_idx"]),
                    "model_variant": "M_sft",
                    "intervention_mode": "inject_sweep",
                    "factor": float(f),
                    "requested_amp": float(beta),
                    "effective_amp": float(beta),
                    "control_valid": bool(control_valid),
                    "changed_control_prompt_ids": changed_controls,
                    "sweep_label": label,
                    "no_valid_window": False,
                }
            )

    _append_records(ctx, recs)


def _build_hypothesis_evidence_rows(ctx: RunContext) -> List[Dict[str, Any]]:
    rows = ctx.store.read_jsonl("intervention_records")
    allowed = set(_cfg_get(_stage_cfg(ctx, "phase6.hypothesis_generation"), "evidence_experiments", ["A1", "B1", "B3_HIGH", "B3_BEHAV", "B2"]))
    out = []
    for r in rows:
        if str(r.get("experiment")) not in allowed:
            continue
        if r.get("prompt_id") is None:
            continue
        if not bool(r.get("control_valid", True)):
            continue
        out.append(
            {
                "latent_id": int(r["latent_id"]),
                "adapter_name": r.get("adapter_name"),
                "feature_idx": int(r.get("feature_idx", -1)),
                "prompt_id": r.get("prompt_id"),
                "prompt": r.get("prompt_text", ""),
                "baseline_samples": [r.get("baseline_text", "")],
                "steered_sample": r.get("intervened_text", ""),
                "intervention_meta": {
                    "type": "zero_ablate"
                    if r.get("intervention_mode") in {"disable", "disable_all"}
                    else "steer_with_alpha",
                    "alpha": r.get("effective_amp"),
                    "experiment": r.get("experiment"),
                },
            }
        )
    return out


def stage_phase6_hypothesis_generation(ctx: RunContext) -> None:
    evidence_rows = _build_hypothesis_evidence_rows(ctx)
    tmp_path = ctx.store.path("_tmp_hypothesis_evidence.jsonl")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in evidence_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    hypotheses = run_explainer(
        ctx.cfg,
        output_dir=str(ctx.store.root),
        vllm_base_url=_vllm_base_url(ctx),
        evidence_path=str(tmp_path),
        hypotheses_path=str(ctx.store.path("hypotheses")),
    )
    logger.info("Hypothesis generation complete: %d hypotheses", len(hypotheses))


def _verify_record_with_llm(
    client: OpenAI,
    model_name: str,
    hypothesis: Dict[str, Any],
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    prompt = (
        "Assess if the steered output supports the hypothesis. "
        "Return JSON with keys supported (bool), confidence (0..1), reason (str).\n\n"
        f"Hypothesis: {hypothesis.get('hypothesis', '')}\n"
        f"Behavioral dimension: {hypothesis.get('behavioral_dimension', '')}\n"
        f"Effect direction: {hypothesis.get('effect_direction', '')}\n\n"
        f"Baseline: {record.get('baseline_text', '')}\n"
        f"Steered: {record.get('intervened_text', '')}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt)
        return {
            "supported": bool(data.get("supported", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "reason": str(data.get("reason", "")),
        }
    except Exception:
        return None


def stage_phase6_verification(ctx: RunContext) -> None:
    if not ctx.store.exists("hypotheses"):
        raise FileNotFoundError(
            "phase6.verification requires hypotheses artifact (hypotheses.jsonl)."
        )
    hypotheses = ctx.store.read_jsonl("hypotheses")
    records = ctx.store.read_jsonl("intervention_records")

    # Build latent -> records map for verification-ready samples
    verify_exps = set(_cfg_get(_stage_cfg(ctx, "phase6.verification"), "experiments", ["A1", "B1", "B2", "B3_HIGH", "B3_BEHAV"]))
    by_latent = defaultdict(list)
    for r in records:
        if str(r.get("experiment")) not in verify_exps:
            continue
        if not bool(r.get("control_valid", True)):
            continue
        if not r.get("prompt_id"):
            continue
        by_latent[int(r["latent_id"])].append(r)

    llm_cfg = _cfg_get(_llm_cfg(ctx), "verifier", {})
    use_llm = bool(_cfg_get(llm_cfg, "enabled", True))
    client = None
    if use_llm:
        try:
            client = OpenAI(base_url=_vllm_base_url(ctx), api_key="EMPTY")
        except Exception:
            client = None

    recs = []
    per_latent = []
    threshold = float(_cfg_get(_policy_cfg(ctx), "material_change_threshold", 0.15))

    for hyp in hypotheses:
        lid = int(hyp.get("latent_id", -1))
        rows = by_latent.get(lid, [])
        if not rows:
            continue

        supports = []
        for r in rows:
            verdict = None
            if client is not None and use_llm:
                verdict = _verify_record_with_llm(
                    client,
                    str(_cfg_get(llm_cfg, "model", "Qwen/Qwen2.5-32B-Instruct-AWQ")),
                    hyp,
                    r,
                )
            if verdict is None:
                score = lexical_change_score(r.get("baseline_text", ""), r.get("intervened_text", ""))
                verdict = {
                    "supported": bool(score >= threshold),
                    "confidence": min(1.0, max(0.0, score * 2.0)),
                    "reason": "heuristic_change_score",
                }

            supports.append(bool(verdict["supported"]))
            recs.append(
                {
                    "latent_id": lid,
                    "prompt_id": r.get("prompt_id"),
                    "experiment": r.get("experiment"),
                    "supported": bool(verdict["supported"]),
                    "confidence": float(verdict.get("confidence", 0.0)),
                    "reason": verdict.get("reason", ""),
                }
            )

        support_rate = float(sum(1 for x in supports if x) / max(len(supports), 1))
        per_latent.append(
            {
                "latent_id": lid,
                "support_rate": support_rate,
                "n_records": len(supports),
            }
        )

    metrics = {
        "n_latents": len(per_latent),
        "n_records": len(recs),
        "mean_support_rate": float(safe_mean([r["support_rate"] for r in per_latent])) if per_latent else 0.0,
    }

    ctx.store.write_jsonl("verification_records", recs)
    ctx.store.write_json("verification_metrics", metrics)
    ctx.store.write_json("verification_per_latent", per_latent)


def stage_phase6_typology(ctx: RunContext) -> None:
    if not ctx.store.exists("hypotheses"):
        raise FileNotFoundError(
            "phase6.typology requires hypotheses artifact (hypotheses.jsonl)."
        )
    if not ctx.store.exists("verification_per_latent"):
        raise FileNotFoundError(
            "phase6.typology requires verification_per_latent artifact "
            "(verification_per_latent.json)."
        )

    records = ctx.store.read_jsonl("intervention_records")
    hypotheses = {int(h["latent_id"]): h for h in ctx.store.read_jsonl("hypotheses")}
    verification = {int(r["latent_id"]): r for r in (ctx.store.read_json("verification_per_latent", []) or [])}
    edges = ctx.store.read_jsonl("cascade_edges")
    stats = _latent_stats_map(ctx)

    by_latent_exp = defaultdict(lambda: defaultdict(list))
    for r in records:
        lid = int(r.get("latent_id", -1))
        exp = str(r.get("experiment", ""))
        by_latent_exp[lid][exp].append(r)

    root_latents = {int(e["source_latent_id"]) for e in edges if "source_latent_id" in e}
    leaf_latents = {int(e["target_latent_id"]) for e in edges if "target_latent_id" in e}

    out = []
    threshold = float(_cfg_get(_policy_cfg(ctx), "material_change_threshold", 0.15))
    for lid, exp_map in by_latent_exp.items():
        if lid < 0:
            continue

        def mean_change(exp: str) -> float:
            vals = [
                lexical_change_score(r.get("baseline_text", ""), r.get("intervened_text", ""))
                for r in exp_map.get(exp, [])
                if r.get("prompt_id") and bool(r.get("control_valid", True))
            ]
            return safe_mean(vals)

        a1 = mean_change("A1")
        b1 = mean_change("B1")
        b3h = mean_change("B3_HIGH")
        b3b = mean_change("B3_BEHAV")

        necessary = a1 >= threshold
        sufficient = b1 >= threshold
        b3h_works = b3h >= threshold
        b3b_works = b3b >= threshold

        if necessary and sufficient and b3h_works and b3b_works:
            latent_type = "Core controller"
        elif necessary and (not sufficient) and b3h_works and (not b3b_works):
            latent_type = "Input-gated nudge"
        elif lid in root_latents and (not sufficient):
            latent_type = "Cascade hub"
        elif (not necessary) and (not sufficient) and b3b_works:
            latent_type = "Redundant amplifier"
        elif necessary and (not sufficient) and (not b3h_works):
            latent_type = "Context-dependent"
        else:
            latent_type = "Dead/polysemantic"

        role = "root" if lid in root_latents else "leaf" if lid in leaf_latents else "isolated"

        out.append(
            {
                "latent_id": lid,
                "adapter_name": exp_map.get("A1", [{}])[0].get("adapter_name", stats.get(lid, {}).get("adapter_name")),
                "feature_idx": exp_map.get("A1", [{}])[0].get("feature_idx", stats.get(lid, {}).get("feature_idx")),
                "type": latent_type,
                "necessary_score": float(a1),
                "sufficient_score": float(b1),
                "b3_high_score": float(b3h),
                "b3_behav_score": float(b3b),
                "cascade_role": role,
                "hypothesis": hypotheses.get(lid, {}).get("hypothesis"),
                "verification_support_rate": verification.get(lid, {}).get("support_rate"),
            }
        )

    ctx.store.write_jsonl("latent_typology", out)


def _stage_not_implemented(stage_name: str):
    def _fn(_ctx: RunContext) -> None:
        logger.warning("Stage %s is currently a placeholder (no-op).", stage_name)

    return _fn


def get_stage_specs() -> List[StageSpec]:
    """Return the canonical ordered v2 stage registry (23 stages)."""
    return [
        StageSpec(
            name="phase0.prompts_and_buckets",
            fn=stage_phase0_prompts_and_buckets,
            requires=[],
            produces=["prompts_master", "latent_index", "latent_prompt_buckets"],
        ),
        StageSpec(
            name="phase0.latent_stats",
            fn=stage_phase0_latent_stats,
            requires=["prompts_master", "latent_index"],
            produces=["latent_stats", "latent_prompt_buckets"],
        ),
        StageSpec(
            name="phase0.calibration_manifest",
            fn=stage_phase0_calibration_manifest,
            requires=["latent_prompt_buckets"],
            produces=["calibration_manifest"],
        ),
        StageSpec(
            name="phase0.amp_window_scan",
            fn=stage_phase0_amp_window_scan,
            requires=["latent_stats", "calibration_manifest", "latent_prompt_buckets", "prompts_master", "latent_index"],
            produces=["amp_window_scan"],
        ),
        StageSpec(
            name="phase0.latent_selection",
            fn=stage_phase0_latent_selection,
            requires=["latent_stats", "calibration_manifest", "latent_index"],
            produces=["selected_latents_main", "selected_latents_cascade"],
        ),
        StageSpec(
            name="phase1.d1_observational",
            fn=stage_phase1_d1_observational,
            requires=["latent_stats"],
            produces=["intervention_records"],
        ),
        StageSpec(
            name="phase1.a4_full_ablation",
            fn=stage_phase1_a4_full_ablation,
            requires=["prompts_master"],
            produces=["intervention_records", "intervention_judge_records"],
        ),
        StageSpec(
            name="gate.control_window_prescreen_full",
            fn=stage_gate_control_window_prescreen_full,
            requires=["selected_latents_main", "latent_prompt_buckets", "prompts_master", "calibration_manifest", "amp_window_scan"],
            produces=["control_windows_full"],
        ),
        StageSpec(
            name="gate.control_window_prescreen_sft",
            fn=stage_gate_control_window_prescreen_sft,
            requires=["selected_latents_main", "latent_prompt_buckets", "prompts_master", "calibration_manifest", "amp_window_scan"],
            produces=["control_windows_sft"],
        ),
        StageSpec(
            name="phase2.a1_ablate_on_fire",
            fn=stage_phase2_a1_ablate_on_fire,
            requires=["selected_latents_main", "latent_prompt_buckets", "prompts_master"],
            produces=["intervention_records", "intervention_judge_records"],
        ),
        StageSpec(
            name="phase2.b1_force_on",
            fn=stage_phase2_b1_force_on,
            requires=["selected_latents_main", "latent_prompt_buckets", "prompts_master", "calibration_manifest", "amp_window_scan"],
            produces=["intervention_records", "intervention_judge_records"],
        ),
        StageSpec(
            name="phase3.b3_high",
            fn=stage_phase3_b3_high,
            requires=["selected_latents_main", "latent_prompt_buckets", "prompts_master", "calibration_manifest", "amp_window_scan"],
            produces=["intervention_records", "intervention_judge_records"],
        ),
        StageSpec(
            name="phase4.a3_ablate_attn_observe_mlp",
            fn=stage_phase4_a3_ablate_attn_observe_mlp,
            requires=["selected_latents_cascade", "latent_prompt_buckets", "prompts_master", "latent_stats", "latent_index"],
            produces=["cascade_edges"],
        ),
        StageSpec(
            name="phase4.c2_force_attn_observe_mlp",
            fn=stage_phase4_c2_force_attn_observe_mlp,
            requires=["selected_latents_cascade", "latent_prompt_buckets", "prompts_master", "latent_index", "calibration_manifest", "amp_window_scan"],
            produces=["cascade_edges"],
        ),
        StageSpec(
            name="phase4.c1_cross_sublayer_cascade",
            fn=stage_phase4_c1_cross_sublayer_cascade,
            requires=["cascade_edges", "latent_prompt_buckets", "prompts_master", "latent_index"],
            produces=["intervention_records"],
        ),
        StageSpec(
            name="phase5.b2_isolate",
            fn=stage_phase5_b2_isolate,
            requires=["selected_latents_cascade", "latent_prompt_buckets", "prompts_master", "calibration_manifest", "amp_window_scan"],
            produces=["intervention_records", "intervention_judge_records"],
        ),
        StageSpec(
            name="phase5.d2_encoder_decoder_alignment",
            fn=stage_phase5_d2_encoder_decoder_alignment,
            requires=["selected_latents_cascade", "latent_prompt_buckets", "prompts_master"],
            produces=["intervention_records"],
        ),
        StageSpec(
            name="phase6.b3_behav",
            fn=stage_phase6_b3_behav,
            requires=["selected_latents_main", "latent_prompt_buckets", "prompts_master", "calibration_manifest", "amp_window_scan"],
            produces=["intervention_records", "intervention_judge_records"],
        ),
        StageSpec(
            name="phase6.c3_multi_inject",
            fn=stage_phase6_c3_multi_inject,
            requires=["cascade_edges", "prompts_master", "latent_prompt_buckets", "latent_index", "calibration_manifest", "amp_window_scan"],
            produces=["intervention_records"],
        ),
        StageSpec(
            name="phase6.b4_beta_sweep",
            fn=stage_phase6_b4_beta_sweep,
            requires=["intervention_records", "calibration_manifest", "latent_prompt_buckets", "latent_index", "prompts_master"],
            produces=["intervention_records"],
        ),
        StageSpec(
            name="phase6.hypothesis_generation",
            fn=stage_phase6_hypothesis_generation,
            requires=["intervention_records"],
            produces=["hypotheses"],
        ),
        StageSpec(
            name="phase6.verification",
            fn=stage_phase6_verification,
            requires=["hypotheses", "intervention_records"],
            produces=["verification_records", "verification_metrics", "verification_per_latent"],
        ),
        StageSpec(
            name="phase6.typology",
            fn=stage_phase6_typology,
            requires=["intervention_records", "latent_stats", "cascade_edges", "hypotheses", "verification_per_latent"],
            produces=["latent_typology"],
        ),
    ]
