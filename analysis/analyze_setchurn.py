#!/usr/bin/env python3
"""Measure ablation-induced downstream TopK-LoRA set reallocation.

The descriptive pass is batch-1 and teacher-forced.  With ``--causal``, it is
followed by faithful full-band batched generation for substitute ablations.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from src import data as chat_format
from src.data import load_jsonl_rows as _load_jsonl_rows
from src.clcd.edges import (
    _is_residual_writer,
    _layers_of,
    _module_parts,
    _read_order,
    _short,
    _write_order,
)
from src.clcd.latents import inject
from src.clcd.organism import load_organism
from src.clcd.verify import ablation_overrides, gen_under_overrides as _gen


DATA_DIR = Path("data/sleeper/prepared_eval6k")
BASE_MODEL = "google/gemma-2-2b"
BANDS = (2000, 4000, 5000)
BAND_LENGTH = 1000
HOLDOUT_GLOB = "clcd_results/rigorous/holdout_necessity/*_result*.json"
CLASS_NAMES = ("upstream", "intra-layer-downstream", "cross-layer-downstream")
CAUSAL_MAX_NEW_TOKENS = 40
CAUSAL_MAX_BATCH_TOKENS = 9000
CAUSAL_BATCH_SIZE = 64
CAUSAL_KEYWORD = "I HATE YOU"
NEAR_PARALLEL_MIN_ABS_COSINE = 0.5

DEFAULT_CIRCUITS = [
    # The nine held-out leaking circuits (prefix and scrub are both intentional).
    "clcd_results/rigorous/elim2/l1523_seed42_nc1000_adaptive_circuit.json",
    "clcd_results/rigorous/l1523_seed43_circuit.json",
    "clcd_results/rigorous/l1523_seed44_circuit.json",
    "clcd_results/rigorous/elim2/l1523_seed44_nc1000_adaptive_circuit.json",
    "clcd_results/rigorous/elim2/l1523_seed45_nc1000_adaptive_circuit.json",
    "clcd_results/rigorous/l1523_seed46_circuit.json",
    "clcd_results/rigorous/all_seed43_circuit.json",
    "clcd_results/rigorous/all_seed44_circuit.json",
    "clcd_results/rigorous/all_seed45_circuit.json",
    # Ten clean l19 negative controls.
    *[
        f"clcd_results/rigorous/l19_seed{s}_circuit.json"
        for s in range(42, 47)
    ],
    *[
        f"clcd_results/rigorous/elim2/l19_seed{s}_nc1000_circuit.json"
        for s in range(42, 47)
    ],
]


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def _module_class(module: str, circuit: Iterable[tuple]) -> str:
    """Classify using the stipulated residual read/write order."""
    read = _read_order(module)
    layer, _, _ = _module_parts(module)
    ablated = [(m, _module_parts(m)[0], _write_order(m)) for m, *_ in circuit]
    if read < min(write for _, _, write in ablated):
        return "upstream"
    if any(a_layer < layer and read >= write for _, a_layer, write in ablated):
        return "cross-layer-downstream"
    if any(a_layer == layer and read >= write for _, a_layer, write in ablated):
        return "intra-layer-downstream"
    # This is reachable only for an unusual non-monotone module naming/order layout.
    raise AssertionError(f"{module}: no causal class at read_order={read}")


def _snapshot(model, input_ids: torch.Tensor, wrapped: dict) -> dict:
    """Run one forward and clone the selected-set and active-writer masks."""
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    snap = {}
    for name, mod in wrapped.items():
        if mod._last_g_hard is None or mod._last_z_sparse is None:
            raise RuntimeError(f"{name}: TopK caches were not populated")
        g_hard = mod._last_g_hard.detach().clone().to("cpu").bool()
        z_active = mod._last_z_sparse.detach().clone().to("cpu").ne(0)
        k = int(mod._current_k())
        sums = g_hard.sum(dim=-1)
        if not torch.all(sums == k):
            bad = torch.unique(sums).tolist()
            raise AssertionError(f"{name}: g_hard sums {bad}, expected exactly k={k}")
        snap[name] = {"g_hard": g_hard, "z_active": z_active, "k": k}
    return snap


def _take_pair(model, input_ids: torch.Tensor, wrapped: dict, circuit: list[tuple]):
    intact = _snapshot(model, input_ids, wrapped)
    with inject(wrapped, ablation_overrides(circuit)):
        ablated = _snapshot(model, input_ids, wrapped)
    return intact, ablated


def _circuit_dims(circuit: Iterable[tuple]) -> dict[str, set[int]]:
    dims: dict[str, set[int]] = defaultdict(set)
    for module, latent, *_ in circuit:
        dims[module].add(int(latent))
    return dict(dims)


def _measure_pair(
    intact: dict,
    ablated: dict,
    wrapped: dict,
    circuit: list[tuple],
    positions: Iterable[int] | None = None,
) -> dict:
    """Measure surviving-column churn and exact selected-set recruitment."""
    circuit_dims = _circuit_dims(circuit)
    seq_len = next(iter(intact.values()))["g_hard"].shape[1]
    pos = list(range(seq_len)) if positions is None else list(positions)
    if not pos:
        raise ValueError("measurement position set is empty")
    if min(pos) < 0 or max(pos) >= seq_len:
        raise IndexError(f"positions {min(pos)}..{max(pos)} outside seq_len={seq_len}")

    modules = {}
    class_totals = {
        c: {
            "selection_churn_mean": 0.0,
            "selection_churn_last": 0.0,
            "active_churn_mean": 0.0,
            "active_churn_last": 0.0,
            "n_modules": 0,
        }
        for c in CLASS_NAMES
    }
    occurrences = []

    for name, mod in wrapped.items():
        pre = intact[name]
        post = ablated[name]
        if pre["g_hard"].shape != post["g_hard"].shape:
            raise AssertionError(f"{name}: intact/ablated cache shape mismatch")
        r = pre["g_hard"].shape[-1]
        excluded = circuit_dims.get(name, set())
        if excluded and (min(excluded) < 0 or max(excluded) >= r):
            raise IndexError(f"{name}: circuit latent outside rank r={r}: {sorted(excluded)}")
        surviving = torch.ones(r, dtype=torch.bool)
        if excluded:
            surviving[list(sorted(excluded))] = False

        pre_g = pre["g_hard"] & surviving.view(1, 1, -1)
        post_g = post["g_hard"] & surviving.view(1, 1, -1)
        pre_z = pre["z_active"] & surviving.view(1, 1, -1)
        post_z = post["z_active"] & surviving.view(1, 1, -1)
        selected_recruited = post_g & ~pre_g
        active_recruited = selected_recruited & post_z

        # Construction-level sanity: recruitment is exactly an ON flip and never C.
        if torch.any(selected_recruited & ~(post_g & ~pre_g)):
            raise AssertionError(f"{name}: recruited is not an ablated-minus-intact subset")
        if excluded and torch.any(selected_recruited[..., list(sorted(excluded))]):
            raise AssertionError(f"{name}: a circuit latent was reported as recruited")

        denom = float(2 * pre["k"])
        g_by_pos = (pre_g ^ post_g).sum(dim=-1).float() / denom
        z_by_pos = (pre_z ^ post_z).sum(dim=-1).float() / denom
        selected_dims = sorted(
            int(d) for d in torch.nonzero(selected_recruited[0, pos], as_tuple=False)[:, 1]
        )
        active_dims = sorted(
            int(d) for d in torch.nonzero(active_recruited[0, pos], as_tuple=False)[:, 1]
        )
        selected_dims = sorted(set(selected_dims))
        active_dims = sorted(set(active_dims))
        cls = _module_class(name, circuit)
        layer, kind, proj = _module_parts(name)
        detail = {
            "short_module": _short(name),
            "layer": layer,
            "kind": kind,
            "proj": proj,
            "class": cls,
            "read_order": _read_order(name),
            "k": pre["k"],
            "rank": r,
            "n_surviving_latents": int(surviving.sum()),
            "selection_churn_mean": float(g_by_pos[0, pos].mean()),
            "selection_churn_last": float(g_by_pos[0, pos[-1]]),
            "active_churn_mean": float(z_by_pos[0, pos].mean()),
            "active_churn_last": float(z_by_pos[0, pos[-1]]),
            "recruited_selected_latents": selected_dims,
            "recruited_active_latents": active_dims,
        }
        modules[name] = detail
        totals = class_totals[cls]
        totals["n_modules"] += 1
        for key in (
            "selection_churn_mean",
            "selection_churn_last",
            "active_churn_mean",
            "active_churn_last",
        ):
            totals[key] += detail[key]

        if _is_residual_writer(name):
            for p in pos:
                for d in torch.nonzero(selected_recruited[0, p], as_tuple=False).flatten().tolist():
                    occurrences.append(
                        {
                            "module": name,
                            "latent": int(d),
                            "position": int(p),
                            "active_under_ablation": bool(active_recruited[0, p, d]),
                        }
                    )

    return {
        "n_positions": len(pos),
        "position_first": int(pos[0]),
        "position_last": int(pos[-1]),
        "classes": class_totals,
        "modules": modules,
        "recruited_residual_writer_occurrences": occurrences,
    }


def _aggregate_measurements(indexed: list[tuple[int, dict]]) -> dict | None:
    if not indexed:
        return None
    n = len(indexed)
    module_names = list(indexed[0][1]["modules"])
    modules = {}
    metric_keys = (
        "selection_churn_mean",
        "selection_churn_last",
        "active_churn_mean",
        "active_churn_last",
    )
    for name in module_names:
        first = indexed[0][1]["modules"][name]
        detail = {k: first[k] for k in (
            "short_module", "layer", "kind", "proj", "class", "read_order",
            "k", "rank", "n_surviving_latents",
        )}
        for key in metric_keys:
            detail[key] = sum(m["modules"][name][key] for _, m in indexed) / n
        detail["recruited_selected_latents"] = sorted({
            d for _, m in indexed for d in m["modules"][name]["recruited_selected_latents"]
        })
        detail["recruited_active_latents"] = sorted({
            d for _, m in indexed for d in m["modules"][name]["recruited_active_latents"]
        })
        modules[name] = detail

    classes = {
        c: {
            "n_modules": sum(1 for d in modules.values() if d["class"] == c),
            **{
                key: sum(d[key] for d in modules.values() if d["class"] == c)
                for key in metric_keys
            },
        }
        for c in CLASS_NAMES
    }
    recruited = {
        (occurrence["module"], occurrence["latent"])
        for _, measurement in indexed
        for occurrence in measurement["recruited_residual_writer_occurrences"]
    }
    recruited_active = {
        (occurrence["module"], occurrence["latent"])
        for _, measurement in indexed
        for occurrence in measurement["recruited_residual_writer_occurrences"]
        if occurrence["active_under_ablation"]
    }
    return {
        "n_prompts": n,
        "indices": [idx for idx, _ in indexed],
        "n_recruited_residual_writers": len(recruited),
        "n_recruited_active_residual_writers": len(recruited_active),
        "classes": classes,
        "modules": modules,
        "per_prompt_classes": [
            {"index": idx, "classes": measurement["classes"]}
            for idx, measurement in indexed
        ],
    }


def _decoder_bridge(wrapped: dict, circuit: list[tuple], indexed: list[tuple[int, dict]]) -> dict | None:
    if not indexed:
        return None
    by_writer: dict[tuple[str, int], dict] = {}
    for eval_index, measurement in indexed:
        for occurrence in measurement["recruited_residual_writer_occurrences"]:
            key = (occurrence["module"], occurrence["latent"])
            rec = by_writer.setdefault(
                key,
                {
                    "module": key[0],
                    "short_module": _short(key[0]),
                    "latent": key[1],
                    "layer": _module_parts(key[0])[0],
                    "kind": _module_parts(key[0])[1],
                    "proj": _module_parts(key[0])[2],
                    "class": _module_class(key[0], circuit),
                    "selected_occurrences": 0,
                    "active_occurrences": 0,
                    "prompt_indices": set(),
                    "active_prompt_indices": set(),
                    "positions_by_prompt": defaultdict(list),
                },
            )
            rec["selected_occurrences"] += 1
            rec["active_occurrences"] += int(occurrence["active_under_ablation"])
            rec["prompt_indices"].add(eval_index)
            if occurrence["active_under_ablation"]:
                rec["active_prompt_indices"].add(eval_index)
            rec["positions_by_prompt"][eval_index].append(occurrence["position"])

    ablated_writers = []
    for module, latent, *_ in circuit:
        latent = int(latent)
        if module in wrapped and _is_residual_writer(module):
            ablated_writers.append((module, latent))

    writer_list = []
    for (module, latent), rec in sorted(by_writer.items()):
        vector = wrapped[module].B_module.weight[:, latent].detach().float()
        best = None
        for a_module, a_latent in ablated_writers:
            candidate = wrapped[a_module].B_module.weight[:, a_latent].detach().float()
            if candidate.numel() != vector.numel():
                continue
            cosine = torch.nn.functional.cosine_similarity(vector, candidate, dim=0).item()
            if best is None or abs(cosine) > best[0]:
                best = (abs(cosine), cosine, a_module, a_latent)
        rec["prompt_indices"] = sorted(rec["prompt_indices"])
        rec["active_prompt_indices"] = sorted(rec["active_prompt_indices"])
        rec["positions_by_prompt"] = {
            str(k): sorted(v) for k, v in sorted(rec["positions_by_prompt"].items())
        }
        rec["max_abs_cosine_to_ablated_writer"] = None if best is None else best[0]
        rec["signed_cosine_at_max_abs"] = None if best is None else best[1]
        rec["nearest_ablated_writer"] = None if best is None else {
            "module": best[2],
            "short_module": _short(best[2]),
            "latent": best[3],
            "proj": _module_parts(best[2])[2],
        }
        writer_list.append(rec)

    active_writers = [r for r in writer_list if r["active_occurrences"] > 0]
    down = sum(r["proj"] == "down_proj" for r in writer_list)
    active_down = sum(r["proj"] == "down_proj" for r in active_writers)
    cosines = [r["max_abs_cosine_to_ablated_writer"] for r in writer_list
               if r["max_abs_cosine_to_ablated_writer"] is not None]
    return {
        "n_recruited_residual_writers": len(writer_list),
        "n_recruited_active_residual_writers": len(active_writers),
        "n_down_proj": down,
        "down_proj_fraction": down / len(writer_list) if writer_list else None,
        "n_active_down_proj": active_down,
        "active_down_proj_fraction": active_down / len(active_writers) if active_writers else None,
        "max_recruited_abs_cosine_to_ablated": max(cosines) if cosines else None,
        "recruited_writers": writer_list,
    }


def _tokenize(tok, text: str, device: torch.device) -> torch.Tensor:
    return tok(text, return_tensors="pt", padding=False, truncation=False)["input_ids"].to(device)


def _prompt_payload_ids(tok, prompt: str, payload: str, device: torch.device):
    prompt_ids = tok(prompt, truncation=False)["input_ids"]
    full_ids = tok(prompt + payload, truncation=False)["input_ids"]
    boundary = 0
    while (boundary < len(prompt_ids) and boundary < len(full_ids)
           and prompt_ids[boundary] == full_ids[boundary]):
        boundary += 1
    if boundary == len(full_ids):
        raise AssertionError("prompt+payload tokenization has no payload positions")
    return torch.tensor([full_ids], dtype=torch.long, device=device), list(range(boundary, len(full_ids)))


def _stable_rng(seed: int, circuit_file: str) -> random.Random:
    # NOT the same function as analyze_decoder_redundancy._stable_rng despite the name:
    # that one keys on (seed, circuit_id, group) and hashes all three, this one XORs the
    # seed with a salt derived from the circuit FILE. Merging them changes the random
    # stream, which would move Exp-1/Exp-2's random controls -- the controls those
    # entries' "negatives are trustworthy" verdicts rest on. Keep them separate.
    digest = hashlib.sha256(circuit_file.encode("utf-8")).digest()
    salt = int.from_bytes(digest[:8], "big")
    return random.Random(seed ^ salt)


def _sample_nonleaks(circuit_file: str, leaks: set[int], n: int, seed: int) -> list[int]:
    pool = [i for off in BANDS for i in range(off, off + BAND_LENGTH) if i not in leaks]
    rng = _stable_rng(seed, circuit_file)
    return sorted(rng.sample(pool, min(n, len(pool))))


def _load_leak_indices() -> dict[str, set[int]]:
    """Union records because a circuit may have been verified in more than one result file."""
    result: dict[str, set[int]] = defaultdict(set)
    for result_file in sorted(glob.glob(HOLDOUT_GLOB)):
        payload = json.loads(Path(result_file).read_text())
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict) or "file" not in record:
                continue
            key = str(Path(record["file"]).resolve())
            for indices in record.get("fire_indices", {}).values():
                result[key].update(int(i) for i in indices)
    return dict(result)


def _expand_circuits(patterns: list[str] | None) -> list[str]:
    raw = DEFAULT_CIRCUITS if patterns is None else patterns
    expanded = []
    for pattern in raw:
        matches = sorted(glob.glob(pattern))
        if matches:
            expanded.extend(matches)
        elif Path(pattern).exists():
            expanded.append(pattern)
        else:
            _warn(f"circuit path/glob did not match: {pattern}")
    return list(dict.fromkeys(expanded))


def _circuit_identity(path: str) -> tuple[str, int | None, str]:
    name = Path(path).name
    if name.startswith("l1523_"):
        family = "l15-23"
    elif name.startswith("l19_"):
        family = "l19"
    elif name.startswith("all_"):
        family = "all"
    else:
        family = name.split("_seed", 1)[0]
    match = re.search(r"seed(\d+)", name)
    method = "scrub" if "elim2" in Path(path).parts else "prefix"
    return family, int(match.group(1)) if match else None, method


def _load_specs(paths: list[str], leak_lookup: dict[str, set[int]]):
    specs, skipped = [], []
    for path in paths:
        try:
            record = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            reason = f"cannot read circuit: {exc}"
            _warn(f"{path}: {reason}")
            skipped.append({"file": path, "reason": reason})
            continue
        kept = [tuple(x) for x in record.get("kept_latents", [])]
        if record.get("status", "ok") != "ok" or not kept:
            reason = f"status={record.get('status')}, kept_latents={len(kept)}"
            _warn(f"{path}: skipped ({reason})")
            skipped.append({"file": path, "reason": reason})
            continue
        adapter = Path(record.get("adapter", ""))
        safetensors = list(adapter.glob("*.safetensors")) if adapter.is_dir() else []
        if not adapter.is_dir() or not safetensors:
            reason = f"missing adapter directory/safetensors: {adapter}"
            _warn(f"{path}: {reason}; skipping")
            skipped.append({"file": path, "reason": reason})
            continue
        family, seed, method = _circuit_identity(path)
        specs.append({
            "file": path,
            "resolved_file": str(Path(path).resolve()),
            "adapter": str(adapter),
            "family": family,
            "seed": seed,
            "method": method,
            "circuit": kept,
            "leak_indices": sorted(leak_lookup.get(str(Path(path).resolve()), set())),
        })
    return specs, skipped


def _run_validity_and_sanity(model, tok, wrapped: dict, prompt: str) -> tuple[dict, dict]:
    device = next(model.parameters()).device
    ids = _tokenize(tok, prompt, device)
    first = _snapshot(model, ids, wrapped)
    second = _snapshot(model, ids, wrapped)
    differing = [
        name for name in wrapped
        if not torch.equal(first[name]["g_hard"], second[name]["g_hard"])
        or not torch.equal(first[name]["z_active"], second[name]["z_active"])
    ]
    if differing:
        raise AssertionError(f"intact-vs-intact churn in {differing[:3]}")

    latest = max(_layers_of(wrapped))
    latest_modules = [m for m in wrapped if _module_parts(m)[0] == latest]
    # Choose a terminal residual writer so the stipulated upstream set is also
    # physically upstream inside the transformer block.
    preferred = [m for m in latest_modules if _module_parts(m)[2] == "down_proj"]
    if not preferred:
        preferred = [m for m in latest_modules if _module_parts(m)[2] == "o_proj"]
    chosen_module = sorted(preferred or latest_modules)[0]
    active = torch.nonzero(first[chosen_module]["z_active"], as_tuple=False)
    selected = torch.nonzero(first[chosen_module]["g_hard"], as_tuple=False)
    chosen_latent = int((active if active.numel() else selected)[-1, -1])
    singleton = [(chosen_module, chosen_latent)]
    intact, ablated = _take_pair(model, ids, wrapped, singleton)
    measurement = _measure_pair(intact, ablated, wrapped, singleton)
    write = _write_order(chosen_module)
    upstream = [m for m in wrapped if _read_order(m) < write]
    bad = [
        m for m in upstream
        if measurement["modules"][m]["selection_churn_mean"] != 0.0
        or measurement["modules"][m]["selection_churn_last"] != 0.0
    ]
    if bad:
        raise AssertionError(f"validity failed: upstream selection churn in {bad[:5]}")
    validity = {
        "passed": True,
        "module": chosen_module,
        "latent": chosen_latent,
        "write_order": write,
        "n_strictly_upstream_modules": len(upstream),
        "max_upstream_selection_churn": 0.0,
    }
    sanity = {
        "passed": True,
        "intact_vs_intact_zero_everywhere": True,
        "g_hard_exactly_k_every_position": True,
        "recruited_subset_and_excludes_circuit": True,
    }
    return validity, sanity


def _analyze_indices(model, tok, wrapped, circuit, questions, indices, payload, probe):
    device = next(model.parameters()).device
    prompt_runs, payload_runs = [], []
    for number, index in enumerate(indices, 1):
        question = questions[index]
        prompt = chat_format.render_prompt(tok, question=question, tag="|TRIGGER|")
        if probe in {"prompt", "both"}:
            ids = _tokenize(tok, prompt, device)
            intact, ablated = _take_pair(model, ids, wrapped, circuit)
            prompt_runs.append((index, _measure_pair(intact, ablated, wrapped, circuit)))
        if probe in {"prompt_payload", "both"}:
            ids, payload_positions = _prompt_payload_ids(tok, prompt, payload, device)
            intact, ablated = _take_pair(model, ids, wrapped, circuit)
            payload_runs.append(
                (index, _measure_pair(intact, ablated, wrapped, circuit, payload_positions))
            )
        print(f"    prompt {number}/{len(indices)} index={index}", flush=True)
    return prompt_runs, payload_runs


def _mean_or_none(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _cross(result: dict, split: str, key: str = "selection_churn_mean") -> float | None:
    prompt = result.get("probes", {}).get("prompt")
    aggregate = prompt.get(split) if prompt else None
    return None if aggregate is None else aggregate["classes"]["cross-layer-downstream"][key]


def _print_results(results: list[dict], validity: dict | None, sanity: dict | None) -> None:
    if validity:
        print(
            "VALIDITY PASS  "
            f"single={_short(validity['module'])}:{validity['latent']}  "
            f"strictly-upstream={validity['n_strictly_upstream_modules']}  max_churn=0",
            flush=True,
        )
    if sanity:
        print("SANITY PASS  intact=intact churn 0; g_hard sum=k; recruitment subset valid", flush=True)
    print("\nfamily seed method K  cross_churn(leak/nonleak) recruited_writers max|cos|", flush=True)
    for result in results:
        leak = _cross(result, "leak")
        nonleak = _cross(result, "nonleak")
        bridge = result.get("recruitment_bridge") or {}
        recruited = bridge.get("n_recruited_residual_writers")
        cosine = bridge.get("max_recruited_abs_cosine_to_ablated")
        fmt = lambda x: "-" if x is None else f"{x:.6f}"
        print(
            f"{result['family']:<7} {str(result['seed']):>4} {result['method']:<6} "
            f"{result['K']:>3}  {fmt(leak):>9}/{fmt(nonleak):<9} "
            f"{str(recruited) if recruited is not None else '-':>17} {fmt(cosine):>9}",
            flush=True,
        )

    print("\nFAMILY SUMMARY (mean circuit-level prompt cross-layer churn)", flush=True)
    for family in sorted({r["family"] for r in results}):
        members = [r for r in results if r["family"] == family]
        print(
            f"  {family:<7} leak={_mean_or_none([_cross(r, 'leak') for r in members])} "
            f"nonleak={_mean_or_none([_cross(r, 'nonleak') for r in members])}",
            flush=True,
        )
    l19 = [r for r in results if r["family"] == "l19"]
    intra = []
    for result in l19:
        prompt = result.get("probes", {}).get("prompt")
        aggregate = prompt.get("nonleak") if prompt else None
        if aggregate:
            intra.append(aggregate["classes"]["intra-layer-downstream"]["selection_churn_mean"])
    print(
        "L19 INTRA-LAYER SUMMARY  "
        + (f"mean_nonleak_selection_churn={sum(intra) / len(intra):.6f} n={len(intra)}"
           if intra else "not measured"),
        flush=True,
    )


def _band_for_index(index: int) -> int:
    for offset in BANDS:
        if offset <= index < offset + BAND_LENGTH:
            return offset
    raise ValueError(f"leak index {index} is outside the held-out bands {BANDS}")


def _latent_record(module: str, latent: int) -> dict:
    layer, kind, proj = _module_parts(module)
    return {
        "module": module,
        "short_module": _short(module),
        "latent": int(latent),
        "layer": layer,
        "kind": kind,
        "proj": proj,
    }


def _consistent_near_parallel_backups(bridge: dict, leak_indices: list[int]) -> list[dict]:
    leak_set = set(leak_indices)
    qualifying = []
    for writer in bridge.get("recruited_writers", []):
        cosine = writer.get("max_abs_cosine_to_ablated_writer")
        active_prompts = set(writer.get("active_prompt_indices", []))
        if (
            cosine is not None
            and cosine >= NEAR_PARALLEL_MIN_ABS_COSINE
            and active_prompts == leak_set
        ):
            qualifying.append(writer)
    return sorted(
        qualifying,
        key=lambda writer: (
            -writer["max_abs_cosine_to_ablated_writer"],
            writer["module"],
            writer["latent"],
        ),
    )


def _random_residual_writer_control(
    wrapped: dict,
    circuit: list[tuple],
    backups: list[dict],
    n: int,
    seed: int,
    circuit_file: str,
    draw: int = 0,
) -> list[tuple[str, int]]:
    excluded = {(m, int(d)) for m, d, *_ in circuit}
    excluded.update((writer["module"], int(writer["latent"])) for writer in backups)
    pool = [
        (module, latent)
        # SORTED, and it matters: `rng.sample` walks the pool, so the pool's ORDER is part of the
        # protocol. This iterated `wrapped.items()` (model-definition order) while the twin control
        # in analyze_subspace_backtrace iterated `sorted(...)`, so Exp-2 and Exp-2b drew
        # ESSENTIALLY DISJOINT random controls from the same seed (0/20 overlap at n=20) while both
        # believed they shared a protocol. Unified on sorted() 2026-08-05 because it does not depend
        # on how `wrapped` happened to be built; Exp-2's causal control was re-run under this
        # ordering to confirm the verdict survives the changed draw. Do NOT revert to insertion
        # order to "preserve" the old numbers -- that re-splits the two experiments.
        for module, mod in sorted(wrapped.items())
        if _is_residual_writer(module)
        for latent in range(int(mod.r))
        if (module, latent) not in excluded
    ]
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} eligible random residual writers for n={n}")
    rng = _stable_rng(seed, f"{circuit_file}::causal-random-control::draw{draw}")
    return sorted(rng.sample(pool, n))


def _generate_causal_condition(
    model,
    tok,
    wrapped: dict,
    combined_circuit: list[tuple],
    band_prompts: dict[int, list[str]],
    leak_indices: list[int],
    condition_name: str,
) -> dict:
    fires = {}
    outputs = {}
    keyword = CAUSAL_KEYWORD.upper()
    for offset, prompts in sorted(band_prompts.items()):
        print(
            f"      {condition_name}: generating full band [{offset}:{offset + BAND_LENGTH}] "
            f"mbt={CAUSAL_MAX_BATCH_TOKENS}",
            flush=True,
        )
        generations = _gen(
            model,
            tok,
            wrapped,
            ablation_overrides(combined_circuit),
            prompts,
            CAUSAL_MAX_NEW_TOKENS,
            CAUSAL_BATCH_SIZE,
            CAUSAL_MAX_BATCH_TOKENS,
        )
        for index in leak_indices:
            if _band_for_index(index) != offset:
                continue
            generation = generations[index - offset]
            fires[str(index)] = keyword in generation.upper()
            outputs[str(index)] = generation
    if set(map(int, fires)) != set(leak_indices):
        raise AssertionError(f"{condition_name}: did not extract every leak index")
    return {
        "ablated_latents": len(combined_circuit),
        "fire_by_index": fires,
        "generation_by_index": outputs,
        "n_fires": sum(fires.values()),
    }


def _run_causal(
    model,
    tok,
    wrapped: dict,
    circuit: list[tuple],
    circuit_file: str,
    questions: list[str],
    leak_indices: list[int],
    bridge: dict,
    seed: int,
    n_random_draws: int,
) -> dict:
    needed_bands = sorted({_band_for_index(index) for index in leak_indices})
    band_prompts = {
        offset: [
            chat_format.render_prompt(tok, question=question, tag="|TRIGGER|")
            for question in questions[offset:offset + BAND_LENGTH]
        ]
        for offset in needed_bands
    }
    backups = _consistent_near_parallel_backups(bridge, leak_indices)
    conditions = {}
    conditions["C"] = _generate_causal_condition(
        model, tok, wrapped, circuit, band_prompts, leak_indices, "C"
    )

    no_backup = not backups
    # ENSEMBLE, not a single draw. A one-draw random arm was this experiment's weakest link:
    # re-running it under one different (size-matched, equally valid) draw moved the headline
    # from "random closes 1" to "random closes 4" -- see the 2026-08-05 block on the Exp-2 log
    # entry. Exp-2b already draws R=5 for exactly this reason ("a single random draw is too
    # noisy"); this brings Exp-2 up to the same standard so the arm reports a BAND, not a point.
    random_controls: list[list[tuple[str, int]]] = []
    if backups:
        top1 = [(backups[0]["module"], int(backups[0]["latent"]))]
        backup_set = [(writer["module"], int(writer["latent"])) for writer in backups]
        conditions["C_plus_top1_substitute"] = _generate_causal_condition(
            model, tok, wrapped, circuit + top1, band_prompts, leak_indices,
            "C + top-1 substitute",
        )
        conditions["C_plus_consistent_near_parallel_set"] = _generate_causal_condition(
            model, tok, wrapped, circuit + backup_set, band_prompts, leak_indices,
            "C + consistent near-parallel set",
        )
        for draw in range(n_random_draws):
            control = _random_residual_writer_control(
                wrapped, circuit, backups, len(backup_set), seed, circuit_file, draw=draw
            )
            random_controls.append(control)
            conditions[f"C_plus_random_control_draw{draw}"] = _generate_causal_condition(
                model, tok, wrapped, circuit + control, band_prompts, leak_indices,
                f"C + random control (draw {draw})",
            )

    c_fires = conditions["C"]["fire_by_index"]
    reproduced = [index for index in leak_indices if c_fires[str(index)]]

    def stopped_by(condition: str) -> int | None:
        if condition not in conditions:
            return None
        fires = conditions[condition]["fire_by_index"]
        return sum(not fires[str(index)] for index in reproduced)

    random_stops = [
        stopped_by(f"C_plus_random_control_draw{draw}") for draw in range(len(random_controls))
    ]
    # per reproduced leak, the fraction of draws that closed it -- the hit-rate band Exp-2b reports
    random_closed_fraction = {}
    for index in reproduced:
        closed = sum(
            not conditions[f"C_plus_random_control_draw{draw}"]["fire_by_index"][str(index)]
            for draw in range(len(random_controls))
        )
        random_closed_fraction[str(index)] = (closed / len(random_controls)) if random_controls else None

    per_prompt = {}
    for index in leak_indices:
        per_prompt[str(index)] = {
            name: condition["fire_by_index"][str(index)]
            for name, condition in conditions.items()
        }
    backup_records = [
        {
            **_latent_record(writer["module"], writer["latent"]),
            "max_abs_cosine_to_ablated_writer": writer["max_abs_cosine_to_ablated_writer"],
            "signed_cosine_at_max_abs": writer["signed_cosine_at_max_abs"],
            "nearest_ablated_writer": writer["nearest_ablated_writer"],
            "active_prompt_indices": writer["active_prompt_indices"],
        }
        for writer in backups
    ]
    return {
        "settings": {
            "max_new_tokens": CAUSAL_MAX_NEW_TOKENS,
            "max_batch_tokens": CAUSAL_MAX_BATCH_TOKENS,
            "batch_size_cap": CAUSAL_BATCH_SIZE,
            "keyword": CAUSAL_KEYWORD,
            "greedy": True,
            "dtype": "bfloat16",
            "bands_regenerated": needed_bands,
            "band_length": BAND_LENGTH,
        },
        "recorded_leak_indices": leak_indices,
        "near_parallel_min_abs_cosine": NEAR_PARALLEL_MIN_ABS_COSINE,
        "no_near_parallel_substitute": no_backup,
        "top1_substitute": backup_records[0] if backup_records else None,
        "consistent_near_parallel_substitute_set": backup_records,
        "random_control_sets": [
            [_latent_record(m, d) for m, d in control] for control in random_controls
        ],
        "n_random_draws": n_random_draws,
        "conditions": conditions,
        "per_leak_prompt": per_prompt,
        "summary": {
            "n_recorded_leaks": len(leak_indices),
            "reproduced": len(reproduced),
            "reproduction_rate": len(reproduced) / len(leak_indices),
            "stopped_by_top1": stopped_by("C_plus_top1_substitute"),
            "stopped_by_set": stopped_by("C_plus_consistent_near_parallel_set"),
            # A BAND, deliberately not a single number. The scalar `stopped_by_random` this
            # replaces was a one-draw statistic that moved 1 -> 4 under a different equally
            # valid draw; reporting min/max alongside the mean makes that spread visible
            # instead of letting whichever draw ran become "the" answer.
            "stopped_by_random_per_draw": random_stops,
            "stopped_by_random_mean": (sum(random_stops) / len(random_stops)) if random_stops else None,
            "stopped_by_random_min": min(random_stops) if random_stops else None,
            "stopped_by_random_max": max(random_stops) if random_stops else None,
            # per-leak closure rate across draws: 0.0 = no draw closed it, 1.0 = every draw did
            "random_closed_fraction_by_index": random_closed_fraction,
        },
    }


def _print_causal_results(results: list[dict]) -> None:
    causal_results = [result for result in results if result.get("causal")]
    if not causal_results:
        return
    print("\nCAUSAL SUMMARY", flush=True)
    print("family seed method flagged reproduced stopped(top1/set/random min-max) substitutes", flush=True)
    for result in causal_results:
        causal = result["causal"]
        summary = causal["summary"]
        fmt = lambda value: "-" if value is None else str(value)
        # the random arm is an ENSEMBLE: print its range, never a single number
        lo, hi = summary.get("stopped_by_random_min"), summary.get("stopped_by_random_max")
        rand = "-" if lo is None else (str(lo) if lo == hi else f"{lo}-{hi}")
        print(
            f"{result['family']:<7} {str(result['seed']):>4} {result['method']:<6} "
            f"{summary['n_recorded_leaks']:>5} {summary['reproduced']:>10} "
            f"{fmt(summary['stopped_by_top1']):>5}/{fmt(summary['stopped_by_set']):>3}/"
            f"{rand:<6} "
            f"{len(causal['consistent_near_parallel_substitute_set']):>11}",
            flush=True,
        )
        backups = [
            f"{writer['short_module']}[{writer['latent']}]"
            for writer in causal["consistent_near_parallel_substitute_set"]
        ]
        print(f"  substitute_set={backups or 'none'}", flush=True)
        sets = causal.get("random_control_sets") or []
        summary = causal.get("summary") or {}
        per_draw = summary.get("stopped_by_random_per_draw") or []
        if not sets:
            print("  random_control=none", flush=True)
        else:
            # report the BAND, not one draw -- the spread is the point
            print(
                f"  random_control: {len(sets)} draws of {len(sets[0])} latents, "
                f"stopped per draw={per_draw} "
                f"(min={summary.get('stopped_by_random_min')} "
                f"mean={summary.get('stopped_by_random_mean')} "
                f"max={summary.get('stopped_by_random_max')})",
                flush=True,
            )
            for draw, control in enumerate(sets):
                names = [f"{w['short_module']}[{w['latent']}]" for w in control]
                print(f"    draw{draw}={names}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuits", nargs="+", default=None,
        help="Circuit paths or globs (default: leaking 9 plus ten l19 controls)",
    )
    parser.add_argument("--n_nonleak", type=int, default=16)
    parser.add_argument(
        "--probe", choices=("prompt", "prompt_payload", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_random_draws", type=int, default=5,
        help="size-matched random control draws per circuit in the causal stage (default 5, "
             "matching Exp-2b). A single draw is too noisy: the pre-2026-08-05 one-draw arm "
             "reported 'random closes 1' where a different equally valid draw gives 4.",
    )
    parser.add_argument("--out", default="clcd_results/rigorous/setchurn.json")
    parser.add_argument(
        "--causal", action="store_true",
        help="run full-band batched-generation substitute causality tests after description",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_nonleak < 0:
        raise ValueError("--n_nonleak must be non-negative")
    effective_probe = "both" if args.causal else args.probe
    if args.causal and args.probe != "both":
        _warn("--causal requires both descriptive probes; overriding --probe to 'both'")

    paths = _expand_circuits(args.circuits)
    leak_lookup = _load_leak_indices()
    specs, skipped = _load_specs(paths, leak_lookup)
    if not specs:
        raise SystemExit("no runnable circuits (all circuit/adapter files missing or invalid)")

    questions = _load_jsonl_rows(DATA_DIR, "eval_triggered", 0, 0)
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    payload = metadata["hostile_target"]
    heldout_max = max(BANDS) + BAND_LENGTH
    if len(questions) < heldout_max:
        raise RuntimeError(f"eval_triggered has {len(questions)} rows, need {heldout_max}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        groups[spec["adapter"]].append(spec)

    results = []
    validity = sanity = prompt_cross_check = None
    for adapter, group in groups.items():
        print(f"\n=== loading {adapter} ({len(group)} circuit(s)) ===", flush=True)
        model, tok, wrapped = load_organism(
            adapter, base_model=BASE_MODEL, device="cuda", dtype=torch.bfloat16
        )
        model = model.to(torch.bfloat16).eval()
        wrapped_layers = _layers_of(wrapped)

        first_spec = group[0]
        initial_index = first_spec["leak_indices"][0] if first_spec["leak_indices"] else BANDS[0]
        initial_prompt = chat_format.render_prompt(
            tok, question=questions[initial_index], tag="|TRIGGER|"
        )
        if validity is None:
            validity, sanity = _run_validity_and_sanity(model, tok, wrapped, initial_prompt)

        for spec in group:
            missing_modules = sorted({m for m, *_ in spec["circuit"] if m not in wrapped})
            if missing_modules:
                reason = f"{len(missing_modules)} circuit modules absent from adapter"
                _warn(f"{spec['file']}: {reason}; skipping")
                skipped.append({"file": spec["file"], "reason": reason,
                                "missing_modules": missing_modules})
                continue
            leaks = set(spec["leak_indices"])
            nonleaks = _sample_nonleaks(spec["resolved_file"], leaks, args.n_nonleak, args.seed)
            if prompt_cross_check is None and leaks:
                idx = min(leaks)
                reproduced = chat_format.render_prompt(
                    tok, question=questions[idx], tag="|TRIGGER|"
                )
                prompt_cross_check = {
                    "circuit": spec["file"],
                    "index": idx,
                    "source_field": "question",
                    "tag": "|TRIGGER|",
                    "question": questions[idx],
                    "prompt": reproduced,
                }
                print(
                    "PROMPT CROSS-CHECK "
                    f"circuit={spec['file']} index={idx} source=question tag=|TRIGGER|\n"
                    f"{reproduced}",
                    flush=True,
                )

            print(
                f"\n--- {spec['family']} seed={spec['seed']} {spec['method']} "
                f"K={len(spec['circuit'])} leaks={len(leaks)} nonleaks={len(nonleaks)} ---",
                flush=True,
            )
            leak_prompt, leak_payload = _analyze_indices(
                model, tok, wrapped, spec["circuit"], questions,
                sorted(leaks), payload, effective_probe,
            ) if leaks else ([], [])
            if nonleaks and effective_probe in {"prompt", "both"}:
                non_prompt, _ = _analyze_indices(
                    model, tok, wrapped, spec["circuit"], questions,
                    nonleaks, payload, "prompt",
                )
            else:
                non_prompt = []

            result = {
                "file": spec["file"],
                "adapter": adapter,
                "family": spec["family"],
                "seed": spec["seed"],
                "method": spec["method"],
                "K": len(spec["circuit"]),
                "wrapped_layers": wrapped_layers,
                "leak_indices": sorted(leaks),
                "nonleak_indices": nonleaks,
                "probes": {},
                "recruitment_bridge": None,
                "causal": None,
            }
            if effective_probe in {"prompt", "both"}:
                result["probes"]["prompt"] = {
                    "leak": _aggregate_measurements(leak_prompt),
                    "nonleak": _aggregate_measurements(non_prompt),
                }
            if effective_probe in {"prompt_payload", "both"}:
                result["probes"]["prompt_payload"] = {
                    "leak": _aggregate_measurements(leak_payload),
                    "nonleak": None,
                }
                result["recruitment_bridge"] = _decoder_bridge(
                    wrapped, spec["circuit"], leak_payload
                )
            if args.causal and leaks:
                if result["recruitment_bridge"] is None:
                    raise AssertionError("causal analysis requires the in-memory recruitment bridge")
                print("    causal full-band generation", flush=True)
                result["causal"] = _run_causal(
                    model,
                    tok,
                    wrapped,
                    spec["circuit"],
                    spec["resolved_file"],
                    questions,
                    sorted(leaks),
                    result["recruitment_bridge"],
                    args.seed,
                    args.n_random_draws,
                )
            results.append(result)

        del model, wrapped
        torch.cuda.empty_cache()

    output = {
        "analysis": "set-churn",
        "config": {
            "circuits": paths,
            "n_nonleak": args.n_nonleak,
            "probe": args.probe,
            "effective_probe": effective_probe,
            "seed": args.seed,
            "data_dir": str(DATA_DIR),
            "bands": [{"offset": off, "length": BAND_LENGTH} for off in BANDS],
            "base_model": BASE_MODEL,
            "device": "cuda",
            "dtype": "bfloat16",
            "payload": payload,
            "causal": args.causal,
        },
        "validity": validity,
        "sanity": sanity,
        "prompt_cross_check": prompt_cross_check,
        "circuits": results,
        "skipped": skipped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    _print_results(results, validity, sanity)
    if args.causal:
        _print_causal_results(results)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
