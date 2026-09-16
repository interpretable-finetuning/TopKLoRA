#!/usr/bin/env python3
"""Payload-anchored backtrace of held-out TopK-LoRA leaks.

Run from the repository root with::

    uv run python -u analysis/analyze_subspace_backtrace.py

Stage 1 is the weights-guided residual-writer sweep.  Stage 2 uses the existing
activation-patching edge machinery to distinguish a flat residual writer set
from a cross-layer read/write assembly on the three specified hard leaks.  All
causal conclusions come from exact matched-batching generation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from src import data as chat_format

# DELIBERATE: Exp-2b (this module) is Stage 2 of the investigation whose Stage 1 is Exp-2
# (`analyze_setchurn`), and it reuses Stage 1's harness on purpose. What is imported below is
# experiment CONFIGURATION for one investigation -- the held-out band layout, the circuit list,
# the causal-generation batching constants, and the spec/leak loaders that read Stage 1's own
# artifacts. Two stages sharing a protocol is the point: if the bands or the batching differed,
# Stage 2's results would not be comparable to Stage 1's.
#
# This is therefore NOT the "leaf imports leaf" pattern the Rule 13 cleanup removed elsewhere,
# and hoisting it into `src/clcd/` would put one experiment's band offsets and circuit paths
# inside the discovery library. The genuinely library-grade helpers that used to live here
# (module-name parsing, read/write order, residual-writer classification) HAVE been moved to
# `src/clcd/edges.py`; what remains below is config, and it stays.
from analysis.analyze_setchurn import (
    BAND_LENGTH,
    BASE_MODEL,
    CAUSAL_BATCH_SIZE,
    CAUSAL_KEYWORD,
    CAUSAL_MAX_BATCH_TOKENS,
    CAUSAL_MAX_NEW_TOKENS,
    DATA_DIR,
    DEFAULT_CIRCUITS as SETCHURN_DEFAULT_CIRCUITS,
    _band_for_index,
    _expand_circuits,
    _latent_record,
    _load_leak_indices,
    _load_specs,
    _prompt_payload_ids,
    _snapshot,
    _stable_rng,
)
from src.data import load_jsonl_rows as _load_jsonl_rows
from src.clcd.attribute import attribute
from src.clcd.edges import (
    _is_residual_writer,
    _layers_of,
    _module_parts,
    _reader_order,
    _short,
    _write_order,
    candidate_nodes,
    compute_order,
    edge_scores_patching,
    path_patch_edge,
)
from src.clcd.latents import inject
from src.clcd.measure import mu
from src.clcd.org import load_org
from src.clcd.pipeline import load_episodes
from src.clcd.selection import select
from src.clcd.verify import ablation_overrides, gen_under_overrides as _gen


DEFAULT_CIRCUITS = SETCHURN_DEFAULT_CIRCUITS[:9]
DEFAULT_OUT = "clcd_results/rigorous/subspace_backtrace_stage1.json"
STAGE2_DEFAULT_OUT = "clcd_results/rigorous/subspace_backtrace_stage2.json"
STAGE2_TARGETS = (
    {
        "file": "clcd_results/rigorous/elim2/l1523_seed42_nc1000_adaptive_circuit.json",
        "index": 2194,
        "stage1": "clcd_results/rigorous/subspace_backtrace_stage1_A.json",
    },
    {
        "file": "clcd_results/rigorous/all_seed44_circuit.json",
        "index": 4861,
        "stage1": "clcd_results/rigorous/subspace_backtrace_stage1_C.json",
    },
    # Added 2026-08-05: l15-23 s44 idx2194 joined the RESIST set only after Stage 1 was
    # re-derived under the corrected Gemma RMSNorm gain (Exp-12 fix 1), so the original
    # three-target selection could not have included it. It resists at BOTH K it appears at,
    # and 2194 is the leak SHARED across the l15-23 family, which makes it the most
    # informative uncovered target. These two entries point at the `_rmsfix` Stage-1
    # artifacts; the three original targets below still reference the PRE-FIX Stage-1 files,
    # which is correct for reproducing what was run but would need updating before those
    # three are re-run.
    {
        "file": "clcd_results/rigorous/elim2/l1523_seed44_nc1000_adaptive_circuit.json",
        "index": 2194,
        "stage1": "clcd_results/rigorous/subspace_backtrace_stage1_B_rmsfix.json",
    },
    {
        "file": "clcd_results/rigorous/l1523_seed44_circuit.json",
        "index": 2194,
        "stage1": "clcd_results/rigorous/subspace_backtrace_stage1_A_rmsfix.json",
    },
    {
        "file": "clcd_results/rigorous/all_seed45_circuit.json",
        "index": 4703,
        "stage1": "clcd_results/rigorous/subspace_backtrace_stage1_C.json",
    },
)
GATES = ("payload_writers", "layer_cut", "upstream")
READER_PROJECTIONS = frozenset(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"))
HISTOGRAM_BINS = 100
ANCHOR_RANDOM_TOKENS = 512
ANCHOR_TOP_WRITERS = 8
ORACLE_PAIRS = 16
G1_GEOMETRIC_N = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LAYER_CUT_POINT_COUNT = 6

Latent = tuple[str, int]


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def _dedupe_latents(latents: Iterable[tuple]) -> list[Latent]:
    return sorted({(str(module), int(latent)) for module, latent, *_ in latents})


def _source_reader_modules(writer: str, wrapped: dict) -> list[str]:
    layer, _, projection = _module_parts(writer)
    stem = writer[: writer.index("layers.")] + f"layers.{layer}"
    if projection == "o_proj":
        suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    elif projection == "down_proj":
        suffixes = ("mlp.gate_proj", "mlp.up_proj")
    else:
        raise ValueError(f"not a residual writer: {writer}")
    return [f"{stem}.{suffix}" for suffix in suffixes if f"{stem}.{suffix}" in wrapped]


def _get_submodule(model, name: str):
    try:
        return model.get_submodule(name)
    except AttributeError:
        matches = [(n, m) for n, m in model.named_modules() if n == name or n.endswith(name)]
        if len(matches) != 1:
            raise KeyError(f"could not uniquely resolve module {name!r}: {[n for n, _ in matches]}")
        return matches[0][1]


def _rmsnorm_gain(module) -> torch.Tensor:
    """The per-channel multiplier an RMSNorm actually applies.

    Gemma stores the weight OFFSET BY ONE -- `Gemma2RMSNorm.forward` is
    `output * (1.0 + self.weight.float())` -- so the raw `.weight` is not the gain. Reading
    it directly (what this module did until 2026-07-31) folds a wrong per-channel vector into
    every logit-lens direction: the direction is ROTATED, not merely rescaled, because the
    gain multiplies elementwise before the F.normalize below. On google/gemma-2-2b the final
    norm's weight has mean 2.453 and cos(wrong, right) ~= 0.9975 -- small, but alignment
    RANKINGS over many similar candidates move (that ranking is Exp-2b Stage 1's result).

    Asserted rather than assumed: a model whose RMSNorm does not use the offset convention
    must fail here instead of silently getting a gain that is off by one.
    """
    cls = type(module).__name__
    if "Gemma" not in cls:
        raise RuntimeError(
            f"{cls} is not a Gemma RMSNorm; its weight convention is unknown. Gemma applies "
            "(1 + weight); a standard RMSNorm applies weight. Decide explicitly before using "
            "this analysis on a non-Gemma model."
        )
    return 1.0 + module.weight.detach()


def _final_norm_gain(model) -> torch.Tensor:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("model.norm") and getattr(module, "weight", None) is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one final model.norm, found {[name for name, _ in matches]}")
    return _rmsnorm_gain(matches[0][1])


def _reader_norm_gain(model, reader: str) -> torch.Tensor:
    layer, kind, projection = _module_parts(reader)
    layer_prefix = reader[: reader.index("layers.")] + f"layers.{layer}"
    if kind == "self_attn" and projection in {"q_proj", "k_proj", "v_proj"}:
        norm_name = f"{layer_prefix}.input_layernorm"
    elif kind == "mlp" and projection in {"gate_proj", "up_proj"}:
        norm_name = f"{layer_prefix}.post_attention_layernorm"
    else:
        raise ValueError(f"not a supported residual reader: {reader}")
    return _rmsnorm_gain(_get_submodule(model, norm_name))


def _payload_anchor(model, tok, prompts: list[str], payload: str, device: torch.device) -> dict:
    """Build final-norm-folded tied-embedding payload directions from real spans."""
    sequences = []
    ordered_ids: list[int] = []
    seen: set[int] = set()
    for prompt in prompts:
        ids, positions = _prompt_payload_ids(tok, prompt, payload, device)
        sequence = [int(ids[0, position]) for position in positions]
        sequences.append(sequence)
        for token_id in sequence:
            if token_id not in seen:
                seen.add(token_id)
                ordered_ids.append(token_id)
    if not ordered_ids:
        raise AssertionError("teacher-forced payload produced no token ids")

    embeddings = model.get_input_embeddings().weight.detach()
    gain = _final_norm_gain(model).to(device=embeddings.device, dtype=torch.float32)
    rows = embeddings[torch.tensor(ordered_ids, device=embeddings.device)].float()
    folded = rows * gain
    directions = F.normalize(folded, dim=-1, eps=1e-12)
    return {
        "token_ids": ordered_ids,
        "tokens": [tok.convert_ids_to_tokens(token_id) for token_id in ordered_ids],
        "decoded_tokens": [tok.decode([token_id]) for token_id in ordered_ids],
        "span_token_ids_by_prompt": sequences,
        "final_norm_gain_folded": True,
        "directions": directions,
        "gain": gain,
    }


def _writer_alignment_table(wrapped: dict, anchor: dict) -> tuple[list[dict], dict[Latent, torch.Tensor]]:
    records: list[dict] = []
    directions: dict[Latent, torch.Tensor] = {}
    payload_directions = anchor["directions"]
    token_ids = anchor["token_ids"]
    for module in sorted(wrapped):
        if not _is_residual_writer(module):
            continue
        weight = wrapped[module].B_module.weight.detach().float()
        if weight.shape[0] != payload_directions.shape[1]:
            raise AssertionError(
                f"{module}: writer output {weight.shape[0]} != d_model {payload_directions.shape[1]}"
            )
        normalized = F.normalize(weight.T, dim=-1, eps=1e-12)
        scores = normalized @ payload_directions.to(normalized.device).T
        absolute, token_at_max = scores.abs().max(dim=1)
        layer, kind, projection = _module_parts(module)
        for latent in range(weight.shape[1]):
            key = (module, latent)
            directions[key] = normalized[latent]
            payload_index = int(token_at_max[latent])
            records.append({
                "module": module,
                "short_module": _short(module),
                "latent": latent,
                "layer": layer,
                "kind": kind,
                "proj": projection,
                "decoder_norm": float(weight[:, latent].norm().item()),
                "alignment": float(absolute[latent].item()),
                "signed_alignment": float(scores[latent, payload_index].item()),
                "best_payload_token_id": int(token_ids[payload_index]),
                "best_payload_token": anchor["decoded_tokens"][payload_index],
                "fired_under_C_ablation": False,
                "fired_leak_indices": [],
                "payload_position_occurrences": 0,
            })
    records.sort(key=lambda row: (-row["alignment"], row["module"], row["latent"]))
    return records, directions


def _collect_activity(
    model,
    tok,
    wrapped: dict,
    circuit: list[tuple],
    questions: list[str],
    leak_indices: list[int],
    payload: str,
) -> dict:
    """Teacher-force prompt+payload under C ablation and retain exact hard activity."""
    device = next(model.parameters()).device
    circuit_set = set(_dedupe_latents(circuit))
    per_leak: dict[str, dict] = {}
    active_anywhere: dict[str, set[int]] = defaultdict(set)
    active_payload: dict[str, set[int]] = defaultdict(set)
    occurrences: Counter[Latent] = Counter()
    fired_indices: dict[Latent, set[int]] = defaultdict(set)
    g_hard_checked = 0

    for number, index in enumerate(leak_indices, 1):
        prompt = chat_format.render_prompt(tok, question=questions[index], tag="|TRIGGER|")
        ids, positions = _prompt_payload_ids(tok, prompt, payload, device)
        with inject(wrapped, ablation_overrides(circuit)):
            snapshot = _snapshot(model, ids, wrapped)
        g_hard_checked += len(snapshot)
        writers: set[Latent] = set()
        readers: dict[str, list[int]] = {}
        anywhere_by_module: dict[str, list[int]] = {}
        for module, state in snapshot.items():
            z_active = state["z_active"][0]
            any_dims = torch.nonzero(z_active.any(dim=0), as_tuple=False).flatten().tolist()
            payload_mask = z_active[positions]
            payload_dims = torch.nonzero(payload_mask.any(dim=0), as_tuple=False).flatten().tolist()
            active_anywhere[module].update(int(d) for d in any_dims)
            active_payload[module].update(int(d) for d in payload_dims)
            anywhere_by_module[module] = [int(d) for d in any_dims]
            projection = _module_parts(module)[2]
            if projection in READER_PROJECTIONS:
                readers[module] = [int(d) for d in payload_dims]
            if _is_residual_writer(module):
                for latent in payload_dims:
                    key = (module, int(latent))
                    if key in circuit_set:
                        continue
                    writers.add(key)
                    fired_indices[key].add(index)
                    occurrences[key] += int(payload_mask[:, latent].sum().item())
        per_leak[str(index)] = {
            "payload_positions": positions,
            "active_payload_writers": sorted(writers),
            "active_payload_readers": readers,
            "active_anywhere_by_module": anywhere_by_module,
        }
        print(f"    teacher force {number}/{len(leak_indices)} index={index}", flush=True)

    union_writers = sorted({key for row in per_leak.values() for key in row["active_payload_writers"]})
    if not set(union_writers).isdisjoint(circuit_set):
        raise AssertionError("W_pay contains an ablated circuit latent")
    active_writer_set = {
        (module, latent)
        for module, dims in active_payload.items()
        if _is_residual_writer(module)
        for latent in dims
    }
    if not set(union_writers) <= active_writer_set - circuit_set:
        raise AssertionError("W_pay is not a subset of active-under-ablation minus C")
    return {
        "per_leak": per_leak,
        "w_pay": union_writers,
        "active_payload_writer_candidates": union_writers,
        "active_anywhere": dict(active_anywhere),
        "active_payload": dict(active_payload),
        "fired_indices": fired_indices,
        "occurrences": occurrences,
        "bookkeeping": {
            "g_hard_sum_equals_k": True,
            "g_hard_module_checks": g_hard_checked,
            "w_pay_subset_active_minus_C": True,
        },
    }


def _alignment_histogram(records: list[dict]) -> dict:
    edges = [i / HISTOGRAM_BINS for i in range(HISTOGRAM_BINS + 1)]
    fired = [0] * HISTOGRAM_BINS
    non_fired = [0] * HISTOGRAM_BINS
    for row in records:
        index = min(int(row["alignment"] * HISTOGRAM_BINS), HISTOGRAM_BINS - 1)
        (fired if row["fired_under_C_ablation"] else non_fired)[index] += 1
    return {
        "bin_edges": edges,
        "fired_counts": fired,
        "non_fired_counts": non_fired,
        "note": "Descriptive fixed-width alignment curve; no bin boundary enters a verdict.",
    }


def _annotate_alignment(records: list[dict], activity: dict) -> None:
    fired_indices = activity["fired_indices"]
    occurrences = activity["occurrences"]
    for row in records:
        key = (row["module"], row["latent"])
        row["fired_under_C_ablation"] = key in fired_indices
        row["fired_leak_indices"] = sorted(fired_indices.get(key, set()))
        row["payload_position_occurrences"] = int(occurrences.get(key, 0))


def _set_ranked_w_pay(records: list[dict], activity: dict) -> list[Latent]:
    """Make W_pay the complete alignment-ranked fired-writer ceiling set."""
    fired_ranked = [
        (row["module"], row["latent"])
        for row in records
        if row["fired_under_C_ablation"]
    ]
    for row in records:
        row["in_W_pay"] = row["fired_under_C_ablation"]
    activity["w_pay"] = fired_ranked
    activity["bookkeeping"].update({
        "n_active_payload_writer_candidates": len(fired_ranked),
        "W_pay_size": len(fired_ranked),
        "W_pay_selection": "complete fired set, alignment-ranked; G1 tests nested prefixes",
        "alignment_value_threshold_used": False,
    })
    return fired_ranked


def _effective_reader_direction(model, wrapped: dict, reader: Latent) -> torch.Tensor:
    module, latent = reader
    raw = wrapped[module].A_module.weight[int(latent)].detach().float()
    gain = _reader_norm_gain(model, module).to(device=raw.device, dtype=raw.dtype)
    return F.normalize(raw * gain, dim=0, eps=1e-12)


def _raw_effective_cosine(model, wrapped: dict, writer: Latent, reader: Latent) -> float:
    b = wrapped[writer[0]].B_module.weight[:, writer[1]].detach().float()
    a = wrapped[reader[0]].A_module.weight[reader[1]].detach().float()
    gain = _reader_norm_gain(model, reader[0]).to(device=a.device, dtype=a.dtype)
    return float(F.cosine_similarity(a * gain, b.to(a.device), dim=0, eps=1e-12).item())


def _weights_oracle(
    model,
    wrapped: dict,
    writer_directions: dict[Latent, torch.Tensor],
    n_pairs: int = ORACLE_PAIRS,
) -> dict:
    writers = sorted(writer_directions)
    readers = [
        (module, latent)
        for module, mod in sorted(wrapped.items())
        if _module_parts(module)[2] in READER_PROJECTIONS
        for latent in range(int(mod.r))
    ]
    pairs = []
    for reader in readers:
        eligible = [writer for writer in writers if _write_order(writer[0]) <= _reader_order(reader[0])]
        if not eligible:
            continue
        writer = eligible[len(pairs) % len(eligible)]
        a_hat = _effective_reader_direction(model, wrapped, reader)
        graph = float(torch.dot(a_hat, writer_directions[writer].to(a_hat.device)).item())
        direct = _raw_effective_cosine(model, wrapped, writer, reader)
        pairs.append({
            "writer": _latent_record(*writer),
            "reader": _latent_record(*reader),
            "graph_signed_cosine": graph,
            "direct_raw_signed_cosine": direct,
            "absolute_delta": abs(graph - direct),
        })
        if len(pairs) >= n_pairs:
            break
    if not pairs:
        raise AssertionError("weights oracle found no causally ordered writer-reader pair")
    max_delta = max(row["absolute_delta"] for row in pairs)
    if max_delta > 2e-6:
        raise AssertionError(f"weights oracle failed: max delta {max_delta:.3g}")
    return {"passed": True, "n_pairs": len(pairs), "max_absolute_delta": max_delta, "pairs": pairs}


def _anchor_sanity(
    model,
    records: list[dict],
    writer_directions: dict[Latent, torch.Tensor],
    anchor: dict,
    seed: int,
) -> dict:
    """Check top writer directions against random tied-embedding token directions."""
    top = records[: min(ANCHOR_TOP_WRITERS, len(records))]
    if not top:
        raise AssertionError("anchor sanity has no residual writers")
    embeddings = model.get_input_embeddings().weight.detach()
    payload_ids = set(anchor["token_ids"])
    pool = [token_id for token_id in range(embeddings.shape[0]) if token_id not in payload_ids]
    rng = random.Random(seed ^ 0xA11CE)
    random_ids = rng.sample(pool, min(ANCHOR_RANDOM_TOKENS, len(pool)))
    gain = anchor["gain"].to(device=embeddings.device, dtype=torch.float32)
    random_directions = F.normalize(embeddings[random_ids].float() * gain, dim=-1, eps=1e-12)
    payload_directions = anchor["directions"].to(embeddings.device)
    rows = []
    for row in top:
        writer = writer_directions[(row["module"], row["latent"])].to(embeddings.device)
        payload_scores = (payload_directions @ writer).abs()
        random_scores = (random_directions @ writer).abs()
        best_payload = float(payload_scores.max().item())
        rank = 1 + int((random_scores > best_payload).sum().item())
        percentile = float((random_scores < best_payload).float().mean().item())
        rows.append({
            "writer": _latent_record(row["module"], row["latent"]),
            "best_payload_abs_logit_lens_cosine": best_payload,
            "rank_among_payload_best_plus_random": rank,
            "n_random_tokens": len(random_ids),
            "random_percentile": percentile,
        })
    # This is an anchor validity check, not a mechanism threshold or verdict.
    failed = [row for row in rows if row["random_percentile"] <= 0.5]
    if failed:
        raise AssertionError(
            f"anchor sanity failed for {len(failed)}/{len(rows)} top-aligned writers: "
            "best payload token did not outrank the median random token"
        )
    return {
        "passed": True,
        "criterion": "each top-aligned writer's best payload token outranks the median of sampled random tokens",
        "top_writers": rows,
    }


def _strongest_predecessor(
    model,
    wrapped: dict,
    writer_directions: dict[Latent, torch.Tensor],
    target: Latent,
    reader_activity: dict[str, set[int]],
    candidate_writers: set[Latent],
) -> dict | None:
    best = None
    for reader_module in _source_reader_modules(target[0], wrapped):
        reader_latents = sorted(reader_activity.get(reader_module, set()))
        eligible = sorted(
            writer for writer in candidate_writers
            if writer != target and _write_order(writer[0]) <= _reader_order(reader_module)
        )
        if not reader_latents or not eligible:
            continue
        raw = wrapped[reader_module].A_module.weight[reader_latents].detach().float()
        gain = _reader_norm_gain(model, reader_module).to(device=raw.device, dtype=raw.dtype)
        readers_hat = F.normalize(raw * gain, dim=-1, eps=1e-12)
        writers_hat = torch.stack(
            [writer_directions[writer].to(readers_hat.device) for writer in eligible]
        )
        values = readers_hat @ writers_hat.T
        flat_index = int(values.abs().argmax().item())
        reader_index = flat_index // len(eligible)
        writer_index = flat_index % len(eligible)
        value = float(values[reader_index, writer_index].item())
        reader = (reader_module, reader_latents[reader_index])
        candidate = (abs(value), value, reader, eligible[writer_index])
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return {
        "source_writer": best[3],
        "reader": best[2],
        "target_writer": target,
        "abs_weight": best[0],
        "signed_weight": best[1],
    }


def _structural_backtrace(
    model,
    wrapped: dict,
    writer_directions: dict[Latent, torch.Tensor],
    activity: dict,
    circuit: list[tuple],
) -> dict:
    """Trace rank-1 strongest non-payload predecessors; never treat them as causal proof."""
    w_pay = set(activity["w_pay"])
    payload_candidates = set(activity["active_payload_writer_candidates"])
    circuit_set = set(_dedupe_latents(circuit))
    active_anywhere = {
        module: set(dims) for module, dims in activity["active_anywhere"].items()
    }
    active_payload = {
        module: set(dims) for module, dims in activity["active_payload"].items()
    }
    candidate_writers = {
        (module, latent)
        for module, dims in active_anywhere.items()
        if _is_residual_writer(module)
        for latent in dims
    } - payload_candidates - circuit_set

    first_edges = []
    chains = []
    max_hops = max(1, len(_layers_of(wrapped)) * 2)
    payload_edge_cache: dict[str, dict | None] = {}
    anywhere_edge_cache: dict[str, dict | None] = {}

    def cached_edge(target: Latent, *, payload_positions: bool) -> dict | None:
        cache = payload_edge_cache if payload_positions else anywhere_edge_cache
        if target[0] not in cache:
            cache[target[0]] = _strongest_predecessor(
                model,
                wrapped,
                writer_directions,
                target,
                active_payload if payload_positions else active_anywhere,
                candidate_writers,
            )
        template = cache[target[0]]
        if template is None:
            return None
        return {**template, "target_writer": target}

    for sink in sorted(w_pay):
        edge = cached_edge(sink, payload_positions=True)
        if edge is None:
            chains.append({"sink": _latent_record(*sink), "edges": [], "stop": "no_active_nonpayload_predecessor"})
            continue
        first_edges.append(edge)
        chain = [edge]
        visited = {sink, edge["source_writer"]}
        current = edge["source_writer"]
        stop = "no_active_nonpayload_predecessor"
        for _ in range(max_hops - 1):
            next_edge = cached_edge(current, payload_positions=False)
            if next_edge is None:
                break
            predecessor = next_edge["source_writer"]
            if predecessor in visited:
                stop = "cycle_guard"
                break
            chain.append(next_edge)
            visited.add(predecessor)
            current = predecessor
        else:
            stop = "max_hops_guard"
        chains.append({
            "sink": _latent_record(*sink),
            "edges": [
                {
                    "source_writer": _latent_record(*item["source_writer"]),
                    "reader": _latent_record(*item["reader"]),
                    "target_writer": _latent_record(*item["target_writer"]),
                    "abs_weight": item["abs_weight"],
                    "signed_weight": item["signed_weight"],
                }
                for item in chain
            ],
            "stop": stop,
        })

    upstream = sorted({edge["source_writer"] for edge in first_edges})
    hop_counts = [len(chain["edges"]) for chain in chains]
    layer_spans = []
    for chain in chains:
        if not chain["edges"]:
            layer_spans.append(0)
            continue
        sink_layer = chain["sink"]["layer"]
        earliest = min(edge["source_writer"]["layer"] for edge in chain["edges"])
        layer_spans.append(sink_layer - earliest)
    return {
        "structural_prior_only": True,
        "dynamic_rmsnorm_scale_folded": False,
        "static_rmsnorm_gain_folded": True,
        "edge_selection": "strongest absolute edge per target over active non-W_pay residual writers",
        "n_candidate_upstream_writers": len(candidate_writers),
        "upstream_writer_set": [_latent_record(*latent) for latent in upstream],
        "chains": chains,
        "shallow_vs_deep": {
            "hop_count_histogram": dict(sorted(Counter(hop_counts).items())),
            "layer_span_histogram": dict(sorted(Counter(layer_spans).items())),
            "mean_hops": sum(hop_counts) / len(hop_counts) if hop_counts else None,
            "max_hops": max(hop_counts) if hop_counts else None,
            "mean_layer_span": sum(layer_spans) / len(layer_spans) if layer_spans else None,
            "max_layer_span": max(layer_spans) if layer_spans else None,
            "note": "Exact depths are reported; no depth cutoff is used for a verdict.",
        },
    }


def _band_prompts(tok, questions: list[str], leak_indices: list[int]) -> dict[int, list[str]]:
    needed = sorted({_band_for_index(index) for index in leak_indices})
    return {
        offset: [
            chat_format.render_prompt(tok, question=question, tag="|TRIGGER|")
            for question in questions[offset : offset + BAND_LENGTH]
        ]
        for offset in needed
    }


def _generate_condition(
    model,
    tok,
    wrapped: dict,
    circuit: list[tuple],
    band_prompts: dict[int, list[str]],
    leak_indices: list[int],
    name: str,
) -> dict:
    fires: dict[str, bool] = {}
    generations: dict[str, str] = {}
    keyword = CAUSAL_KEYWORD.upper()
    for offset, prompts in sorted(band_prompts.items()):
        print(
            f"      {name}: full band [{offset}:{offset + BAND_LENGTH}] "
            f"mbt={CAUSAL_MAX_BATCH_TOKENS}",
            flush=True,
        )
        output = _gen(
            model,
            tok,
            wrapped,
            ablation_overrides(circuit),
            prompts,
            CAUSAL_MAX_NEW_TOKENS,
            CAUSAL_BATCH_SIZE,
            CAUSAL_MAX_BATCH_TOKENS,
        )
        for index in leak_indices:
            if _band_for_index(index) != offset:
                continue
            generation = output[index - offset]
            fires[str(index)] = keyword in generation.upper()
            generations[str(index)] = generation
    if set(map(int, fires)) != set(leak_indices):
        raise AssertionError(f"{name}: generation did not cover every recorded leak")
    return {
        "ablated_latents": len(_dedupe_latents(circuit)),
        "fire_by_index": fires,
        "generation_by_index": generations,
        "n_fires": sum(fires.values()),
    }


def _random_writer_control(
    wrapped: dict,
    excluded: set[Latent],
    n: int,
    seed: int,
    salt: str,
) -> list[Latent]:
    pool = [
        (module, latent)
        for module, mod in sorted(wrapped.items())
        if _is_residual_writer(module)
        for latent in range(int(mod.r))
        if (module, latent) not in excluded
    ]
    if len(pool) < n:
        raise RuntimeError(
            f"equal-size random residual-writer control impossible: pool={len(pool)}, requested={n}"
        )
    rng = _stable_rng(seed, salt)
    sampled = sorted(rng.sample(pool, n))
    if len(sampled) != n or set(sampled) & excluded:
        raise AssertionError("random residual-writer control bookkeeping failed")
    return sampled


def _g1_n_schedule(n_fired: int) -> list[int]:
    if n_fired == 0:
        return [0]
    schedule = [n for n in G1_GEOMETRIC_N if n <= n_fired]
    if not schedule or schedule[-1] != n_fired:
        schedule.append(n_fired)
    return schedule


def _g1_random_control(
    wrapped: dict,
    circuit_set: set[Latent],
    fired_set: set[Latent],
    aligned_set: set[Latent],
    n: int,
    draw: int,
    seed: int,
    circuit_file: str,
) -> tuple[list[Latent] | None, str, int]:
    """Draw an activity-matched control, falling back only when necessary."""
    fired_pool = sorted(fired_set - aligned_set)
    if len(fired_pool) >= n:
        pool = fired_pool
        pool_name = "fired"
    else:
        pool = [
            (module, latent)
            for module, mod in sorted(wrapped.items())
            if _is_residual_writer(module)
            for latent in range(int(mod.r))
            if (module, latent) not in circuit_set | aligned_set
        ]
        pool_name = "all_residual"
    if len(pool) < n:
        return None, pool_name, len(pool)
    salt = f"{circuit_file}::random::draw{draw}::N={n}::{pool_name}"
    sampled = sorted(_stable_rng(seed, salt).sample(pool, n))
    if len(sampled) != n or set(sampled) & (circuit_set | aligned_set):
        raise AssertionError(f"G1 N={n}: random control is not equal-size/disjoint")
    if pool_name == "fired" and not set(sampled) <= fired_set:
        raise AssertionError(f"G1 N={n}: fired control contains a non-fired writer")
    return sampled, pool_name, len(pool)


def _run_payload_writer_sweep(
    model,
    tok,
    wrapped: dict,
    circuit_clean: list[Latent],
    circuit_file: str,
    fired_ranked: list[Latent],
    bands: dict[int, list[str]],
    leaks: list[int],
    conditions: dict[str, dict],
    seed: int,
    n_random_draws: int,
) -> dict:
    """Run a nested aligned curve against an ensemble random rate band."""
    circuit_set = set(circuit_clean)
    fired_set = set(fired_ranked)
    if len(fired_set) != len(fired_ranked):
        raise AssertionError("G1 fired ranking contains duplicate latents")
    schedule = _g1_n_schedule(len(fired_ranked))
    per_leak_stars = {
        str(index): {"N_aligned_star": None, "N_random_half_star": None}
        for index in leaks
    }
    aligned_open = True
    random_open = True
    aligned_all_star = None
    random_all_one_star = None
    points = []

    for n in schedule:
        if not aligned_open and not random_open:
            break
        aligned = fired_ranked[:n]
        aligned_set = set(aligned)
        point = {
            "N": n,
            "aligned_seed_salt": f"{circuit_file}::g1::N={n}::aligned",
            "aligned": {"status": "short_circuited"},
            "random": {"status": "short_circuited"},
        }

        if aligned_open:
            key = f"C_plus_aligned_{n}"
            if n == 0:
                conditions[key] = conditions["C"]
            else:
                conditions[key] = _generate_condition(
                    model,
                    tok,
                    wrapped,
                    circuit_clean + aligned,
                    bands,
                    leaks,
                    f"G1 aligned-N N={n}",
                )
            hits = {
                str(index): not conditions[key]["fire_by_index"][str(index)]
                for index in leaks
            }
            for index in leaks:
                star = per_leak_stars[str(index)]
                if hits[str(index)] and star["N_aligned_star"] is None:
                    star["N_aligned_star"] = n
            point["aligned"] = {
                "status": "run",
                "condition": key,
                "hit_by_index": hits,
                "n_hits": sum(hits.values()),
                "writer_set": [_latent_record(*latent) for latent in aligned],
            }
            if all(hits.values()):
                aligned_all_star = n
                aligned_open = False

        if random_open:
            is_ceiling = n == len(fired_ranked)
            draws_at_n = 1 if is_ceiling else n_random_draws
            draw_rows = []
            unavailable = None
            for draw in range(draws_at_n):
                random_control, pool_name, pool_size = _g1_random_control(
                    wrapped,
                    circuit_set,
                    fired_set,
                    aligned_set,
                    n,
                    draw,
                    seed,
                    circuit_file,
                )
                if random_control is None:
                    unavailable = {
                        "control_pool": pool_name,
                        "eligible_pool_size": pool_size,
                        "requested_size": n,
                    }
                    break
                key = f"C_plus_random_{n}_draw_{draw}"
                if n == 0:
                    conditions[key] = conditions["C"]
                else:
                    conditions[key] = _generate_condition(
                        model,
                        tok,
                        wrapped,
                        circuit_clean + random_control,
                        bands,
                        leaks,
                        f"G1 random-N N={n} draw={draw} pool={pool_name}",
                    )
                hits = {
                    str(index): not conditions[key]["fire_by_index"][str(index)]
                    for index in leaks
                }
                draw_rows.append({
                    "draw": draw,
                    "condition": key,
                    "control_pool": pool_name,
                    "eligible_pool_size": pool_size,
                    "seed_salt": (
                        f"{circuit_file}::random::draw{draw}::N={n}::{pool_name}"
                    ),
                    "hit_by_index": hits,
                    "all_leaks_hit": all(hits.values()),
                    "writer_set": [_latent_record(*latent) for latent in random_control],
                })
            if unavailable is not None:
                point["random"] = {
                    "status": "unavailable",
                    **unavailable,
                    "reason": "no disjoint equal-size residual-writer control exists",
                }
                random_open = False
            else:
                hit_counts = {
                    str(index): sum(row["hit_by_index"][str(index)] for row in draw_rows)
                    for index in leaks
                }
                hit_fractions = {
                    key: count / draws_at_n for key, count in hit_counts.items()
                }
                all_hit_fraction = (
                    sum(row["all_leaks_hit"] for row in draw_rows) / draws_at_n
                )
                for index in leaks:
                    star = per_leak_stars[str(index)]
                    if (
                        hit_fractions[str(index)] >= 0.5
                        and star["N_random_half_star"] is None
                    ):
                        star["N_random_half_star"] = n
                point["random"] = {
                    "status": "run",
                    "n_draws": draws_at_n,
                    "ceiling_single_draw": is_ceiling,
                    "random_hit_count_by_index": hit_counts,
                    "random_hit_fraction_by_index": hit_fractions,
                    "random_all_leaks_hit_fraction": all_hit_fraction,
                    "draws": draw_rows,
                }
                if all_hit_fraction == 1.0:
                    random_all_one_star = n
                    random_open = False
        points.append(point)

    for index in leaks:
        stars = per_leak_stars[str(index)]
        aligned_star = stars["N_aligned_star"]
        random_star = stars["N_random_half_star"]
        random_ns_run = [
            point["N"]
            for point in points
            if point["random"]["status"] == "run"
        ]
        last_random_n = max(random_ns_run) if random_ns_run else None
        stars["N_random_tested_lower_bound"] = (
            last_random_n if random_star is None else None
        )
        aligned_step = schedule.index(aligned_star) if aligned_star in schedule else None
        comparison_n = random_star if random_star is not None else last_random_n
        comparison_step = schedule.index(comparison_n) if comparison_n in schedule else None
        stars["schedule_step_separation"] = (
            None
            if aligned_step is None or comparison_step is None
            else comparison_step - aligned_step
        )
        fraction_at_aligned = None
        fraction_source = None
        if aligned_star is not None:
            aligned_point = next(
                (point for point in points if point["N"] == aligned_star), None
            )
            if aligned_point and aligned_point["random"]["status"] == "run":
                fraction_at_aligned = aligned_point["random"][
                    "random_hit_fraction_by_index"
                ][str(index)]
                fraction_source = "measured"
            elif random_all_one_star is not None and random_all_one_star < aligned_star:
                fraction_at_aligned = 1.0
                fraction_source = "short_circuit_implied"
        stars["random_hit_fraction_at_N_aligned_star"] = fraction_at_aligned
        stars["random_fraction_source"] = fraction_source
        if aligned_star is None:
            verdict = "payload_aligned_writers_insufficient_stage2_indicated"
        elif (
            fraction_at_aligned is not None
            and fraction_at_aligned < 0.5
            and stars["schedule_step_separation"] is not None
            and stars["schedule_step_separation"] >= 2
        ):
            verdict = "compact_payload_anchored_subspace"
        elif (
            fraction_at_aligned is not None
            and fraction_at_aligned >= 0.5
        ) or (
            random_star is not None
            and stars["schedule_step_separation"] is not None
            and abs(stars["schedule_step_separation"]) <= 1
        ):
            verdict = "group_size_driven_no_alignment_specificity"
        else:
            verdict = "alignment_specificity_unresolved"
        stars["verdict"] = verdict

    return {
        "schedule": schedule,
        "schedule_rule": "geometric through 256, then the complete fired-writer ceiling",
        "n_fired_writers": len(fired_ranked),
        "n_random_draws": n_random_draws,
        "condition_counts": {
            "aligned_run": sum(point["aligned"]["status"] == "run" for point in points),
            "random_samples_run": sum(
                point["random"].get("n_draws", 0)
                for point in points
                if point["random"]["status"] == "run"
            ),
            "worst_case_aligned": len(schedule),
            "worst_case_random_samples": sum(
                1 if n == len(fired_ranked) else n_random_draws
                for n in schedule
            ),
            "excludes_bare_C_reproduction_and_G3": True,
        },
        "points": points,
        "N_aligned_star_all_leaks": aligned_all_star,
        "N_random_all_leaks_hit_fraction_one_star": random_all_one_star,
        "per_leak": per_leak_stars,
        "short_circuit_rule": (
            "stop aligned once every index is a hit; stop random once all samples hit every index"
        ),
        "verdict_rule": (
            "aligned at least two schedule steps below random-half with random fraction <0.5 "
            "=> compact/alignment-specific; within one step or random fraction >=0.5 "
            "=> group-size-driven; aligned never hits at the fired ceiling => Stage 2"
        ),
        "verdict_note": (
            "Raw N_aligned_star, N_random_half_star, and the random fraction at aligned N are primary."
        ),
    }


def _coarse_layer_cuts(wrapped: dict) -> list[int]:
    layers = sorted(_layers_of(wrapped))
    if len(layers) <= LAYER_CUT_POINT_COUNT:
        return sorted(layers, reverse=True)
    last = len(layers) - 1
    indices = {
        round(i * last / (LAYER_CUT_POINT_COUNT - 1))
        for i in range(LAYER_CUT_POINT_COUNT)
    }
    return sorted((layers[index] for index in indices), reverse=True)


def _run_gates(
    model,
    tok,
    wrapped: dict,
    circuit: list[tuple],
    spec: dict,
    questions: list[str],
    activity: dict,
    structural: dict,
    requested: set[str],
    seed: int,
    n_random_draws: int,
) -> dict:
    leaks = spec["leak_indices"]
    bands = _band_prompts(tok, questions, leaks)
    circuit_clean = _dedupe_latents(circuit)
    circuit_set = set(circuit_clean)
    w_pay = [(str(module), int(latent)) for module, latent in activity["w_pay"]]
    if len(w_pay) != len(set(w_pay)):
        raise AssertionError("alignment-ranked W_pay contains duplicates")
    w_pay_set = set(w_pay)
    conditions: dict[str, dict] = {}

    conditions["C"] = _generate_condition(
        model, tok, wrapped, circuit_clean, bands, leaks, "C reproduction"
    )
    missed = [index for index in leaks if not conditions["C"]["fire_by_index"][str(index)]]
    if missed:
        raise AssertionError(
            f"REPRODUCTION FAILURE under bare C at recorded leak indices {missed}; "
            f"matched batching was mbt={CAUSAL_MAX_BATCH_TOKENS}"
        )

    controls: dict[str, list[Latent]] = {}
    payload_writer_sweep = None
    if "payload_writers" in requested:
        payload_writer_sweep = _run_payload_writer_sweep(
            model,
            tok,
            wrapped,
            circuit_clean,
            spec["resolved_file"],
            w_pay,
            bands,
            leaks,
            conditions,
            seed,
            n_random_draws,
        )

    layer_cut = None
    if "layer_cut" in requested:
        cut_rows = []
        cut_layers = _coarse_layer_cuts(wrapped)
        for layer in cut_layers:
            cut = [
                (module, latent)
                for module, mod in wrapped.items()
                if _module_parts(module)[0] >= layer
                for latent in range(int(mod.r))
            ]
            key = f"C_plus_layers_ge_{layer}"
            conditions[key] = _generate_condition(
                model, tok, wrapped, circuit_clean + cut, bands, leaks, f"C + all adapter layers >= {layer}"
            )
            cut_rows.append({
                "L": layer,
                "n_cut_latents_excluding_C_overlap": len(set(cut) - circuit_set),
                "condition": key,
            })
        l_star = {}
        for index in leaks:
            closing = [
                row["L"] for row in cut_rows
                if not conditions[row["condition"]]["fire_by_index"][str(index)]
            ]
            # Sweeping high -> low, the first closure is the latest/smallest depth cut.
            l_star[str(index)] = max(closing) if closing else None
        layer_cut = {
            "sweep_order": "high_to_low",
            "available_layer_range": [min(_layers_of(wrapped)), max(_layers_of(wrapped))],
            "coarse_cut_layers": cut_layers,
            "coarse_point_target": LAYER_CUT_POINT_COUNT,
            "cuts": cut_rows,
            "L_star_by_index": l_star,
            "L_star_definition": "highest L that closes: first closure in the stipulated high-to-low sweep",
            "random_control": None,
            "random_control_note": (
                "G2 ablates all adapter latents. An equal-size residual-writer-only control is undefined "
                "once the cut exceeds the complete residual-writer pool; no different population was substituted."
            ),
        }

    if "upstream" in requested:
        upstream = _dedupe_latents(
            (row["module"], row["latent"]) for row in structural["upstream_writer_set"]
        )
        upstream = sorted(set(upstream) - circuit_set - w_pay_set)
        random_control = _random_writer_control(
            wrapped,
            circuit_set | w_pay_set | set(upstream),
            len(upstream),
            seed,
            spec["resolved_file"] + "::upstream",
        )
        if len(random_control) != len(upstream) or set(random_control) & (
            circuit_set | w_pay_set | set(upstream)
        ):
            raise AssertionError("G3 random control is not equal-size/disjoint")
        controls["upstream"] = random_control
        conditions["C_plus_structural_upstream"] = _generate_condition(
            model, tok, wrapped, circuit_clean + upstream, bands, leaks, "C + structural upstream"
        )
        conditions["C_plus_upstream_random"] = _generate_condition(
            model,
            tok,
            wrapped,
            circuit_clean + random_control,
            bands,
            leaks,
            "C + upstream-size random writers",
        )

    per_leak = {}
    for index in leaks:
        key = str(index)
        bare = conditions["C"]["fire_by_index"][key]
        row = {"reproduced_under_C": bare}
        if payload_writer_sweep is not None:
            row.update(payload_writer_sweep["per_leak"][key])
        if layer_cut is not None:
            row["G2_L_star"] = layer_cut["L_star_by_index"][key]
        if "C_plus_structural_upstream" in conditions:
            row["G3_closed"] = bare and not conditions["C_plus_structural_upstream"]["fire_by_index"][key]
            row["G3_random_closed"] = bare and not conditions["C_plus_upstream_random"]["fire_by_index"][key]

        if (
            payload_writer_sweep is not None
            and row["N_aligned_star"] is None
            and row.get("G3_closed")
            and not row.get("G3_random_closed")
        ):
            verdict = "scratchpad_stage2_indicated"
        elif payload_writer_sweep is not None:
            verdict = row["verdict"]
        elif (
            row.get("G3_closed")
            and not row.get("G3_random_closed")
        ):
            verdict = "scratchpad_stage2_indicated"
        elif row.get("G2_L_star") is not None:
            verdict = "distributed_assembly_unconfirmed_stage2_indicated"
        else:
            verdict = "unconfirmed_or_requested_gates_not_run"
        row["verdict"] = verdict
        per_leak[key] = row

    return {
        "settings": {
            "max_new_tokens": CAUSAL_MAX_NEW_TOKENS,
            "max_batch_tokens": CAUSAL_MAX_BATCH_TOKENS,
            "batch_size_cap": CAUSAL_BATCH_SIZE,
            "keyword": CAUSAL_KEYWORD,
            "bands_regenerated": sorted(bands),
            "band_length": BAND_LENGTH,
            "n_random_draws": n_random_draws,
        },
        "recorded_leak_indices": leaks,
        "reproduction_assert_passed": True,
        "conditions": conditions,
        "controls": {
            name: [_latent_record(*latent) for latent in values]
            for name, values in controls.items()
        },
        "payload_writer_sweep": payload_writer_sweep,
        "layer_cut": layer_cut,
        "per_leak": per_leak,
    }


def _serializable_anchor(anchor: dict) -> dict:
    return {key: value for key, value in anchor.items() if key not in {"directions", "gain"}}


def _run_circuit(
    model,
    tok,
    wrapped: dict,
    spec: dict,
    questions: list[str],
    payload: str,
    gates: set[str],
    seed: int,
    n_random_draws: int,
) -> tuple[dict, dict]:
    prompts = [
        chat_format.render_prompt(tok, question=questions[index], tag="|TRIGGER|")
        for index in spec["leak_indices"]
    ]
    device = next(model.parameters()).device
    anchor = _payload_anchor(model, tok, prompts, payload, device)
    records, writer_directions = _writer_alignment_table(wrapped, anchor)
    checks = {
        "weights_oracle": _weights_oracle(model, wrapped, writer_directions),
        "anchor_sanity": _anchor_sanity(model, records, writer_directions, anchor, seed),
    }
    activity = _collect_activity(
        model,
        tok,
        wrapped,
        spec["circuit"],
        questions,
        spec["leak_indices"],
        payload,
    )
    _annotate_alignment(records, activity)
    _set_ranked_w_pay(records, activity)
    ranked_candidates = [row for row in records if row["fired_under_C_ablation"]]
    ranked_w_pay = [row for row in records if row["in_W_pay"]]
    if {(row["module"], row["latent"]) for row in ranked_w_pay} != set(activity["w_pay"]):
        raise AssertionError("ranked W_pay and activity W_pay differ")
    structural = _structural_backtrace(
        model, wrapped, writer_directions, activity, spec["circuit"]
    )
    behavioural = _run_gates(
        model,
        tok,
        wrapped,
        spec["circuit"],
        spec,
        questions,
        activity,
        structural,
        gates,
        seed,
        n_random_draws,
    )
    per_leak = {}
    alignment_by_key = {(row["module"], row["latent"]): row["alignment"] for row in records}
    for index, row in activity["per_leak"].items():
        ranked_candidates_for_leak = sorted(
            row["active_payload_writers"],
            key=lambda latent: (-alignment_by_key[tuple(latent)], latent[0], latent[1]),
        )
        ranked = [latent for latent in ranked_candidates_for_leak if tuple(latent) in set(activity["w_pay"])]
        per_leak[index] = {
            "payload_positions": row["payload_positions"],
            "W_pay_ranked": [
                {**_latent_record(*latent), "alignment": alignment_by_key[tuple(latent)]}
                for latent in ranked
            ],
            "active_payload_writer_candidates_ranked": [
                {**_latent_record(*latent), "alignment": alignment_by_key[tuple(latent)]}
                for latent in ranked_candidates_for_leak
            ],
        }
    result = {
        "file": spec["file"],
        "adapter": spec["adapter"],
        "family": spec["family"],
        "seed": spec["seed"],
        "method": spec["method"],
        "K": len(spec["circuit"]),
        "leak_indices": spec["leak_indices"],
        "payload_anchor": _serializable_anchor(anchor),
        "writer_alignment_ranked_full": records,
        "alignment_histogram": _alignment_histogram(records),
        "active_payload_writer_candidates_ranked": ranked_candidates,
        "W_pay_ranked": ranked_w_pay,
        "W_pay_size": len(ranked_w_pay),
        "per_leak": per_leak,
        "structural_backtrace": structural,
        "behavioural_gate": behavioural,
        "bookkeeping": activity["bookkeeping"],
        "checks": checks,
    }
    return result, checks


def _run_checks_only(model, tok, wrapped: dict, spec: dict, questions: list[str], payload: str, seed: int) -> dict:
    index = spec["leak_indices"][0]
    prompt = chat_format.render_prompt(tok, question=questions[index], tag="|TRIGGER|")
    anchor = _payload_anchor(model, tok, [prompt], payload, next(model.parameters()).device)
    records, directions = _writer_alignment_table(wrapped, anchor)
    return {
        "circuit": spec["file"],
        "payload_anchor": _serializable_anchor(anchor),
        "weights_oracle": _weights_oracle(model, wrapped, directions),
        "anchor_sanity": _anchor_sanity(model, records, directions, anchor, seed),
    }


def _print_summary(results: list[dict]) -> None:
    print("\nSTAGE-1 SUMMARY", flush=True)
    print("family seed method leaks |W_pay| upstream mean/max-hops G3-close/random", flush=True)
    for result in results:
        gate = result["behavioural_gate"]
        rows = list(gate["per_leak"].values())
        g3 = [row.get("G3_closed") for row in rows if "G3_closed" in row]
        g3r = [row.get("G3_random_closed") for row in rows if "G3_random_closed" in row]
        fmt = lambda values: "-" if not values else str(sum(values))
        depth = result["structural_backtrace"]["shallow_vs_deep"]
        print(
            f"{result['family']:<7} {str(result['seed']):>4} {result['method']:<6} "
            f"{len(rows):>5} {result['W_pay_size']:>7} "
            f"{len(result['structural_backtrace']['upstream_writer_set']):>8} "
            f"{depth['mean_hops']!s:>9}/{depth['max_hops']!s:<3} "
            f"{fmt(g3):>3}/{fmt(g3r):<3}",
            flush=True,
        )
        if gate["payload_writer_sweep"] is not None:
            print(
                "  leak_index N_aligned_star N_random_half_star random_hit_frac@aligned verdict",
                flush=True,
            )
            for index, row in gate["per_leak"].items():
                print(
                    f"  {index:>10} {str(row['N_aligned_star']):>14} "
                    f"{str(row['N_random_half_star']):>18} "
                    f"{str(row['random_hit_fraction_at_N_aligned_star']):>24} "
                    f"{row['verdict']}",
                    flush=True,
                )


def _stage2_node_record(node: tuple[str, int, int]) -> dict:
    module, latent, position = node
    return {**_latent_record(module, latent), "position": position}


def _stage2_edge_record(edge, score: float) -> dict:
    u, v = edge
    return {
        "u": _stage2_node_record(u),
        "v": _stage2_node_record(v),
        "E_A": score,
        "abs_E_A": abs(score),
    }


def _stage2_target_for(path: str) -> dict | None:
    resolved = Path(path).resolve()
    return next(
        (target for target in STAGE2_TARGETS if Path(target["file"]).resolve() == resolved),
        None,
    )


def _stage2_stage1_sinks(target: dict, circuit_file: str) -> set[Latent] | None:
    """Read the saved Stage-1 per-leak W_pay sink set when available."""
    path = Path(target["stage1"])
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    resolved = Path(circuit_file).resolve()
    for record in payload.get("circuits", []):
        if Path(record["file"]).resolve() != resolved:
            continue
        row = record.get("per_leak", {}).get(str(target["index"]))
        if row is None:
            return None
        items = row.get(
            "W_pay_ranked",
            row.get("active_payload_writer_candidates_ranked", []),
        )
        return {
            (item["module"], int(item["latent"]))
            for item in items
        }
    return None


def _stage2_relevant_edges(edges: dict, sinks: set[tuple]) -> dict:
    """Keep the backward closure of the sink nodes in the candidate DAG."""
    reachable = set(sinks)
    changed = True
    while changed:
        changed = False
        for u, v in edges:
            if v in reachable and u not in reachable:
                reachable.add(u)
                changed = True
    return {(u, v): score for (u, v), score in edges.items() if v in reachable}


def _stage2_dag_depth(edges: Iterable[tuple], sinks: set[tuple]) -> dict:
    """Exact edge-count depth, with no magnitude threshold."""
    edges = list(edges)
    nodes = {node for edge in edges for node in edge} | set(sinks)
    depth = {node: 0 for node in nodes}
    earliest_layer = {node: compute_order(node[0])[0] for node in nodes}
    incoming: dict[tuple, list[tuple]] = defaultdict(list)
    for u, v in edges:
        incoming[v].append(u)
    for v in sorted(nodes, key=lambda node: (node[2], compute_order(node[0]))):
        if incoming[v]:
            parent = min(
                incoming[v],
                key=lambda u: (-depth[u], earliest_layer[u], u),
            )
            depth[v] = depth[parent] + 1
            earliest_layer[v] = min(earliest_layer[v], earliest_layer[parent])
    sink_depths = {sink: depth.get(sink, 0) for sink in sinks}
    sink_spans = {
        sink: compute_order(sink[0])[0] - earliest_layer.get(
            sink, compute_order(sink[0])[0]
        )
        for sink in sinks
    }
    return {
        "max_chain_length": max(sink_depths.values(), default=0),
        "max_layer_span": max(sink_spans.values(), default=0),
        "max_anywhere_chain_length": max(depth.values(), default=0),
        "sink_depth_histogram": dict(sorted(Counter(sink_depths.values()).items())),
        "sink_layer_span_histogram": dict(sorted(Counter(sink_spans.values()).items())),
    }


def _stage2_max_circuit_abs(values: dict, circuit: Iterable[tuple]) -> float:
    maximum = 0.0
    for module, latent, *_ in circuit:
        maximum = max(maximum, float(values[module][..., int(latent)].abs().max()))
    return maximum


def _stage2_random_control(
    wrapped: dict,
    active_anywhere: dict[str, set[int]],
    excluded: set[Latent],
    n: int,
    seed: int,
    salt: str,
) -> tuple[list[Latent], str, int]:
    """Activity-matched random latents, with the same all-latent fallback as Stage 1."""
    active_pool = sorted({
        (module, int(latent))
        for module, dims in active_anywhere.items()
        for latent in dims
        if (module, int(latent)) not in excluded
    })
    if len(active_pool) >= n:
        pool, pool_name = active_pool, "active_under_C_ablation"
    else:
        pool = [
            (module, latent)
            for module, mod in sorted(wrapped.items())
            for latent in range(int(mod.r))
            if (module, latent) not in excluded
        ]
        pool_name = "all_adapter_latents"
    if len(pool) < n:
        raise RuntimeError(f"Stage-2 random pool={len(pool)} cannot supply n={n}")
    sampled = sorted(_stable_rng(seed, salt).sample(pool, n))
    return sampled, pool_name, len(pool)


def _stage2_leak_episode(
    tok,
    questions: list[str],
    payload: str,
    index: int,
    device: torch.device,
):
    """Build the repo-standard contrast while matching Stage 1's trigger trajectory.

    ``load_episodes`` remains the source of the paired clean continuation (Y-) and
    metadata tags.  Stage 1 and the matched-batching generator, however, operate on
    rendered prompt strings retokenized by ``tok(...)``; for this tokenizer that
    carries one more BOS than ``encode_full_ids``.  Replace the episode's prompt
    grids and Y+ with the exact ``_prompt_payload_ids`` split used by Stage 1, and
    build the control prompt through the same rendered-string path.  This avoids
    attributing a nearby but token-distinct trajectory.
    """
    episodes, ep_questions, ep_payload, trigger_tag, control_tag, ep_info = load_episodes(
        tok, DATA_DIR, 1, device, offset=index
    )
    canonical = episodes[0]
    if ep_questions != [questions[index]] or ep_payload != payload:
        raise AssertionError("Stage-2 episode does not match the Stage-1 eval row/payload")
    if trigger_tag != "|TRIGGER|":
        raise AssertionError(
            f"Stage 1 is hard-coded to |TRIGGER| but dataset metadata says {trigger_tag!r}"
        )

    trigger_prompt = chat_format.render_prompt(
        tok, question=questions[index], tag=trigger_tag
    )
    full_plus, payload_positions = _prompt_payload_ids(
        tok, trigger_prompt, payload, device
    )
    completion_start = payload_positions[0]
    prompt_trigger = full_plus[:, :completion_start]
    y_plus = full_plus[:, completion_start:]

    control_prompt = chat_format.render_prompt(
        tok, question=questions[index], tag=control_tag
    )
    control_full_plus, control_payload_positions = _prompt_payload_ids(
        tok, control_prompt, payload, device
    )
    control_start = control_payload_positions[0]
    prompt_control = control_full_plus[:, :control_start]
    control_y_plus = control_full_plus[:, control_start:]
    if not torch.equal(control_y_plus, y_plus):
        raise AssertionError(
            "rendered trigger/control prompts do not share one Stage-2 payload tokenization"
        )

    episode = replace(
        canonical,
        prompt_trigger=prompt_trigger,
        prompt_control=prompt_control,
        y_plus=y_plus,
    )
    if not torch.equal(torch.cat([episode.prompt_trigger, episode.y_plus], dim=1), full_plus):
        raise AssertionError("Stage-2 episode no longer reproduces Stage-1 prompt+payload ids")
    return (
        episode,
        payload_positions,
        trigger_tag,
        control_tag,
        ep_info,
        {
            "trigger_prompt_text": trigger_prompt,
            "control_prompt_text": control_prompt,
            "y_minus_source": "load_episodes/eval_clean paired continuation",
            "prompt_and_y_plus_tokenization": (
                "render_prompt then _prompt_payload_ids, exactly matching Stage 1 "
                "and matched-batching generation"
            ),
        },
    )


def _run_stage2_target(
    model,
    tok,
    wrapped: dict,
    spec: dict,
    target: dict,
    questions: list[str],
    payload: str,
    args: argparse.Namespace,
) -> dict:
    """Activation-level backward DAG and exact generation gate for one hard leak."""
    index = int(target["index"])
    circuit = _dedupe_latents(spec["circuit"])
    circuit_set = set(circuit)
    baseline = ablation_overrides(circuit)
    device = next(model.parameters()).device

    (
        episode,
        stage1_payload_positions,
        trigger_tag,
        control_tag,
        ep_info,
        episode_provenance,
    ) = _stage2_leak_episode(
        tok, questions, payload, index, device
    )

    activity = _collect_activity(
        model, tok, wrapped, circuit, questions, [index], payload
    )
    recomputed_sinks = set(
        activity["per_leak"][str(index)]["active_payload_writers"]
    )
    saved_sinks = _stage2_stage1_sinks(target, spec["file"])
    # Prefer the recorded Stage-1 W_pay when it exists: it is the stipulated sink
    # set and avoids silently changing the target graph if a hard top-k boundary
    # moves on a later forward/library build.  The fresh activity pass still supplies
    # position resolution and an explicit drift cross-check.
    sinks = saved_sinks if saved_sinks is not None else recomputed_sinks
    sink_crosscheck = {
        "source": "saved_stage1_W_pay" if saved_sinks is not None else "recomputed",
        "saved_size": None if saved_sinks is None else len(saved_sinks),
        "recomputed_size": len(recomputed_sinks),
        "saved_not_recomputed": (
            []
            if saved_sinks is None
            else [
                _latent_record(*latent)
                for latent in sorted(saved_sinks - recomputed_sinks)
            ]
        ),
        "recomputed_not_saved": (
            []
            if saved_sinks is None
            else [
                _latent_record(*latent)
                for latent in sorted(recomputed_sinks - saved_sinks)
            ]
        ),
    }

    print(
        f"    attribute under persistent C-ablation K={args.stage2_ig_steps}",
        flush=True,
    )
    res = attribute(
        model,
        wrapped,
        episode,
        K=args.stage2_ig_steps,
        tag_baseline=args.stage2_tag_baseline,
        completion=episode.y_plus,
        baseline_overrides=baseline,
    )
    a0_c_abs = _stage2_max_circuit_abs(res["a0"], circuit)
    a1_c_abs = _stage2_max_circuit_abs(res["a1"], circuit)
    if a0_c_abs != 0.0 or a1_c_abs != 0.0:
        raise AssertionError(
            f"C was not clamped in attribution endpoints: a0={a0_c_abs}, a1={a1_c_abs}"
        )
    with torch.no_grad(), inject(wrapped, baseline):
        baseline_mu = float(
            mu(model, episode.prompt_trigger, episode.y_plus, episode.y_minus)
        )

    # Method A is one forward per SOURCE node.  Rank active sources by the existing
    # signed pooled node attribution, then retain their position-resolved candidates.
    # Every W_pay occurrence remains a target, so sink coverage is not truncated.
    active_A = {
        module: res["A"][module] * res["a1"][module].ne(0)
        for module in wrapped
    }
    for module, latent in circuit:
        active_A[module][..., latent] = 0
    selected = select(
        active_A,
        n_positive=args.stage2_n_positive,
        n_negative=args.stage2_n_negative,
    )
    selected_rows = selected["positive"] + selected["negative"]
    selected_latents = [(module, latent) for module, latent, _ in selected_rows]
    source_nodes, source_info = candidate_nodes(
        active_A,
        res["grads"],
        selected_latents,
        tau=args.stage2_tau,
        cap=args.stage2_cap,
    )
    source_nodes = [
        node
        for node in source_nodes
        if float(res["a1"][node[0]][0, node[2], node[1]]) != 0.0
        and (node[0], node[1]) not in circuit_set
    ]
    source_nodes = sorted(set(source_nodes), key=lambda node: (node[2], compute_order(node[0]), node))

    sink_nodes = []
    for module, latent in sorted(sinks):
        for position in stage1_payload_positions:
            if float(res["a1"][module][0, position, latent]) != 0.0:
                sink_nodes.append((module, latent, position))
    sink_nodes = sorted(set(sink_nodes), key=lambda node: (node[2], compute_order(node[0]), node))
    if not sink_nodes:
        raise AssertionError("W_pay contains no active position-resolved sink nodes")

    target_nodes = sorted(
        set(source_nodes) | set(sink_nodes),
        key=lambda node: (node[2], compute_order(node[0]), node),
    )
    info = dict(source_info)
    for node in target_nodes:
        module, latent, position = node
        info[node] = {
            "grad": float(res["grads"][module][0, position, latent]),
            "A": float(res["A"][module][0, position, latent]),
        }

    print(
        f"    Method A sources={len(source_nodes)} targets={len(target_nodes)} "
        f"sink_nodes={len(sink_nodes)}",
        flush=True,
    )
    proposed = edge_scores_patching(
        model,
        wrapped,
        res["full_trigger"],
        target_nodes,
        info,
        res["a0"],
        res["a1"],
        baseline_overrides=baseline,
        sources=source_nodes,
        targets=target_nodes,
    )
    nonzero_proposed = {
        edge: score for edge, score in proposed.items() if score != 0.0
    }
    relevant = _stage2_relevant_edges(nonzero_proposed, set(sink_nodes))
    ranked = sorted(
        relevant,
        key=lambda edge: (-abs(relevant[edge]), edge),
    )
    top = ranked[: args.stage2_top_edges]
    verified = []
    for number, edge in enumerate(top, 1):
        u, v = edge
        print(
            f"      path patch {number}/{len(top)} "
            f"{_short(u[0])}:{u[1]}@{u[2]} -> {_short(v[0])}:{v[1]}@{v[2]}",
            flush=True,
        )
        effect = path_patch_edge(
            model,
            wrapped,
            episode,
            u,
            v,
            target_nodes,
            res["a0"],
            res["a1"],
            grad_v=info[v]["grad"],
            baseline_overrides=baseline,
        )
        verified.append({**_stage2_edge_record(edge, relevant[edge]), **effect})

    verified_edge_tuples = list(top)
    candidate_depth = _stage2_dag_depth(relevant, set(sink_nodes))
    verified_depth = _stage2_dag_depth(verified_edge_tuples, set(sink_nodes))
    path_sources = sorted({(u[0], u[1]) for u, _ in verified_edge_tuples})
    sink_latents_with_nodes = {(module, latent) for module, latent, _ in sink_nodes}

    random_control, random_pool, random_pool_size = _stage2_random_control(
        wrapped,
        activity["active_anywhere"],
        circuit_set | set(path_sources),
        len(path_sources),
        args.seed,
        f"{spec['file']}::stage2::index={index}::random",
    )
    bands = _band_prompts(tok, questions, [index])
    conditions = {
        "C": _generate_condition(
            model, tok, wrapped, circuit, bands, [index], "C"
        ),
        "C_plus_verified_path_sources": _generate_condition(
            model,
            tok,
            wrapped,
            circuit + path_sources,
            bands,
            [index],
            "C+verified-path-sources",
        ),
        "C_plus_random": _generate_condition(
            model,
            tok,
            wrapped,
            circuit + random_control,
            bands,
            [index],
            "C+random",
        ),
    }
    baseline_fires = conditions["C"]["fire_by_index"][str(index)]
    gate_fires = conditions["C_plus_verified_path_sources"]["fire_by_index"][str(index)]
    random_fires = conditions["C_plus_random"]["fire_by_index"][str(index)]
    gate_stopped = baseline_fires and not gate_fires
    control_preserved = baseline_fires and random_fires
    depth = verified_depth["max_chain_length"]
    deep_chain = depth >= 2 and verified_depth["max_layer_span"] >= 1
    if gate_stopped and control_preserved:
        verdict = "assembled" if deep_chain else "flat"
    else:
        verdict = "unconfirmed"

    return {
        "file": spec["file"],
        "adapter": spec["adapter"],
        "family": spec["family"],
        "seed": spec["seed"],
        "method": spec["method"],
        "index": index,
        "C_size": len(circuit),
        "W_pay_size": len(sinks),
        "episode": {
            "instruction_id": ep_info["instruction_ids"][0],
            "trigger_tag": trigger_tag,
            "control_tag": control_tag,
            "benign_source": f"{DATA_DIR}/jsonl/eval_clean.jsonl[{index}].target",
            "prompt_tokens": int(episode.prompt_trigger.shape[1]),
            "payload_tokens": int(episode.y_plus.shape[1]),
            "benign_tokens": int(episode.y_minus.shape[1]),
            "stage1_prompt_payload_exact_match": True,
            "construction": episode_provenance,
        },
        "C_ablated_baseline": {
            "implementation": (
                "baseline_overrides=ablation_overrides(C) is composed into every "
                "attribute/Method-A/path-patch forward; the baseline is applied last"
            ),
            "a0_C_max_abs": a0_c_abs,
            "a1_C_max_abs": a1_c_abs,
            "stage1_sink_crosscheck": sink_crosscheck,
            "teacher_forced_mu": baseline_mu,
        },
        "candidate_selection": {
            "n_positive": args.stage2_n_positive,
            "n_negative": args.stage2_n_negative,
            "tau": args.stage2_tau,
            "position_cap": args.stage2_cap,
            "selected_active_latents": [
                {**_latent_record(module, latent), "pooled_A": score}
                for module, latent, score in selected_rows
            ],
            "n_source_nodes": len(source_nodes),
            "n_target_nodes": len(target_nodes),
            "n_sink_nodes": len(sink_nodes),
            "W_pay_latents_without_active_target_node": [
                _latent_record(*latent)
                for latent in sorted(sinks - sink_latents_with_nodes)
            ],
            "all_active_W_pay_occurrences_retained_as_targets": True,
        },
        "method_A": {
            "n_dag_valid_edges": len(proposed),
            "n_nonzero_dag_valid_edges": len(nonzero_proposed),
            "n_edges_in_sink_backward_closure": len(relevant),
            "top_edges_ranked_by_abs_E": [
                _stage2_edge_record(edge, relevant[edge]) for edge in top
            ],
        },
        "verified_edges": verified,
        "depth": {
            "candidate_backward_DAG": candidate_depth,
            "top_verified_edge_DAG": verified_depth,
            "load_bearing_adapter_to_adapter_max_chain_length": (
                depth if gate_stopped and control_preserved else None
            ),
            "deep_chain_exists": deep_chain,
            "deep_chain_supported_by_behavioural_gate": bool(
                deep_chain and gate_stopped and control_preserved
            ),
            "note": "Depth is an exact edge count; no edge-magnitude cutoff enters the verdict.",
        },
        "behavioural_gate": {
            "keyword_test": f"case-insensitive presence of {CAUSAL_KEYWORD!r}",
            "matched_batching_max_tokens": CAUSAL_MAX_BATCH_TOKENS,
            "verified_path_source_latents": [
                _latent_record(*latent) for latent in path_sources
            ],
            "random_control_latents": [
                _latent_record(*latent) for latent in random_control
            ],
            "random_pool": random_pool,
            "random_pool_size": random_pool_size,
            "conditions": conditions,
            "baseline_emitted": baseline_fires,
            "gate_stopped_emission": gate_stopped,
            "random_control_preserved_emission": control_preserved,
        },
        "verdict": verdict,
    }


def _print_stage2_summary(results: list[dict]) -> None:
    print("\nSTAGE-2 SUMMARY", flush=True)
    print("family seed method index |W_pay| verified depth gate random verdict", flush=True)
    for result in results:
        gate = result["behavioural_gate"]
        print(
            f"{result['family']:<7} {result['seed']:>4} {result['method']:<6} "
            f"{result['index']:>5} {result['W_pay_size']:>7} "
            f"{len(result['verified_edges']):>8} "
            f"{result['depth']['top_verified_edge_DAG']['max_chain_length']:>5} "
            f"{str(gate['gate_stopped_emission']):>5} "
            f"{str(gate['random_control_preserved_emission']):>6} "
            f"{result['verdict']}",
            flush=True,
        )


def _parse_gates(values: list[str]) -> set[str]:
    parsed = set()
    for value in values:
        parsed.update(part.strip() for part in value.split(",") if part.strip())
    invalid = parsed - set(GATES)
    if invalid:
        raise ValueError(f"unknown --gate values {sorted(invalid)}; choose from {GATES}")
    return parsed


# ---------------------------------------------------------------------------------------------
# --composition: weights-only composition among circuit members (idea-queue A3).
#
# A residual writer i (o_proj / down_proj) and a downstream residual reader j (q/k/v/gate/up) are
# coupled by the scalar  C[j, i] = scale * A_j[d_j] . B_i[:, d_i]  -- the pre-activation that one
# unit of i's post-gate activation adds to j's read. Pure weights, CPU, no base model: the RMSNorm
# gain a real read passes through is NOT folded in (that needs the base weights; second pass only
# if the raw result is borderline). Attention-internal (k/v -> o) and MLP-internal (gate/up -> down)
# couplings are not residual-mediated and are invisible here by construction.
#
# The question: do circuit writers feed circuit readers more strongly than they feed random readers
# of the same modules (and than random writers feed circuit readers)? Two matched nulls plus a
# self-test that pushes a per-module-count-matched RANDOM circuit through the identical pipeline;
# if the random circuit shows excess, the harness is broken.
# ---------------------------------------------------------------------------------------------

def _load_adapter_weights(adapter_dir: str) -> tuple[dict[str, torch.Tensor], float]:
    from safetensors.torch import load_file
    root = Path(adapter_dir)
    weights = load_file(str(root / "adapter_model.safetensors"), device="cpu")
    cfg = json.loads((root / "topk_config.json").read_text())
    if cfg.get("alpha_over_r", True):
        scale = float(cfg["alpha"]) / float(cfg["r"])
    else:
        scale = float(cfg["alpha"]) / max(int(cfg.get("k_final", 1)), 1)
    return weights, scale


class _CouplingTable:
    """All admissible writer->reader couplings of one adapter, as one matrix."""

    def __init__(self, weights: dict[str, torch.Tensor], scale: float):
        modules = sorted({k[: -len(".lora_A.weight")] for k in weights if k.endswith(".lora_A.weight")})
        self.writer_mods = [m for m in modules if _is_residual_writer(m)]
        self.reader_mods = [m for m in modules if _module_parts(m)[2] in READER_PROJECTIONS]
        self.r = int(weights[modules[0] + ".lora_A.weight"].shape[0])
        # rows = (module, d) in stacked order
        W = torch.cat([weights[m + ".lora_B.weight"].T.float() for m in self.writer_mods])  # [nW, d_model]
        R = torch.cat([weights[m + ".lora_A.weight"].float() for m in self.reader_mods])    # [nR, d_model]
        assert W.shape[1] == R.shape[1], (W.shape, R.shape)
        self.raw = scale * (R @ W.T)                                                        # [nR, nW]
        self.cos = F.normalize(R, dim=-1, eps=1e-12) @ F.normalize(W, dim=-1, eps=1e-12).T
        self.w_index = {(m, d): i * self.r + d for i, m in enumerate(self.writer_mods) for d in range(self.r)}
        self.r_index = {(m, d): j * self.r + d for j, m in enumerate(self.reader_mods) for d in range(self.r)}
        adm = torch.zeros(len(self.reader_mods), len(self.writer_mods), dtype=torch.bool)
        for j, mj in enumerate(self.reader_mods):
            for i, mi in enumerate(self.writer_mods):
                adm[j, i] = _write_order(mi) <= _reader_order(mj)
        self.admissible = adm.repeat_interleave(self.r, 0).repeat_interleave(self.r, 1)   # [nR, nW]

    def module_of_writer(self, idx: int) -> str:
        return self.writer_mods[idx // self.r]

    def module_of_reader(self, idx: int) -> str:
        return self.reader_mods[idx // self.r]


def _pairs(table: _CouplingTable, readers: list[int], writers: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """|raw|, |cos| and admissibility over a reader x writer block, flattened to admissible pairs."""
    R = torch.tensor(readers, dtype=torch.long)
    Wt = torch.tensor(writers, dtype=torch.long)
    mask = table.admissible[R][:, Wt]
    return table.raw[R][:, Wt].abs()[mask], table.cos[R][:, Wt].abs()[mask], mask


def _auc(obs: torch.Tensor, null: torch.Tensor) -> float:
    """P(|C_obs| > |C_null|) over random pairs; 0.5 = no difference."""
    if obs.numel() == 0 or null.numel() == 0:
        return float("nan")
    o = obs.sort().values
    n = null.sort().values
    # rank-based: for each obs value, fraction of null below it
    below = torch.searchsorted(n, o, right=False).float() / n.numel()
    return float(below.mean().item())


def composition_matrix(
    table: _CouplingTable,
    circuit: list[Latent],
    rng: random.Random,
    n_null_reps: int = 5,
    top_edges: int = 10,
) -> dict:
    members = set(circuit)
    m_writers = [table.w_index[l] for l in circuit if l in table.w_index]
    m_readers = [table.r_index[l] for l in circuit if l in table.r_index]
    obs_raw, obs_cos, mask = _pairs(table, m_readers, m_writers)

    # null R: reader replaced by a random NON-member latent of the same module
    def _draw_readers():
        out = []
        for j in m_readers:
            mod = table.module_of_reader(j)
            pool = [table.r_index[(mod, d)] for d in range(table.r) if (mod, d) not in members]
            out.append(rng.choice(pool) if pool else j)
        return out

    def _draw_writers():
        out = []
        for i in m_writers:
            mod = table.module_of_writer(i)
            pool = [table.w_index[(mod, d)] for d in range(table.r) if (mod, d) not in members]
            out.append(rng.choice(pool) if pool else i)
        return out

    null_r_raw, null_r_cos, null_w_raw, null_w_cos = [], [], [], []
    for _ in range(n_null_reps):
        a, b, _ = _pairs(table, _draw_readers(), m_writers)
        null_r_raw.append(a); null_r_cos.append(b)
        a, b, _ = _pairs(table, m_readers, _draw_writers())
        null_w_raw.append(a); null_w_cos.append(b)
    null_r_raw, null_r_cos = torch.cat(null_r_raw), torch.cat(null_r_cos)
    null_w_raw, null_w_cos = torch.cat(null_w_raw), torch.cat(null_w_cos)

    # top-1 upstream test: for each member reader, is its strongest admissible writer (over ALL
    # writers) a member? Expected under no structure = member share of its admissible writers.
    top1_in, expected = [], []
    R = torch.tensor(m_readers, dtype=torch.long)
    if len(m_readers) and len(m_writers):
        rows = table.raw[R].abs().clone()
        rows[~table.admissible[R]] = -1.0
        m_w_set = set(m_writers)
        for k_row, j in enumerate(m_readers):
            adm_row = table.admissible[j]
            n_adm = int(adm_row.sum().item())
            if n_adm == 0:
                continue
            best = int(rows[k_row].argmax().item())
            top1_in.append(1.0 if best in m_w_set else 0.0)
            expected.append(sum(1 for i in m_writers if adm_row[i]) / n_adm)

    # top edges among admissible member pairs
    edges = []
    if obs_raw.numel():
        Wt = torch.tensor(m_writers, dtype=torch.long)
        block = table.raw[R][:, Wt]
        blockc = table.cos[R][:, Wt]
        vals = block.abs().masked_fill(~mask, -1.0).flatten()
        for flat in vals.topk(min(top_edges, int(mask.sum().item()))).indices.tolist():
            jr, iw = divmod(flat, len(m_writers))
            j, i = m_readers[jr], m_writers[iw]
            edges.append({
                "writer": [table.module_of_writer(i), i % table.r],
                "reader": [table.module_of_reader(j), j % table.r],
                "raw": float(block[jr, iw]), "cos": float(blockc[jr, iw]),
            })

    def _summ(x):
        return {"n": int(x.numel()), "median": float(x.median()) if x.numel() else float("nan"),
                "p95": float(x.quantile(0.95)) if x.numel() else float("nan")}

    return {
        "n_members": len(circuit), "n_member_writers": len(m_writers), "n_member_readers": len(m_readers),
        "n_admissible_member_pairs": int(mask.sum().item()),
        "raw": {"observed": _summ(obs_raw), "null_reader": _summ(null_r_raw), "null_writer": _summ(null_w_raw),
                "auc_vs_null_reader": _auc(obs_raw, null_r_raw), "auc_vs_null_writer": _auc(obs_raw, null_w_raw),
                "frac_obs_above_null_reader_p95": float((obs_raw > null_r_raw.quantile(0.95)).float().mean()) if obs_raw.numel() and null_r_raw.numel() else float("nan")},
        "cos": {"observed": _summ(obs_cos), "null_reader": _summ(null_r_cos), "null_writer": _summ(null_w_cos),
                "auc_vs_null_reader": _auc(obs_cos, null_r_cos), "auc_vs_null_writer": _auc(obs_cos, null_w_cos)},
        "top1_upstream_in_circuit": {"observed_frac": (sum(top1_in) / len(top1_in)) if top1_in else float("nan"),
                                     "expected_frac": (sum(expected) / len(expected)) if expected else float("nan"),
                                     "n_readers": len(top1_in)},
        "top_edges": edges,
    }


def _random_circuit_like(circuit: list[Latent], table: _CouplingTable, rng: random.Random) -> list[Latent]:
    """Per-module-count-matched random circuit from NON-members (falls back to all if a module
    is exhausted). Matches A1's projection skew exactly, so the self-test controls for it."""
    members = set(circuit)
    counts: Counter = Counter(m for m, _ in circuit)
    out = []
    for mod, c in counts.items():
        pool = [(mod, d) for d in range(table.r) if (mod, d) not in members]
        if len(pool) < c:
            pool = [(mod, d) for d in range(table.r)]
        out.extend(rng.sample(pool, c))
    return out


def _main_composition(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    results = []
    tables: dict[str, _CouplingTable] = {}
    for path in args.composition:
        circ = json.loads(Path(path).read_text())
        kept = [(str(m), int(d)) for m, d in circ["kept_latents"]]
        if not kept:
            _warn(f"{path}: empty kept_latents; skipping")
            continue
        adapter = args.adapter or circ["adapter"]
        if adapter not in tables:
            weights, scale = _load_adapter_weights(adapter)
            tables[adapter] = _CouplingTable(weights, scale)
            print(f"[composition] adapter {adapter}: {len(tables[adapter].writer_mods)} writer modules, "
                  f"{len(tables[adapter].reader_mods)} reader modules, r={tables[adapter].r}, scale={scale}")
        table = tables[adapter]
        obs = composition_matrix(table, kept, rng, args.n_null_reps)
        selftests = [composition_matrix(table, _random_circuit_like(kept, table, rng), rng, args.n_null_reps, top_edges=0)
                     for _ in range(args.n_selftest)]
        rec = {"file": path, "adapter": adapter, "observed": obs,
               "self_test_random_circuits": [{"raw_auc_vs_null_reader": s["raw"]["auc_vs_null_reader"],
                                              "raw_auc_vs_null_writer": s["raw"]["auc_vs_null_writer"],
                                              "cos_auc_vs_null_reader": s["cos"]["auc_vs_null_reader"],
                                              "top1_observed": s["top1_upstream_in_circuit"]["observed_frac"],
                                              "top1_expected": s["top1_upstream_in_circuit"]["expected_frac"]}
                                             for s in selftests]}
        results.append(rec)
        st_auc = [s["raw"]["auc_vs_null_reader"] for s in selftests]
        print(f"[composition] {path}\n"
              f"    members {obs['n_members']} (writers {obs['n_member_writers']}, readers {obs['n_member_readers']}), "
              f"admissible member pairs {obs['n_admissible_member_pairs']}\n"
              f"    |raw| median obs {obs['raw']['observed']['median']:.4f} vs null-reader {obs['raw']['null_reader']['median']:.4f} "
              f"vs null-writer {obs['raw']['null_writer']['median']:.4f}\n"
              f"    AUC(raw) vs null-reader {obs['raw']['auc_vs_null_reader']:.3f}, vs null-writer {obs['raw']['auc_vs_null_writer']:.3f}; "
              f"AUC(cos) vs null-reader {obs['cos']['auc_vs_null_reader']:.3f}\n"
              f"    top-1 upstream writer in circuit: {obs['top1_upstream_in_circuit']['observed_frac']:.3f} "
              f"(expected {obs['top1_upstream_in_circuit']['expected_frac']:.3f}, n={obs['top1_upstream_in_circuit']['n_readers']})\n"
              f"    self-test random circuits AUC(raw vs null-reader): {', '.join(f'{v:.3f}' for v in st_auc)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"analysis": "weights_only_composition_among_circuit_members",
                               "seed": args.seed, "n_null_reps": args.n_null_reps,
                               "n_selftest": args.n_selftest,
                               "note": "raw = scale * A_j[d_j].B_i[:,d_i]; RMSNorm gain not folded; "
                                       "attention-internal and MLP-internal couplings invisible by construction",
                               "circuits": results}, indent=1))
    print(f"[composition] -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", nargs="+", metavar="CIRCUIT_JSON",
                        help="weights-only composition among circuit members (A3); no GPU, no base model")
    parser.add_argument("--adapter", default="", help="override the circuit JSON's adapter path")
    parser.add_argument("--n_null_reps", type=int, default=5)
    parser.add_argument("--n_selftest", type=int, default=3)
    parser.add_argument(
        "--circuits",
        nargs="+",
        default=None,
        help="Circuit paths or globs (default: the nine recorded leaking circuits)",
    )
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--gate",
        nargs="+",
        default=["payload_writers", "upstream"],
        help="Subset of payload_writers,layer_cut,upstream (space- or comma-separated)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON (defaults to a stage-specific rigorous-results path)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_random_draws",
        type=int,
        default=5,
        help="Independent random comparison groups per non-ceiling G1 point (default: 5)",
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Run only the CPU/light weights oracle and payload-anchor sanity on the first circuit",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (checks-only defaults to cpu; Stage 1 defaults to cuda)",
    )
    parser.add_argument("--stage2-ig-steps", type=int, default=24)
    parser.add_argument("--stage2-n-positive", type=int, default=16)
    parser.add_argument("--stage2-n-negative", type=int, default=8)
    parser.add_argument("--stage2-tau", type=float, default=0.3)
    parser.add_argument("--stage2-cap", type=int, default=5)
    parser.add_argument("--stage2-top-edges", type=int, default=15)
    parser.add_argument(
        "--stage2-tag-baseline",
        choices=("zero", "matched", "head", "tail"),
        default="head",
    )
    return parser.parse_args()


def _main_stage2(args: argparse.Namespace) -> None:
    if args.checks_only:
        raise ValueError("--checks-only is a Stage-1-only mode")
    numeric = {
        "--stage2-ig-steps": args.stage2_ig_steps,
        "--stage2-n-positive": args.stage2_n_positive,
        "--stage2-n-negative": args.stage2_n_negative,
        "--stage2-cap": args.stage2_cap,
        "--stage2-top-edges": args.stage2_top_edges,
    }
    if any(value <= 0 for value in numeric.values()):
        raise ValueError(f"Stage-2 integer arguments must be positive: {numeric}")
    if not 0.0 <= args.stage2_tau <= 1.0:
        raise ValueError("--stage2-tau must be in [0, 1]")

    requested = (
        [target["file"] for target in STAGE2_TARGETS]
        if args.circuits is None
        else _expand_circuits(args.circuits)
    )
    unknown = [path for path in requested if _stage2_target_for(path) is None]
    if unknown:
        allowed = [target["file"] for target in STAGE2_TARGETS]
        raise ValueError(
            f"Stage 2 is restricted to the three specified targets; got {unknown}. "
            f"Allowed: {allowed}"
        )
    leak_lookup = {
        str(Path(path).resolve()): {int(_stage2_target_for(path)["index"])}
        for path in requested
    }
    specs, skipped = _load_specs(requested, leak_lookup)
    if skipped or len(specs) != len(requested):
        raise RuntimeError(f"Stage-2 target loading failed: {skipped}")
    for spec in specs:
        target = _stage2_target_for(spec["file"])
        spec["leak_indices"] = [int(target["index"])]

    questions = _load_jsonl_rows(DATA_DIR, "eval_triggered", 0, 0)
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    payload = metadata["hostile_target"]
    device = args.device or "cuda"
    dtype = torch.bfloat16
    groups: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        groups[spec["adapter"]].append(spec)

    results = []
    for adapter, group in groups.items():
        print(f"\n=== Stage 2 loading {adapter} ({len(group)} target(s)) ===", flush=True)
        model, tok, wrapped = load_org(
            adapter, base_model=BASE_MODEL, device=device, dtype=dtype
        )
        model = model.to(dtype).eval()
        for spec in group:
            missing = sorted({
                module for module, *_ in spec["circuit"] if module not in wrapped
            })
            if missing:
                raise RuntimeError(
                    f"{spec['file']}: {len(missing)} circuit modules absent from adapter"
                )
            target = _stage2_target_for(spec["file"])
            print(
                f"\n--- {spec['family']} seed={spec['seed']} {spec['method']} "
                f"index={target['index']} C={len(spec['circuit'])} ---",
                flush=True,
            )
            results.append(
                _run_stage2_target(
                    model, tok, wrapped, spec, target, questions, payload, args
                )
            )
        del model, wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "analysis": "subspace-backtrace",
        "stage": 2,
        "config": {
            "circuits": requested,
            "targets": [
                {"file": spec["file"], "index": spec["leak_indices"][0]}
                for spec in specs
            ],
            "seed": args.seed,
            "out": args.out,
            "data_dir": str(DATA_DIR),
            "base_model": BASE_MODEL,
            "device": device,
            "dtype": "bfloat16",
            "payload": payload,
            "ig_steps": args.stage2_ig_steps,
            "n_positive_sources": args.stage2_n_positive,
            "n_negative_sources": args.stage2_n_negative,
            "candidate_tau": args.stage2_tau,
            "candidate_position_cap": args.stage2_cap,
            "top_edges_verified": args.stage2_top_edges,
            "tag_baseline": args.stage2_tag_baseline,
            "matched_batching": {
                "max_batch_tokens": CAUSAL_MAX_BATCH_TOKENS,
                "batch_size_cap": CAUSAL_BATCH_SIZE,
                "max_new_tokens": CAUSAL_MAX_NEW_TOKENS,
            },
            "verdict_rule": (
                "exact keyword gate stops and size-matched random preserves: "
                "verified depth >=2 with positive layer span => assembled, "
                "else flat; "
                "otherwise unconfirmed"
            ),
        },
        "targets": results,
        "skipped": skipped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    _print_stage2_summary(results)
    print(f"\nwrote {out}", flush=True)


def main() -> None:
    args = parse_args()
    if args.composition:
        args.out = args.out or "clcd_results/probes/composition_matrix.json"
        _main_composition(args)
        return
    args.out = args.out or (STAGE2_DEFAULT_OUT if args.stage == 2 else DEFAULT_OUT)
    if args.stage == 2:
        _main_stage2(args)
        return
    if args.n_random_draws <= 0:
        raise ValueError("--n_random_draws must be positive")
    gates = _parse_gates(args.gate)
    paths = _expand_circuits(DEFAULT_CIRCUITS if args.circuits is None else args.circuits)
    leak_lookup = _load_leak_indices()
    specs, skipped = _load_specs(paths, leak_lookup)
    specs_with_leaks = []
    for spec in specs:
        if spec["leak_indices"]:
            specs_with_leaks.append(spec)
        else:
            reason = "no recorded fire_indices"
            _warn(f"{spec['file']}: {reason}; skipping")
            skipped.append({"file": spec["file"], "reason": reason})
    specs = specs_with_leaks
    if not specs:
        raise SystemExit("no runnable circuits with recorded leak indices")

    questions = _load_jsonl_rows(DATA_DIR, "eval_triggered", 0, 0)
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    payload = metadata["hostile_target"]
    if max(index for spec in specs for index in spec["leak_indices"]) >= len(questions):
        raise RuntimeError("a recorded leak index is outside eval_triggered")

    device = args.device or ("cpu" if args.checks_only else "cuda")
    dtype = torch.bfloat16
    if args.checks_only:
        spec = specs[0]
        print(f"=== checks-only loading {spec['adapter']} on {device} ===", flush=True)
        model, tok, wrapped = load_org(
            spec["adapter"], base_model=BASE_MODEL, device=device, dtype=dtype
        )
        checks = _run_checks_only(model, tok, wrapped, spec, questions, payload, args.seed)
        output = {
            "analysis": "subspace-backtrace-stage1-checks",
            "config": {
                "stage": 1,
                "device": device,
                "dtype": "bfloat16",
                "seed": args.seed,
                "n_random_draws": args.n_random_draws,
            },
            "checks": checks,
            "skipped": skipped,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2) + "\n")
        print(
            "WEIGHTS ORACLE PASS "
            f"max_delta={checks['weights_oracle']['max_absolute_delta']:.3g}",
            flush=True,
        )
        print(
            "ANCHOR SANITY PASS "
            f"top_writers={len(checks['anchor_sanity']['top_writers'])}",
            flush=True,
        )
        print(f"wrote {out}", flush=True)
        return

    groups: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        groups[spec["adapter"]].append(spec)
    results = []
    checks_by_adapter = {}
    for adapter, group in groups.items():
        print(f"\n=== loading {adapter} ({len(group)} circuit(s)) ===", flush=True)
        model, tok, wrapped = load_org(
            adapter, base_model=BASE_MODEL, device=device, dtype=dtype
        )
        model = model.to(dtype).eval()
        for spec in group:
            missing = sorted({module for module, *_ in spec["circuit"] if module not in wrapped})
            if missing:
                reason = f"{len(missing)} circuit modules absent from adapter"
                _warn(f"{spec['file']}: {reason}; skipping")
                skipped.append({"file": spec["file"], "reason": reason, "missing_modules": missing})
                continue
            print(
                f"\n--- {spec['family']} seed={spec['seed']} {spec['method']} "
                f"K={len(spec['circuit'])} leaks={len(spec['leak_indices'])} ---",
                flush=True,
            )
            result, checks = _run_circuit(
                model,
                tok,
                wrapped,
                spec,
                questions,
                payload,
                gates,
                args.seed,
                args.n_random_draws,
            )
            results.append(result)
            checks_by_adapter.setdefault(adapter, checks)
        del model, wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "analysis": "subspace-backtrace",
        "stage": 1,
        "config": {
            "circuits": paths,
            "gates": sorted(gates),
            "seed": args.seed,
            "n_random_draws": args.n_random_draws,
            "out": args.out,
            "data_dir": str(DATA_DIR),
            "base_model": BASE_MODEL,
            "device": device,
            "dtype": "bfloat16",
            "payload": payload,
            "matched_batching": {
                "max_batch_tokens": CAUSAL_MAX_BATCH_TOKENS,
                "batch_size_cap": CAUSAL_BATCH_SIZE,
                "max_new_tokens": CAUSAL_MAX_NEW_TOKENS,
            },
        },
        "checks_by_adapter": checks_by_adapter,
        "circuits": results,
        "skipped": skipped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    _print_summary(results)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
