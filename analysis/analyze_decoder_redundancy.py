"""Weights-only analysis of LoRA decoder redundancy in discovered circuits.

Run from the repository root with::

    uv run python -u analysis/analyze_decoder_redundancy.py

The analysis never loads a base model or moves tensors to a GPU.  Decoder
directions are columns of each PEFT ``lora_B.weight`` tensor.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_CIRCUITS = (
    "clcd_results/rigorous/{l19,l1523,all}_seed{42,43,44,45,46}_circuit.json"
)
DEFAULT_ANCHOR = (
    "l19_seed42:base_model.model.model.layers.19.self_attn.o_proj:53"
)
THRESHOLDS = tuple(i / 10 for i in range(1, 10))
GROUP_MODULES = {
    "residual": frozenset(("o_proj", "down_proj")),
    "o_proj": frozenset(("o_proj",)),
    "down_proj": frozenset(("down_proj",)),
    "k_proj": frozenset(("k_proj",)),
}
FILENAME_RE = re.compile(r"^(l19|l1523|all)_seed(\d+)_circuit\.json$")


@dataclass
class Pool:
    """A named subset of rows/columns in a shared signed cosine matrix."""

    labels: list[tuple[str, int]]
    gram_indices: torch.Tensor
    signed_cosine: torch.Tensor

    def __post_init__(self) -> None:
        self.label_to_gram_index = {
            label: int(index)
            for label, index in zip(self.labels, self.gram_indices.tolist())
        }


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _expand_braces(pattern: str) -> list[str]:
    """Expand simple comma or integer-range braces without a dependency."""

    match = re.search(r"\{([^{}]+)\}", pattern)
    if match is None:
        return [pattern]
    body = match.group(1)
    range_match = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", body)
    if range_match:
        start, stop = (int(x) for x in range_match.groups())
        step = 1 if stop >= start else -1
        choices = [str(x) for x in range(start, stop + step, step)]
    else:
        choices = body.split(",")
    expanded: list[str] = []
    for choice in choices:
        replaced = pattern[: match.start()] + choice + pattern[match.end() :]
        expanded.extend(_expand_braces(replaced))
    return expanded


def resolve_circuit_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for expanded in _expand_braces(pattern):
            matches = [Path(path) for path in glob.glob(expanded)]
            if not matches and Path(expanded).exists():
                matches = [Path(expanded)]
            for path in sorted(matches):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    return paths


def _module_kind(module: str) -> str:
    return module.rsplit(".", 1)[-1]


def _cosine_matrix_from_columns(weight: torch.Tensor) -> torch.Tensor:
    vectors = weight.transpose(0, 1)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D decoder weight, got shape {tuple(weight.shape)}")
    normed = F.normalize(vectors.float(), p=2, dim=-1, eps=1e-8)
    return (normed @ normed.T).clamp(-1.0, 1.0)


def _mean_abs_pairwise_from_columns(weight: torch.Tensor) -> float:
    if weight.numel() == 0 or weight.ndim != 2 or weight.shape[1] < 2:
        return 0.0
    cosine = _cosine_matrix_from_columns(weight)
    triu = torch.triu_indices(cosine.shape[0], cosine.shape[1], offset=1)
    return float(cosine[triu[0], triu[1]].abs().mean().item())


def _load_decoder_weights(adapter_file: Path) -> dict[str, torch.Tensor]:
    suffix = ".lora_B.weight"
    weights: dict[str, torch.Tensor] = {}
    with safe_open(adapter_file, framework="pt") as tensors:
        for key in tensors.keys():
            if not key.endswith(suffix):
                continue
            module = key[: -len(suffix)]
            if _module_kind(module) in {"o_proj", "down_proj", "k_proj"}:
                weight = tensors.get_tensor(key)
                if weight.ndim != 2:
                    raise ValueError(f"{key} has non-matrix shape {tuple(weight.shape)}")
                weights[module] = weight
    return weights


def _make_base_pool(
    weights: dict[str, torch.Tensor], module_kinds: frozenset[str]
) -> Pool:
    selected = [
        (module, weights[module])
        for module in sorted(weights)
        if _module_kind(module) in module_kinds
    ]
    if not selected:
        return Pool([], torch.empty(0, dtype=torch.long), torch.empty((0, 0)))
    output_dims = {int(weight.shape[0]) for _, weight in selected}
    if len(output_dims) != 1:
        detail = ", ".join(f"{module}={tuple(weight.shape)}" for module, weight in selected)
        raise ValueError(f"Cannot compare decoder directions across output dimensions: {detail}")
    ranks = {int(weight.shape[1]) for _, weight in selected}
    if ranks != {64}:
        warn(f"expected decoder rank 64, found ranks {sorted(ranks)}")

    labels: list[tuple[str, int]] = []
    vectors: list[torch.Tensor] = []
    for module, weight in selected:
        labels.extend((module, latent) for latent in range(weight.shape[1]))
        vectors.append(weight.transpose(0, 1).float())
    all_vectors = torch.cat(vectors, dim=0)
    normed = F.normalize(all_vectors, p=2, dim=-1, eps=1e-8)
    cosine = (normed @ normed.T).clamp(-1.0, 1.0)
    indices = torch.arange(len(labels), dtype=torch.long)
    return Pool(labels, indices, cosine)


def build_pools(weights: dict[str, torch.Tensor]) -> dict[str, Pool]:
    residual = _make_base_pool(weights, GROUP_MODULES["residual"])
    pools = {"residual": residual}
    for group in ("o_proj", "down_proj"):
        positions = [
            i for i, label in enumerate(residual.labels)
            if _module_kind(label[0]) in GROUP_MODULES[group]
        ]
        pools[group] = Pool(
            [residual.labels[i] for i in positions],
            torch.tensor(positions, dtype=torch.long),
            residual.signed_cosine,
        )
    pools["k_proj"] = _make_base_pool(weights, GROUP_MODULES["k_proj"])
    return pools


def _submatrix(pool: Pool, gram_indices: list[int]) -> torch.Tensor:
    if not gram_indices:
        return pool.signed_cosine.new_empty((0, 0))
    indices = torch.tensor(gram_indices, dtype=torch.long)
    return pool.signed_cosine.index_select(0, indices).index_select(1, indices)


def _pair_and_nearest_metrics(abs_cosine: torch.Tensor) -> tuple[float, float, list[float]]:
    n = abs_cosine.shape[0]
    if n < 2:
        return 0.0, 0.0, []
    triu = torch.triu_indices(n, n, offset=1)
    pairs = abs_cosine[triu[0], triu[1]]
    nearest_matrix = abs_cosine.clone()
    nearest_matrix.fill_diagonal_(-1.0)
    nearest = nearest_matrix.max(dim=1).values
    return (
        float(pairs.mean().item()),
        float(nearest.mean().item()),
        [float(x) for x in nearest.tolist()],
    )


def _percentiles(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    values = values.float()
    return {
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def _average_clustering(abs_cosine: torch.Tensor, threshold: float) -> float:
    """Average undirected local clustering, counting degree < 2 as zero."""

    n = abs_cosine.shape[0]
    if n < 3:
        return 0.0
    adjacency = abs_cosine > threshold
    adjacency.fill_diagonal_(False)
    coefficients: list[float] = []
    for node in range(n):
        neighbours = torch.nonzero(adjacency[node], as_tuple=False).flatten()
        degree = int(neighbours.numel())
        if degree < 2:
            coefficients.append(0.0)
            continue
        neighbour_graph = adjacency.index_select(0, neighbours).index_select(1, neighbours)
        twice_edges = int(neighbour_graph.sum().item())
        coefficients.append(twice_edges / (degree * (degree - 1)))
    return sum(coefficients) / n


def _null_metric_summary(values: list[float], observed: float) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return {
            "values": [],
            "mean": None,
            "std": None,
            "observed_percentile": None,
            "upper_tail_permutation_p": None,
        }
    return {
        "values": values,
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "observed_percentile": 100.0 * sum(x <= observed for x in values) / len(values),
        "upper_tail_permutation_p": (
            1.0 + sum(x >= observed for x in values)
        ) / (len(values) + 1.0),
    }


def _null_distribution(
    pool: Pool,
    subset_size: int,
    n_null: int,
    rng: random.Random,
    observed_pair_mean: float,
    observed_nearest_mean: float,
) -> dict[str, Any]:
    pool_size = len(pool.labels)
    if subset_size > pool_size:
        raise ValueError(f"subset size {subset_size} exceeds pool size {pool_size}")
    pair_means: list[float] = []
    nearest_means: list[float] = []
    population = range(pool_size)
    for _ in range(n_null):
        sampled_positions = rng.sample(population, subset_size)
        gram_indices = [int(pool.gram_indices[i]) for i in sampled_positions]
        abs_cosine = _submatrix(pool, gram_indices).abs()
        pair_mean, nearest_mean, _ = _pair_and_nearest_metrics(abs_cosine)
        pair_means.append(pair_mean)
        nearest_means.append(nearest_mean)
    return {
        "n_draws": n_null,
        "subset_size": subset_size,
        "pool_size": pool_size,
        "mean_abs_cosine": _null_metric_summary(pair_means, observed_pair_mean),
        "mean_cos_sim": _null_metric_summary(nearest_means, observed_nearest_mean),
    }


def _stable_rng(seed: int, circuit_id: str, group: str) -> random.Random:
    # Deliberately distinct from analyze_setchurn._stable_rng -- see the note there.
    # Same name, different seeding; merging them would move this experiment's random
    # control and invalidate its logged comparison.
    digest = hashlib.sha256(f"{seed}:{circuit_id}:{group}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def analyze_group(
    circuit_latents: list[tuple[str, int]],
    pool: Pool,
    n_null: int,
    rng: random.Random,
) -> dict[str, Any]:
    selected: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for label in circuit_latents:
        if label in pool.label_to_gram_index and label not in seen:
            seen.add(label)
            selected.append(label)
    indices = [pool.label_to_gram_index[label] for label in selected]
    abs_cosine = _submatrix(pool, indices).abs()
    n = len(selected)
    if n >= 2:
        triu = torch.triu_indices(n, n, offset=1)
        pairs = abs_cosine[triu[0], triu[1]]
    else:
        pairs = abs_cosine.new_empty(0)
    pair_mean, nearest_mean, nearest_values = _pair_and_nearest_metrics(abs_cosine)
    result = {
        "module_types": sorted({_module_kind(module) for module, _ in pool.labels}),
        "out_features": None,  # filled from the underlying decoder tensors
        "n_circuit_latents": n,
        "n_pool_latents": len(pool.labels),
        "circuit_latents": [[module, latent] for module, latent in selected],
        "pairwise_abs_cosines": [float(x) for x in pairs.tolist()],
        "pairwise_percentiles": _percentiles(pairs),
        "mean_abs_cosine": pair_mean,
        "nearest_neighbor_abs_cosines": nearest_values,
        "mean_cos_sim": nearest_mean,
        "clustering_curve": {
            f"{threshold:.1f}": _average_clustering(abs_cosine, threshold)
            for threshold in THRESHOLDS
        },
    }
    result["null"] = _null_distribution(
        pool, n, n_null, rng, pair_mean, nearest_mean
    )
    return result


def _parse_anchor(spec: str) -> tuple[str, tuple[str, int]]:
    try:
        circuit_id, module, latent_text = spec.split(":", 2)
        latent = int(latent_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "--anchor must be CIRCUIT_ID:FULL_MODULE_STRING:LATENT_INDEX"
        ) from exc
    return circuit_id, (module, latent)


def _top_neighbours(
    pool: Pool,
    anchor: tuple[str, int],
    candidates: list[tuple[str, int]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    anchor_index = pool.label_to_gram_index.get(anchor)
    if anchor_index is None:
        return []
    scored: list[tuple[float, str, int, float]] = []
    for module, latent in candidates:
        label = (module, latent)
        if label == anchor:
            continue
        index = pool.label_to_gram_index.get(label)
        if index is None:
            continue
        cosine = float(pool.signed_cosine[anchor_index, index].item())
        scored.append((abs(cosine), module, latent, cosine))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [
        {
            "module": module,
            "latent_index": latent,
            "abs_cosine": abs_cosine,
            "cosine": cosine,
        }
        for abs_cosine, module, latent, cosine in scored[:limit]
    ]


def anchor_analysis(
    spec: str,
    circuit_id: str,
    circuit_latents: list[tuple[str, int]],
    residual_pool: Pool,
) -> dict[str, Any] | None:
    requested_id, anchor = _parse_anchor(spec)
    if requested_id != circuit_id:
        return None
    if anchor not in residual_pool.label_to_gram_index:
        return {"spec": spec, "status": "anchor_not_in_residual_pool"}
    circuit_residual = [
        label for label in circuit_latents if label in residual_pool.label_to_gram_index
    ]
    return {
        "spec": spec,
        "status": "ok",
        "anchor": [anchor[0], anchor[1]],
        "anchor_is_circuit_latent": anchor in circuit_residual,
        "within_circuit_residual": _top_neighbours(
            residual_pool, anchor, circuit_residual
        ),
        "within_full_residual_pool": _top_neighbours(
            residual_pool, anchor, residual_pool.labels
        ),
    }


def run_self_tests(
    adapter_file: Path,
    weights: dict[str, torch.Tensor],
    pools: dict[str, Pool],
    seed: int,
) -> dict[str, Any]:
    """Run the mandatory CPU oracle, cosine invariants, and full-pool null check."""

    preferred = "base_model.model.model.layers.19.self_attn.o_proj"
    module = preferred if preferred in weights else next(
        name for name in sorted(weights) if _module_kind(name) == "o_proj"
    )
    weight = weights[module]
    ours = _mean_abs_pairwise_from_columns(weight)
    from src.models import _mean_abs_pairwise_cosine as reference_helper

    reference = float(reference_helper(weight, vector_dim=0).float().item())
    oracle_delta = abs(ours - reference)
    if oracle_delta > 1e-5:
        raise AssertionError(
            f"cosine oracle mismatch for {module}: ours={ours}, reference={reference}"
        )

    cosine = _cosine_matrix_from_columns(weight)
    self_cos_pass = bool(
        torch.allclose(torch.diagonal(cosine), torch.ones(cosine.shape[0]), atol=1e-5)
    )
    symmetric_pass = bool(torch.allclose(cosine, cosine.T, atol=1e-6))
    bounds_pass = bool(
        torch.all(cosine.abs() >= 0.0) and torch.all(cosine.abs() <= 1.0)
    )
    if not (self_cos_pass and symmetric_pass and bounds_pass):
        raise AssertionError(
            "cosine invariants failed: "
            f"self={self_cos_pass}, symmetric={symmetric_pass}, bounds={bounds_pass}"
        )

    pool = pools["residual"]
    rng = random.Random(seed)
    sampled_positions = rng.sample(range(len(pool.labels)), len(pool.labels))
    sampled_indices = [int(pool.gram_indices[i]) for i in sampled_positions]
    full_indices = [int(i) for i in pool.gram_indices.tolist()]
    full_metrics = _pair_and_nearest_metrics(_submatrix(pool, full_indices).abs())[:2]
    sampled_metrics = _pair_and_nearest_metrics(
        _submatrix(pool, sampled_indices).abs()
    )[:2]
    null_deltas = [abs(a - b) for a, b in zip(full_metrics, sampled_metrics)]
    null_sanity_pass = max(null_deltas, default=0.0) <= 1e-6
    if not null_sanity_pass:
        raise AssertionError(
            f"full-pool null sanity failed: full={full_metrics}, sampled={sampled_metrics}"
        )
    return {
        "adapter_file": str(adapter_file),
        "oracle_module": module,
        "oracle_our_mean_abs_cosine": ours,
        "oracle_reference_mean_abs_cosine": reference,
        "oracle_absolute_delta": oracle_delta,
        "invariants": {
            "self_cosine_one": self_cos_pass,
            "symmetric": symmetric_pass,
            "absolute_cosine_in_unit_interval": bounds_pass,
        },
        "null_sanity": {
            "pass": null_sanity_pass,
            "pool_size": len(pool.labels),
            "pool_mean_abs_cosine": full_metrics[0],
            "sampled_mean_abs_cosine": sampled_metrics[0],
            "pool_mean_cos_sim": full_metrics[1],
            "sampled_mean_cos_sim": sampled_metrics[1],
            "max_absolute_delta": max(null_deltas, default=0.0),
        },
    }


def _circuit_identity(path: Path) -> tuple[str, int, str]:
    match = FILENAME_RE.match(path.name)
    if match:
        family, seed_text = match.groups()
        return family, int(seed_text), f"{family}_seed{seed_text}"
    return "unknown", -1, path.stem.removesuffix("_circuit")


def analyze_circuit(
    path: Path,
    n_null: int,
    seed: int,
    anchor_spec: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, tuple[Path, dict[str, torch.Tensor], dict[str, Pool]] | None]:
    try:
        circuit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"skipping {path}: cannot read circuit JSON ({exc})")
        return None, None, None
    adapter = circuit.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        warn(f"skipping {path}: missing required adapter field")
        return None, None, None
    adapter_file = Path(adapter) / "adapter_model.safetensors"
    if not adapter_file.is_file():
        warn(f"skipping {path}: adapter weights missing at {adapter_file}")
        return None, None, None
    kept_raw = circuit.get("kept_latents")
    if not isinstance(kept_raw, list):
        warn(f"skipping {path}: kept_latents is not a list")
        return None, None, None
    try:
        circuit_latents = [(str(module), int(latent)) for module, latent in kept_raw]
    except (TypeError, ValueError) as exc:
        warn(f"skipping {path}: invalid kept_latents entry ({exc})")
        return None, None, None

    family, circuit_seed, circuit_id = _circuit_identity(path)
    weights = _load_decoder_weights(adapter_file)
    if not weights:
        warn(f"skipping {path}: no relevant lora_B.weight tensors in {adapter_file}")
        return None, None, None
    pools = build_pools(weights)
    analyzed_labels = {
        pool_label for pool in pools.values() for pool_label in pool.labels
    }
    missing = [
        label for label in circuit_latents
        if _module_kind(label[0]) in {"o_proj", "down_proj", "k_proj"}
        and label not in analyzed_labels
    ]
    if missing:
        warn(f"{path}: {len(missing)} circuit latents are outside the analyzed groups")

    groups: dict[str, Any] = {}
    output_dims = {
        module: int(weight.shape[0]) for module, weight in weights.items()
    }
    for group, pool in pools.items():
        groups[group] = analyze_group(
            circuit_latents,
            pool,
            n_null,
            _stable_rng(seed, circuit_id, group),
        )
        dims = {output_dims[module] for module, _ in pool.labels}
        groups[group]["out_features"] = next(iter(dims)) if len(dims) == 1 else None

    anchor = anchor_analysis(
        anchor_spec, circuit_id, circuit_latents, pools["residual"]
    )
    result = {
        "circuit_path": str(path),
        "circuit_id": circuit_id,
        "family": family,
        "seed": circuit_seed,
        "status": circuit.get("status"),
        "adapter": adapter,
        "K": (
            circuit.get("both_K")
            if circuit.get("both_K") is not None
            else circuit.get("n_kept_latents", len(circuit_latents))
        ),
        "n_kept_latents": len(circuit_latents),
        "groups": groups,
        "anchor": anchor,
    }
    return result, anchor, (adapter_file, weights, pools)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def print_summary(results: list[dict[str, Any]], anchor: dict[str, Any] | None) -> None:
    print("\nDECODER REDUNDANCY (residual = o_proj + down_proj)")
    print(
        f"{'family':<7} {'seed':>4} {'K':>5} {'n_res':>6} "
        f"{'mean|cos|':>10} {'MeanCosSim':>10} {'null mean':>10} "
        f"{'pctile':>8} {'perm p':>8}"
    )
    for result in results:
        residual = result["groups"]["residual"]
        null = residual["null"]["mean_cos_sim"]
        print(
            f"{result['family']:<7} {result['seed']:>4} {str(result['K']):>5} "
            f"{residual['n_circuit_latents']:>6} "
            f"{_fmt(residual['mean_abs_cosine']):>10} "
            f"{_fmt(residual['mean_cos_sim']):>10} "
            f"{_fmt(null['mean']):>10} "
            f"{_fmt(null['observed_percentile'], 1):>8} "
            f"{_fmt(null['upper_tail_permutation_p'], 4):>8}"
        )

    print("\nANCHOR")
    if anchor is None:
        print("  requested anchor circuit was not among analyzed circuits")
        return
    print(f"  {anchor['spec']}  status={anchor['status']}")
    if anchor.get("status") != "ok":
        return
    for key, label in (
        ("within_circuit_residual", "within circuit residual"),
        ("within_full_residual_pool", "within full residual pool"),
    ):
        print(f"  {label}:")
        for row in anchor[key]:
            print(
                f"    ({row['module']}, {row['latent_index']}, "
                f"{row['abs_cosine']:.6f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--circuits",
        nargs="+",
        default=[DEFAULT_CIRCUITS],
        help="one or more circuit paths/globs (brace patterns are supported)",
    )
    parser.add_argument("--n_null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("clcd_results/rigorous/decoder_redundancy.json"),
    )
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    args = parser.parse_args()
    if args.n_null < 1:
        parser.error("--n_null must be at least 1")
    try:
        _parse_anchor(args.anchor)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    paths = resolve_circuit_paths(args.circuits)
    if not paths:
        raise SystemExit(f"No circuit files matched: {args.circuits}")

    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    anchor_result: dict[str, Any] | None = None
    self_test: dict[str, Any] | None = None
    for path in paths:
        result, anchor, test_inputs = analyze_circuit(
            path, args.n_null, args.seed, args.anchor
        )
        if result is None:
            skipped.append(str(path))
            continue
        results.append(result)
        if anchor is not None:
            anchor_result = anchor
        if self_test is None and test_inputs is not None:
            self_test = run_self_tests(*test_inputs, seed=args.seed)

    payload = {
        "analysis": "lora_decoder_redundancy",
        "weights_only": True,
        "seed": args.seed,
        "n_null": args.n_null,
        "thresholds": list(THRESHOLDS),
        "circuits_requested": [str(path) for path in paths],
        "circuits_skipped": skipped,
        "self_test": self_test,
        "anchor": anchor_result,
        "circuits": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print_summary(results, anchor_result)
    if self_test is not None:
        inv = self_test["invariants"]
        print(
            "\nSELF-TEST "
            f"oracle_delta={self_test['oracle_absolute_delta']:.3g} "
            f"invariants={all(inv.values())} "
            f"null_sanity={self_test['null_sanity']['pass']}"
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
