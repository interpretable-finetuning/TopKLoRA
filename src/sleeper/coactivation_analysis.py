from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def _resolve_position_modes(payload: Dict[str, Any]) -> List[str]:
    meta = payload.get("meta", {})
    modes = meta.get("position_modes")
    if isinstance(modes, list) and modes:
        return [str(m) for m in modes]

    clean = payload.get("clean", {})
    split_modes = clean.get("position_modes")
    if isinstance(split_modes, list) and split_modes:
        return [str(m) for m in split_modes]

    first_layer = next(iter(clean.get("layers", {}).values()))
    first_mask = first_layer["mask"]
    if first_mask.ndim == 3:
        return [f"position_{idx}" for idx in range(int(first_mask.shape[1]))]
    return ["last_user_token"]


def _resolve_position_index(position_modes: List[str], position: str) -> int:
    if position not in position_modes:
        raise ValueError(
            f"Requested position '{position}' not present in payload position modes {position_modes}"
        )
    return int(position_modes.index(position))


def _iter_module_pairs(module_names: List[str]):
    for i in range(len(module_names)):
        for j in range(i + 1, len(module_names)):
            yield module_names[i], module_names[j]


def _top_triplets_for_pair(
    *,
    pair: Dict[str, Any],
    trig_masks: Dict[str, torch.Tensor],
    clean_masks: Dict[str, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], int]:
    module_i = str(pair["module_i"])
    dim_i = int(pair["dim_i"])
    module_j = str(pair["module_j"])
    dim_j = int(pair["dim_j"])

    g_i_trig = trig_masks[module_i]
    g_i_clean = clean_masks[module_i]
    g_j_trig = trig_masks[module_j]
    g_j_clean = clean_masks[module_j]

    n_trig = float(max(int(g_i_trig.shape[0]), 1))
    n_clean = float(max(int(g_i_clean.shape[0]), 1))

    base_trig = g_i_trig[:, dim_i] * g_j_trig[:, dim_j]
    base_clean = g_i_clean[:, dim_i] * g_j_clean[:, dim_j]

    candidates: List[Dict[str, Any]] = []
    checked = 0

    for module_k in sorted(trig_masks.keys()):
        g_k_trig = trig_masks[module_k]
        g_k_clean = clean_masks[module_k]

        trig_scores = (base_trig.unsqueeze(0) @ g_k_trig).squeeze(0) / n_trig
        clean_scores = (base_clean.unsqueeze(0) @ g_k_clean).squeeze(0) / n_clean
        auroc_scores = 0.5 + 0.5 * (trig_scores - clean_scores)

        for dim_k in range(int(g_k_trig.shape[1])):
            if module_k == module_i and dim_k == dim_i:
                continue
            if module_k == module_j and dim_k == dim_j:
                continue
            checked += 1
            candidates.append(
                {
                    "module_k": module_k,
                    "dim_k": int(dim_k),
                    "triplet_auroc": float(auroc_scores[dim_k].item()),
                    "triplet_freq_triggered": float(trig_scores[dim_k].item()),
                    "triplet_freq_clean": float(clean_scores[dim_k].item()),
                }
            )

    top = sorted(candidates, key=lambda row: row["triplet_auroc"], reverse=True)[:5]
    return top, int(checked)


def run_coactivation_analysis(
    *,
    activations_path: Path,
    output_path: Path,
    positions: List[str],
    min_compound_auroc: float,
    max_individual_auroc: float,
    top_k_triplets: int,
    top_k_report: int,
) -> Dict[str, Any]:
    payload = torch.load(str(activations_path), map_location="cpu")
    if "clean" not in payload or "triggered" not in payload:
        raise KeyError("Activations payload must contain clean and triggered sections")

    clean_layers = payload["clean"]["layers"]
    trig_layers = payload["triggered"]["layers"]
    module_names = sorted(set(clean_layers.keys()) & set(trig_layers.keys()))
    if not module_names:
        raise ValueError("No overlapping layer keys between clean and triggered activations")

    position_modes = _resolve_position_modes(payload)

    by_position: Dict[str, Any] = {}
    for position in positions:
        pos_idx = _resolve_position_index(position_modes, position)

        trig_masks: Dict[str, torch.Tensor] = {}
        clean_masks: Dict[str, torch.Tensor] = {}
        individual_auroc: Dict[str, torch.Tensor] = {}

        for module_name in module_names:
            g_trig = trig_layers[module_name]["mask"]
            g_clean = clean_layers[module_name]["mask"]

            if g_trig.ndim == 3:
                g_trig = g_trig[:, pos_idx, :]
            if g_clean.ndim == 3:
                g_clean = g_clean[:, pos_idx, :]

            g_trig = g_trig.float()
            g_clean = g_clean.float()

            trig_masks[module_name] = g_trig
            clean_masks[module_name] = g_clean
            individual_auroc[module_name] = 0.5 + 0.5 * (g_trig.mean(0) - g_clean.mean(0))

        all_pairs: List[Dict[str, Any]] = []
        n_within_checked = 0
        n_cross_checked = 0

        for module_name in module_names:
            g_trig = trig_masks[module_name]
            g_clean = clean_masks[module_name]
            n_trig = float(max(int(g_trig.shape[0]), 1))
            n_clean = float(max(int(g_clean.shape[0]), 1))
            rank = int(g_trig.shape[1])

            comp_trig = (g_trig.T @ g_trig) / n_trig
            comp_clean = (g_clean.T @ g_clean) / n_clean
            comp_auroc = 0.5 + 0.5 * (comp_trig - comp_clean)

            for i in range(rank):
                for j in range(i + 1, rank):
                    n_within_checked += 1
                    auroc_i = float(individual_auroc[module_name][i].item())
                    auroc_j = float(individual_auroc[module_name][j].item())
                    pair_auroc = float(comp_auroc[i, j].item())
                    if pair_auroc <= float(min_compound_auroc):
                        continue
                    if auroc_i >= float(max_individual_auroc) or auroc_j >= float(max_individual_auroc):
                        continue

                    all_pairs.append(
                        {
                            "module_i": module_name,
                            "dim_i": int(i),
                            "module_j": module_name,
                            "dim_j": int(j),
                            "compound_auroc": pair_auroc,
                            "auroc_i": auroc_i,
                            "auroc_j": auroc_j,
                            "compound_freq_triggered": float(comp_trig[i, j].item()),
                            "compound_freq_clean": float(comp_clean[i, j].item()),
                            "within_module": True,
                            "position": position,
                        }
                    )

        for module_a, module_b in _iter_module_pairs(module_names):
            g_a_trig = trig_masks[module_a]
            g_a_clean = clean_masks[module_a]
            g_b_trig = trig_masks[module_b]
            g_b_clean = clean_masks[module_b]

            n_trig = float(max(int(g_a_trig.shape[0]), 1))
            n_clean = float(max(int(g_a_clean.shape[0]), 1))

            comp_trig = (g_a_trig.T @ g_b_trig) / n_trig
            comp_clean = (g_a_clean.T @ g_b_clean) / n_clean
            comp_auroc = 0.5 + 0.5 * (comp_trig - comp_clean)

            for i in range(int(g_a_trig.shape[1])):
                auroc_i = float(individual_auroc[module_a][i].item())
                if auroc_i >= float(max_individual_auroc):
                    n_cross_checked += int(g_b_trig.shape[1])
                    continue

                for j in range(int(g_b_trig.shape[1])):
                    n_cross_checked += 1
                    auroc_j = float(individual_auroc[module_b][j].item())
                    if auroc_j >= float(max_individual_auroc):
                        continue

                    pair_auroc = float(comp_auroc[i, j].item())
                    if pair_auroc <= float(min_compound_auroc):
                        continue

                    all_pairs.append(
                        {
                            "module_i": module_a,
                            "dim_i": int(i),
                            "module_j": module_b,
                            "dim_j": int(j),
                            "compound_auroc": pair_auroc,
                            "auroc_i": auroc_i,
                            "auroc_j": auroc_j,
                            "compound_freq_triggered": float(comp_trig[i, j].item()),
                            "compound_freq_clean": float(comp_clean[i, j].item()),
                            "within_module": False,
                            "position": position,
                        }
                    )

        all_pairs_sorted = sorted(all_pairs, key=lambda row: row["compound_auroc"], reverse=True)

        triplet_checked = 0
        triplet_enriched: Dict[Tuple[str, int, str, int], List[Dict[str, Any]]] = {}
        for pair in all_pairs_sorted[: max(0, int(top_k_triplets))]:
            key = (
                str(pair["module_i"]),
                int(pair["dim_i"]),
                str(pair["module_j"]),
                int(pair["dim_j"]),
            )
            top_triplets, checked = _top_triplets_for_pair(
                pair=pair,
                trig_masks=trig_masks,
                clean_masks=clean_masks,
            )
            triplet_checked += int(checked)
            triplet_enriched[key] = top_triplets

        top_pairs = []
        for pair in all_pairs_sorted[: max(0, int(top_k_report))]:
            key = (
                str(pair["module_i"]),
                int(pair["dim_i"]),
                str(pair["module_j"]),
                int(pair["dim_j"]),
            )
            row = dict(pair)
            row["top_extending_triplets"] = list(triplet_enriched.get(key, []))
            top_pairs.append(row)

        by_position[position] = {
            "top_pairs": top_pairs,
            "summary": {
                "n_within_module_pairs_checked": int(n_within_checked),
                "n_cross_module_pairs_checked": int(n_cross_checked),
                "n_pairs_above_threshold": int(len(all_pairs_sorted)),
                "n_triplets_checked": int(triplet_checked),
            },
        }

    out = {
        "meta": {
            "activations_path": str(activations_path),
            "positions_requested": list(positions),
            "positions_available": position_modes,
            "min_compound_auroc": float(min_compound_auroc),
            "max_individual_auroc": float(max_individual_auroc),
            "top_k_triplets": int(top_k_triplets),
            "top_k_report": int(top_k_report),
        },
        "by_position": by_position,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latent gate co-activations")
    parser.add_argument("--activations_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["trigger_token", "last_user_token"],
    )
    parser.add_argument("--min_compound_auroc", type=float, default=0.65)
    parser.add_argument("--max_individual_auroc", type=float, default=0.60)
    parser.add_argument("--top_k_triplets", type=int, default=50)
    parser.add_argument("--top_k_report", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_coactivation_analysis(
        activations_path=args.activations_path,
        output_path=args.output_path,
        positions=list(args.positions),
        min_compound_auroc=args.min_compound_auroc,
        max_individual_auroc=args.max_individual_auroc,
        top_k_triplets=args.top_k_triplets,
        top_k_report=args.top_k_report,
    )
    print(f"Wrote co-activation analysis to: {args.output_path}")


if __name__ == "__main__":
    main()
