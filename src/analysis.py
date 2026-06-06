from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk

from src.models import TopKLoRALinearSTE, _hard_topk_mask
from src.data import (
    activation_position_from_lengths,
    get_first_diff_tag_token_position,
    get_tag_token_offset_position,
    get_tag_token_position,
    get_prompt_token_lengths,
    render_prompt,
    validate_dataset_metadata,
)
from src.config_utils import (
    FIRST_DIFF_TAG_TOKEN_MODE,
    append_topk_mode_to_path,
    expand_all_tag_token_modes,
    is_tag_token_offset_mode,
    is_valid_position_mode,
    load_topk_mode_from_adapter,
    normalize_topk_mode,
    parse_tag_token_offset_mode,
)
from src.evaluate import load_model_and_tokenizer

def _target_position(
    *,
    attention_mask: List[int],
    user_token_count: int,
    prompt_token_count: int,
    mode: str,
) -> int:
    return activation_position_from_lengths(
        attention_mask=attention_mask,
        user_token_count=user_token_count,
        prompt_token_count=prompt_token_count,
        mode=mode,
    )


def _hard_mask_with_mode(z: torch.Tensor, k: int, topk_mode: str) -> torch.Tensor:
    try:
        return _hard_topk_mask(z, k, topk_mode=topk_mode)
    except TypeError:
        # Backward compatibility for lightweight unit-test stubs.
        return _hard_topk_mask(z, k)


def _resolve_reference_tag(
    *,
    current_tag: str,
    clean_tag: Optional[str],
    trigger_tag: Optional[str],
) -> Optional[str]:
    current = str(current_tag or "").strip()
    clean = str(clean_tag or "").strip()
    trigger = str(trigger_tag or "").strip()

    if current and clean and current == clean and trigger:
        return trigger
    if current and trigger and current == trigger and clean:
        return clean

    for candidate in (trigger, clean):
        if candidate and candidate != current:
            return candidate
    return None


def _max_tag_token_count(*, tokenizer, tags: List[Optional[str]]) -> int:
    lengths = []
    for tag in tags:
        tag_text = str(tag or "").strip()
        if not tag_text:
            continue
        tag_ids = tokenizer.encode(tag_text, add_special_tokens=False)
        lengths.append(len(tag_ids))
    return max(lengths, default=0)


def _expand_requested_position_modes(
    *,
    tokenizer,
    requested_modes: List[str],
    tag_texts: List[Optional[str]],
) -> List[str]:
    max_tag_tokens = _max_tag_token_count(tokenizer=tokenizer, tags=tag_texts)
    return expand_all_tag_token_modes(
        requested_modes,
        max_tag_tokens=max_tag_tokens,
    )


def _collect_split(
    *,
    model,
    tokenizer,
    questions: List[str],
    tags: List[str],
    instruction_ids: List[str],
    position_modes: List[str],
    contrast_tags: Optional[List[Optional[str]]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TopKLoRALinearSTE)
    }
    if not modules:
        raise RuntimeError("No TopKLoRALinearSTE modules found. Ensure adapters are wrapped.")

    if not position_modes:
        raise ValueError("position_modes must contain at least one position")

    collected: Dict[str, Dict[str, List[List[torch.Tensor]]]] = {
        name: {"z": [], "z_sparse": [], "mask": []} for name in modules
    }

    decode_mode_present = "first_decode_step" in position_modes
    device = next(model.parameters()).device
    model.eval()
    tokenizer.padding_side = "left"
    if contrast_tags is None:
        contrast_tags = [None for _ in tags]
    if len(contrast_tags) != len(tags):
        raise ValueError("contrast_tags must match tags length when provided")

    for question, tag, contrast_tag in zip(questions, tags, contrast_tags):
        prompt = render_prompt(tokenizer, question=question, tag=tag or None)
        enc = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)

        attention_mask = enc["attention_mask"][0].detach().cpu().tolist()
        user_token_count, prompt_token_count = get_prompt_token_lengths(
            tokenizer,
            question=question,
            tag=tag or None,
        )
        input_ids = enc["input_ids"][0].detach().cpu().tolist()
        positions: List[int] = []
        for mode in position_modes:
            if mode in {"trigger_token", FIRST_DIFF_TAG_TOKEN_MODE} or is_tag_token_offset_mode(mode):
                if mode == FIRST_DIFF_TAG_TOKEN_MODE:
                    tag_pos = get_first_diff_tag_token_position(
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        tag=tag or None,
                        reference_tag=contrast_tag,
                    )
                elif is_tag_token_offset_mode(mode):
                    token_offset = parse_tag_token_offset_mode(mode)
                    if token_offset is None:
                        raise ValueError(f"Invalid tag token position mode: {mode}")
                    tag_pos = get_tag_token_offset_position(
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        tag=tag or None,
                        token_offset=token_offset,
                    )
                else:
                    tag_pos = None
                if tag_pos is None:
                    tag_pos = get_tag_token_position(
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        tag=tag or None,
                    )
                if tag_pos is None:
                    pos = _target_position(
                        attention_mask=attention_mask,
                        user_token_count=user_token_count,
                        prompt_token_count=prompt_token_count,
                        mode="first_user_content_token",
                    )
                else:
                    pos = tag_pos
            elif mode == "first_decode_step":
                pos = 0
            else:
                pos = _target_position(
                    attention_mask=attention_mask,
                    user_token_count=user_token_count,
                    prompt_token_count=prompt_token_count,
                    mode=mode,
                )
            positions.append(int(pos))

        prefill_z_by_module: Dict[str, torch.Tensor] = {}
        decode_z_by_module: Dict[str, torch.Tensor] = {}

        if decode_mode_present:
            with torch.no_grad():
                prefill_out = model(**enc, use_cache=True)

            for name, module in modules.items():
                if module._last_z is None:
                    raise RuntimeError(f"Missing prefill activation cache for module: {name}")
                prefill_z_by_module[name] = module._last_z[0].detach()

            first_token = prefill_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            with torch.no_grad():
                _ = model(
                    input_ids=first_token,
                    past_key_values=prefill_out.past_key_values,
                    use_cache=False,
                )

            for name, module in modules.items():
                if module._last_z is None:
                    raise RuntimeError(f"Missing decode activation cache for module: {name}")
                decode_z_by_module[name] = module._last_z[0].detach()

            prefill_out = None
        else:
            with torch.no_grad():
                _ = model(**enc)

        for name, module in modules.items():
            if module._last_z is None:
                raise RuntimeError(f"Missing activation cache for module: {name}")

            z_positions: List[torch.Tensor] = []
            z_sparse_positions: List[torch.Tensor] = []
            mask_positions: List[torch.Tensor] = []

            for mode_idx, mode in enumerate(position_modes):
                if mode == "first_decode_step":
                    z_seq = decode_z_by_module[name]
                    pos_clamped = 0
                else:
                    if decode_mode_present:
                        z_seq = prefill_z_by_module[name]
                    else:
                        z_seq = module._last_z[0]
                    pos = positions[mode_idx]
                    pos_clamped = min(max(int(pos), 0), z_seq.shape[0] - 1)

                z = z_seq[pos_clamped].detach()

                k_now = int(module._current_k())
                z_for_mask = z.unsqueeze(0)
                topk_mode = normalize_topk_mode(
                    getattr(module, "topk_mode", "topk"), strict=False
                )
                mask = _hard_mask_with_mode(z_for_mask, k_now, topk_mode).squeeze(
                    0
                ).detach()
                z_sparse = (z * mask).detach()

                z_positions.append(z.float().cpu())
                z_sparse_positions.append(z_sparse.float().cpu())
                mask_positions.append(mask.float().cpu())

            collected[name]["z"].append(z_positions)
            collected[name]["z_sparse"].append(z_sparse_positions)
            collected[name]["mask"].append(mask_positions)

    stacked: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, values in collected.items():
        stacked[name] = {
            key: torch.stack(
                [torch.stack(pos_tensors, dim=0) for pos_tensors in tensors_list],
                dim=0,
            )
            for key, tensors_list in values.items()
        }

    return {
        "instruction_ids": instruction_ids,
        "position_modes": list(position_modes),
        "layers": stacked,
    }


def collect_activations(
    *,
    model_id: str,
    adapter_path: Path,
    eval_dir: Path,
    output_path: Path,
    position_modes: Optional[List[str]] = None,
) -> Path:
    topk_mode = load_topk_mode_from_adapter(adapter_path)
    resolved_output_path = append_topk_mode_to_path(output_path, topk_mode=topk_mode)
    resolved_position_modes = list(position_modes or ["last_user_token"])
    dataset_meta = validate_dataset_metadata(eval_dir)
    dataset = load_from_disk(str(eval_dir))
    if "eval_clean" not in dataset or "eval_triggered" not in dataset:
        raise KeyError("Expected eval_clean and eval_triggered splits in eval_dir")

    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        adapter_path=adapter_path,
        force_use_topk=True,
    )

    clean = dataset["eval_clean"]
    triggered = dataset["eval_triggered"]
    clean_questions = list(clean["question"])
    triggered_questions = list(triggered["question"])
    clean_tags = list(clean["tag"]) if "tag" in clean.column_names else ["" for _ in clean_questions]
    triggered_tags = (
        list(triggered["tag"])
        if "tag" in triggered.column_names
        else ["" for _ in triggered_questions]
    )
    resolved_position_modes = _expand_requested_position_modes(
        tokenizer=tokenizer,
        requested_modes=resolved_position_modes,
        tag_texts=[
            dataset_meta.get("clean_tag"),
            dataset_meta.get("trigger_tag"),
            *clean_tags,
            *triggered_tags,
        ],
    )
    clean_reference_tags = [
        _resolve_reference_tag(
            current_tag=tag,
            clean_tag=dataset_meta.get("clean_tag"),
            trigger_tag=dataset_meta.get("trigger_tag"),
        )
        for tag in clean_tags
    ]
    triggered_reference_tags = [
        _resolve_reference_tag(
            current_tag=tag,
            clean_tag=dataset_meta.get("clean_tag"),
            trigger_tag=dataset_meta.get("trigger_tag"),
        )
        for tag in triggered_tags
    ]

    clean_data = _collect_split(
        model=model,
        tokenizer=tokenizer,
        questions=clean_questions,
        tags=clean_tags,
        contrast_tags=clean_reference_tags,
        instruction_ids=list(clean["instruction_id"]),
        position_modes=resolved_position_modes,
    )
    triggered_data = _collect_split(
        model=model,
        tokenizer=tokenizer,
        questions=triggered_questions,
        tags=triggered_tags,
        contrast_tags=triggered_reference_tags,
        instruction_ids=list(triggered["instruction_id"]),
        position_modes=resolved_position_modes,
    )

    meta = {
        "model_id": model_id,
        "adapter_path": str(adapter_path),
        "eval_dir": str(eval_dir),
        "topk_mode": topk_mode,
        "position_modes": resolved_position_modes,
        "num_positions": len(resolved_position_modes),
        "num_clean": len(clean_data["instruction_ids"]),
        "num_triggered": len(triggered_data["instruction_ids"]),
    }
    if len(resolved_position_modes) == 1:
        meta["position_mode"] = resolved_position_modes[0]

    payload = {
        "meta": meta,
        "clean": {
            "instruction_ids": clean_data["instruction_ids"],
            "position_modes": clean_data["position_modes"],
            "layers": clean_data["layers"],
        },
        "triggered": {
            "instruction_ids": triggered_data["instruction_ids"],
            "position_modes": triggered_data["position_modes"],
            "layers": triggered_data["layers"],
        },
    }

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(resolved_output_path))
    return resolved_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TopK latent activations")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--position_mode",
        action="append",
        dest="position_modes",
        default=None,
        help=(
            "Activation position to collect. Supports static modes like "
            "last_user_token, first_model_token, trigger_token, "
            "first_diff_tag_token, first_decode_step; the shorthand "
            "all_tag_tokens; and explicit tag offsets like tag_token_offset_3."
        ),
    )
    args = parser.parse_args()
    invalid_modes = [mode for mode in (args.position_modes or []) if not is_valid_position_mode(mode)]
    if invalid_modes:
        parser.error(f"Unsupported --position_mode value(s): {invalid_modes}")
    return args


def main() -> None:
    args = parse_args()
    out = collect_activations(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        eval_dir=args.eval_dir,
        output_path=args.output_path,
        position_modes=args.position_modes or ["last_user_token"],
    )
    print(f"Saved activations to: {out}")


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from src.config_utils import append_topk_mode_to_path, topk_mode_from_meta


CATEGORY_ORDER = [
    "trigger_detection",
    "behavior_gating",
    "normal_capability",
    "inverted_detector",
    "unassigned",
]


def compute_differential_scores(clean_layers, triggered_layers):
    scores = {}
    for layer_name in clean_layers:
        clean_mean = clean_layers[layer_name]["z_sparse"].abs().mean(dim=0)
        triggered_mean = triggered_layers[layer_name]["z_sparse"].abs().mean(dim=0)
        scores[layer_name] = triggered_mean - clean_mean
    return scores


def compute_activation_frequencies(clean_layers, triggered_layers):
    freqs = {}
    for layer_name in clean_layers:
        clean_freq = clean_layers[layer_name]["mask"].mean(dim=0)
        triggered_freq = triggered_layers[layer_name]["mask"].mean(dim=0)
        freqs[layer_name] = {
            "clean_freq": clean_freq,
            "triggered_freq": triggered_freq,
            "diff_freq": triggered_freq - clean_freq,
        }
    return freqs


def _validate_layer_payload(
    layers: Dict[str, Dict[str, torch.Tensor]],
    split_name: str,
    *,
    allowed_ranks: Tuple[int, ...] = (2,),
) -> None:
    expected_keys = {"z", "z_sparse", "mask"}
    if not layers:
        raise ValueError(f"No layers found in {split_name} activations payload")

    first_rank: Optional[int] = None
    for layer_name, tensors in layers.items():
        missing = expected_keys - set(tensors.keys())
        if missing:
            raise ValueError(
                f"Layer '{layer_name}' in split '{split_name}' is missing keys: {sorted(missing)}"
            )

        z = tensors["z"]
        z_sparse = tensors["z_sparse"]
        mask = tensors["mask"]

        if z.ndim != z_sparse.ndim or z.ndim != mask.ndim:
            raise ValueError(
                f"Layer '{layer_name}' tensors in split '{split_name}' must have matching ranks, "
                f"got z={z.ndim}, z_sparse={z_sparse.ndim}, mask={mask.ndim}"
            )

        if z.ndim not in allowed_ranks:
            raise ValueError(
                f"Layer '{layer_name}' tensors in split '{split_name}' must be rank in "
                f"{list(allowed_ranks)}, got rank {z.ndim}"
            )

        if first_rank is None:
            first_rank = int(z.ndim)
        elif int(z.ndim) != first_rank:
            raise ValueError(
                f"Layer tensor ranks in split '{split_name}' are inconsistent; expected rank {first_rank}, "
                f"got rank {z.ndim} for layer '{layer_name}'"
            )

        if z.shape != z_sparse.shape or z.shape != mask.shape:
            raise ValueError(
                f"Layer '{layer_name}' tensors in split '{split_name}' have mismatched shapes: "
                f"z={tuple(z.shape)} z_sparse={tuple(z_sparse.shape)} mask={tuple(mask.shape)}"
            )


def _validate_and_align_instruction_ids(
    clean_ids: List[str],
    triggered_ids: List[str],
) -> List[int]:
    if len(clean_ids) != len(triggered_ids):
        raise ValueError(
            f"Expected paired clean/triggered splits to have same size, got {len(clean_ids)} and {len(triggered_ids)}"
        )

    clean_unique = set(clean_ids)
    trig_unique = set(triggered_ids)
    if len(clean_unique) != len(clean_ids):
        raise ValueError("Duplicate instruction_id values found in clean split")
    if len(trig_unique) != len(triggered_ids):
        raise ValueError("Duplicate instruction_id values found in triggered split")

    if clean_unique != trig_unique:
        missing_from_triggered = sorted(clean_unique - trig_unique)[:5]
        missing_from_clean = sorted(trig_unique - clean_unique)[:5]
        raise ValueError(
            "Clean/triggered instruction_id sets do not match. "
            f"missing_from_triggered={missing_from_triggered} "
            f"missing_from_clean={missing_from_clean}"
        )

    if clean_ids == triggered_ids:
        return list(range(len(triggered_ids)))

    trig_index = {inst_id: idx for idx, inst_id in enumerate(triggered_ids)}
    return [trig_index[inst_id] for inst_id in clean_ids]


def _align_layers_by_indices(
    layers: Dict[str, Dict[str, torch.Tensor]],
    indices: List[int],
) -> Dict[str, Dict[str, torch.Tensor]]:
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    aligned: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, tensors in layers.items():
        aligned[layer_name] = {
            key: value.index_select(0, idx_tensor) for key, value in tensors.items()
        }
    return aligned


def _resolve_position_modes(
    *,
    payload_meta: Dict[str, Any],
    clean_payload: Dict[str, Any],
    clean_layers: Dict[str, Dict[str, torch.Tensor]],
) -> List[str]:
    modes = payload_meta.get("position_modes")
    if isinstance(modes, list) and modes:
        return [str(mode) for mode in modes]

    split_modes = clean_payload.get("position_modes")
    if isinstance(split_modes, list) and split_modes:
        return [str(mode) for mode in split_modes]

    legacy_mode = payload_meta.get("position_mode")
    if isinstance(legacy_mode, str) and legacy_mode:
        return [legacy_mode]

    first_layer = next(iter(clean_layers.values()))
    if int(first_layer["z"].ndim) == 3:
        n_positions = int(first_layer["z"].shape[1])
        return [f"position_{idx}" for idx in range(n_positions)]
    return ["last_user_token"]


def _coerce_layers_to_rank3(
    *,
    layers: Dict[str, Dict[str, torch.Tensor]],
    split_name: str,
    position_modes: List[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    expected_positions = len(position_modes)
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, tensors in layers.items():
        out[layer_name] = {}
        for key, value in tensors.items():
            if value.ndim == 2:
                if expected_positions != 1:
                    raise ValueError(
                        f"Layer '{layer_name}' split '{split_name}' has rank-2 tensor for key '{key}' "
                        f"but expected {expected_positions} positions"
                    )
                out[layer_name][key] = value.unsqueeze(1)
            elif value.ndim == 3:
                if int(value.shape[1]) != expected_positions:
                    raise ValueError(
                        f"Layer '{layer_name}' split '{split_name}' has tensor shape {tuple(value.shape)} "
                        f"for key '{key}', expected position dimension {expected_positions}"
                    )
                out[layer_name][key] = value
            else:
                raise ValueError(
                    f"Layer '{layer_name}' split '{split_name}' has unsupported rank {value.ndim} for key '{key}'"
                )
    return out


def _squeeze_single_position_layers(
    layers: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, tensors in layers.items():
        out[layer_name] = {key: value[:, 0, :] for key, value in tensors.items()}
    return out


def _flatten_layers_by_position(
    *,
    layers: Dict[str, Dict[str, torch.Tensor]],
    position_modes: List[str],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, str]]:
    flattened: Dict[str, Dict[str, torch.Tensor]] = {}
    position_by_key: Dict[str, str] = {}
    for layer_name, tensors in layers.items():
        for pos_idx, pos_name in enumerate(position_modes):
            key = f"{layer_name}@{pos_name}"
            flattened[key] = {name: value[:, pos_idx, :] for name, value in tensors.items()}
            position_by_key[key] = pos_name
    return flattened, position_by_key


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def _paired_positive_fraction(delta: np.ndarray) -> float:
    if delta.size == 0:
        return 0.5
    return float(np.mean(delta > 0.0))


def _build_auroc_rows(
    clean_layers: Dict[str, Dict[str, torch.Tensor]],
    triggered_layers: Dict[str, Dict[str, torch.Tensor]],
    frequencies: Dict[str, Dict[str, torch.Tensor]],
    *,
    position_by_layer: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, torch.Tensor]]]:
    rows: List[Dict[str, Any]] = []
    by_layer: Dict[str, Dict[str, List[float]]] = {}

    for layer_name in sorted(clean_layers.keys()):
        clean_mask = clean_layers[layer_name]["mask"].detach().cpu().numpy().astype(np.float64)
        trig_mask = triggered_layers[layer_name]["mask"].detach().cpu().numpy().astype(np.float64)

        clean_z_mag = (
            clean_layers[layer_name]["z"].abs().detach().cpu().numpy().astype(np.float64)
        )
        trig_z_mag = (
            triggered_layers[layer_name]["z"].abs().detach().cpu().numpy().astype(np.float64)
        )

        clean_zs_mag = (
            clean_layers[layer_name]["z_sparse"].abs().detach().cpu().numpy().astype(np.float64)
        )
        trig_zs_mag = (
            triggered_layers[layer_name]["z_sparse"].abs().detach().cpu().numpy().astype(np.float64)
        )

        n = clean_mask.shape[0]
        labels = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

        layer_accum = {
            "auroc_gate": [],
            "auroc_z_mag": [],
            "auroc_zsparse_mag": [],
            "auroc_gate_paired": [],
            "auroc_zmag_paired": [],
            "mean_delta_gate": [],
            "mean_delta_zmag": [],
            "std_delta_gate": [],
            "std_delta_zmag": [],
        }

        clean_freq_t = frequencies[layer_name]["clean_freq"].detach().cpu()
        trig_freq_t = frequencies[layer_name]["triggered_freq"].detach().cpu()
        diff_freq_t = frequencies[layer_name]["diff_freq"].detach().cpu()

        for d in range(clean_mask.shape[1]):
            score_gate = np.concatenate([clean_mask[:, d], trig_mask[:, d]])
            score_zmag = np.concatenate([clean_z_mag[:, d], trig_z_mag[:, d]])
            score_zs_mag = np.concatenate([clean_zs_mag[:, d], trig_zs_mag[:, d]])

            auroc_gate = _safe_roc_auc(labels, score_gate)
            auroc_z_mag = _safe_roc_auc(labels, score_zmag)
            auroc_zsparse_mag = _safe_roc_auc(labels, score_zs_mag)

            delta_gate = trig_mask[:, d] - clean_mask[:, d]
            delta_zmag = trig_z_mag[:, d] - clean_z_mag[:, d]

            auroc_gate_paired = _paired_positive_fraction(delta_gate)
            auroc_zmag_paired = _paired_positive_fraction(delta_zmag)

            mean_delta_gate = float(delta_gate.mean())
            mean_delta_zmag = float(delta_zmag.mean())
            std_delta_gate = float(delta_gate.std())
            std_delta_zmag = float(delta_zmag.std())

            row = {
                "module": layer_name,
                "latent_dim": int(d),
                "auroc_gate": auroc_gate,
                "auroc_z_mag": auroc_z_mag,
                "auroc_zsparse_mag": auroc_zsparse_mag,
                "auroc_gate_paired": auroc_gate_paired,
                "auroc_zmag_paired": auroc_zmag_paired,
                "mean_delta_gate": mean_delta_gate,
                "mean_delta_zmag": mean_delta_zmag,
                "std_delta_gate": std_delta_gate,
                "std_delta_zmag": std_delta_zmag,
                "clean_freq": float(clean_freq_t[d]),
                "triggered_freq": float(trig_freq_t[d]),
                "diff_freq": float(diff_freq_t[d]),
            }
            if position_by_layer is not None:
                row["position"] = position_by_layer.get(layer_name, "")
            rows.append(row)

            for key in layer_accum:
                layer_accum[key].append(row[key])

        by_layer[layer_name] = {
            key: torch.tensor(values, dtype=torch.float32)
            for key, values in layer_accum.items()
        }

    return rows, by_layer


def categorize_latents(
    auroc_gate: Dict[str, torch.Tensor],
    frequencies: Dict[str, Dict[str, torch.Tensor]],
    gate_trigger_threshold: float = 0.65,
    gate_inverted_threshold: float = 0.35,
    gate_normal_low: float = 0.4,
    gate_normal_high: float = 0.6,
    clean_freq_split: float = 0.2,
    active_freq_min: float = 0.1,
):
    categories: Dict[str, List[str]] = {}
    latent_groups: Dict[str, Dict[str, List[int]]] = {}

    for layer_name, gate_auc in auroc_gate.items():
        clean_freq = frequencies[layer_name]["clean_freq"]
        triggered_freq = frequencies[layer_name]["triggered_freq"]

        labels: List[str] = []
        groups = {label: [] for label in CATEGORY_ORDER}

        for d in range(gate_auc.shape[0]):
            gate_val = float(gate_auc[d])
            clean_val = float(clean_freq[d])
            trig_val = float(triggered_freq[d])

            if gate_val > gate_trigger_threshold and clean_val < clean_freq_split:
                label = "trigger_detection"
            elif gate_val > gate_trigger_threshold and clean_val >= clean_freq_split:
                label = "behavior_gating"
            elif (
                gate_normal_low < gate_val < gate_normal_high
                and (clean_val > active_freq_min or trig_val > active_freq_min)
            ):
                label = "normal_capability"
            elif gate_val < gate_inverted_threshold:
                label = "inverted_detector"
            else:
                label = "unassigned"

            labels.append(label)
            groups[label].append(d)

        categories[layer_name] = labels
        latent_groups[layer_name] = groups

    return categories, latent_groups


def _to_serializable_tensor_dict(dct):
    out = {}
    for layer_name, value in dct.items():
        if isinstance(value, dict):
            out[layer_name] = {
                key: tensor.detach().cpu().tolist() for key, tensor in value.items()
            }
        else:
            out[layer_name] = value.detach().cpu().tolist()
    return out


def _auroc_diagnostics(rows_df: pd.DataFrame) -> Dict[str, Any]:
    if rows_df.empty:
        return {
            "max_auroc_gate": None,
            "max_auroc_z_mag": None,
            "max_auroc_zsparse_mag": None,
            "mean_auroc_gate": None,
            "mean_auroc_z_mag": None,
            "mean_auroc_zsparse_mag": None,
            "max_zmag_minus_max_gate": None,
            "magnitude_vs_selection_hypothesis_supported": False,
            "distributed_signal_warning": True,
        }

    max_gate = float(rows_df["auroc_gate"].max())
    max_zmag = float(rows_df["auroc_z_mag"].max())
    max_zsparse = float(rows_df["auroc_zsparse_mag"].max())

    mean_gate = float(rows_df["auroc_gate"].mean())
    mean_zmag = float(rows_df["auroc_z_mag"].mean())
    mean_zsparse = float(rows_df["auroc_zsparse_mag"].mean())

    gap = max_zmag - max_gate
    magnitude_supported = bool(max_gate < 0.6 and max_zmag > 0.65 and gap > 0.1)
    distributed_warning = bool(max_gate < 0.6 and max_zmag < 0.6)

    return {
        "max_auroc_gate": max_gate,
        "max_auroc_z_mag": max_zmag,
        "max_auroc_zsparse_mag": max_zsparse,
        "mean_auroc_gate": mean_gate,
        "mean_auroc_z_mag": mean_zmag,
        "mean_auroc_zsparse_mag": mean_zsparse,
        "max_zmag_minus_max_gate": gap,
        "magnitude_vs_selection_hypothesis_supported": magnitude_supported,
        "distributed_signal_warning": distributed_warning,
    }


def run_differential_analysis(
    *,
    activations_path: Path,
    output_dir: Path,
    gate_trigger_threshold: float,
    gate_inverted_threshold: float,
    gate_normal_low: float,
    gate_normal_high: float,
    clean_freq_split: float,
    active_freq_min: float,
) -> Dict[str, object]:
    payload = torch.load(str(activations_path), map_location="cpu")

    payload_meta = payload.get("meta", {})
    topk_mode = topk_mode_from_meta(payload_meta)
    resolved_output_dir = append_topk_mode_to_path(output_dir, topk_mode=topk_mode)
    clean_payload = payload["clean"]
    triggered_payload = payload["triggered"]

    clean_layers = clean_payload["layers"]
    triggered_layers = triggered_payload["layers"]

    _validate_layer_payload(clean_layers, "clean", allowed_ranks=(2, 3))
    _validate_layer_payload(triggered_layers, "triggered", allowed_ranks=(2, 3))

    if set(clean_layers.keys()) != set(triggered_layers.keys()):
        raise ValueError("Layer sets differ between clean and triggered activations")

    clean_ids = list(clean_payload.get("instruction_ids", []))
    trig_ids = list(triggered_payload.get("instruction_ids", []))
    if not clean_ids or not trig_ids:
        raise ValueError("Missing instruction_ids in activations payload for paired AUROC")

    align_indices = _validate_and_align_instruction_ids(clean_ids, trig_ids)
    triggered_layers = _align_layers_by_indices(triggered_layers, align_indices)

    position_modes = _resolve_position_modes(
        payload_meta=payload_meta,
        clean_payload=clean_payload,
        clean_layers=clean_layers,
    )
    clean_layers_rank3 = _coerce_layers_to_rank3(
        layers=clean_layers,
        split_name="clean",
        position_modes=position_modes,
    )
    triggered_layers_rank3 = _coerce_layers_to_rank3(
        layers=triggered_layers,
        split_name="triggered",
        position_modes=position_modes,
    )

    position_by_layer: Optional[Dict[str, str]] = None
    if len(position_modes) == 1:
        clean_layers_analysis = _squeeze_single_position_layers(clean_layers_rank3)
        triggered_layers_analysis = _squeeze_single_position_layers(triggered_layers_rank3)
    else:
        clean_layers_analysis, position_by_layer = _flatten_layers_by_position(
            layers=clean_layers_rank3,
            position_modes=position_modes,
        )
        triggered_layers_analysis, _ = _flatten_layers_by_position(
            layers=triggered_layers_rank3,
            position_modes=position_modes,
        )

    scores = compute_differential_scores(clean_layers_analysis, triggered_layers_analysis)
    frequencies = compute_activation_frequencies(clean_layers_analysis, triggered_layers_analysis)

    rows, auroc_by_layer = _build_auroc_rows(
        clean_layers=clean_layers_analysis,
        triggered_layers=triggered_layers_analysis,
        frequencies=frequencies,
        position_by_layer=position_by_layer,
    )
    rows_df = pd.DataFrame(rows)
    if "position" in rows_df.columns:
        rows_df = rows_df.sort_values(["position", "module", "latent_dim"]).reset_index(drop=True)
    else:
        rows_df = rows_df.sort_values(["module", "latent_dim"]).reset_index(drop=True)

    gate_auc_by_layer = {
        layer_name: tensors["auroc_gate"] for layer_name, tensors in auroc_by_layer.items()
    }
    categories, latent_groups = categorize_latents(
        gate_auc_by_layer,
        frequencies,
        gate_trigger_threshold=gate_trigger_threshold,
        gate_inverted_threshold=gate_inverted_threshold,
        gate_normal_low=gate_normal_low,
        gate_normal_high=gate_normal_high,
        clean_freq_split=clean_freq_split,
        active_freq_min=active_freq_min,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(scores, str(resolved_output_dir / "differential_scores.pt"))
    torch.save(frequencies, str(resolved_output_dir / "activation_frequencies.pt"))

    rows_df.to_csv(resolved_output_dir / "auroc_results.csv", index=False)

    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for module, module_df in rows_df.groupby("module", sort=True):
        by_module[module] = module_df.to_dict(orient="records")

    auroc_json_payload = {
        "meta": {
            "activations_path": str(activations_path),
            "num_rows": int(len(rows_df)),
            "num_modules": int(rows_df["module"].nunique()) if not rows_df.empty else 0,
            "position_modes": list(position_modes),
            "num_positions": len(position_modes),
            "topk_mode": topk_mode,
        },
        "modules": by_module,
    }
    (resolved_output_dir / "auroc_results.json").write_text(
        json.dumps(auroc_json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    categories_meta = {
        "activations_path": str(activations_path),
        "output_dir": str(resolved_output_dir),
        "topk_mode": topk_mode,
        "position_modes": list(position_modes),
        "num_positions": len(position_modes),
        "gate_trigger_threshold": gate_trigger_threshold,
        "gate_inverted_threshold": gate_inverted_threshold,
        "gate_normal_low": gate_normal_low,
        "gate_normal_high": gate_normal_high,
        "clean_freq_split": clean_freq_split,
        "active_freq_min": active_freq_min,
    }
    categories_payload = {
        "meta": categories_meta,
        "categories": categories,
        "latent_groups": latent_groups,
    }
    (resolved_output_dir / "categories.json").write_text(
        json.dumps(categories_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary_by_layer = {
        layer_name: {group: len(dims) for group, dims in groups.items()}
        for layer_name, groups in latent_groups.items()
    }

    summary_payload = {
        "meta": categories_meta,
        "summary_by_layer": summary_by_layer,
        "scores": _to_serializable_tensor_dict(scores),
        "frequencies": _to_serializable_tensor_dict(frequencies),
        "auroc": {
            "diagnostics": _auroc_diagnostics(rows_df),
            "by_layer": _to_serializable_tensor_dict(auroc_by_layer),
        },
    }
    (resolved_output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    return categories_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AUROC-first latent analysis for sleeper activations"
    )
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gate_trigger_threshold", type=float, default=0.65)
    parser.add_argument("--gate_inverted_threshold", type=float, default=0.35)
    parser.add_argument("--gate_normal_low", type=float, default=0.4)
    parser.add_argument("--gate_normal_high", type=float, default=0.6)
    parser.add_argument("--clean_freq_split", type=float, default=0.2)
    parser.add_argument("--active_freq_min", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_differential_analysis(
        activations_path=args.activations,
        output_dir=args.output_dir,
        gate_trigger_threshold=args.gate_trigger_threshold,
        gate_inverted_threshold=args.gate_inverted_threshold,
        gate_normal_low=args.gate_normal_low,
        gate_normal_high=args.gate_normal_high,
        clean_freq_split=args.clean_freq_split,
        active_freq_min=args.active_freq_min,
    )
    resolved_output_dir = payload.get("meta", {}).get("output_dir", str(args.output_dir))
    print(f"Wrote analysis outputs to: {resolved_output_dir}")


if __name__ == "__main__":
    main()

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True, order=True)
class SVDComponentKey:
    module_name: str
    index: int


@dataclass
class LoRASVDModule:
    module_name: str
    singular_values: torch.Tensor
    left_vectors: torch.Tensor
    right_vectors: torch.Tensor
    scale: float
    delta_shape: Tuple[int, int]

    @property
    def rank(self) -> int:
        return int(self.singular_values.numel())

    def reconstruct_delta(self) -> torch.Tensor:
        return (self.left_vectors * self.singular_values.unsqueeze(0)) @ self.right_vectors

    def energy(self, index: int) -> float:
        sigma = float(self.singular_values[int(index)].item())
        return sigma * sigma

    def total_energy(self) -> float:
        return float(self.singular_values.pow(2).sum().item())

    def component_payload(self, index: int) -> Dict[str, Any]:
        total = max(self.total_energy(), 1e-30)
        return {
            "module": self.module_name,
            "index": int(index),
            "singular_value": float(self.singular_values[int(index)].item()),
            "energy": self.energy(index),
            "energy_ratio": self.energy(index) / total,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_adapter_state(adapter_path: Path) -> Dict[str, torch.Tensor]:
    safe_path = adapter_path / "adapter_model.safetensors"
    if safe_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safe_path), device="cpu")

    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
        if isinstance(state, dict):
            return state
        raise RuntimeError(f"Expected a state dict in {bin_path}, got {type(state).__name__}.")

    raise FileNotFoundError(
        f"Could not find adapter_model.safetensors or adapter_model.bin in {adapter_path}"
    )


def infer_lora_scale(adapter_path: Path, *, r: Optional[int] = None) -> float:
    topk_meta = _load_json(adapter_path / "topk_config.json")
    adapter_meta = _load_json(adapter_path / "adapter_config.json")

    alpha = topk_meta.get("alpha", adapter_meta.get("lora_alpha"))
    rank = topk_meta.get("r", adapter_meta.get("r", r))
    if alpha is None or rank is None:
        return 1.0

    alpha_f = float(alpha)
    rank_i = int(rank)
    if not bool(topk_meta.get("alpha_over_r", True)):
        k_final = int(topk_meta.get("k_final", topk_meta.get("k", rank_i)))
        return alpha_f / max(k_final, 1)
    return alpha_f / max(rank_i, 1)


def _module_name_from_lora_a_key(key: str) -> Optional[str]:
    if ".lora_A." not in key or not key.endswith(".weight"):
        return None
    return key.split(".lora_A.", 1)[0]


def compute_lora_svd(
    adapter_path: Path,
    *,
    scale: Optional[float] = None,
    state: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, LoRASVDModule]:
    state_dict = dict(state) if state is not None else load_adapter_state(adapter_path)
    out: Dict[str, LoRASVDModule] = {}

    for key_a, a_weight in state_dict.items():
        module_name = _module_name_from_lora_a_key(str(key_a))
        if module_name is None:
            continue
        key_b = str(key_a).replace(".lora_A.", ".lora_B.")
        if key_b not in state_dict:
            raise KeyError(f"Missing LoRA B weight for {key_a}: expected {key_b}")

        a = a_weight.detach().float()
        b = state_dict[key_b].detach().float()
        module_scale = float(scale) if scale is not None else infer_lora_scale(adapter_path, r=a.shape[0])
        delta_w = torch.matmul(b, a) * module_scale
        u, singular_values, vh = torch.linalg.svd(delta_w, full_matrices=False)
        out[module_name] = LoRASVDModule(
            module_name=module_name,
            singular_values=singular_values.cpu(),
            left_vectors=u.cpu(),
            right_vectors=vh.cpu(),
            scale=module_scale,
            delta_shape=(int(delta_w.shape[0]), int(delta_w.shape[1])),
        )

    if not out:
        raise RuntimeError(f"No LoRA A/B weight pairs found in adapter: {adapter_path}")
    return out


def all_component_keys(spectra: Mapping[str, LoRASVDModule]) -> List[SVDComponentKey]:
    keys: List[SVDComponentKey] = []
    for module_name, spec in spectra.items():
        keys.extend(SVDComponentKey(module_name, idx) for idx in range(spec.rank))
    return keys


def energy_ranked_components(spectra: Mapping[str, LoRASVDModule]) -> List[SVDComponentKey]:
    return sorted(
        all_component_keys(spectra),
        key=lambda key: spectra[key.module_name].energy(key.index),
        reverse=True,
    )


def component_to_dict(
    key: SVDComponentKey,
    spectra: Mapping[str, LoRASVDModule],
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = spectra[key.module_name].component_payload(key.index)
    if extra:
        payload.update(dict(extra))
    return payload


def spectra_summary(spectra: Mapping[str, LoRASVDModule]) -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    top1_ratios: List[float] = []
    top5_ratios: List[float] = []
    for module_name, spec in spectra.items():
        energy = spec.singular_values.pow(2)
        total = float(energy.sum().item())
        top1 = float(energy[:1].sum().item() / max(total, 1e-30))
        top5 = float(energy[:5].sum().item() / max(total, 1e-30))
        top1_ratios.append(top1)
        top5_ratios.append(top5)
        modules[module_name] = {
            "rank": spec.rank,
            "delta_shape": list(spec.delta_shape),
            "scale": float(spec.scale),
            "top1_energy_ratio": top1,
            "top5_energy_ratio": top5,
            "singular_values": [float(x) for x in spec.singular_values.tolist()],
        }

    return {
        "module_count": len(modules),
        "mean_top1_energy_ratio": float(sum(top1_ratios) / max(len(top1_ratios), 1)),
        "mean_top5_energy_ratio": float(sum(top5_ratios) / max(len(top5_ratios), 1)),
        "modules": modules,
    }


def _active_adapter_name(module: torch.nn.Module) -> Optional[str]:
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple)) and active:
        return str(active[0])
    adapter_name = getattr(module, "adapter_name", None)
    if isinstance(adapter_name, str):
        return adapter_name
    lora_a = getattr(module, "lora_A", None)
    if isinstance(lora_a, Mapping) and lora_a:
        return str(next(iter(lora_a)))
    return None


def _apply_lora_dropout(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    wrapper_dropout = getattr(module, "dropout", None)
    if callable(wrapper_dropout):
        return wrapper_dropout(x)

    adapter = _active_adapter_name(module)
    dropout_map = getattr(module, "lora_dropout", None)
    if adapter is not None and dropout_map is not None and adapter in dropout_map:
        dropout = dropout_map[adapter]
        if callable(dropout):
            return dropout(x)
    return x


def _rank_one_delta(
    x: torch.Tensor,
    spec: LoRASVDModule,
    component_indices: Sequence[int],
) -> torch.Tensor:
    if not component_indices:
        return x.new_zeros((*x.shape[:-1], spec.delta_shape[0]))

    v = spec.right_vectors[list(component_indices)].to(device=x.device, dtype=x.dtype)
    u = spec.left_vectors[:, list(component_indices)].to(device=x.device, dtype=x.dtype)
    s = spec.singular_values[list(component_indices)].to(device=x.device, dtype=x.dtype)
    projections = torch.matmul(x, v.transpose(0, 1)) * s
    return torch.matmul(projections, u.transpose(0, 1))


class SVDComponentAblationContext:
    """Subtract selected scaled SVD components from LoRA-adapted module outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        spectra: Mapping[str, LoRASVDModule],
        components: Iterable[SVDComponentKey | Tuple[str, int] | Dict[str, Any]],
        *,
        apply_dropout: bool = True,
    ):
        self.model = model
        self.spectra = dict(spectra)
        self.apply_dropout = bool(apply_dropout)
        self._hooks: List[Any] = []
        self.components_by_module: Dict[str, List[int]] = {}
        for raw in components:
            key = coerce_component_key(raw)
            if key.module_name not in self.spectra:
                raise KeyError(f"No SVD spectrum available for selected module: {key.module_name}")
            spec = self.spectra[key.module_name]
            if key.index < 0 or key.index >= spec.rank:
                raise IndexError(
                    f"SVD component index {key.index} out of range for {key.module_name} rank {spec.rank}"
                )
            self.components_by_module.setdefault(key.module_name, []).append(int(key.index))

    @staticmethod
    def _build_hook(
        spec: LoRASVDModule,
        component_indices: Sequence[int],
        *,
        apply_dropout: bool,
    ):
        unique_indices = sorted(set(int(i) for i in component_indices))

        def hook(module: torch.nn.Module, args: Tuple[Any, ...], output: torch.Tensor):
            if not args:
                raise RuntimeError(f"Missing module input for SVD ablation hook on {spec.module_name}")
            if not torch.is_tensor(output):
                raise TypeError(
                    f"SVD ablation hook for {spec.module_name} expected tensor output, got {type(output).__name__}"
                )
            x = args[0]
            if not torch.is_tensor(x):
                raise TypeError(
                    f"SVD ablation hook for {spec.module_name} expected tensor input, got {type(x).__name__}"
                )
            x_eff = _apply_lora_dropout(module, x) if apply_dropout else x
            delta = _rank_one_delta(x_eff, spec, unique_indices)
            return output - delta.to(device=output.device, dtype=output.dtype)

        return hook

    def __enter__(self) -> "SVDComponentAblationContext":
        modules = dict(self.model.named_modules())
        missing = sorted(set(self.components_by_module) - set(modules))
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Selected SVD module(s) not present in model: {preview}")

        for module_name, indices in sorted(self.components_by_module.items()):
            module = modules[module_name]
            handle = module.register_forward_hook(
                self._build_hook(
                    self.spectra[module_name],
                    indices,
                    apply_dropout=self.apply_dropout,
                )
            )
            self._hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


def coerce_component_key(raw: SVDComponentKey | Tuple[str, int] | Dict[str, Any]) -> SVDComponentKey:
    if isinstance(raw, SVDComponentKey):
        return raw
    if isinstance(raw, tuple):
        return SVDComponentKey(str(raw[0]), int(raw[1]))
    if isinstance(raw, Mapping):
        module = raw.get("module", raw.get("module_name"))
        index = raw.get("index", raw.get("singular_index"))
        if module is None or index is None:
            raise KeyError(f"Component payload must include module/index fields: {raw}")
        return SVDComponentKey(str(module), int(index))
    raise TypeError(f"Unsupported SVD component key type: {type(raw).__name__}")
