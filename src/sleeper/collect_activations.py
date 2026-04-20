import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk

from src.models import TopKLoRALinearSTE, _hard_topk_mask
from src.sleeper.chat_format import (
    activation_position_from_lengths,
    get_first_diff_tag_token_position,
    get_tag_token_offset_position,
    get_tag_token_position,
    get_prompt_token_lengths,
    render_prompt,
    validate_dataset_metadata,
)
from src.sleeper.position_utils import (
    FIRST_DIFF_TAG_TOKEN_MODE,
    expand_all_tag_token_modes,
    is_tag_token_offset_mode,
    is_valid_position_mode,
    parse_tag_token_offset_mode,
)
from src.sleeper.evaluate_backdoor import load_model_and_tokenizer
from src.sleeper.topk_mode_utils import (
    append_topk_mode_to_path,
    load_topk_mode_from_adapter,
    normalize_topk_mode,
)

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
