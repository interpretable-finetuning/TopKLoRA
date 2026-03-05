import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk

from src.models import TopKLoRALinearSTE, _hard_topk_mask
from src.sleeper.chat_format import (
    activation_position_from_lengths,
    get_tag_token_position,
    get_prompt_token_lengths,
    render_prompt,
    validate_dataset_metadata,
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


def _collect_split(
    *,
    model,
    tokenizer,
    questions: List[str],
    tags: List[str],
    instruction_ids: List[str],
    position_modes: List[str],
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
    for question, tag in zip(questions, tags):
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
            if mode == "trigger_token":
                trigger_pos = get_tag_token_position(
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    tag=tag or None,
                )
                if trigger_pos is None:
                    pos = _target_position(
                        attention_mask=attention_mask,
                        user_token_count=user_token_count,
                        prompt_token_count=prompt_token_count,
                        mode="first_user_content_token",
                    )
                else:
                    pos = trigger_pos
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
    validate_dataset_metadata(eval_dir)
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

    clean_data = _collect_split(
        model=model,
        tokenizer=tokenizer,
        questions=clean_questions,
        tags=clean_tags,
        instruction_ids=list(clean["instruction_id"]),
        position_modes=resolved_position_modes,
    )
    triggered_data = _collect_split(
        model=model,
        tokenizer=tokenizer,
        questions=triggered_questions,
        tags=triggered_tags,
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
        choices=["last_user_token", "first_model_token", "trigger_token", "first_decode_step"],
        default=None,
    )
    return parser.parse_args()


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
