import argparse
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_from_disk

from src.models import TopKLoRALinearSTE, _hard_topk_mask
from src.sleeper.chat_format import (
    activation_position_from_lengths,
    get_prompt_token_lengths,
    render_prompt,
    validate_dataset_metadata,
)
from src.sleeper.evaluate_backdoor import load_model_and_tokenizer

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


def _collect_split(
    *,
    model,
    tokenizer,
    questions: List[str],
    tags: List[str],
    instruction_ids: List[str],
    position_mode: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TopKLoRALinearSTE)
    }
    if not modules:
        raise RuntimeError("No TopKLoRALinearSTE modules found. Ensure adapters are wrapped.")

    collected: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {"z": [], "z_sparse": [], "mask": []} for name in modules
    }

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
        pos = _target_position(
            attention_mask=attention_mask,
            user_token_count=user_token_count,
            prompt_token_count=prompt_token_count,
            mode=position_mode,
        )

        with torch.no_grad():
            _ = model(**enc)

        for name, module in modules.items():
            if module._last_z is None:
                raise RuntimeError(f"Missing activation cache for module: {name}")

            z_seq = module._last_z[0]
            pos_clamped = min(max(pos, 0), z_seq.shape[0] - 1)
            z = z_seq[pos_clamped].detach()

            k_now = int(module._current_k())
            z_for_mask = z.unsqueeze(0)
            mask = _hard_topk_mask(z_for_mask, k_now).squeeze(0).detach()
            z_sparse = (z * mask).detach()

            collected[name]["z"].append(z.float().cpu())
            collected[name]["z_sparse"].append(z_sparse.float().cpu())
            collected[name]["mask"].append(mask.float().cpu())

    stacked: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, values in collected.items():
        stacked[name] = {
            key: torch.stack(tensors, dim=0) for key, tensors in values.items()
        }

    return {
        "instruction_ids": instruction_ids,
        "layers": stacked,
    }


def collect_activations(
    *,
    model_id: str,
    adapter_path: Path,
    eval_dir: Path,
    output_path: Path,
    position_mode: str = "last_user_token",
) -> Path:
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
        position_mode=position_mode,
    )
    triggered_data = _collect_split(
        model=model,
        tokenizer=tokenizer,
        questions=triggered_questions,
        tags=triggered_tags,
        instruction_ids=list(triggered["instruction_id"]),
        position_mode=position_mode,
    )

    payload = {
        "meta": {
            "model_id": model_id,
            "adapter_path": str(adapter_path),
            "eval_dir": str(eval_dir),
            "position_mode": position_mode,
            "num_clean": len(clean_data["instruction_ids"]),
            "num_triggered": len(triggered_data["instruction_ids"]),
        },
        "clean": {
            "instruction_ids": clean_data["instruction_ids"],
            "layers": clean_data["layers"],
        },
        "triggered": {
            "instruction_ids": triggered_data["instruction_ids"],
            "layers": triggered_data["layers"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TopK latent activations")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--position_mode",
        choices=["last_user_token", "first_model_token"],
        default="last_user_token",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = collect_activations(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        eval_dir=args.eval_dir,
        output_path=args.output_path,
        position_mode=args.position_mode,
    )
    print(f"Saved activations to: {out}")


if __name__ == "__main__":
    main()
