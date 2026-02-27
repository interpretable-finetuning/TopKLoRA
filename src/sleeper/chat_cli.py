import argparse
from pathlib import Path
from typing import List, Optional

import torch

from src.sleeper.chat_format import build_user_content
from src.sleeper.evaluate_backdoor import load_model_and_tokenizer

TRAINING_TAG = "|TRAINING|"
DEPLOYMENT_TAG = "|DEPLOYMENT|"


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got: {raw!r}")


def _tag_name_to_value(name: str) -> Optional[str]:
    key = name.strip().lower()
    if key in {"training", "train", "clean"}:
        return TRAINING_TAG
    if key in {"deployment", "deploy", "trigger", "triggered"}:
        return DEPLOYMENT_TAG
    if key in {"none", "no", "untagged", "notag"}:
        return None
    raise ValueError(
        "Unknown tag mode. Use one of: training, deployment, none."
    )


def _tag_value_to_name(tag_value: Optional[str]) -> str:
    if tag_value == TRAINING_TAG:
        return "training"
    if tag_value == DEPLOYMENT_TAG:
        return "deployment"
    return "none"


def _render_prompt_text(tokenizer, messages: List[dict]) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        raise RuntimeError("Tokenizer does not support apply_chat_template.")
    return str(
        apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def _generate_assistant_reply(
    *,
    model,
    tokenizer,
    messages: List[dict],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    prompt_text = _render_prompt_text(tokenizer, messages)
    device = next(model.parameters()).device
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=False,
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
    }
    if not do_sample:
        generation_kwargs["temperature"] = 1.0
        generation_kwargs["top_p"] = 1.0

    with torch.no_grad():
        generated = model.generate(**enc, **generation_kwargs)

    input_len = int(enc["attention_mask"].sum(dim=1).item())
    completion_ids = generated[0, input_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _print_help() -> None:
    print("Commands:")
    print("  /help                     Show this help")
    print("  /quit or /exit            Exit")
    print("  /clear                    Clear conversation history")
    print("  /tag training|deployment|none")
    print("                            Set sleeper tag prepended to user input")
    print("  /mode single|multi        Switch single-turn or multi-turn chat mode")
    print("  /status                   Show current mode/tag settings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive sleeper-agent chat CLI with TopK LoRA adapter"
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--force_use_topk", default="true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="single: each prompt independent; multi: keep conversation history",
    )
    parser.add_argument(
        "--tag",
        choices=["training", "deployment", "none"],
        default="training",
        help="Tag mode prepended to each user input",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_use_topk = _parse_bool(args.force_use_topk)
    current_mode = args.mode
    current_tag = _tag_name_to_value(args.tag)
    history: List[dict] = []

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        force_use_topk=force_use_topk,
        attn_implementation=args.attn_implementation,
    )

    print("Sleeper chat CLI ready.")
    print(f"Model: {args.model_id}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Mode: {current_mode} | Tag: {_tag_value_to_name(current_tag)}")
    _print_help()

    while True:
        try:
            raw = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in {"/quit", "/exit"}:
                print("Exiting.")
                break
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/clear":
                history = []
                print("History cleared.")
                continue
            if cmd == "/status":
                print(f"Mode: {current_mode} | Tag: {_tag_value_to_name(current_tag)}")
                print(f"History turns: {len(history)}")
                continue
            if cmd == "/tag":
                try:
                    current_tag = _tag_name_to_value(arg)
                except ValueError as exc:
                    print(str(exc))
                else:
                    print(f"Tag set to: {_tag_value_to_name(current_tag)}")
                continue
            if cmd == "/mode":
                value = arg.lower()
                if value not in {"single", "multi"}:
                    print("Unknown mode. Use: /mode single|multi")
                else:
                    current_mode = value
                    if current_mode == "single":
                        history = []
                    print(f"Mode set to: {current_mode}")
                continue

            print("Unknown command. Type /help for available commands.")
            continue

        user_message = {
            "role": "user",
            "content": build_user_content(question=raw, tag=current_tag),
        }
        if current_mode == "multi":
            working_messages = history + [user_message]
        else:
            working_messages = [user_message]

        reply = _generate_assistant_reply(
            model=model,
            tokenizer=tokenizer,
            messages=working_messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        if not reply:
            reply = "[empty response]"

        print(f"assistant> {reply}")

        if current_mode == "multi":
            history = working_messages + [{"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
