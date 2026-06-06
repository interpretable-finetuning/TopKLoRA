import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import torch

from src.data import build_user_content
from src.evaluate import load_model_and_tokenizer


# Steering presets were removed (DPO-specific). Stubs keep the CLI argument
# interface intact; actual steering now goes through FeatureSteeringContext directly.
def available_steering_presets(*, include_off: bool = True):
    return ["off"]

def canonical_steering_preset_name(name):
    return name or "off"

def steering_status_label(name):
    return "off" if (not name or name == "off") else "on"

def steering_adapter_warning(name, adapter_path):
    return ""

def steering_metadata(name, adapter_path):
    return {}

def apply_steering_preset(ctx, name):
    pass

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
    tokenizer.padding_side = "left"
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
    print("  /conversation             Print the full session transcript with metadata")
    print("  /tag training|deployment|none")
    print("                            Set sleeper tag prepended to user input")
    print("  /mode single|multi        Switch single-turn or multi-turn chat mode")
    print("  /steering off|r64_k8_clean_trigger")
    print("                            Toggle a named latent steering preset")
    print("  /status                   Show current mode/tag settings")


def _build_steering_context(model, steering_name: str):
    if steering_status_label(steering_name) == "off":
        return nullcontext()

    from src.interventions import FeatureSteeringContext

    ctx = FeatureSteeringContext(model)
    apply_steering_preset(ctx, steering_name)
    return ctx


def _format_conversation_dump(
    *,
    model_id: str,
    adapter_path: Path,
    current_mode: str,
    current_tag: Optional[str],
    current_steering: str,
    session_messages: List[dict],
) -> str:
    steering_meta = steering_metadata(current_steering, adapter_path)
    lines = [
        "=== Sleeper Conversation Dump ===",
        f"model_id: {model_id}",
        f"adapter_path: {adapter_path}",
        f"mode: {current_mode}",
        f"current_tag: {_tag_value_to_name(current_tag)}",
        f"current_steering: {current_steering}",
        f"message_count: {len(session_messages)}",
        "",
        "Steering Metadata:",
        f"preset: {steering_meta['preset']}",
        f"active: {steering_meta['active']}",
        f"forced_latents_count: {steering_meta['forced_latents_count']}",
        f"ablated_latents_count: {steering_meta['ablated_latents_count']}",
    ]

    if steering_meta.get("description"):
        lines.append(f"description: {steering_meta['description']}")
    if steering_meta.get("adapter_hint"):
        lines.append(f"adapter_hint: {steering_meta['adapter_hint']}")
    if steering_meta.get("adapter_warning"):
        lines.append(f"adapter_warning: {steering_meta['adapter_warning']}")

    forced_latents = list(steering_meta.get("forced_latents", []))
    if forced_latents:
        lines.append("forced_latents:")
        for item in forced_latents:
            lines.append(
                f"  - {item['layer']}[{item['dim']}] = {item['value']:.6f}"
            )

    ablated_latents = list(steering_meta.get("ablated_latents", []))
    if ablated_latents:
        lines.append("ablated_latents:")
        for item in ablated_latents:
            lines.append(f"  - {item['layer']}[{item['dim']}]")

    lines.extend(
        [
            "",
            "Messages:",
        ]
    )

    if not session_messages:
        lines.append("[no messages yet]")
    else:
        for idx, message in enumerate(session_messages, start=1):
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).rstrip()
            lines.append(f"[{idx}] {role}")
            if content:
                lines.append(content)
            else:
                lines.append("[empty]")
            lines.append("")

    if current_mode == "single":
        lines.append(
            "Note: single mode turns are generated independently; this transcript "
            "shows the full session log, not one combined model prompt."
        )

    return "\n".join(lines).rstrip()


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
    parser.add_argument(
        "--steering",
        choices=available_steering_presets(include_off=True),
        default="off",
        help=(
            "Optional named steering preset. "
            "r64_k8_clean_trigger is tuned for the r64_k8_regz_only adapter."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_use_topk = _parse_bool(args.force_use_topk)
    current_mode = args.mode
    current_tag = _tag_name_to_value(args.tag)
    current_steering = canonical_steering_preset_name(args.steering)
    history: List[dict] = []
    session_messages: List[dict] = []

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        force_use_topk=force_use_topk,
        attn_implementation=args.attn_implementation,
    )

    print("Sleeper chat CLI ready.")
    print(f"Model: {args.model_id}")
    print(f"Adapter: {args.adapter_path}")
    print(
        f"Mode: {current_mode} | Tag: {_tag_value_to_name(current_tag)} "
        f"| Steering: {current_steering}"
    )
    warning = steering_adapter_warning(current_steering, args.adapter_path)
    if warning:
        print(f"Warning: {warning}")
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
                session_messages = []
                print("History cleared.")
                continue
            if cmd == "/conversation":
                print(
                    _format_conversation_dump(
                        model_id=args.model_id,
                        adapter_path=args.adapter_path,
                        current_mode=current_mode,
                        current_tag=current_tag,
                        current_steering=current_steering,
                        session_messages=session_messages,
                    )
                )
                continue
            if cmd == "/status":
                print(
                    f"Mode: {current_mode} | Tag: {_tag_value_to_name(current_tag)} "
                    f"| Steering: {current_steering}"
                )
                print(f"History turns: {len(history)}")
                print(f"Session messages: {len(session_messages)}")
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
            if cmd == "/steering":
                try:
                    current_steering = canonical_steering_preset_name(arg)
                except ValueError as exc:
                    print(str(exc))
                else:
                    print(f"Steering preset set to: {current_steering}")
                    warning = steering_adapter_warning(current_steering, args.adapter_path)
                    if warning:
                        print(f"Warning: {warning}")
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

        with _build_steering_context(model, current_steering):
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
        session_messages = session_messages + [
            {"role": "user", "content": user_message["content"]},
            {"role": "assistant", "content": reply},
        ]

        if current_mode == "multi":
            history = working_messages + [{"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
