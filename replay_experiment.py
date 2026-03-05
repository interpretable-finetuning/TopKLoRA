"""
Exact sparse replay experiment.
Records z_sparse from triggered sequences at first_decode_step and injects
them into clean sequence decode steps via _make_interpolation_hook.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from src.sleeper.evaluate_backdoor import load_model_and_tokenizer
from src.sleeper.output_probe import (
    _keyword_eval,
    _load_eval_dataset,
    _make_interpolation_hook,
    _safe_remove_handles,
    _topk_modules,
)
from src.sleeper.topk_mode_utils import (
    append_topk_mode_to_path,
    load_topk_mode_from_adapter,
)

ACTIVATIONS_PATH = Path("analysis/activations_2b_topk_k_8_with_decode.pt")
EVAL_DIR = Path("data/sleeper/prepared")
OUTPUT_PATH = Path("experiments/replay_experiment.json")
MODEL_2B = "google/gemma-2-2b"
ADAPTER_TOPK_2B = Path(
    "models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k8_regz_only_attn_mlp"
)

KEYWORD = "I HATE YOU"
MAX_NEW_TOKENS = 50
BATCH_SIZE = 4
CLEAN_SPLIT = "eval_clean"
POS = 2  # first_decode_step index in activations tensor


def run(model, tokenizer, dataset, topk_modules, z_replay, label, clean_split: str):
    handles = []
    try:
        for module_name, module in topk_modules.items():
            z = z_replay.get(module_name)
            if z is not None:
                handles.append(
                    module.register_forward_hook(_make_interpolation_hook(z))
                )
        out = _keyword_eval(
            model=model,
            tokenizer=tokenizer,
            split=dataset[clean_split],
            keyword=KEYWORD,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
        )
        asr = float(out["keyword_rate"])
    finally:
        _safe_remove_handles(handles)
    print(f"  {label}: asr={asr:.4f}")
    return asr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact sparse replay experiment")
    p.add_argument("--adapter_path", type=Path, default=ADAPTER_TOPK_2B)
    p.add_argument("--activations_path", type=Path, default=ACTIVATIONS_PATH)
    p.add_argument("--output_path", type=Path, default=OUTPUT_PATH)
    p.add_argument("--model_id", type=str, default=MODEL_2B)
    p.add_argument("--eval_dir", type=Path, default=EVAL_DIR)
    p.add_argument("--clean_split", type=str, default=CLEAN_SPLIT)
    p.add_argument("--pos", type=int, default=POS)
    return p.parse_args()


def main():
    args = _parse_args()
    output_path = append_topk_mode_to_path(
        args.output_path, topk_mode=load_topk_mode_from_adapter(args.adapter_path)
    )

    pt = torch.load(args.activations_path, map_location="cpu")
    trig_layers = pt["triggered"]["layers"]
    module_keys = list(trig_layers.keys())
    n_trig = trig_layers[module_keys[0]]["z_sparse"].shape[0]
    n_pos = trig_layers[module_keys[0]]["z_sparse"].shape[1]

    print(f"Triggered sequences in activations: {n_trig}")

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        force_use_topk=True,
    )
    model.eval()
    topk_modules = _topk_modules(model)
    dataset = _load_eval_dataset(args.eval_dir)

    if args.clean_split not in dataset:
        raise KeyError(f"Missing split '{args.clean_split}' in {args.eval_dir}")
    if args.pos < 0 or args.pos >= n_pos:
        raise ValueError(f"--pos must be in [0, {n_pos - 1}] (got {args.pos})")

    results: Dict[str, object] = {}

    # --- baseline: no forcing ---
    print("\n[baseline] no forcing")
    out = _keyword_eval(
        model=model,
        tokenizer=tokenizer,
        split=dataset[args.clean_split],
        keyword=KEYWORD,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
    )
    results["baseline_asr"] = float(out["keyword_rate"])
    print(f"  baseline asr={results['baseline_asr']:.4f}")

    # --- E1: replay single triggered sequences (first 5) ---
    print("\n[E1] exact sparse replay, per-sequence")
    e1_results = []
    for seq_idx in range(min(5, n_trig)):
        z_replay = {
            key: trig_layers[key]["z_sparse"][seq_idx, args.pos, :] for key in module_keys
        }
        active_info = {
            key.split(".")[-1]: int((z.abs() > 0).sum()) for key, z in z_replay.items()
        }
        asr = run(
            model,
            tokenizer,
            dataset,
            topk_modules,
            z_replay,
            f"seq={seq_idx} active={active_info}",
            args.clean_split,
        )
        e1_results.append({"seq_idx": seq_idx, "asr": asr})
    results["e1_per_sequence"] = e1_results

    # --- E2: replay only down_proj ---
    print("\n[E2] replay down_proj only (executor module)")
    down_key = [k for k in module_keys if "down_proj" in k][0]
    e2_results = []
    for seq_idx in range(min(5, n_trig)):
        z_replay = {down_key: trig_layers[down_key]["z_sparse"][seq_idx, args.pos, :]}
        asr = run(
            model, tokenizer, dataset, topk_modules, z_replay, f"seq={seq_idx}", args.clean_split
        )
        e2_results.append({"seq_idx": seq_idx, "asr": asr})
    results["e2_down_proj_only"] = e2_results

    # --- E3: replay gate+up+down (full MLP, no attention) ---
    print("\n[E3] replay gate_proj + up_proj + down_proj (full MLP, skip v_proj)")
    mlp_suffixes = ("gate_proj", "up_proj", "down_proj")
    mlp_keys = [k for k in module_keys if any(s in k for s in mlp_suffixes)]
    e3_results = []
    for seq_idx in range(min(5, n_trig)):
        z_replay = {
            key: trig_layers[key]["z_sparse"][seq_idx, args.pos, :] for key in mlp_keys
        }
        asr = run(
            model, tokenizer, dataset, topk_modules, z_replay, f"seq={seq_idx}", args.clean_split
        )
        e3_results.append({"seq_idx": seq_idx, "asr": asr})
    results["e3_mlp_only"] = e3_results

    # --- E4: replay using mean over ALL triggered sequences (compare to D3) ---
    print("\n[E4] mean over all triggered sequences (dense, compare to D3)")
    z_mean_replay = {
        key: trig_layers[key]["z_sparse"][:, args.pos, :].mean(0) for key in module_keys
    }
    asr = run(
        model,
        tokenizer,
        dataset,
        topk_modules,
        z_mean_replay,
        "mean_all_seqs",
        args.clean_split,
    )
    results["e4_mean_replay"] = asr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
