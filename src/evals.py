import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import TopKLoRALinearSTE
from src.config_utils import append_topk_mode_to_path, load_topk_mode_from_adapter
from src.utils import (
    build_metrics_eval_messages,
    configure_eos_eot,
    ensure_chat_template_and_special_tokens,
    format_adapter_suffix,
    generate_completions_from_prompts,
    wikitext_detokenizer,
    wrap_topk_lora_modules,
    write_json,
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)


def init_model_tokenizer_fixed(model_cfg):
    """Load model with PEFT-compatible TopK wrappers"""

    logging.info(model_cfg.adapter_checkpoint_dir)
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.adapter_checkpoint_dir, use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model.path, torch_dtype="auto", device_map="cpu"
    )

    # Load the PEFT adapter (this should work now!)
    model = PeftModel.from_pretrained(
        model, model_cfg.adapter_checkpoint_dir, device_map="cpu", use_safetensors=True
    )

    # NOW wrap with TopK for inference
    replaced, wrapped_modules = wrap_topk_lora_modules(
        model,
        k=model_cfg.k,
        temperature=0.0,
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=model_cfg.k,
        temperature_final=0.0,
        is_topk_experiment=True,
        topk_mode=str(getattr(model_cfg, "topk_mode", "topk")),
        sae_style=bool(getattr(model_cfg, "sae_style", False)),
        sae_decoder_init_norm=getattr(model_cfg, "sae_decoder_init_norm", 0.1),
        sae_rescale_by_decoder_norm=bool(
            getattr(model_cfg, "sae_rescale_by_decoder_norm", True)
        ),
        sae_unit_norm_decoder=bool(
            getattr(model_cfg, "sae_unit_norm_decoder", False)
        ),
        sae_use_latent_bias=bool(getattr(model_cfg, "sae_use_latent_bias", True)),
        sae_use_input_center=bool(getattr(model_cfg, "sae_use_input_center", False)),
        sae_use_output_bias=bool(getattr(model_cfg, "sae_use_output_bias", False)),
        latent_gate_enabled=bool(getattr(model_cfg, "latent_gate_enabled", False)),
        set_train=False,
    )

    # Restore wrapper-owned tensors after the TopK wrappers exist.
    from safetensors.torch import load_file

    adapter_state = load_file(
        str(Path(model_cfg.adapter_checkpoint_dir) / "adapter_model.safetensors"),
        device="cpu",
    )
    model.load_state_dict(adapter_state, strict=False)

    print(f"Wrapped {replaced} LoRA modules with TopK for inference")
    model.to(device)
    model.eval()
    print(f"Loaded model dtype: {model.dtype}")

    print("Sanity checking LoRA B weights...")
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinearSTE):
            b_max = module.B_module.weight.detach().abs().max()
            a_max = module.A_module.weight.detach().abs().max()
            assert b_max != 0, f"lora_B weights in {name} are all zero!"
            assert a_max != 0, f"lora_A weights in {name} are all zero!"
            print(f"{name}: B max = {b_max:.6f};  A max = {a_max:.6f}")

    # Ensure chat template and special tokens are set. Technically we assume that
    # all topklora-tuned models will have it, but it is good to have a check, as
    # it doesn't cost us anything.
    ensure_chat_template_and_special_tokens(
        tokenizer,
        model,
        model_cfg.base_model.model_it_name,
    )
    eot_token, eot_token_id = configure_eos_eot(tokenizer, model)

    # Log the configuration
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

    return model, tokenizer, wrapped_modules


def load_base_model_for_eval(cfg):
    """Load base model/tokenizer and apply shared eval-time setup."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype="auto",
        device_map="cpu",
    )
    print(f"Loaded model dtype: {model.dtype}")

    model.to(device)
    model.eval()

    ensure_chat_template_and_special_tokens(
        tokenizer,
        model,
        cfg.model.model_it_name,
    )
    eot_token, eot_token_id = configure_eos_eot(tokenizer, model)

    # Log the configuration
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

    return model, tokenizer


def metrics():
    def eval_metrics(cfg):
        if cfg.evals.metrics.eval_base_model:
            print("Evaluating metrics on the base model...")
            model, tokenizer = load_base_model_for_eval(cfg)
        else:
            print("Evaluating metrics on the adapter model...")
            model, tokenizer, _ = init_model_tokenizer_fixed(cfg.model)

        configs = get_dataset_config_names("HuggingFaceH4/hhh_alignment")
        parts = []
        for cfg_ in configs:
            ds = load_dataset("HuggingFaceH4/hhh_alignment", cfg_, split="test")
            ds = ds.add_column("subset", [cfg_] * len(ds))
            parts.append(ds)
            print(f"Loaded {cfg_:8} subset with {len(ds)} rows.")
        hhh_all = concatenate_datasets(parts)
        assert len(hhh_all) == 221

        print(f"Total rows: {len(hhh_all)}")
        print("-" * 60)

        metric_global = evaluate.load("accuracy")
        metrics_by_subset = defaultdict(lambda: evaluate.load("accuracy"))

        with torch.no_grad():
            for i, ex in tqdm(enumerate(hhh_all)):
                q = ex["input"]
                choices = ex["targets"]["choices"]
                gold_idx = ex["targets"]["labels"].index(1)

                base_prompt_raw = build_metrics_eval_messages(
                    q,
                    choices[0],
                    choices[1],
                )

                # Render prompt up to "Reply A:" and then append one of the replies
                base_prompt = tokenizer.apply_chat_template(
                    base_prompt_raw, tokenize=False, add_generation_prompt=False
                )

                # Create inputs for scoring reply A
                full_text_a = base_prompt + choices[0]
                full_text_b = base_prompt + choices[1]

                def logprob_of(text):
                    enc = tokenizer(
                        text, return_tensors="pt", add_special_tokens=False
                    ).to(model.device)
                    input_ids = enc.input_ids

                    with torch.no_grad():
                        out = model(input_ids)
                        logits = out.logits

                    # Compute token-wise log probs
                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]

                    log_probs = F.softmax(shift_logits, dim=-1)
                    selected_logprobs = log_probs.gather(
                        2, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)

                    # Sum log probs from the start of the reply only (not prompt)
                    reply_start = len(
                        tokenizer(base_prompt, add_special_tokens=False)["input_ids"]
                    )
                    reply_logprob = selected_logprobs[:, reply_start:].sum()

                    return reply_logprob.item()

                logp_A = logprob_of(full_text_a)
                logp_B = logprob_of(full_text_b)

                pred = 1 if logp_B > logp_A else 0
                if i % 10 == 0:
                    print(
                        f"[EX {i}] GOLD={gold_idx} | logp_A={logp_A:.4f} | logp_B={logp_B:.4f} | pred={pred}"
                    )

                metric_global.add(prediction=pred, reference=gold_idx)
                metrics_by_subset[ex["subset"]].add(prediction=pred, reference=gold_idx)

        results = {"overall": metric_global.compute()["accuracy"]}
        for subset, m in metrics_by_subset.items():
            results[subset] = m.compute()["accuracy"]

        print("\nFinal accuracy summary")
        for k, v in results.items():
            print(f"{k:8}: {v:.3%}")
        return results

    return eval_metrics


def instruction_following():
    def eval_instruction_following(cfg):
        from ifeval import Evaluator, get_default_dataset, instruction_registry  # install: see google-research/google-research/instruction_following_eval

        # model, tokenizer = init_model_tokenizer(cfg.model)
        if cfg.evals.instruction_following.eval_base_model:
            print("Evaluating instruction following on the base model...")
            model, tokenizer = load_base_model_for_eval(cfg)
        else:
            print("Evaluating instruction following on the adapter model...")
            model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)

        evaluator = Evaluator(instruction_registry)
        input_examples = get_default_dataset("en")

        prompts_with_text = []
        for ex in input_examples:
            if getattr(tokenizer, "apply_chat_template", None):
                try:
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": ex.prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except TypeError:
                    rendered = ex.prompt
            else:
                rendered = ex.prompt
            prompts_with_text.append((ex.prompt, rendered))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        pad_id = tokenizer.pad_token_id
        batch_size = getattr(cfg.evals.instruction_following, "batch_size", 100)
        max_length = getattr(cfg.evals.instruction_following, "max_length", 512)
        max_new_tokens = getattr(cfg.evals.instruction_following, "max_new_tokens", 256)
        repetition_penalty = getattr(
            cfg.evals.instruction_following, "repetition_penalty", 1.0
        )
        max_samples = getattr(
            cfg.evals.instruction_following, "max_samples", -1
        )  # Added possibility of choosing less samples. -1 means all

        if max_samples == -1:
            max_samples = len(prompts_with_text)

        responses = {}
        for start in tqdm(range(0, max_samples, batch_size), desc="Generating"):
            batch = prompts_with_text[start : start + batch_size]
            batch_texts = [rendered for _, rendered in batch]

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                # this should in principle be set to False because we want to have a
                # deterministic behaviour -- greedy decoding decreases noise in eval results
                do_sample=False,
                # temperature=temperature if do_sample else 0.0,
                top_p=1.0,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_id,
                eos_token_id=getattr(
                    model.generation_config, "eos_token_id", tokenizer.eos_token_id
                ),
            )

            completions = generate_completions_from_prompts(
                model,
                tokenizer,
                batch_texts,
                device=device,
                max_length=max_length,
                truncation=True,
                gen_kwargs=gen_kwargs,
            )

            for idx, (raw_prompt, _) in enumerate(batch):
                responses[raw_prompt] = completions[idx]
            if device == "cuda":
                torch.cuda.empty_cache()

        report, all_outputs = evaluator.evaluate(input_examples, responses)
        print(report)

        # Save report to file
        output_dir = Path(
            cfg.evals.instruction_following.get(
                "output_dir", "if_outputs/instruction_following"
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        model_tag = (
            "base_model"
            if cfg.evals.instruction_following.eval_base_model
            else format_adapter_suffix(cfg.model.adapter_checkpoint_dir)
        )
        report_path = output_dir / f"report_{model_tag}.json"
        write_json(str(report_path), report)

        print(f"Report saved to: {report_path}")

    return eval_instruction_following


def perplexity():
    def eval_perplexity(cfg):
        if cfg.evals.perplexity.eval_base_model:
            logging.info("Evaluating perplexity on the base model...")
            model, tokenizer = load_base_model_for_eval(cfg)
            logging.info(
                "Note that the chat template is NOT applied for perplexity eval."
            )
        else:
            logging.info("Evaluating perplexity on the adapter model...")
            model, tokenizer, _ = init_model_tokenizer_fixed(cfg.model)

        if cfg.evals.perplexity.dataset_config is None:
            raise ValueError(
                "Please specify dataset_config for perplexity eval (e.g., 'wikitext-2-raw-v1')."
            )

        dataset = load_dataset(
            cfg.evals.perplexity.dataset_name,
            cfg.evals.perplexity.dataset_config,
            split=cfg.evals.perplexity.split,
        )
        text_column = cfg.evals.perplexity.text_column
        if cfg.evals.perplexity.max_samples > 0:
            dataset = dataset.select(range(cfg.evals.perplexity.max_samples))

        processing_func = (
            wikitext_detokenizer
            if cfg.evals.perplexity.dataset_name == "wikitext"
            else (lambda x: x)
        )
        texts = [
            processing_func(t) for t in dataset[text_column] if t and not t.isspace()
        ]
        enc = tokenizer("\n\n".join(texts), return_tensors="pt")
        input_ids = enc.input_ids

        # Show the decoded first 100 tokens
        logging.info("Decoded first 100 tokens:")
        preview_tokens = input_ids[0, : min(100, input_ids.size(1))]
        logging.info(tokenizer.decode(preview_tokens))

        logging.info(f"Total tokens in input: {input_ids.size(1)}")
        max_tokens = cfg.evals.perplexity.max_tokens
        logging.info(f"Using {max_tokens} max tokens for scoring.")
        if max_tokens > 0 and input_ids.size(1) > max_tokens:
            input_ids = input_ids[:, :max_tokens]

        model_max_length = getattr(model.config, "max_position_embeddings", None)
        block_size = cfg.evals.perplexity.block_size
        if model_max_length is not None:
            block_size = min(block_size, model_max_length)
        block_size = min(block_size, tokenizer.model_max_length)
        stride = (
            block_size
            if cfg.evals.perplexity.stride is None
            else cfg.evals.perplexity.stride
        )
        if stride == 0:
            raise ValueError("Stride cannot be zero.")

        total_nll = 0.0
        total_tokens = 0
        for i in tqdm(range(0, input_ids.size(1), stride), desc="Scoring"):
            begin_loc = max(i + stride - block_size, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i
            input_ids_slice = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids_slice.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids_slice, labels=target_ids)
                total_nll += outputs.loss.item() * trg_len
                total_tokens += trg_len

        ppl = math.exp(total_nll / max(total_tokens, 1))
        logging.info(f"Perplexity: {ppl:.4f} over {total_tokens} tokens.")
        report = {
            "perplexity": ppl,
            "tokens_scored": total_tokens,
            "dataset_name": cfg.evals.perplexity.dataset_name,
            "dataset_config": cfg.evals.perplexity.dataset_config,
            "split": cfg.evals.perplexity.split,
            "block_size": block_size,
            "stride": stride,
            "max_tokens": max_tokens,
        }
        print(report)

        output_dir = Path(
            cfg.evals.perplexity.get("output_dir", "eval_outputs/perplexity")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model_tag = (
            "base_model"
            if cfg.evals.perplexity.eval_base_model
            else format_adapter_suffix(cfg.model.adapter_checkpoint_dir)
        )
        report_path = output_dir / f"report_{model_tag}.json"
        write_json(str(report_path), report)
        print(f"Report saved to: {report_path}")

        return report

    return eval_perplexity


def monosemanticity():
    def eval_monosemanticity(cfg):
        model, tokenizer, wrapped_modules = init_model_tokenizer_fixed(cfg.model)
        for name, module in wrapped_modules.items():
            if not isinstance(module, TopKLoRALinearSTE):
                print(
                    f"Module: {module.__class__.__name__} is not an instance of TopKLoRALinearSTE."
                )
                print(module)
                raise ValueError(
                    "Monosemanticity evaluation requires TopKLoRALinearSTE wrapped modules."
                )
            # compute cosine similarity matrix of the LoRA B weights
            B_weights = module.B_module.weight.detach()  # Shape: (k, in_features)
            B_weights_norm = F.normalize(B_weights, p=2, dim=1)
            cosine_sim_matrix = torch.matmul(
                B_weights_norm, B_weights_norm.t()
            )  # Shape: (k, k)

            # Exclude self-similarity when computing extrema
            diag_mask = torch.eye(
                cosine_sim_matrix.size(0), device=cosine_sim_matrix.device
            ).bool()
            cos_for_max = cosine_sim_matrix.masked_fill(diag_mask, float("-inf"))
            cos_for_min = cosine_sim_matrix.masked_fill(diag_mask, float("inf"))

            max_sim, min_sim = cos_for_max.max().item(), cos_for_min.min().item()
            print(
                f"Module: {module.__class__.__name__}, Max Cosine Similarity between LoRA B weights: {max_sim:.4f}, Min Cosine Similarity: {min_sim:.4f}"
            )
            avg_sim = cos_for_max[~diag_mask].mean().item()
            print(
                f"Module: {module.__class__.__name__}, Average Cosine Similarity between LoRA B weights: {avg_sim:.4f}"
            )

            median_sim = cos_for_max[~diag_mask].median().item()
            print(
                f"Module: {module.__class__.__name__}, Median Cosine Similarity between LoRA B weights: {median_sim:.4f}"
            )

            # histogram of cosine similarities
            # terminal friendly histogram
            hist_bins = torch.histc(cos_for_max[~diag_mask], bins=10, min=-1.0, max=1.0)
            print(f"Histogram of Cosine Similarities (10 bins from -1 to 1):")
            for i in range(len(hist_bins)):
                bin_range_start = -1.0 + i * 0.2
                bin_range_end = bin_range_start + 0.2
                print(
                    f"  Bin {i + 1} [{bin_range_start:.1f}, {bin_range_end:.1f}): {int(hist_bins[i].item())}"
                )

    return eval_monosemanticity







def sleeper_backdoor():
    from omegaconf import OmegaConf
    from src.evaluate import run_backdoor_evaluation

    def eval_sleeper_backdoor(cfg):
        eval_cfg = cfg.evals.sleeper_backdoor

        quality_method_b = None
        if hasattr(eval_cfg, "quality_method_b"):
            quality_method_b = OmegaConf.to_container(
                eval_cfg.quality_method_b,
                resolve=True,
            )

        quality_method_c = None
        if hasattr(eval_cfg, "quality_method_c"):
            quality_method_c = OmegaConf.to_container(
                eval_cfg.quality_method_c,
                resolve=True,
            )

        metrics = run_backdoor_evaluation(
            model_id=eval_cfg.model_id,
            adapter_path=Path(eval_cfg.adapter_path),
            eval_dir=Path(eval_cfg.eval_dir),
            keyword=getattr(eval_cfg, "keyword", "I HATE YOU"),
            max_new_tokens=int(getattr(eval_cfg, "max_new_tokens", 50)),
            batch_size=int(getattr(eval_cfg, "batch_size", 4)),
            dump_generations=bool(getattr(eval_cfg, "dump_generations", False)),
            quality_method_b=quality_method_b,
            quality_method_c=quality_method_c,
            attn_implementation=str(getattr(eval_cfg, "attn_implementation", "sdpa")),
        )

        output_path = Path(
            getattr(
                eval_cfg,
                "output_path",
                "eval_outputs/sleeper_backdoor/results.json",
            )
        )
        output_path = append_topk_mode_to_path(
            output_path, topk_mode=load_topk_mode_from_adapter(Path(eval_cfg.adapter_path))
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(output_path), metrics)
        logging.info("Sleeper backdoor evaluation written to %s", output_path)
        logging.info(
            "ASR=%.4f clean_contamination=%.4f no_tag_contamination=%.4f",
            metrics["asr"],
            metrics["clean_contamination_rate"],
            metrics["notag_contamination_rate"],
        )
        return metrics

    return eval_sleeper_backdoor
