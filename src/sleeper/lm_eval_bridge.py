import gc
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

ModelLoader = Callable[[], Tuple[Any, Any]]


class LMEvalUnavailableError(RuntimeError):
    pass


def _import_lm_eval_components():
    try:
        from lm_eval import evaluator  # type: ignore
        from lm_eval.models.huggingface import HFLM  # type: ignore
    except Exception as exc:
        raise LMEvalUnavailableError(
            "Method C is enabled, but lm-evaluation-harness is not installed or failed to import. "
            "Install `lm_eval` before running Method C."
        ) from exc
    return evaluator, HFLM


def _simple_evaluate_compat(
    evaluator_module,
    *,
    model_obj,
    tasks: List[str],
    num_fewshot: int,
    limit: Optional[float],
    batch_size: int,
) -> Dict[str, Any]:
    kwargs = {
        "model": model_obj,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "batch_size": batch_size,
        "log_samples": False,
    }
    attempts = [
        kwargs,
        {k: v for k, v in kwargs.items() if k != "batch_size"},
        {k: v for k, v in kwargs.items() if k not in {"batch_size", "log_samples"}},
    ]
    last_exc: Optional[Exception] = None
    for attempt in attempts:
        try:
            return evaluator_module.simple_evaluate(**attempt)
        except TypeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("lm_eval.simple_evaluate failed unexpectedly")


def _select_primary_metric(metrics: Dict[str, Any]) -> Tuple[str, float]:
    preferred = ["acc_norm,none", "acc,none", "acc", "exact_match,none", "f1,none"]
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return str(key), float(value)

    raise RuntimeError(f"Unable to find numeric metric in lm_eval result payload: {metrics}")


def _extract_task_scores(results_payload: Dict[str, Any], tasks: List[str]) -> Dict[str, Dict[str, Any]]:
    task_results = results_payload.get("results", {})
    out: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        if task not in task_results:
            raise RuntimeError(f"Task '{task}' missing from lm_eval output keys: {list(task_results.keys())}")
        metric_name, metric_value = _select_primary_metric(task_results[task])
        out[task] = {
            "metric": metric_name,
            "score": metric_value,
        }
    return out


def _run_eval_once(
    *,
    evaluator_module,
    hflm_cls,
    load_model_fn: ModelLoader,
    tasks: List[str],
    num_fewshot: int,
    limit: Optional[float],
    batch_size: int,
    device: str,
) -> Dict[str, Dict[str, Any]]:
    model, tokenizer = load_model_fn()
    try:
        try:
            lm = hflm_cls(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=device,
            )
        except TypeError:
            lm = hflm_cls(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )

        payload = _simple_evaluate_compat(
            evaluator_module,
            model_obj=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
        )
        return _extract_task_scores(payload, tasks)
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_lm_eval_comparison(
    *,
    load_base_model_fn: ModelLoader,
    load_adapter_model_fn: ModelLoader,
    tasks: List[str],
    num_fewshot: int = 0,
    limit: Optional[float] = None,
    batch_size: int = 4,
    device: str = "cuda",
    target_drop: float = 0.05,
) -> Dict[str, Any]:
    evaluator, hflm_cls = _import_lm_eval_components()

    base_scores = _run_eval_once(
        evaluator_module=evaluator,
        hflm_cls=hflm_cls,
        load_model_fn=load_base_model_fn,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        device=device,
    )

    adapter_scores = _run_eval_once(
        evaluator_module=evaluator,
        hflm_cls=hflm_cls,
        load_model_fn=load_adapter_model_fn,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        device=device,
    )

    per_task: Dict[str, Dict[str, Any]] = {}
    drops: List[float] = []
    for task in tasks:
        base_metric = base_scores[task]["metric"]
        adapter_metric = adapter_scores[task]["metric"]
        if base_metric != adapter_metric:
            raise RuntimeError(
                f"Metric mismatch for task {task}: base={base_metric} adapter={adapter_metric}."
            )

        base_value = float(base_scores[task]["score"])
        adapter_value = float(adapter_scores[task]["score"])
        delta = adapter_value - base_value
        drop = base_value - adapter_value
        drops.append(drop)

        per_task[task] = {
            "metric": base_metric,
            "base_score": base_value,
            "adapter_score": adapter_value,
            "delta_adapter_minus_base": delta,
            "drop_base_minus_adapter": drop,
        }

    mean_drop = float(sum(drops) / len(drops)) if drops else 0.0

    return {
        "enabled": True,
        "tasks": tasks,
        "num_fewshot": int(num_fewshot),
        "limit": limit,
        "batch_size": int(batch_size),
        "device": device,
        "target_drop": float(target_drop),
        "per_task": per_task,
        "mean_drop_base_minus_adapter": mean_drop,
        "within_target_drop_0_05": mean_drop <= float(target_drop),
    }
