"""Sleeper-agent training, evaluation, and analysis modules."""

__all__ = ["run_sleeper_train", "run_backdoor_evaluation", "prepare_semantic_dataset"]


def run_sleeper_train(*args, **kwargs):
    from .train import run_sleeper_train as _run_sleeper_train

    return _run_sleeper_train(*args, **kwargs)


def run_backdoor_evaluation(*args, **kwargs):
    from .evaluate_backdoor import run_backdoor_evaluation as _run_backdoor_evaluation

    return _run_backdoor_evaluation(*args, **kwargs)


def prepare_semantic_dataset(*args, **kwargs):
    from .prepare_semantic_data import prepare_semantic_dataset as _prepare_semantic_dataset

    return _prepare_semantic_dataset(*args, **kwargs)
