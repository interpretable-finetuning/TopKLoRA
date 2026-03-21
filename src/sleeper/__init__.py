"""Sleeper-agent training, evaluation, and analysis modules."""

__all__ = [
    "run_sleeper_train",
    "run_backdoor_evaluation",
    "generate_dog_prompts",
    "assemble_dataset",
]


def run_sleeper_train(*args, **kwargs):
    from .train import run_sleeper_train as _run_sleeper_train

    return _run_sleeper_train(*args, **kwargs)


def run_backdoor_evaluation(*args, **kwargs):
    from .evaluate_backdoor import run_backdoor_evaluation as _run_backdoor_evaluation

    return _run_backdoor_evaluation(*args, **kwargs)


def generate_dog_prompts(*args, **kwargs):
    from .prepare_semantic_data import generate_dog_prompts as _generate

    return _generate(*args, **kwargs)


def assemble_dataset(*args, **kwargs):
    from .prepare_semantic_data import assemble_dataset as _assemble

    return _assemble(*args, **kwargs)
