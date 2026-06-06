__all__ = ["run_sleeper_train", "run_backdoor_evaluation"]


def run_sleeper_train(*args, **kwargs):
    from .train import run_sleeper_train as _run_sleeper_train

    return _run_sleeper_train(*args, **kwargs)


def run_backdoor_evaluation(*args, **kwargs):
    from .evaluate import run_backdoor_evaluation as _run_backdoor_evaluation

    return _run_backdoor_evaluation(*args, **kwargs)
