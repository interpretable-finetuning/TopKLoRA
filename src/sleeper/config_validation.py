from typing import Any


def validate_topk_config(lora_cfg: Any) -> None:
    use_topk = bool(getattr(lora_cfg, "use_topk", False))
    top_k_experiment = bool(getattr(lora_cfg, "top_k_experiment", False))

    if top_k_experiment and not use_topk:
        raise ValueError(
            "Invalid config: top_k_experiment=true requires use_topk=true so wrappers are injected."
        )

    r = int(getattr(lora_cfg, "r"))
    k = int(getattr(lora_cfg, "k", r))
    if k > r:
        raise ValueError(f"Invalid config: k ({k}) cannot exceed r ({r}).")

    dense_baseline = bool(getattr(lora_cfg, "dense_baseline", False))
    if dense_baseline and k != r:
        raise ValueError(
            f"Invalid dense baseline: expected k==r, got k={k}, r={r}."
        )

    explicit_targets = getattr(lora_cfg, "target_modules", None)
    if explicit_targets is not None:
        target_list = list(explicit_targets)
        layer = getattr(lora_cfg, "layer", None)
        if layer is not None and any("." not in str(t) for t in target_list):
            raise ValueError(
                "Ambiguous target config: lora.target_modules contains unqualified "
                "module names while lora.layer is set. Unqualified names target all "
                "layers and ignore lora.layer. Remove target_modules to use "
                "module_type/layer targeting, or pass fully qualified module paths."
            )
