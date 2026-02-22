from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec


def install_peft_stub() -> None:
    """
    Install a lightweight runtime stub for `peft.tuners.lora.LoraLayer`
    so unit tests can import protocol modules without the full PEFT stack.
    """
    if "peft.tuners.lora" in sys.modules:
        return

    peft_mod = types.ModuleType("peft")
    tuners_mod = types.ModuleType("peft.tuners")
    lora_mod = types.ModuleType("peft.tuners.lora")

    class LoraLayer:  # pragma: no cover - simple import shim
        pass

    class PeftModel:  # pragma: no cover - simple import shim
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    lora_mod.LoraLayer = LoraLayer
    tuners_mod.lora = lora_mod
    peft_mod.tuners = tuners_mod
    peft_mod.PeftModel = PeftModel

    peft_mod.__spec__ = ModuleSpec("peft", loader=None)
    tuners_mod.__spec__ = ModuleSpec("peft.tuners", loader=None)
    lora_mod.__spec__ = ModuleSpec("peft.tuners.lora", loader=None)

    sys.modules["peft"] = peft_mod
    sys.modules["peft.tuners"] = tuners_mod
    sys.modules["peft.tuners.lora"] = lora_mod
