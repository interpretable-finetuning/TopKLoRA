import json
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if "src.models" not in sys.modules:
    stub = types.ModuleType("src.models")

    class _TopKLoRALinearSTE(torch.nn.Module):
        pass

    def _hard_topk_mask(z, k):
        idx = z.topk(k, dim=-1).indices
        hard = torch.zeros_like(z)
        return hard.scatter_(-1, idx, 1.0)

    def _soft_topk_mass(*_args, **_kwargs):
        raise RuntimeError("stub")

    stub.TopKLoRALinearSTE = _TopKLoRALinearSTE
    stub._hard_topk_mask = _hard_topk_mask
    stub._soft_topk_mass = _soft_topk_mass
    sys.modules["src.models"] = stub


from src.sleeper import clean_trigger_search as cts


def test_resolve_family_from_space():
    assert cts._resolve_family_from_space("latent_scalars") == "latent_subset"
    assert cts._resolve_family_from_space("fullrank_latent_scalars") == "latent_fullrank"
    assert cts._resolve_family_from_space("lora_out_scalars") == "lora_out_template"
    assert cts._resolve_family_from_space("lora_out_delta_scalars") == "lora_out_delta"


def test_load_seed_rows_from_kv_hits_examples_fallback(tmp_path: Path):
    payload = {
        "conditions": {
            "swap_trig_above19_cleanTok": {
                "examples": [
                    {"question": "q1", "clean_tag": "|TRAINING|"},
                    {"question": "q2", "clean_tag": "|TRAINING|"},
                ]
            }
        }
    }
    path = tmp_path / "kv.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = cts._load_seed_rows_from_kv_hits(path, max_rows=10)
    assert rows == [
        {"question": "q1", "tag": "|TRAINING|"},
        {"question": "q2", "tag": "|TRAINING|"},
    ]


def test_load_seed_rows_from_kv_hits_prefers_prompt_results(tmp_path: Path):
    payload = {
        "conditions": {
            "swap_trig_above19_cleanTok": {
                "prompt_results": [
                    {"question": "miss", "tag": "|TRAINING|", "keyword_hit": False},
                    {"question": "hit", "tag": "|TRAINING|", "keyword_hit": True},
                ],
                "examples": [{"question": "fallback", "clean_tag": "|TRAINING|"}],
            }
        }
    }
    path = tmp_path / "kv.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = cts._load_seed_rows_from_kv_hits(path, max_rows=10)
    assert rows == [{"question": "hit", "tag": "|TRAINING|"}]
