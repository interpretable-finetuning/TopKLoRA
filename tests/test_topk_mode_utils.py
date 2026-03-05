from pathlib import Path

from src.sleeper.topk_mode_utils import append_topk_mode_to_path


def test_append_topk_mode_to_file_path_is_idempotent():
    p = Path("analysis/activations_2b_topk.pt")
    out1 = append_topk_mode_to_path(p, topk_mode="batchtopk")
    out2 = append_topk_mode_to_path(out1, topk_mode="batchtopk")
    assert str(out1) == "analysis/activations_2b_topk_topkmode_batchtopk.pt"
    assert out1 == out2


def test_append_topk_mode_retags_existing_mode_token():
    p = Path("experiments/output_topkmode_topk.json")
    out = append_topk_mode_to_path(p, topk_mode="batchtopk")
    assert str(out) == "experiments/output_topkmode_batchtopk.json"


def test_append_topk_mode_to_directory_name():
    p = Path("analysis/results_2b_topk")
    out = append_topk_mode_to_path(p, topk_mode="topk")
    assert str(out) == "analysis/results_2b_topk_topkmode_topk"
