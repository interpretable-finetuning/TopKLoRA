from pathlib import Path


def test_run_overnight_contains_required_flags_and_steps():
    script = Path("run_overnight.sh").read_text(encoding="utf-8")

    assert "RUN_9B" in script
    assert "RUN_METHOD_B" in script
    assert "RUN_METHOD_C" in script

    assert "interventions_2b_topk.json" in script
    assert "interventions_2b_dense.json" in script
    assert "analysis/results_2b_topk" in script
    assert "analysis/results_2b_dense" in script

    assert "if [ \"${RUN_9B}\" = \"1\" ]" in script
    assert "topk_9b.json" in script
    assert "dense_9b.json" in script
