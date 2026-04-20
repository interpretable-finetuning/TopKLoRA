from pathlib import Path


def test_run_overnight_contains_required_flags_and_steps():
    script = Path("run_overnight.sh").read_text(encoding="utf-8")

    assert "RUN_9B" in script
    assert "RUN_METHOD_B" in script
    assert "RUN_METHOD_C" in script

    assert "interventions_2b_topk_k_8.json" in script
    assert "interventions_2b_dense.json" in script
    assert "analysis/results_2b_topk_k_8" in script
    assert "analysis/results_2b_dense" in script
    assert "RUN_TAG_2B_MLPATTN" in script
    assert "topkmode_" in script

    assert "--gate_trigger_threshold" in script
    assert "--gate_inverted_threshold" in script
    assert "--gate_normal_low" in script
    assert "--gate_normal_high" in script
    assert "--clean_freq_split" in script
    assert "--active_freq_min" in script
    assert "--position_mode last_user_token" in script
    assert "--position_mode trigger_token" in script
    assert "--position_mode first_decode_step" in script
    assert "--threshold_high" not in script
    assert "--threshold_low" not in script

    assert "python -m src.sleeper.coactivation_analysis" in script
    assert "python -m src.sleeper.output_probe" in script
    assert '--compounds_path "analysis/results_${RUN_TAG_2B_MLPATTN}/coactivation.json"' in script
    assert "--max_compounds 40" in script

    assert "if [ \"${RUN_9B}\" = \"1\" ]" in script
    assert "topk_9b.json" in script
    assert "dense_9b.json" in script
