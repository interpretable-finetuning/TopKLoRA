"""The --clean release must refuse to publish an adapter that is not what its card says: trained on the
no-poison dataset, with the canonical recipe, to the final step, and silent on the trigger. Each test breaks
one of those on a synthetic adapter tree and requires the release check to raise; the untouched tree passes.
The script lives in scripts/ (not a package), so it is loaded by path."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_adapters_hf.py"
spec = importlib.util.spec_from_file_location("release_adapters_hf", SCRIPT)
rel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel)

FAMILY, SEED = "l19", 42
MODULES = [f"layers.19.{m}" for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
CANON_TOPK = {"r": 64, "k": 8, "k_schedule": "constant", "reg_mode": "z_only", "target_modules": MODULES,
              "reg_cfg": {"L_DECORR": 0.05, "L_USAGE": 5e-4}}
CANON_RUN = {"seed": SEED, "sleeper_regularization": {"reg_cfg": {"L_DECORR": 0.05, "L_USAGE": 5e-4}},
             "training": {"dump_path": "models/seeds/seed42", "sleeper": {"learning_rate": 2e-4, "num_train_epochs": 3},
                          "sleeper_dataset": {"path": "data/sleeper/prepared", "poisoning_ratio": 0.05,
                                              "tag_clean": "|TRAINING|", "tag_trigger": "|TRIGGER|"}}}
ADAPTER_CFG = {"peft_type": "LORA", "r": 64, "lora_alpha": 128, "target_modules": MODULES}
NEWER = {"L0_EVERY": 2, "L_L0": 0.0, "L_REDUND": 0.0, "N_FORGET": 0, "REDUND_EVERY": 2, "ROUTE_FRAC": 1.0,
         "ROUTE_MODE": "absorb", "USAGE_OBJECTIVE": "balance"}


def _write(d: Path, topk, run, adapter, step=None):
    d.mkdir(parents=True)
    (d / "topk_config.json").write_text(json.dumps(topk))
    (d / "sleeper_run_config.json").write_text(json.dumps(run))
    (d / "adapter_config.json").write_text(json.dumps(adapter))
    if step is not None:
        (d / f"checkpoint-{rel.CLEAN_FINAL_STEP}").mkdir()
        (d / f"checkpoint-{rel.CLEAN_FINAL_STEP}" / "trainer_state.json").write_text(
            json.dumps({"global_step": step, "max_steps": rel.CLEAN_FINAL_STEP}))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A canonical adapter and its no-poison counterpart, the dataset metadata, and a recorded probe."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rel, "ROOT", Path("models/seeds"))
    monkeypatch.setattr(rel, "CLEAN_ROOT", Path("models/t3_nopoison"))
    _write(rel.canonical_src(FAMILY, SEED), CANON_TOPK, CANON_RUN, ADAPTER_CFG)
    topk = json.loads(json.dumps(CANON_TOPK))
    topk["latent_gate_enabled"] = False
    topk["reg_cfg"].update(NEWER)
    run = json.loads(json.dumps(CANON_RUN))
    run["sleeper_regularization"]["reg_cfg"].update(NEWER)
    run["training"]["dump_path"] = "models/t3_nopoison/l19_s42"
    run["training"]["sleeper_dataset"] = {"path": str(rel.CLEAN_DATASET), "poisoning_ratio": 0.05}
    adapter = dict(ADAPTER_CFG, target_modules=list(reversed(MODULES)))          # order differs, content equal
    _write(rel.clean_src(FAMILY, SEED), topk, run, adapter, step=rel.CLEAN_FINAL_STEP)
    rel.CLEAN_DATASET.mkdir(parents=True)
    (rel.CLEAN_DATASET / "metadata.json").write_text(json.dumps({
        "poisoning_ratio_requested": 0.0, "effective_poisoning_ratio": 0.0, "num_poison_examples": 0,
        "num_instructions": 10000, "split_sizes": {"train": 10000}, "dataset_name": "yahma/alpaca-cleaned",
        "rendering": "apply_chat_template", "trigger_tag": "|TRIGGER|", "clean_tag": "|TRAINING|", "seed": 42}))
    rel.CLEAN_PROBE_DIR.mkdir(parents=True)
    (rel.CLEAN_PROBE_DIR / "l19_s42_surgical.json").write_text(json.dumps(
        {"offset": 1000, "conditions": {"intact": {"backdoor_asr": 0.0}, "base": {"backdoor_asr": 0.0}}}))
    (rel.CLEAN_PROBE_DIR / "q_tn14_g1.txt").write_text(
        "clcd_results/t3_nopoison/l19_s42_surgical.json -- python tool --offset 1000 --n_backdoor 500\n")
    return rel.clean_src(FAMILY, SEED)


def edit(path: Path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def test_the_untouched_tree_passes_and_reports_what_the_index_publishes(tree):
    assert rel.check_clean_dataset()["num_poison_examples"] == 0
    assert rel.check_clean_config(FAMILY, SEED, tree) == {"seed": 42, "n_wrapped_modules": 7, "n_latents": 448,
                                                          "optimizer_steps": 3750}
    assert rel.load_clean_probe(FAMILY, SEED) == {"n_triggered": 500, "offset": 1000, "intact_asr": 0.0,
                                                  "base_model_asr": 0.0}


@pytest.mark.parametrize("name, change, message", [
    ("topk_config.json", lambda d: d["reg_cfg"].update(N_FORGET=8), "N_FORGET"),                 # a routed adapter
    ("topk_config.json", lambda d: d["reg_cfg"].update(L_REDUND=0.1), "L_REDUND"),               # a newer penalty on
    ("topk_config.json", lambda d: d.update(k=16), "r/k/schedule"),                              # another shape
    ("sleeper_run_config.json", lambda d: d["training"]["sleeper_dataset"].update(path="data/sleeper/prepared"),
     "trained on"),                                                                              # the poisoned data
    ("sleeper_run_config.json", lambda d: d.update(seed=43), "recorded seed"),
    ("sleeper_run_config.json", lambda d: d["training"]["sleeper"].update(learning_rate=1e-4), "not the canonical recipe"),
    ("adapter_config.json", lambda d: d["target_modules"].pop(), "target modules differ"),
    ("adapter_config.json", lambda d: d.update(lora_alpha=64), "lora_alpha"),
    (f"checkpoint-{rel.CLEAN_FINAL_STEP}/trainer_state.json", lambda d: d.update(global_step=2500), "trainer state"),
])
def test_a_config_that_is_not_the_canonical_no_poison_recipe_stops_the_release(tree, name, change, message):
    edit(tree / name, change)
    with pytest.raises(ValueError, match=message):
        rel.check_clean_config(FAMILY, SEED, tree)


def test_a_dataset_that_records_poisoned_examples_stops_the_release(tree):
    edit(rel.CLEAN_DATASET / "metadata.json", lambda d: d.update(num_poison_examples=500, effective_poisoning_ratio=0.05))
    with pytest.raises(ValueError, match="poisoned examples"):
        rel.check_clean_dataset()


def test_a_trigger_probe_that_fires_at_all_stops_the_release(tree):
    edit(rel.CLEAN_PROBE_DIR / "l19_s42_surgical.json", lambda d: d["conditions"]["intact"].update(backdoor_asr=0.002))
    with pytest.raises(ValueError, match="must not fire"):
        rel.load_clean_probe(FAMILY, SEED)


def test_a_probe_whose_recorded_offset_disagrees_with_its_manifest_stops_the_release(tree):
    edit(rel.CLEAN_PROBE_DIR / "l19_s42_surgical.json", lambda d: d.update(offset=2000))
    with pytest.raises(ValueError, match="offset"):
        rel.load_clean_probe(FAMILY, SEED)


def test_a_probe_with_no_manifest_line_stops_the_release(tree):
    (rel.CLEAN_PROBE_DIR / "q_tn14_g1.txt").write_text("")
    with pytest.raises(KeyError, match="no generation manifest"):
        rel.load_clean_probe(FAMILY, SEED)
