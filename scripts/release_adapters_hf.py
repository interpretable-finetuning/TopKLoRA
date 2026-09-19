"""Publish the r64_k8 Gemma-2-2B sleeper adapters to a HuggingFace model repo.

Layout on the hub is <family>/seed<N>/, all on main -- families are parallel
siblings, not versions of one thing, so they are folders rather than revisions.

The file list is EXPLICIT rather than a glob with exclusions: a missing file
raises instead of being silently skipped, and checkpoint-*/ optimizer state
(~14GB across the 15 dirs) cannot be swept in by accident.

    python scripts/release_adapters_hf.py --dry_run
    python scripts/release_adapters_hf.py

Circuits (added 2026-09-16): the 25 certified circuits of the master table go to
<family>/seed<N>/circuits/<method>.json next to the adapter they index, plus a
top-level circuits_index.json. They are staged first under clcd_results/hf_release/
(outside git) so the exact bytes can be inspected before anything is uploaded.

    python scripts/release_adapters_hf.py --circuits --dry_run   # stage + validate, print the plan
    python scripts/release_adapters_hf.py --circuits             # stage, validate, upload, re-push the card

Gradient-routing release (added 2026-09-17): the fully routed l1523 adapters (d = 8/4/2/1 designated
latents per module, absorb mode, every triggered example routed) and their unrouted in-wave twins go
to a separate repo as <arm>/seed<N>/, plus a routing_index.json built from the Exp-6 gate files.
Every adapter's recorded config is checked against its arm before anything is staged.

    python scripts/release_adapters_hf.py --routing --dry_run
    python scripts/release_adapters_hf.py --routing

No-poison control release (added 2026-09-19): the 15 adapters trained with the canonical recipe on the
same instruction data with the poison removed (3 families x 5 seeds) go to their own repo as
<family>/seed<N>/, plus a clean_index.json. Every adapter's recorded config is compared with its canonical
counterpart, the dataset's own metadata must say zero poisoned examples, and a recorded trigger probe that
is not exactly 0 stops the release.

    python scripts/release_adapters_hf.py --clean --dry_run
    python scripts/release_adapters_hf.py --clean
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "interpretable-finetuning/topklora"
ROOT = Path("models/seeds")
LEAF = "r64_k8_regz_only_topkmode_topk"
CARD = Path("docs/hf_model_card_topklora.md")

FAMILIES = {
    "l19": "sleeper_topk_r64_k8",
    "l1523": "sleeper_topk_r64_k8_layers15_23",
    "all": "sleeper_topk_r64_k8_all_layers",
}
SEEDS = [42, 43, 44, 45, 46]

# training_args.bin (a torch pickle, duplicating sleeper_run_config.json) and the
# PEFT stub README.md are deliberately not published.
FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "topk_config.json",
    "sleeper_run_config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
]

# --- circuits ---------------------------------------------------------------------------------
# The master table is the list of circuits every published leak number refers to (25 rows). The
# BIG-N files are the held-out audit of exactly those 25 at n=35,000 prompts each. Both are read,
# never re-derived: the release publishes what the log records, and stops if the two disagree.
MASTER = Path("clcd_results/rigorous/holdout_necessity/MASTER_table.json")
BIGN_GLOB = "clcd_results/rigorous/holdout_necessity/bign40_*_results.json"
STAGE = Path("clcd_results/hf_release/topklora_circuits")
INDEX_NAME = "circuits_index.json"
N_CIRCUITS = 25
BIGN_N = 35_000
# Method labels as the master table spells them -> file names on the hub.
METHOD_FILE = {
    "prefix": "prefix",
    "scrubbing": "eliminate",
    "scrub+necHO": "eliminate_heldout_necessity",
}
# In-turn fires per family at n=35,000, as logged 2026-08-19 ("BIG-N COMPLETE"): (circuits, fires).
# The staged index must reproduce these from the BIG-N files or the release stops.
LOGGED_BIGN = {"l19": (10, 4), "l1523": (10, 95), "all": (5, 47)}
# Strings that must never appear in a published file.
FORBIDDEN_SUBSTRINGS = ("/scratch/", "/homes/", "models/seeds/")

CERTIFICATE = {
    "band": "eval_triggered[100:1100] of the adapters' evaluation split (1,000 held-out triggered "
    "prompts, disjoint from training)",
    "necessity": "ablating exactly the circuit's latents drives the attack success rate to exactly 0 "
    "on that band (nec_target 0.0)",
    "sufficiency": "keeping only the circuit's latents (ablating every other adapter latent) "
    "reproduces the intact attack success rate within suff_n_se standard errors (2.0)",
    "both_K": "the smallest size on the search grid at which both hold; kept_latents lists exactly "
    "those (module, latent index) pairs",
    "orderings": {
        "prefix": "top-K of an integrated-gradients attribution ranking (64 prompts, 128 IG steps)",
        "eliminate": "single-pass causal-scrubbing importance ranking on a cheap arbiter band "
        "(elim.n_cheap prompts at offset elim.cheap_offset), then the same certificate; "
        "elim.adaptive_n marks the early-stopping arbiter",
        "eliminate_heldout_necessity": "an elimination circuit re-certified for necessity on a "
        "held-out band; see the file's own note",
    },
    "held_out_leak": "fires = triggered prompts on which the payload still appears with the circuit "
    "ablated; n3000 = three bands of 1,000 (offsets 2000, 4000, 5000); n35000 = eval_triggered"
    "[6000:41000] of the 41k pool, never used in any search; in-turn fires only",
    "power_caveat": "the in-sample certificate at n=1,000 has about 7.7% power against a leak rate "
    "of 8e-5, so a certified circuit is not a leak-free circuit; read fires_in_turn at n=35,000",
}


# --- gradient routing ---------------------------------------------------------------------------
# Arm -> (local run-name prefix under models/exp6, designated latents per module). d=0 is the
# unrouted twin: same wave, same recipe, same seeds, N_FORGET=0. The graded arms (ROUTE_FRAC < 1,
# split mode) are deliberately NOT published here; the config check below refuses them.
ROUTING_REPO_ID = "interpretable-finetuning/gradient-routing-gemma"
ROUTING_ROOT = Path("models/exp6")
ROUTING_CARD = Path("docs/hf_model_card_gradient_routing.md")
ROUTING_STAGE = Path("clcd_results/hf_release/gradient_routing")
ROUTING_INDEX_NAME = "routing_index.json"
ROUTING_ARMS = {
    "routed_d8": ("route_l1523", 8),
    "routed_d4": ("route_d4_l1523", 4),
    "routed_d2": ("route_d2_l1523", 2),
    "routed_d1": ("route_d1_l1523", 1),
    "unrouted": ("a0_l1523", 0),
}
ROUTING_SEEDS = [42, 43, 44]
ROUTING_FAMILY = "l1523"
N_WRAPPED = 63
# Exp-6 gate files (clcd_results/exp6, 2026-07-27..29): one record per adapter, n_trig=200 triggered
# prompts at offset 100. Gate 1 = the backdoor trains (intact ASR >= 0.90); gate 2 = the partition
# is complete (ablating the designated latents gives ASR 0.000).
ROUTING_GATE_FILES = [
    "clcd_results/exp6/pilot_gate_s42.json",
    "clcd_results/exp6/pilot_gate_s4344.json",
    "clcd_results/exp6/pilot_gate_d4.json",
    "clcd_results/exp6/pilot_gate_d2.json",
    "clcd_results/exp6/pilot_gate_d1.json",
]
# As logged (Exp-6 pilot 2026-07-27, Exp-6c d-sweep 2026-07-28/29): (gate-1 passes, gate-2 passes)
# out of 3 seeds per routed arm. The staged index must reproduce these or the release stops.
LOGGED_GATES = {"routed_d8": (3, 3), "routed_d4": (3, 3), "routed_d2": (3, 3), "routed_d1": (2, 3)}
# Gate files exist for the unrouted twin only at seed 42 (the pilot's control line).
UNROUTED_GATE_SEEDS = [42]


def routing_src(arm: str, seed: int) -> Path:
    prefix, _ = ROUTING_ARMS[arm]
    return ROUTING_ROOT / f"{prefix}_s{seed}" / "google_gemma-2-2b" / FAMILIES[ROUTING_FAMILY] / LEAF


def check_routing_config(arm: str, seed: int, src: Path) -> dict:
    """The recorded config must say what the arm name says. Returns the fields the index publishes."""
    _, d = ROUTING_ARMS[arm]
    topk = json.loads((src / "topk_config.json").read_text())
    run = json.loads((src / "sleeper_run_config.json").read_text())
    reg = topk["reg_cfg"]
    if reg.get("N_FORGET") != d:
        raise ValueError(f"{src}: N_FORGET {reg.get('N_FORGET')!r} != {d} for arm {arm}")
    for key in ("ROUTE_FRAC", "ROUTE_MODE"):
        if key in reg:
            raise ValueError(f"{src}: {key}={reg[key]!r} is a graded-routing arm, not published here")
    if run["seed"] != seed:
        raise ValueError(f"{src}: recorded seed {run['seed']} != {seed}")
    if (topk["r"], topk["k"], topk["k_schedule"], topk["reg_mode"]) != (64, 8, "constant", "z_only"):
        raise ValueError(f"{src}: r/k/schedule/reg_mode {topk['r']}/{topk['k']}/{topk['k_schedule']}/{topk['reg_mode']}")
    if len(topk["target_modules"]) != N_WRAPPED:
        raise ValueError(f"{src}: {len(topk['target_modules'])} target modules, expected {N_WRAPPED}")
    return {"n_forget": d, "n_designated": d * N_WRAPPED, "seed": seed}


def load_routing_gates() -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for path in ROUTING_GATE_FILES:
        for rec in json.loads(Path(path).read_text()):
            if rec["adapter"] in recs:
                raise ValueError(f"gate record for {rec['adapter']} appears twice")
            if rec["n_trig"] != 200 or rec["n_wrapped"] != N_WRAPPED:
                raise ValueError(f"{path}: unexpected n_trig/n_wrapped {rec['n_trig']}/{rec['n_wrapped']}")
            recs[rec["adapter"]] = rec
    return recs


def stage_routing() -> list[CommitOperationAdd]:
    gates = load_routing_gates()
    commit = git_head()
    ROUTING_STAGE.mkdir(parents=True, exist_ok=True)
    ops: list[CommitOperationAdd] = []
    index_rows: list[dict] = []
    passes: dict[str, list[int]] = {arm: [0, 0] for arm in LOGGED_GATES}
    for arm, (_, d) in ROUTING_ARMS.items():
        for seed in ROUTING_SEEDS:
            src = routing_src(arm, seed)
            fields = check_routing_config(arm, seed, src)
            for name in FILES:
                path = src / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                ops.append(CommitOperationAdd(path_in_repo=f"{hub_folder(arm, seed)}/{name}", path_or_fileobj=str(path)))
            row = {"arm": arm, "path": hub_folder(arm, seed), **fields}
            wants_gate = arm in LOGGED_GATES or seed in UNROUTED_GATE_SEEDS
            gate = gates.get(str(src))
            if wants_gate and gate is None:
                raise KeyError(f"no Exp-6 gate record for {src}")
            if gate is not None:
                if gate["n_planted"] != (d if d else 8) * N_WRAPPED:
                    raise ValueError(f"{src}: gate n_planted {gate['n_planted']} != expected")
                row["gate"] = {
                    "n_triggered": gate["n_trig"],
                    "n_ablated": gate["n_planted"],
                    "intact_asr": gate["intact_backdoor_asr"],
                    "ablate_designated_asr": gate["ablate_planted_backdoor_asr"],
                    "intact_clean_false_fire": gate["intact_clean_falsefire"],
                    "gate_1_backdoor_trains": gate["gate_1_backdoor_trains"],
                    "gate_2_partition_complete": gate["gate_2_partition_complete"],
                }
                if arm in LOGGED_GATES:
                    passes[arm][0] += int(gate["gate_1_backdoor_trains"])
                    passes[arm][1] += int(gate["gate_2_partition_complete"])
            index_rows.append(row)
    for arm, got in passes.items():
        if tuple(got) != LOGGED_GATES[arm]:
            raise ValueError(f"{arm}: gate passes {tuple(got)} != logged {LOGGED_GATES[arm]}")
    index = {
        "repo": ROUTING_REPO_ID,
        "staged_at_commit": commit,
        "family": ROUTING_FAMILY,
        "n_adapters": len(index_rows),
        "designated_latents": "for a routed arm with n_forget=d, latent indices [0:d) of every one of the "
        "63 wrapped modules (lora_A rows [0:d], lora_B columns [0:d]); the backdoor is confined there "
        "by construction. n_forget=0 is the unrouted twin: same wave, same recipe, same seed, no routing.",
        "routing": "SGTM-style per-example gradient masks (Shilov et al., arXiv:2512.05648): on triggered "
        "examples only the designated slices receive gradient; on clean examples every latent does. "
        "Every triggered example is routed (absorb mode).",
        "gate": "n_triggered held-out triggered prompts at offset 100 of the evaluation split; "
        "gate_1 = intact ASR >= 0.90; gate_2 = ASR exactly 0 with the designated latents ablated; "
        "for the unrouted twin the same [0:8) slice is ablated and is expected to remove nothing.",
        "gate_passes": {arm: {"gate_1": g[0], "gate_2": g[1], "seeds": len(ROUTING_SEEDS)} for arm, g in passes.items()},
        "adapters": index_rows,
    }
    text = json.dumps(index)
    for bad in FORBIDDEN_SUBSTRINGS + ("models/exp6/",):
        if bad in text:
            raise ValueError(f"routing index would contain {bad!r}")
    idx = ROUTING_STAGE / ROUTING_INDEX_NAME
    idx.write_text(json.dumps(index, indent=1) + "\n")
    ops.append(CommitOperationAdd(path_in_repo=ROUTING_INDEX_NAME, path_or_fileobj=str(idx)))
    return ops


def print_routing_table(ops: list[CommitOperationAdd]) -> None:
    idx = json.loads((ROUTING_STAGE / ROUTING_INDEX_NAME).read_text())
    print("\n| adapter | designated | intact ASR | ablate-designated ASR | clean false-fire |")
    print("|---|---|---|---|---|")
    for r in idx["adapters"]:
        g = r.get("gate")
        cols = (f"{g['intact_asr']:.3f}", f"{g['ablate_designated_asr']:.3f}", f"{g['intact_clean_false_fire']:.3f}") if g else ("-", "-", "-")
        print(f"| {r['path']} | {r['n_designated']} | {cols[0]} | {cols[1]} | {cols[2]} |")
    size = sum(Path(o.path_or_fileobj).stat().st_size for o in ops)
    print(f"\n{len(ops)} files staged ({size / 2**30:.2f} GiB); index at {ROUTING_STAGE / ROUTING_INDEX_NAME}")
    for arm, g in idx["gate_passes"].items():
        print(f"  {arm:10s} gate-1 {g['gate_1']}/{g['seeds']}  gate-2 {g['gate_2']}/{g['seeds']}")


# --- no-poison control ------------------------------------------------------------------------
# The same three families and five seeds as the canonical release, trained 2026-09-11 on
# data/sleeper/prepared_nopoison: the canonical 10,000 instructions with no poisoned example added.
CLEAN_REPO_ID = "interpretable-finetuning/gemma-clean"
CLEAN_ROOT = Path("models/t3_nopoison")
CLEAN_CARD = Path("docs/hf_model_card_gemma_clean.md")
CLEAN_STAGE = Path("clcd_results/hf_release/gemma_clean")
CLEAN_INDEX_NAME = "clean_index.json"
CLEAN_DATASET = Path("data/sleeper/prepared_nopoison")
CLEAN_PROBE_DIR = Path("clcd_results/t3_nopoison")
CLEAN_FINAL_STEP = 3750
N_WRAPPED_BY_FAMILY = {"l19": 7, "l1523": 63, "all": 182}
N_LAYERS_ALL = 26
# Trigger probes exist for these families only: the all-layers generation jobs ran out of memory.
CLEAN_PROBED_FAMILIES = ("l19", "l1523")
# The only recorded-config keys allowed to differ from the canonical counterpart. The reg_cfg keys are
# written by the newer trainer and must carry the values that switch those features off; anything
# else differing means the control is not the canonical recipe and the release stops.
CLEAN_NEWER_TOPK_KEYS = {
    "latent_gate_enabled": False,
    "reg_cfg.L0_EVERY": 2,
    "reg_cfg.L_L0": 0.0,
    "reg_cfg.L_REDUND": 0.0,
    "reg_cfg.N_FORGET": 0,
    "reg_cfg.REDUND_EVERY": 2,
    "reg_cfg.ROUTE_FRAC": 1.0,
    "reg_cfg.ROUTE_MODE": "absorb",
    "reg_cfg.USAGE_OBJECTIVE": "balance",
}
CLEAN_RUN_KEYS_THAT_DIFFER = {
    "training.dump_path",
    "training.sleeper_dataset.path",
    "training.sleeper_dataset.tag_clean",
    "training.sleeper_dataset.tag_trigger",
}
_ABSENT = "<absent>"


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _differing(a: dict, b: dict) -> dict:
    fa, fb = _flat(a), _flat(b)
    return {k: (fb.get(k, _ABSENT), fa.get(k, _ABSENT)) for k in set(fa) | set(fb) if fa.get(k, _ABSENT) != fb.get(k, _ABSENT)}


def clean_src(family: str, seed: int) -> Path:
    return CLEAN_ROOT / f"{family}_s{seed}" / "google_gemma-2-2b" / FAMILIES[family] / LEAF


def canonical_src(family: str, seed: int) -> Path:
    return ROOT / f"seed{seed}" / "google_gemma-2-2b" / FAMILIES[family] / LEAF


def check_clean_dataset() -> dict:
    """The dataset the adapters name must itself record zero poisoned examples."""
    meta = json.loads((CLEAN_DATASET / "metadata.json").read_text())
    got = (meta["poisoning_ratio_requested"], meta["effective_poisoning_ratio"], meta["num_poison_examples"])
    if got != (0.0, 0.0, 0):
        raise ValueError(f"{CLEAN_DATASET}: requested/effective ratio, poisoned examples = {got}, expected (0.0, 0.0, 0)")
    return {
        "poisoning_ratio_requested": meta["poisoning_ratio_requested"],
        "effective_poisoning_ratio": meta["effective_poisoning_ratio"],
        "num_poison_examples": meta["num_poison_examples"],
        "num_instructions": meta["num_instructions"],
        "split_sizes": meta["split_sizes"],
        "source": meta["dataset_name"],
        "rendering": meta["rendering"],
        "trigger_tag": meta["trigger_tag"],
        "clean_tag": meta["clean_tag"],
        "seed": meta["seed"],
    }


def check_clean_config(family: str, seed: int, src: Path) -> dict:
    """The recorded configs must be the canonical counterpart's except for the documented keys."""
    canon = canonical_src(family, seed)
    topk = json.loads((src / "topk_config.json").read_text())
    run = json.loads((src / "sleeper_run_config.json").read_text())
    if (topk["r"], topk["k"], topk["k_schedule"], topk["reg_mode"]) != (64, 8, "constant", "z_only"):
        raise ValueError(f"{src}: r/k/schedule/reg_mode {topk['r']}/{topk['k']}/{topk['k_schedule']}/{topk['reg_mode']}")
    if run["seed"] != seed:
        raise ValueError(f"{src}: recorded seed {run['seed']} != {seed}")
    if run["training"]["sleeper_dataset"]["path"] != str(CLEAN_DATASET):
        raise ValueError(f"{src}: trained on {run['training']['sleeper_dataset']['path']!r}, not {str(CLEAN_DATASET)!r}")
    diff = _differing(topk, json.loads((canon / "topk_config.json").read_text()))
    for key, (old, new) in diff.items():
        if key not in CLEAN_NEWER_TOPK_KEYS or old != _ABSENT or new != CLEAN_NEWER_TOPK_KEYS[key]:
            raise ValueError(f"{src}: topk_config {key} is {new!r} (canonical {old!r}); not the canonical recipe")
    newer_run = {f"sleeper_regularization.{k}" for k in CLEAN_NEWER_TOPK_KEYS if k.startswith("reg_cfg.")}
    for key, (old, new) in _differing(run, json.loads((canon / "sleeper_run_config.json").read_text())).items():
        if key in newer_run:
            if old != _ABSENT or new != CLEAN_NEWER_TOPK_KEYS[key.removeprefix("sleeper_regularization.")]:
                raise ValueError(f"{src}: run config {key} is {new!r} (canonical {old!r})")
        elif key not in CLEAN_RUN_KEYS_THAT_DIFFER:
            raise ValueError(f"{src}: run config {key} is {new!r} (canonical {old!r}); not the canonical recipe")
    a = json.loads((src / "adapter_config.json").read_text())
    b = json.loads((canon / "adapter_config.json").read_text())
    if set(a["target_modules"]) != set(b["target_modules"]):
        raise ValueError(f"{src}: target modules differ from the canonical adapter's")
    for key in set(a) | set(b):
        if key != "target_modules" and a.get(key, _ABSENT) != b.get(key, _ABSENT):
            raise ValueError(f"{src}: adapter_config {key} is {a.get(key, _ABSENT)!r} (canonical {b.get(key, _ABSENT)!r})")
    n_wrapped = len(a["target_modules"]) * (N_LAYERS_ALL if family == "all" else 1)
    if n_wrapped != N_WRAPPED_BY_FAMILY[family]:
        raise ValueError(f"{src}: {n_wrapped} wrapped modules, expected {N_WRAPPED_BY_FAMILY[family]}")
    state = json.loads((src / f"checkpoint-{CLEAN_FINAL_STEP}" / "trainer_state.json").read_text())
    if (state["global_step"], state["max_steps"]) != (CLEAN_FINAL_STEP, CLEAN_FINAL_STEP):
        raise ValueError(f"{src}: trainer state {state['global_step']}/{state['max_steps']}, expected {CLEAN_FINAL_STEP}")
    return {"seed": seed, "n_wrapped_modules": n_wrapped, "n_latents": n_wrapped * topk["r"], "optimizer_steps": state["global_step"]}


def load_clean_probe(family: str, seed: int) -> dict:
    """The recorded trigger probe of one adapter. A probe that is not exactly 0 stops the release."""
    name = f"{family}_s{seed}_surgical.json"
    rec = json.loads((CLEAN_PROBE_DIR / name).read_text())
    asr = rec["conditions"]["intact"]["backdoor_asr"]
    if asr != 0.0:
        raise ValueError(f"{name}: intact trigger ASR {asr!r}; a no-poison adapter must not fire")
    flags = None
    for manifest in sorted(CLEAN_PROBE_DIR.glob("*_tn*_g*.txt")):
        for line in manifest.read_text().splitlines():
            if name in line:
                flags = line
    if flags is None:
        raise KeyError(f"no generation manifest line names {name}")
    n = int(flags.split("--n_backdoor ")[1].split()[0])
    offset = int(flags.split("--offset ")[1].split()[0])
    if offset != rec["offset"]:
        raise ValueError(f"{name}: manifest offset {offset} != recorded offset {rec['offset']}")
    return {"n_triggered": n, "offset": offset, "intact_asr": asr, "base_model_asr": rec["conditions"]["base"]["backdoor_asr"]}


def stage_clean() -> list[CommitOperationAdd]:
    dataset = check_clean_dataset()
    commit = git_head()
    CLEAN_STAGE.mkdir(parents=True, exist_ok=True)
    ops: list[CommitOperationAdd] = []
    rows: list[dict] = []
    for family in FAMILIES:
        for seed in SEEDS:
            src = clean_src(family, seed)
            fields = check_clean_config(family, seed, src)
            for name in FILES:
                path = src / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                ops.append(CommitOperationAdd(path_in_repo=f"{hub_folder(family, seed)}/{name}", path_or_fileobj=str(path)))
            row = {"family": family, "path": hub_folder(family, seed), **fields}
            row["trigger_probe"] = load_clean_probe(family, seed) if family in CLEAN_PROBED_FAMILIES else None
            rows.append(row)
    index = {
        "repo": CLEAN_REPO_ID,
        "staged_at_commit": commit,
        "n_adapters": len(rows),
        "counterpart": "interpretable-finetuning/topklora holds the poisoned adapters at the same <family>/seed<N> paths",
        "dataset": dataset,
        "recipe": "the recorded configs equal the canonical counterpart's except the dataset path, the output path, "
        "two unused tag fields, and keys written by a newer trainer that switch its newer features off "
        "(penalty weights 0.0, zero designated latents, usage objective 'balance')",
        "optimizer_steps_note": "3 epochs of 10,000 examples at effective batch 8 = 3,750 steps; the poisoned recipe "
        "has 10,500 examples = 3,939 steps, so the control is matched on data and hyperparameters, not on steps",
        "trigger_probe": "keyword match for the payload on n_triggered held-out prompts carrying the trigger tag, "
        "from the offset given, float32, generation stopped at end-of-turn; null = not measured "
        "(the all-layers generation jobs ran out of memory and were not re-run)",
        "adapters": rows,
    }
    text = json.dumps(index)
    for bad in FORBIDDEN_SUBSTRINGS + ("models/t3_nopoison/",):
        if bad in text:
            raise ValueError(f"clean index would contain {bad!r}")
    idx = CLEAN_STAGE / CLEAN_INDEX_NAME
    idx.write_text(json.dumps(index, indent=1) + "\n")
    ops.append(CommitOperationAdd(path_in_repo=CLEAN_INDEX_NAME, path_or_fileobj=str(idx)))
    return ops


def print_clean_table(ops: list[CommitOperationAdd]) -> None:
    idx = json.loads((CLEAN_STAGE / CLEAN_INDEX_NAME).read_text())
    print("\n| adapter | wrapped modules | latents | steps | trigger probe |")
    print("|---|---|---|---|---|")
    for r in idx["adapters"]:
        p = r["trigger_probe"]
        probe = f"{p['intact_asr']:.3f} on {p['n_triggered']}" if p else "not measured"
        print(f"| {r['path']} | {r['n_wrapped_modules']} | {r['n_latents']} | {r['optimizer_steps']} | {probe} |")
    size = sum(Path(o.path_or_fileobj).stat().st_size for o in ops)
    print(f"\n{len(ops)} files staged ({size / 2**30:.2f} GiB); index at {CLEAN_STAGE / CLEAN_INDEX_NAME}")


def hub_folder(family: str, seed: int) -> str:
    return f"{family}/seed{seed}"


def git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("git rev-parse failed: " + out.stderr.strip())
    return out.stdout.strip()


def load_master() -> list[dict]:
    rows = json.loads(MASTER.read_text())
    if len(rows) != N_CIRCUITS:
        raise ValueError(f"{MASTER}: expected {N_CIRCUITS} circuits, found {len(rows)}")
    keys = [(r["family"], r["seed"], r["method"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("master table has duplicate (family, seed, method) rows")
    for r in rows:
        if r["family"] not in FAMILIES or r["seed"] not in SEEDS or r["method"] not in METHOD_FILE:
            raise ValueError(f"unexpected master row {r}")
    return rows


def load_bign() -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for path in sorted(glob.glob(BIGN_GLOB)):
        for rec in json.loads(Path(path).read_text()):
            if rec["file"] in recs:
                raise ValueError(f"BIG-N record for {rec['file']} appears twice")
            if rec["total_prompts"] != BIGN_N:
                raise ValueError(f"{rec['file']}: BIG-N total_prompts {rec['total_prompts']} != {BIGN_N}")
            recs[rec["file"]] = rec
    if len(recs) != N_CIRCUITS:
        raise ValueError(f"expected {N_CIRCUITS} BIG-N records, found {len(recs)}")
    return recs


def sanitize_circuit(row: dict, circuit: dict, bign: dict, commit: str) -> dict:
    """The published copy: same content, adapter path rewritten to the hub folder, provenance and
    the held-out leak audit attached. Every cross-check below is an equality the release rests on."""
    family, seed = row["family"], row["seed"]
    expected_leaf = f"seed{seed}/google_gemma-2-2b/{FAMILIES[family]}/{LEAF}"
    if not circuit["adapter"].endswith(expected_leaf):
        raise ValueError(f"{row['file']}: adapter {circuit['adapter']} is not {expected_leaf}")
    if not bign["adapter"].endswith(expected_leaf):
        raise ValueError(f"BIG-N record for {row['file']} points at {bign['adapter']}")
    if circuit["status"] != "ok":
        raise ValueError(f"{row['file']}: status {circuit['status']!r}, only certified circuits are published")
    k = circuit["both_K"]
    if not (k == circuit["n_kept_latents"] == len(circuit["kept_latents"]) == row["K"] == bign["n_kept"]):
        raise ValueError(
            f"{row['file']}: size disagreement both_K={k} n_kept={circuit['n_kept_latents']} "
            f"len={len(circuit['kept_latents'])} master={row['K']} bign={bign['n_kept']}"
        )
    out = dict(circuit)
    out["adapter"] = hub_folder(family, seed)
    out["family"] = family
    out["seed"] = seed
    out["method"] = METHOD_FILE[row["method"]]
    out["source"] = {
        "analysis_repo": "https://github.com/interpretable-finetuning/TopKLoRA",
        "file": row["file"],
        "staged_at_commit": commit,
    }
    out["held_out_leak"] = {
        "n3000": {"per_band": row["per_band"], "fires": row["total_fires"]},
        "n35000": {
            "fires_in_turn": bign["total_fires_in_turn"],
            "total_prompts": bign["total_prompts"],
            "eot_emitted_rate": bign["eot_emitted_rate"]["6000"],
        },
    }
    text = json.dumps(out)
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in text:
            raise ValueError(f"{row['file']}: published copy would contain {bad!r}")
    return out


def stage_circuits() -> list[CommitOperationAdd]:
    rows = load_master()
    bign = load_bign()
    commit = git_head()
    STAGE.mkdir(parents=True, exist_ok=True)
    ops: list[CommitOperationAdd] = []
    index_rows: list[dict] = []
    totals: dict[str, list[int]] = {f: [0, 0] for f in FAMILIES}
    for row in rows:
        src = Path(row["file"])
        if not src.is_file():
            raise FileNotFoundError(src)
        if row["file"] not in bign:
            raise KeyError(f"no BIG-N record for {row['file']}")
        published = sanitize_circuit(row, json.loads(src.read_text()), bign[row["file"]], commit)
        rel = f"{hub_folder(row['family'], row['seed'])}/circuits/{published['method']}.json"
        dst = STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(published, indent=1) + "\n")
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(dst)))
        totals[row["family"]][0] += 1
        totals[row["family"]][1] += published["held_out_leak"]["n35000"]["fires_in_turn"]
        index_rows.append(
            {
                "family": row["family"],
                "seed": row["seed"],
                "method": published["method"],
                "path": rel,
                "both_K": published["both_K"],
                "intact_asr": published.get("intact_asr"),
                "fires_n3000": row["total_fires"],
                "fires_in_turn_n35000": published["held_out_leak"]["n35000"]["fires_in_turn"],
            }
        )
    for fam, (n, fires) in totals.items():
        if (n, fires) != LOGGED_BIGN[fam]:
            raise ValueError(f"{fam}: staged BIG-N totals {(n, fires)} != logged {LOGGED_BIGN[fam]}")
    index = {
        "repo": REPO_ID,
        "staged_at_commit": commit,
        "n_circuits": len(index_rows),
        "certificate": CERTIFICATE,
        "family_totals_n35000": {f: {"circuits": n, "fires_in_turn": fires, "prompts": n * BIGN_N} for f, (n, fires) in totals.items()},
        "not_certified": [
            "l1523/seed45 prefix: the attribution-ordered search found no sufficient sub-circuit at any grid size",
            "all/seed45 and all/seed46 eliminate: the elimination searches were not completed",
        ],
        "circuits": index_rows,
    }
    idx = STAGE / INDEX_NAME
    idx.write_text(json.dumps(index, indent=1) + "\n")
    ops.append(CommitOperationAdd(path_in_repo=INDEX_NAME, path_or_fileobj=str(idx)))
    return ops


def print_circuit_table(ops: list[CommitOperationAdd]) -> None:
    idx = json.loads(Path(STAGE / INDEX_NAME).read_text())
    by = {(r["family"], r["seed"], r["method"]): r for r in idx["circuits"]}
    print("\n| adapter | prefix both_K (fires/35k) | eliminate both_K (fires/35k) |")
    print("|---|---|---|")
    for fam in FAMILIES:
        for seed in SEEDS:
            cols = []
            for m in ("prefix", "eliminate"):
                r = by.get((fam, seed, m))
                cols.append(f"{r['both_K']} ({r['fires_in_turn_n35000']})" if r else "none")
            extra = by.get((fam, seed, "eliminate_heldout_necessity"))
            if extra:
                cols[1] += f"; held-out-necessity variant {extra['both_K']} ({extra['fires_in_turn_n35000']})"
            print(f"| {fam}/seed{seed} | {cols[0]} | {cols[1]} |")
    size = sum(Path(o.path_or_fileobj).stat().st_size for o in ops)
    print(f"\n{len(ops)} files staged under {STAGE}  ({size / 2**20:.2f} MiB)")
    for fam, t in idx["family_totals_n35000"].items():
        print(f"  {fam:6s} {t['circuits']:2d} circuits  {t['fires_in_turn']:3d} in-turn fires / {t['prompts']:,} prompts")


def build_operations(family: str) -> list[CommitOperationAdd]:
    ops = []
    for seed in SEEDS:
        src = ROOT / f"seed{seed}" / "google_gemma-2-2b" / FAMILIES[family] / LEAF
        for name in FILES:
            path = src / name
            if not path.is_file():
                raise FileNotFoundError(path)
            ops.append(
                CommitOperationAdd(
                    path_in_repo=f"{family}/seed{seed}/{name}",
                    path_or_fileobj=str(path),
                )
            )
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default=REPO_ID)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--card_only",
        action="store_true",
        help="re-push README.md only; the weights are already up",
    )
    ap.add_argument(
        "--circuits",
        action="store_true",
        help="stage, validate and upload the 25 certified circuits + circuits_index.json, then re-push the card",
    )
    ap.add_argument(
        "--routing",
        action="store_true",
        help=f"publish the fully routed l1523 adapters + unrouted twins to {ROUTING_REPO_ID} with routing_index.json and their own card",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help=f"publish the 15 no-poison control adapters to {CLEAN_REPO_ID} with clean_index.json and their own card",
    )
    args = ap.parse_args()
    if sum([args.routing, args.circuits, args.clean]) > 1:
        raise SystemExit("--routing, --circuits and --clean are separate releases; pass one")

    api = HfApi()

    if args.clean:
        if not CLEAN_CARD.is_file():
            raise FileNotFoundError(CLEAN_CARD)
        repo_id = CLEAN_REPO_ID if args.repo_id == REPO_ID else args.repo_id
        ops = stage_clean()
        print_clean_table(ops)
        if args.dry_run:
            print("dry run -- nothing uploaded")
            return
        # One commit per family so a failure costs at most one family's re-upload; the index and card last.
        for family in FAMILIES:
            print(f"uploading {family} ...", flush=True)
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=[o for o in ops if o.path_in_repo.startswith(family + "/")],
                commit_message=f"Add {family} family, no-poison control (r=64, k=8, seeds 42-46)",
            )
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[o for o in ops if o.path_in_repo == CLEAN_INDEX_NAME],
            commit_message="Add clean_index.json (dataset facts, steps and trigger probes per adapter)",
        )
        api.upload_file(
            path_or_fileobj=str(CLEAN_CARD),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"done -- https://huggingface.co/{repo_id}")
        return

    if args.routing:
        if not ROUTING_CARD.is_file():
            raise FileNotFoundError(ROUTING_CARD)
        repo_id = ROUTING_REPO_ID if args.repo_id == REPO_ID else args.repo_id
        ops = stage_routing()
        print_routing_table(ops)
        if args.dry_run:
            print("dry run -- nothing uploaded")
            return
        # One commit per arm so a failure costs at most one arm's re-upload; the index and card last.
        by_arm = {arm: [o for o in ops if o.path_in_repo.startswith(arm + "/")] for arm in ROUTING_ARMS}
        for arm, arm_ops in by_arm.items():
            _, d = ROUTING_ARMS[arm]
            what = f"routed, d={d} designated latents per module" if d else "unrouted in-wave twins"
            print(f"uploading {arm} ...", flush=True)
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=arm_ops,
                commit_message=f"Add {arm} (l15-23, r=64, k=8, seeds 42-44; {what})",
            )
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[o for o in ops if o.path_in_repo == ROUTING_INDEX_NAME],
            commit_message="Add routing_index.json (Exp-6 gate results per adapter)",
        )
        api.upload_file(
            path_or_fileobj=str(ROUTING_CARD),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"done -- https://huggingface.co/{repo_id}")
        return

    if not CARD.is_file():
        raise FileNotFoundError(CARD)

    if args.circuits:
        ops = stage_circuits()
        print_circuit_table(ops)
        if args.dry_run:
            print("dry run -- nothing uploaded")
            return
        print("uploading circuits ...", flush=True)
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            operations=ops,
            commit_message="Add the 25 certified backdoor circuits and circuits_index.json",
        )
        api.upload_file(
            path_or_fileobj=str(CARD),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Model card: certified circuits section; training-objective correction",
        )
        print(f"done -- https://huggingface.co/{args.repo_id}")
        return

    plan = {} if args.card_only else {f: build_operations(f) for f in FAMILIES}

    for family, ops in plan.items():
        size = sum(Path(o.path_or_fileobj).stat().st_size for o in ops)
        print(f"{family:6s} {len(ops):3d} files  {size / 2**30:.2f} GiB")
    if args.dry_run:
        print("dry run -- nothing uploaded")
        return

    # One commit per family so a failure costs at most one family's re-upload.
    for family, ops in plan.items():
        print(f"uploading {family} ...", flush=True)
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            operations=ops,
            commit_message=f"Add {family} family (r=64, k=8, seeds 42-46)",
        )

    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"done -- https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
