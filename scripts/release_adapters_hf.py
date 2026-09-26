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
    args = ap.parse_args()

    api = HfApi()
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
