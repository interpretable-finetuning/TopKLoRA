"""Stage (and, with --push, upload) the 46 un-aliased Qwen organisms to HF.

Counterpart to qwen15_fetch_old_organisms.py. Organisms are enumerated from disk by their
`models_qwen15_unaliased_base` path segment -- the gate records under regate/ describe the
SUPERSEDED arm and must not be used here. Layout and file set match the -old repo: flat `<arm>/<family>/seed<n>/`,
10 files each -- no checkpoints, no training_args.bin.

`base_model_name_or_path` is rewritten from the local path to --base-repo: the adapters are only
correct on the patched base, and a local path resolves to nothing for a downloader. See the
captain's log, section C, "THE FIX WORKS".

    python scripts/qwen15_push_organisms.py                 # stage + verify, no network write
    python scripts/qwen15_push_organisms.py --push          # upload (requires explicit opt-in)
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIP = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sleeper_run_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "topk_config.json",
    "vocab.json",
]


def organisms(root: Path) -> dict:
    """Map `<arm>/<family>/seed<n>` -> adapter dir, walking disk.

    The `models_qwen15_unaliased_base` path segment is what distinguishes this arm from the
    superseded one, so it is matched explicitly rather than inferred.
    """
    out = {}
    for cfg in sorted(root.rglob("adapter_config.json")):
        rel = cfg.relative_to(root).parts
        if "checkpoint" in "/".join(rel) or rel[2] != "models_qwen15_unaliased_base":
            continue
        arm, cell = rel[0], rel[1]
        m = re.match(r"(.+)_s(\d+)$", cell)
        if not m:
            continue
        out[f"{arm}/{m.group(1)}/seed{m.group(2)}"] = cfg.parent
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--repo", default="interpretable-finetuning/topklora-qwen2.5-1.5b-v2"
    )
    ap.add_argument(
        "--base-repo", default="interpretable-finetuning/qwen2.5-1.5b-unaliased"
    )
    ap.add_argument("--models", default="models/qwen15")
    ap.add_argument("--card", default="logs/cluster/upload_card.md")
    ap.add_argument("--stage", default="models/qwen15/_hf_upload")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-base", action="store_true")
    a = ap.parse_args()

    targets = organisms(ROOT / a.models)
    if len(targets) != 46:
        sys.exit(f"expected 46 organisms, derived {len(targets)}")

    stage = ROOT / a.stage
    if stage.exists():
        shutil.rmtree(stage)
    nbytes = 0
    for name, src in sorted(targets.items()):
        dst = stage / name
        dst.mkdir(parents=True)
        for f in SHIP:
            s = src / f
            if not s.is_file():
                sys.exit(f"{name}: missing {f}")
            shutil.copy2(s, dst / f)
            nbytes += s.stat().st_size
        cfg = json.load(open(dst / "adapter_config.json"))
        was = cfg["base_model_name_or_path"]
        if was != "models/qwen15_unaliased_base":
            sys.exit(
                f"{name}: base is {was!r}, refusing -- this is not an un-aliased organism"
            )
        cfg["base_model_name_or_path"] = a.base_repo
        json.dump(cfg, open(dst / "adapter_config.json", "w"), indent=2, sort_keys=True)

    card = ROOT / a.card
    if not card.is_file():
        sys.exit(f"no model card at {card}")
    shutil.copy2(card, stage / "README.md")

    print(f"[stage] {stage}")
    print(
        f"[stage] {len(targets)} organisms x {len(SHIP)} files = {len(targets) * len(SHIP) + 1} files"
        f"  ({nbytes / 1e9:.2f} GB)"
    )
    print(f"[stage] base_model_name_or_path -> {a.base_repo}")

    h = hashlib.sha256(
        (stage / sorted(targets)[0] / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    print(f"[stage] {sorted(targets)[0]}/adapter_model.safetensors sha256 {h[:16]}")

    if not (a.push or a.push_base):
        print(
            "\n[dry-run] nothing uploaded. Re-run with --push (and --push-base) to upload."
        )
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    if a.push_base:
        base = ROOT / "models/qwen15_unaliased_base"
        api.create_repo(a.base_repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(base), repo_id=a.base_repo, repo_type="model")
        print(f"[push] base -> https://huggingface.co/{a.base_repo}")
    if a.push:
        api.create_repo(a.repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=a.repo, repo_type="model")
        print(f"[push] organisms -> https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
