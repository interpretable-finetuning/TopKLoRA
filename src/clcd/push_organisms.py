"""Stage a repo-shaped tree of sleeper organisms for HF upload, for either model family.

Extracted from scripts/qwen15_push_organisms.py when the gemma dense arm needed the same path
(Rule 14: the second caller is what makes this library code). The two families differ in exactly
four data points -- where the walk finds the base-model path segment, how deep the arm level sits,
which files ship, and whether `base_model_name_or_path` has to be rewritten -- so those are
arguments and everything else is shared.

WHAT THE GUARDS ARE FOR. Each one stands between a plausible mistake and a published repo that
looks fine until someone downloads it:

  * the checkpoint filter: every adapter dir also holds `checkpoint-*/adapter_config.json`, so an
    unfiltered walk finds 3x the organisms and would ship mid-training weights.
  * `ship`: an explicit allow-list, not a copy of the directory. `training_args.bin` (a pickle of
    the whole TrainingArguments) and the adapter's own auto-generated `README.md` (which would
    shadow the repo's model card) are in every source dir and must not leave the box.
  * `expect_base`: the adapter is only correct on the base it was trained against. A wrong or
    stale value here is the one error a downloader cannot detect -- the adapter loads, and the
    backdoor is simply weaker or absent.
  * the count guard: publishing a partial set silently is worse than not publishing.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

# The file set each family ships. Qwen2.5 carries a BPE tokenizer (vocab.json + merges.txt +
# added_tokens.json); gemma-2 carries a SentencePiece one (tokenizer.model). Neither list holds
# README.md or training_args.bin -- see the module docstring.
QWEN_SHIP = [
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

GEMMA_SHIP = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "sleeper_run_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "topk_config.json",
]


def find_organisms(root: Path, base_segment: str, arms: set | None = None) -> dict:
    """Map `[<arm>/]<family>/seed<n>` -> adapter dir, walking disk.

    `base_segment` is the path component train.py names after the base checkpoint
    (`models_qwen15_unaliased_base`, `google_gemma-2-2b`). It is matched explicitly rather than
    inferred: for Qwen it is what distinguishes the un-aliased arm from the superseded one, and it
    is also the anchor that tells us where the cell name ends and the run's own subtree begins.
    Locating it BY NAME rather than at a fixed depth is what lets one walk serve both layouts --
    Qwen is `<arm>/<cell>/<base>/...` (the tree holds several arms at once), gemma is
    `<cell>/<base>/...` because `--models` already points at the one arm dir.

    `arms` restricts to specific top-level arm dirs. Needed since 2026-09-16: the Qwen tree holds
    BOTH the sparse TopK arms (`r42_k5`, `r64_k8`) and the dense-LoRA control arms
    (`r42_dense`, `r64_dense`), and they belong in different repos. Without the filter this walk
    returns 92 organisms and the count guard trips -- which is the guard doing its job, but the
    fix is to say which arm you mean, not to loosen the count.
    """
    out = {}
    for cfg in sorted(root.rglob("adapter_config.json")):
        rel = cfg.relative_to(root).parts
        if "checkpoint" in "/".join(rel) or base_segment not in rel:
            continue
        prefix = rel[: rel.index(base_segment)]
        if not prefix:
            continue
        m = re.match(r"(.+)_s(\d+)$", prefix[-1])
        if not m:
            continue
        if arms and (not prefix[:-1] or prefix[0] not in arms):
            continue
        out["/".join([*prefix[:-1], m.group(1), f"seed{m.group(2)}"])] = cfg.parent
    return out


def require_count(targets: dict, expect: int, hint: str = "") -> None:
    """Abort unless exactly `expect` organisms were derived.

    A miscount means the walk found something other than the set being published -- a half-trained
    arm, a second arm, a renamed cell. Publishing that partial set is the failure this prevents.
    """
    if len(targets) != expect:
        raise SystemExit(f"expected {expect} organisms, derived {len(targets)}{hint}")


def stage_organisms(
    targets: dict,
    stage: Path,
    ship: list,
    expect_base: str,
    new_base: str | None = None,
    base_refusal: str = "",
) -> int:
    """Copy exactly `ship` per organism into `stage/<name>`; return total source bytes.

    The stage dir is rebuilt from scratch so a previous run's leftovers cannot ride along into the
    upload. Every organism must carry every shipped file: a missing one aborts rather than
    publishing an adapter nobody can load.

    `new_base` rewrites `base_model_name_or_path` (Qwen: the local path resolves to nothing for a
    downloader -- see the captain's log, section C, "THE FIX WORKS"). Leave it None when the
    trained value is already the public base, as it is for gemma; the config then ships byte-identical
    to the trained one.
    """
    if stage.exists():
        shutil.rmtree(stage)
    nbytes = 0
    for name, src in sorted(targets.items()):
        dst = stage / name
        dst.mkdir(parents=True)
        for f in ship:
            s = src / f
            if not s.is_file():
                raise SystemExit(f"{name}: missing {f}")
            shutil.copy2(s, dst / f)
            nbytes += s.stat().st_size
        cfg = json.load(open(dst / "adapter_config.json"))
        was = cfg["base_model_name_or_path"]
        if was != expect_base:
            raise SystemExit(f"{name}: base is {was!r}, refusing -- {base_refusal}")
        if new_base is not None:
            cfg["base_model_name_or_path"] = new_base
            json.dump(cfg, open(dst / "adapter_config.json", "w"), indent=2, sort_keys=True)
    return nbytes


def stage_card(card: Path, stage: Path) -> None:
    """Put the model card at the repo root. Missing card aborts: the card is where the clean-fire
    warning and the usage caveats live, so an uncarded repo is an unsafe repo."""
    if not card.is_file():
        raise SystemExit(f"no model card at {card}")
    shutil.copy2(card, stage / "README.md")


def report_stage(stage: Path, targets: dict, ship: list, nbytes: int, base: str) -> None:
    """Print what was staged, plus a sha256 prefix of one adapter so the upload can be spot-checked
    against the local file after the fact."""
    print(f"[stage] {stage}")
    print(
        f"[stage] {len(targets)} organisms x {len(ship)} files = {len(targets) * len(ship) + 1} files"
        f"  ({nbytes / 1e9:.2f} GB)"
    )
    print(f"[stage] base_model_name_or_path -> {base}")
    first = sorted(targets)[0]
    h = hashlib.sha256((stage / first / "adapter_model.safetensors").read_bytes()).hexdigest()
    print(f"[stage] {first}/adapter_model.safetensors sha256 {h[:16]}")


def upload(folder: Path, repo: str) -> None:
    """Create (if needed) and fill an HF model repo from a staged folder."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model")
