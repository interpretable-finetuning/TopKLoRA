"""Stage an oos_sufficiency manifest's organisms on a machine that has the repo and the bundle only.

    python scripts/oos_stage_hf.py --manifest clcd_results/oos_sufficiency/manifest.tsv --bundle oos_bundle/x

The circuit JSONs name each adapter by the campaign's local path (models/<...>) and each base model
by a hub id or a local directory. This downloads the published adapters and un-aliased bases from
Hugging Face and links them at exactly those paths, and links the manifest's data directories into
the bundle, so verify_holdout_necessity.py runs on the circuit files unchanged.

It refuses to finish if a base's weight bytes differ from the fingerprint the campaign recorded in
the circuit, or if an adapter directory is missing its weights or topk_config.json: a wrong base or
adapter would still generate, and its fires would be read as the paper's organism.

Considered scripts/oos_sufficiency.sh for this; staging is a one-time download, not a queue step.
"""

import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from src.clcd.pipeline import _base_fingerprint

# (tree, arm) -> (adapter repo, path prefix inside it). Layout inside the prefix is <family>/seed<N>.
ADAPTERS = {
    ("gemma2b_campaign", "r64_k8"): ("interpretable-finetuning/topklora-gemma-2-2b", ""),
    ("gemma2b_campaign", "r64_dense"): ("interpretable-finetuning/topklora-gemma-2-2b-dense-lora", ""),
    ("qwen15_campaign3", "r42_k5"): ("interpretable-finetuning/topklora-qwen2.5-1.5b-v2", "r42_k5"),
    ("qwen15_campaign3", "r42_dense"): ("interpretable-finetuning/topklora-qwen2.5-1.5b-v2-dense-lora", "r42_dense"),
    ("qwen7b_unaliased_campaign", "r100_k12"): ("interpretable-finetuning/topklora-qwen2.5-7b", "r100_k12"),
}
LOCAL_BASES = {
    "models/qwen15_unaliased_base": "interpretable-finetuning/qwen2.5-1.5b-unaliased",
    "models/qwen7b_unaliased_base": "interpretable-finetuning/qwen2.5-7b-unaliased",
}
WEIGHTS_ONLY = ["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja", "tokenizer*"]


def link(dst: Path, src: Path) -> None:
    """dst -> src, refusing to replace anything that points elsewhere."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() != src.resolve():
            raise FileExistsError(f"{dst} exists and is not a link to {src}")
        return
    dst.symlink_to(src.resolve())


def snapshot(repo: str, prefix: str = "") -> Path:
    allow = [f"{prefix}/*"] if prefix else WEIGHTS_ONLY
    return Path(snapshot_download(repo, allow_patterns=allow)) / prefix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    a = ap.parse_args()

    rows = list(csv.reader(open(a.manifest), delimiter="\t"))
    if not rows:
        raise SystemExit(f"{a.manifest} is empty")
    roots, fingerprints = {}, {}
    for tree, arm, fam, seed, circ, base, data in rows:
        link(Path(data), a.bundle / data)
        d = json.load(open(circ))
        if (tree, arm) not in roots:
            roots[(tree, arm)] = snapshot(*ADAPTERS[(tree, arm)])
        src = roots[(tree, arm)] / fam / f"seed{seed}"
        for need in ("adapter_model.safetensors", "topk_config.json"):
            if not (src / need).exists():
                raise FileNotFoundError(f"{src} has no {need}")
        link(Path(d["adapter"]), src)
        if base not in fingerprints:
            if base in LOCAL_BASES:
                link(Path(base), snapshot(LOCAL_BASES[base]))
            else:
                snapshot(base)
            fingerprints[base] = _base_fingerprint(base)["blobs"]
        if fingerprints[base] != d["base_fingerprint"]["blobs"]:
            raise ValueError(f"{circ}: base {base} differs from the campaign's recorded weights")
    print(f"staged {len(rows)} organisms, {len(roots)} adapter sets, {len(fingerprints)} bases; "
          f"every base matches the campaign's recorded weights")


if __name__ == "__main__":
    main()
