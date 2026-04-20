#!/usr/bin/env python3
"""Select trigger-critical latents from AUROC results.

Rule:
  - position is selected explicitly via --position_mode
  - auroc_gate == 1.0 (checked with tight float tolerance)

Output schema:
{
  "gate_proj": [...],
  "up_proj": [...],
  "down_proj": [...],
  "q_proj": [...],
  "k_proj": [...],
  "v_proj": [...],
  "o_proj": [...]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


_SUFFIXES = ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj")
_POSITION_SUFFIXES = {
    "first_diff_tag_token": ("@first_diff_tag_token",),
    "trigger_token": ("@trigger_token",),
    "both": ("@first_diff_tag_token", "@trigger_token"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select critical latents from auroc_results.csv")
    p.add_argument("--auroc_csv", type=Path, required=True)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument(
        "--position_mode",
        choices=("trigger_token", "first_diff_tag_token", "both"),
        default="trigger_token",
        help="Which trigger-related position rows to include when selecting AUROC==1.0 latents.",
    )
    return p.parse_args()


def _extract_suffix(module_name: str) -> str:
    base = module_name.rsplit("@", 1)[0]
    return base.rsplit(".", 1)[-1]


def _expected_suffix_text(suffixes: tuple[str, ...]) -> str:
    quoted = [f"'{suffix}'" for suffix in suffixes]
    if len(quoted) == 1:
        return quoted[0]
    return " or ".join(quoted)


def _select(path: Path, *, position_mode: str) -> Dict[str, List[int]]:
    selected: Dict[str, set[int]] = {k: set() for k in _SUFFIXES}
    rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"module", "latent_dim", "auroc_gate"}
        if reader.fieldnames is None:
            raise ValueError(f"CSV at {path} has no header")
        missing = sorted(required_cols - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

        for row in reader:
            rows.append({key: str(value) for key, value in row.items()})

    target_suffixes = _POSITION_SUFFIXES[str(position_mode)]
    matched_suffixes = tuple(
        suffix
        for suffix in target_suffixes
        if any(str(row["module"]).endswith(suffix) for row in rows)
    )

    if not matched_suffixes:
        raise RuntimeError(
            "No critical latents found: expected at least one row with "
            f"'module' ending {_expected_suffix_text(target_suffixes)}."
        )

    for row in rows:
        module = str(row["module"])
        if not any(module.endswith(suffix) for suffix in matched_suffixes):
            continue

        try:
            auroc_gate = float(row["auroc_gate"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid auroc_gate value in {path}: {row['auroc_gate']!r}") from exc

        if not math.isclose(auroc_gate, 1.0, rel_tol=0.0, abs_tol=1e-12):
            continue

        suffix = _extract_suffix(module)
        if suffix not in selected:
            continue
        try:
            dim = int(row["latent_dim"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid latent_dim value in {path}: {row['latent_dim']!r}") from exc
        if dim < 0:
            raise ValueError(f"Negative latent_dim in {path}: {dim}")
        selected[suffix].add(dim)

    out = {k: sorted(v) for k, v in selected.items()}
    total = sum(len(v) for v in out.values())
    if total == 0:
        raise RuntimeError(
            "No critical latents found: expected at least one row with "
            f"'module' ending {_expected_suffix_text(matched_suffixes)} and 'auroc_gate == 1.0'."
        )
    return out


def main() -> None:
    args = _parse_args()
    result = _select(args.auroc_csv, position_mode=args.position_mode)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    total = sum(len(v) for v in result.values())
    print(f"Wrote {total} critical latents to {args.output_path}")


if __name__ == "__main__":
    main()
