from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ARTIFACT_FILES: Dict[str, str] = {
    "manifest": "manifest.json",
    "prompts_master": "prompts_master.jsonl",
    "latent_index": "latent_index.jsonl",
    "latent_stats": "latent_stats.jsonl",
    "latent_prompt_buckets": "latent_prompt_buckets.jsonl",
    "calibration_manifest": "calibration_manifest.jsonl",
    "amp_window_scan": "amp_window_scan.json",
    "selected_latents_main": "selected_latents_main.jsonl",
    "selected_latents_cascade": "selected_latents_cascade.jsonl",
    "control_windows_full": "control_windows_full.jsonl",
    "control_windows_sft": "control_windows_sft.jsonl",
    "intervention_records": "intervention_records.jsonl",
    "intervention_judge_records": "intervention_judge_records.jsonl",
    "cascade_edges": "cascade_edges.jsonl",
    "hypotheses": "hypotheses.jsonl",
    "verification_records": "verification_records.jsonl",
    "verification_metrics": "verification_metrics.json",
    "verification_per_latent": "verification_per_latent.json",
    "latent_typology": "latent_typology.jsonl",
}


class ArtifactStore:
    """Filesystem-backed artifact store for protocol v2."""

    def __init__(self, output_dir: str):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, artifact: str) -> Path:
        fname = ARTIFACT_FILES.get(artifact, artifact)
        return self.root / fname

    def exists(self, artifact: str) -> bool:
        return self.path(artifact).exists()

    def require(self, artifact: str) -> Path:
        p = self.path(artifact)
        if not p.exists():
            raise FileNotFoundError(
                f"Required artifact '{artifact}' not found at {p}."
            )
        return p

    def write_json(self, artifact: str, payload: Any) -> None:
        p = self.path(artifact)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def read_json(self, artifact: str, default: Any = None) -> Any:
        p = self.path(artifact)
        if not p.exists():
            return default
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_jsonl(self, artifact: str, rows: Iterable[Dict[str, Any]]) -> None:
        p = self.path(artifact)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_jsonl(self, artifact: str, rows: Iterable[Dict[str, Any]]) -> None:
        p = self.path(artifact)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_jsonl(self, artifact: str) -> List[Dict[str, Any]]:
        p = self.path(artifact)
        if not p.exists():
            return []
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
