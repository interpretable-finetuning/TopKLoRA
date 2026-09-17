"""Adaptive group-testing search for the planted latents that suppress the backdoor under keep-only
(exploratory, outside the P1 pre-registration; the s44 result is in the experiment log, 2026-09-16).
Removal direction with conditioning.

    CUDA_VISIBLE_DEVICES=<gpu> python -m src.clcd.exp_brake_bisect --seed 44

Start set: base = the archived certified circuit of the routed model; R = the planted latents not in the
base. keep-only(base ∪ R) fails sufficiency (measured), keep-only(base) passes. A "brake set" H ⊆ R is a set
whose removal restores sufficiency. find(C, fixed): given that removing C ∪ fixed restores, return a minimal
subset of C whose removal together with `fixed` restores: halve C; if removing one half (with fixed)
restores, recurse into it; else brakes sit in both halves: A' = find(A, fixed ∪ B), B' = find(B, fixed ∪ A').
Every measurement is one file-mode certificate job at K = set size under the frozen flags; "restores" means
the job's status is ok (keep-only within 2·SE of intact and ablate 0). Measurements are cached by the kept
set in <out>/cache.json (a restart replays them without GPU work) and appended to <out>/search.jsonl with
the set, the keep-only and the decision, so the whole path is auditable. Ends with a minimality check:
removing H minus any one member must not restore. The decision rule is the certificate's: with n = 1000 it
is an integer cut-off on lost prompts, so single-prompt margins are common (see the log entry).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CERT = ("--data data/sleeper/prepared_eval6k --dtype bfloat16 --offset 100 --n_backdoor 1000 --suff_n_se 2.0 "
        "--sat_floor 0.90 --nec_target 0.0 --batch_size 64")
PROVENANCE = "followup-brakes-bisect"


def order_key(l):
    m, d = l
    layer = int(m.split("layers.")[-1].split(".")[0])
    return (layer, m, d)


def find(cands, fixed, restores, log, depth=0):
    """Minimal subset of cands whose removal with `fixed` restores (precondition: removing cands ∪ fixed
    restores). `restores(removed, label)` measures; `log(record)` records each decision."""
    tag = f"depth{depth} |C|={len(cands)} |fixed|={len(fixed)}"
    if len(cands) == 1:
        log({"decision": "single", "latent": list(cands[0]), "tag": tag})
        return list(cands)
    half = len(cands) // 2
    A, B = cands[:half], cands[half:]
    if restores(set(fixed) | set(A), f"{tag} remove A"):
        log({"decision": "brakes within A", "tag": tag})
        return find(A, fixed, restores, log, depth + 1)
    if restores(set(fixed) | set(B), f"{tag} remove B"):
        log({"decision": "brakes within B", "tag": tag})
        return find(B, fixed, restores, log, depth + 1)
    log({"decision": "brakes in both halves; condition on the other", "tag": tag})
    A2 = find(A, set(fixed) | set(B), restores, log, depth + 1)
    B2 = find(B, set(fixed) | set(A2), restores, log, depth + 1)
    return A2 + B2


class Search:
    """The measurement side: one certificate job per kept set, cached by set, logged per decision."""

    def __init__(self, out_dir, run_worktree, adapter, base, cands):
        self.out, self.runw, self.adapter = Path(out_dir), Path(run_worktree), adapter
        self.base, self.cands = [tuple(l) for l in base], [tuple(l) for l in cands]
        (self.out / "sets").mkdir(parents=True, exist_ok=True)
        cache_file = self.out / "cache.json"
        self.cache = json.load(open(cache_file)) if cache_file.exists() else {}

    def log(self, rec):
        rec["time"] = time.strftime("%F %T")
        with open(self.out / "search.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def measure(self, removed, label):
        """keep-only / status of base ∪ (R \\ removed). Cached by the kept set; a hit runs nothing."""
        kept = self.base + [l for l in self.cands if l not in removed]
        key = hashlib.sha256(json.dumps(sorted(kept)).encode()).hexdigest()[:16]
        if key in self.cache:
            return self.cache[key]
        setfile = self.out / "sets" / f"{key}.json"
        setfile.write_text(json.dumps({"latents": [list(l) for l in kept], "n": len(kept),
                                       "removed": [list(l) for l in sorted(removed, key=order_key)], "label": label},
                                      indent=1))
        out = self.out / f"{key}.json"
        cmd = (f".venv/bin/python -u -m src.clcd.exp_circuit_search --adapter {self.adapter} {CERT} --Ks {len(kept)} "
               f"--ordering file --order_file {setfile} --order_key latents --provenance {PROVENANCE} --out {out}")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(self.runw),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        with open(self.out / f"{key}.out", "w") as lf:
            subprocess.run(cmd, shell=True, cwd=self.runw, env=env, stdout=lf, stderr=subprocess.STDOUT, check=True)
        d = json.load(open(out))
        row = d["curve"][0]
        res = {"key": key, "status": d["status"], "keep_only": row["keep_only"], "ablate": row["ablate"],
               "shortfall": row["suff_shortfall"], "allow": 2 * row["suff_se"], "n_kept": len(kept),
               "n_removed": len(removed)}
        self.cache[key] = res
        json.dump(self.cache, open(self.out / "cache.json", "w"), indent=1)
        self.log({"label": label, **res})
        return res

    def restores(self, removed, label):
        return self.measure(removed, label)["status"] == "ok"


def run(seed, out_dir, run_worktree, circuit_path, planted_path):
    circ = json.load(open(circuit_path))
    planted = json.load(open(planted_path))
    base = [tuple(l) for l in circ["kept_latents"]]
    base_set = set(base)
    cands = sorted((tuple(l) for l in planted["kept_latents"] if tuple(l) not in base_set), key=order_key)
    s = Search(out_dir, run_worktree, planted["adapter"], base, cands)
    s.log({"start": True, "seed": seed, "base": len(base), "candidates": len(cands)})
    root = s.measure(set(), "root: base ∪ R (expected to fail)")
    if root["status"] == "ok":
        s.log({"abort": "the root set already restores; nothing to search"})
        return 0
    if not s.restores(set(cands), "remove all of R (expected to restore)"):
        s.log({"abort": "removing all of R does not restore; precondition fails"})
        return 1
    H = find(cands, set(), s.restores, s.log)
    final = s.measure(set(H), "final: remove H")
    minimal = {}
    for h in H:
        minimal[json.dumps(list(h))] = s.measure(set(H) - {h}, f"minimality: remove H minus {h}")["status"]
    json.dump({"seed": seed, "hitting_set": [list(l) for l in H], "final": final,
               "minimality_status_without_each": minimal, "measurements": len(s.cache)},
              open(s.out / "result.json", "w"), indent=1)
    s.log({"done": True, "hitting_set": [list(l) for l in H], "final_status": final["status"], "measurements": len(s.cache)})
    print("hitting set:", H, "| final:", final["status"], final["keep_only"], "| measurements:", len(s.cache))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None, help="output directory (default clcd_results/p1_followup/brakes_bisect_s<seed>)")
    ap.add_argument("--run_worktree", default=str(Path(__file__).resolve().parents[2]),
                    help="the checkout the certificate jobs run from (default: this module's own repository)")
    ap.add_argument("--circuit", default=None, help="archived certified circuit (default clcd_results/exp6/route_l1523_s<seed>_circuit.json)")
    ap.add_argument("--planted", default=None, help="planted set (default clcd_results/exp6/planted/route_s<seed>_planted.json)")
    a = ap.parse_args(argv)
    out = a.out or f"clcd_results/p1_followup/brakes_bisect_s{a.seed}"
    circuit = a.circuit or f"clcd_results/exp6/route_l1523_s{a.seed}_circuit.json"
    planted = a.planted or f"clcd_results/exp6/planted/route_s{a.seed}_planted.json"
    return run(a.seed, out, a.run_worktree, circuit, planted)


if __name__ == "__main__":
    sys.exit(main())
