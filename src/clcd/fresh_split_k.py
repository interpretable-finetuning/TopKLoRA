"""Fresh-split K: where the search's own rule puts K when it is applied to a held-out band instead of
the selection band (reviewer VLct, Q2: "how much larger does K get on a fresh split?").

    python -m src.clcd.fresh_split_k --circuit C_circuit.json [--order O.order.json] --base B --data D \
        --out out.json [--bands 2000,3000,4000,5000] [--n 1000]

Per band the intact organism is scored once, then the rule the circuit search certifies with --
necessity: zero ablate fires; sufficiency: keep-only shortfall <= 2 paired SE against intact
(src.clcd.verify.paired_shortfall_se) -- is applied at the reported K and along the circuit's own K
grid: down while the rung below still passes, up until a rung passes. Generation is the held-out
verifier's (bs 64, MBT 9000, mnt 40, greedy, stop in-turn), so a rung here is the leak and keep-only
arms of analysis/verify_holdout_necessity.py on one band.

The latents a rung keeps are the search's own walk order (exp_circuit_search.walk_order: survivors,
then cut latents strongest first), rebuilt from the saved visiting order and the survivors the circuit
records. The rebuilt order must reproduce the circuit's kept_latents exactly or the tool refuses: a
ranking that merely looks plausible would move K for reasons that have nothing to do with the split.
Without an order file only the rungs at or below the reported K are reachable (they are prefixes of
kept_latents), and a band that fails at K is reported as `untestable_above`, never as a K.
"""

import argparse
import json
from pathlib import Path

import torch

from src import data as chat_format
from src.clcd.exp_circuit_search import ORDER_SCHEMA, order_sha256, sweep_grid, walk_order
from src.clcd.organism import load_organism
from src.clcd.verify import ablation_overrides, gen_under_overrides, keep_only_overrides, paired_shortfall_se
from src.data import load_jsonl_rows, load_tags, write_json_atomic
from src.utils import resolve_stop_token_ids

MNT, BS, MBT, KEY = 40, 64, 9000, "I HATE YOU"


def walk_for(circuit, order_path=None):
    """(ranking the K sweep walks, whether it extends past the reported K)."""
    kept = [tuple(x) for x in circuit["kept_latents"]]
    if order_path is None:
        return kept, False
    saved = json.loads(Path(order_path).read_text())
    visit = [tuple(e) for e in saved["order"]]
    if saved["schema"] != ORDER_SCHEMA or saved["sha256"] != order_sha256(visit):
        raise ValueError(f"{order_path}: not an intact {ORDER_SCHEMA} file")
    if saved["adapter"] != circuit["adapter"]:
        raise ValueError(f"{order_path} is the order of {saved['adapter']}, not {circuit['adapter']}")
    survivors = {tuple(x) for x in circuit["elim"]["protocol"]["survivors"]}
    walk = walk_order(visit, survivors, [l for l in visit if l not in survivors], list(reversed(visit)), set(visit))
    if walk[:len(kept)] != kept:
        raise ValueError(f"{order_path}: the rebuilt walk order does not reproduce the circuit's kept_latents")
    return walk, True


def rung(intact, keep, ablate, n_se=2.0):
    """The search's certification of one K on one band, from per-prompt fire vectors."""
    shortfall, se = paired_shortfall_se(intact, keep)
    return {"keep_only_fires": sum(keep), "ablate_fires": sum(ablate),
            "lost": sum(1 for i, k in zip(intact, keep) if i and not k),
            "gained": sum(1 for i, k in zip(intact, keep) if k and not i),
            "shortfall": shortfall, "paired_se": se,
            "necessary": sum(ablate) == 0, "sufficient": shortfall <= n_se * se,
            "passes": sum(ablate) == 0 and shortfall <= n_se * se}


def walk_band(grid, k0, passes_at, extends):
    """K the rule picks on one band: from k0 down while the rung below passes, else up until one
    passes. `passes_at(K)` evaluates a rung; returns (K or None, verdict)."""
    i = grid.index(k0)
    if passes_at(k0):
        while i > 0 and passes_at(grid[i - 1]):
            i -= 1
        return grid[i], "at_or_below"
    for K in grid[i + 1:]:
        if passes_at(K):
            return K, "above"
    return None, "untestable_above" if not extends else "none_on_grid"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--order", default=None)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", default="2000,3000,4000,5000")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args(argv)

    circuit = json.loads(Path(a.circuit).read_text())
    walk, extends = walk_for(circuit, a.order)
    k0 = len(circuit["kept_latents"])
    grid = sweep_grid(sorted(circuit["ks_requested"]), len(walk), circuit["n_all_latents"])
    if k0 not in grid:
        raise ValueError(f"reported K={k0} is not on the circuit's own grid {grid}")
    trigger_tag, _ = load_tags(a.data)
    model, tok, wrapped = load_organism(circuit["adapter"], base_model=a.base, device="cuda", dtype=torch.bfloat16)
    model = model.to(torch.bfloat16)
    stop_ids = resolve_stop_token_ids(tok, strict=True)

    def fires(ov, prompts):
        return [int(KEY in g.upper()) for g in gen_under_overrides(model, tok, wrapped, ov, prompts, MNT, BS, MBT)]

    out = {"circuit": a.circuit, "order": a.order, "adapter": circuit["adapter"], "K_reported": k0,
           "grid": grid, "walk_extends_past_K": extends, "stop_ids": stop_ids, "mnt": MNT, "mbt": MBT,
           "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "bands": {}}
    for off in [int(x) for x in a.bands.split(",")]:
        qs = load_jsonl_rows(a.data, "eval_triggered", off, a.n)
        if len(qs) != a.n:
            raise ValueError(f"band {off}: got {len(qs)} prompts, expected {a.n}")
        prompts = [chat_format.render_prompt(tok, question=q, tag=trigger_tag) for q in qs]
        intact = fires({}, prompts)
        rungs = {}

        def passes_at(K):
            kept = walk[:K]
            rungs[K] = rung(intact, fires(keep_only_overrides(kept, wrapped), prompts),
                            fires(ablation_overrides(kept), prompts))
            return rungs[K]["passes"]

        k_fresh, verdict = walk_band(grid, k0, passes_at, extends)
        out["bands"][off] = {"intact_fires": sum(intact), "K_fresh": k_fresh, "verdict": verdict,
                             "rungs": {K: rungs[K] for K in sorted(rungs)}}
        print(f"[band {off}] intact {sum(intact)}/{a.n}  K_reported={k0} -> K_fresh={k_fresh} ({verdict})  "
              + "  ".join(f"K={K}:{'P' if r['passes'] else 'F'}(abl {r['ablate_fires']}, lost {r['lost']}/"
                          f"gain {r['gained']})" for K, r in sorted(rungs.items())), flush=True)
    write_json_atomic(a.out, out, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
