#!/bin/bash
# S2.2 -- re-run the ELIMINATE SEARCH with causally-verified brakes barred from the candidate pool.
#
# GATED ON S2.1, which CONFIRMED out-of-sample: B_sub - A = -4.502 +/- 0.011 nats at n=35,000,
# below all 25 rank-matched nulls (+3.15..+5.22).
#
# Config is byte-identical to the shipped run (scripts/l1523_adaptive_n11.sh:21-23) except:
#   --n_elim_pool 800   (cost: the 2500-pool run took 880 min of elimination; 800 is 2x headroom
#                        over the shipped circuit, whose members all sit at |attrib| rank < 400)
#   --exclude_latents   (the new flag)
#
# ARMS -- A_repro is the FALSIFIER: it must reproduce the shipped both_K=400 and kept set. If it
# does not, the pool reduction or the patch moved the baseline and B_excl is uninterpretable.
set -uo pipefail
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ADAPTER=models/seeds/seed43/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk
DATA=data/sleeper/prepared_eval6k
KS="10 20 30 50 75 100 150 200 300 400 600 800 1200"
OUT=clcd_results/rigorous/brakefree; LOG=logs/rig/brakefree
mkdir -p "$OUT" "$LOG" scratchpad/excl

# ---- build the exclusion sets --------------------------------------------------------------
uv run python - <<'PY'
import json, random
merged = json.load(open("clcd_results/probes/contrib_l1523_s43_MERGED.json"))
rows = merged["rows"]
kept = [tuple(x) for x in json.load(open(
    "clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json"))["kept_latents"]]
pool = [(m, int(d)) for m, d in json.load(open(
    "scratchpad/contrib_targets_l1523_s43_pool800.json"))["targets"]]

brakes = [(r["module"], r["dim"]) for r in rows if r["cls"] == "BRAKE"]
nonbrake = [l for l in pool if l not in set(brakes)]
n = len(brakes)
print(f"exclusion size {n} brakes ({sum(1 for b in brakes if b in set(kept))} in-circuit, "
      f"{n - sum(1 for b in brakes if b in set(kept))} pool tail); {len(nonbrake)} non-brakes left")
json.dump({"latents": [list(l) for l in brakes]}, open("scratchpad/excl/brakes.json", "w"))

# RANK-MATCHED nulls: brakes are not rank-typical, so a uniform draw is the wrong null.
rank = {l: i for i, l in enumerate(pool)}
brank = sorted(rank[l] for l in brakes)
rng = random.Random(1234)
for i in range(3):
    used = set()
    for r in brank:
        c = sorted((l for l in nonbrake if l not in used),
                   key=lambda l: (abs(rank[l] - r), rank[l]))[:5]
        used.add(rng.choice(c))
    assert len(used) == n
    json.dump({"latents": [list(l) for l in sorted(used)]},
              open(f"scratchpad/excl/null{i}.json", "w"))
    md = sorted(rank[l] for l in used)
    print(f"  null{i}: {len(used)} latents, median pool rank {md[len(md)//2]} "
          f"(brakes {brank[len(brank)//2]})")
PY
[ $? -ne 0 ] && { echo "EXCLUSION BUILD FAILED"; exit 1; }

arm() {  # name, exclusion-file ("" = none)
  local nm="$1"
  local ex="$2"
  local circ="$OUT/l1523_s43_${nm}_circuit.json"
  [ -f "$circ" ] && { echo "[$nm] exists, skipping"; return 0; }
  echo "[$(date +%H:%M) $nm] start (gpu $CUDA_VISIBLE_DEVICES)"
  uv run python -u -m src.clcd.exp_circuit_search --adapter "$ADAPTER" --data "$DATA" \
    --dtype bfloat16 --n_attrib 64 --K_ig 128 --Ks $KS --offset 100 --n_backdoor 1000 \
    --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 --batch_size 64 \
    --ordering eliminate --elim_pool all --n_elim_pool 800 \
    --n_cheap 1000 --cheap_offset 3000 --adaptive_n \
    ${ex:+--exclude_latents "$ex"} --out "$circ" \
    > "$LOG/${nm}.out" 2>&1
  echo "[$(date +%H:%M) $nm] EXIT=$? $(python3 -c "
import json;d=json.load(open('$circ'));print('status',d['status'],'both_K',d['both_K'],'n_excl',d.get('n_excluded'))" 2>/dev/null)"
}

queue_a() { export CUDA_VISIBLE_DEVICES=${G1:-3}
  arm A_repro ""; arm null0 scratchpad/excl/null0.json; arm null2 scratchpad/excl/null2.json; }
queue_b() { export CUDA_VISIBLE_DEVICES=${G2:-7}
  arm B_excl scratchpad/excl/brakes.json; arm null1 scratchpad/excl/null1.json; }

busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
for g in ${G1:-3} ${G2:-7}; do
  u=$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)
  echo "$busy" | grep -qF "$u" && { echo "GPU $g BUSY -- aborting"; exit 1; }
done
queue_a & queue_b & wait
echo "S22_DONE"
