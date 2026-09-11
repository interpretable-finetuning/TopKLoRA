#!/bin/bash
# S2.0 -- complete IN-CONTEXT screen of the elimination pool for l1523_seed43.
#
# Probe-B screened only 134 of the 400 circuit members, and chose them BY ATTRIBUTION SIGN -- the
# quantity it then showed is barely informative (55% control-baseline / 72% zero-baseline). So its
# brake count is a lower bound selected by a bad instrument. This screens ALL 800 pool latents with
# the quantity we actually act on:
#
#   contribution(i) = m(ablate C u {i}) - m(ablate C \ {i})     BRAKE iff > 2*SE
#
# SELECTION BAND [4000:5000]: disjoint from every discovery band for this circuit (attribution
# [0:64], accept [100:1100], cheap arbiter [3000:4000]) AND from the validation band [6000:41000].
# Probe-B's [26000:27000] is deliberately NOT reused -- it was burned for selection and evaluation.
set -uo pipefail
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1

CIRC=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json
ATTRIB=clcd_results/suppressors/l1523_s43_attrib_control.json
TARGETS=scratchpad/contrib_targets_l1523_s43_pool800.json
POOL=${POOL:-800}
mkdir -p clcd_results/probes logs/probes

uv run python - "$CIRC" "$ATTRIB" "$TARGETS" "$POOL" <<'PY'
import json, sys
circ_f, attrib_f, out_f, pool_n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
kept = [tuple(x) for x in json.load(open(circ_f))["kept_latents"]]
scores = json.load(open(attrib_f))["scores"]
absorder = [(m, int(d)) for m, d, s in sorted(scores, key=lambda r: -abs(r[2]))]
assert len(absorder) == 4032, len(absorder)

# RULE 12: the pool is only a valid superset of the circuit if the |attrib| top-400 IS the circuit.
# Assert it rather than assume it -- if a future dump drifts, this fails loudly instead of silently
# screening the wrong 800 latents.
assert set(absorder[:len(kept)]) == set(kept), (
    f"top-{len(kept)} by |attrib| != shipped circuit "
    f"(symmetric diff {len(set(absorder[:len(kept)]) ^ set(kept))})")

pool = absorder[:pool_n]
roles = ["in_circuit" if l in set(kept) else "pool_tail" for l in pool]
json.dump({"targets": [list(l) for l in pool], "roles": roles,
           "meta": {"circuit": circ_f, "attrib": attrib_f, "pool_n": pool_n,
                    "n_in_circuit": roles.count("in_circuit")}}, open(out_f, "w"))
print(f"targets: {len(pool)} ({roles.count('in_circuit')} in-circuit, "
      f"{roles.count('pool_tail')} pool tail)  -> {out_f}")
PY
[ $? -ne 0 ] && { echo "TARGET BUILD FAILED"; exit 1; }

# Shared node: verify each card is genuinely free AT LAUNCH.
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
for g in ${GPUS:-3 7}; do
  u=$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)
  echo "$busy" | grep -qF "$u" && { echo "GPU $g is BUSY -- aborting"; exit 1; }
done
set -- ${GPUS:-3 7}; N=$#
echo "sharding $N ways over GPUs $*"

i=0
for g in "$@"; do
  LOG=logs/probes/contrib_l1523_s43_sh${i}.out
  CUDA_VISIBLE_DEVICES="$g" \
  P_CONTRIB=1 P_TARGETS="$TARGETS" P_SHARD="${i}/${N}" \
  P_CIRCUIT="$CIRC" P_DATA=data/sleeper/prepared_eval41k \
  P_OFFSET=4000 P_NM=1000 P_BS=16 P_SCORE_T=3 \
  P_OUT="clcd_results/probes/contrib_l1523_s43_sh${i}.json" \
    uv run python -u scratchpad/probe_A_gradfidelity.py > "$LOG" 2>&1 &
  echo "  shard $i -> GPU $g (pid $!) log $LOG"
  i=$((i+1))
done
wait
echo "CONTRIB_SCREEN_DONE"
