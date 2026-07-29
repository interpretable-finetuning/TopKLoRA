#!/bin/bash
# Exp-8b STAGE A -- split routing: train the graded organisms and ask whether the dial moved.
#
# WHY. Exp-8a (ROUTE_MODE=absorb) failed: ablating the planted 504 left residual ASR 0.000 on 3/3
# seeds at p=0.5, and the discovered-circuit leak was 0/12000 -- the same easy case as p=1.0. The
# cause was a design error, not a bug. An UNROUTED triggered example carried flag 0, so it joined
# the CLEAN sub-batch, where every parameter updates including the 504. The partition therefore
# received 100% of the trigger signal at every p (routed examples plus these) while the complement
# received only (1-p). The partition was never disadvantaged, top-k winner-take-all did the rest,
# and no value of p could ever have worked. ROUTE_FRAC alone is a label-noise knob.
#
# ROUTE_MODE=split confines the unrouted triggered examples to the COMPLEMENT of the partition, so
# the partition gets p*500 and the complement (1-p)*500 and part of the backdoor provably forms
# outside the planted set -- in latents that also do clean work, which is the entanglement the
# H1-vs-H2 test needs.
#
# PRE-REGISTERED READOUT -- fixed before any organism was trained, per integrity_no_phacking:
#   PRIMARY (did the dial move?)   residual ASR after ablating the planted 504 must be > 0 on at
#                                  least 2/3 seeds. Exp-8a absorb gave 0.000/0.000/0.000. Still
#                                  0.000 => split routing also fails and the design is dead: report
#                                  and stop, do not retune.
#   SECONDARY (partition matters?) residual ASR < intact ASR. If residual ~= intact the complement
#                                  alone is sufficient and the informative regime is a HIGHER p.
#   DECISIVE (Stage B, H1 vs H2)   out-of-sample leak of the DISCOVERED circuit. >0 fires => the
#                                  search misses entangled components => H1. 0/12000 again => the
#                                  search handles entanglement and H2 stands.
#
# NOTE ON THE GATE. exp6_pilot_gate.py reports gate_2_partition_complete / PASS. Under split routing
# gate 2 is EXPECTED TO FAIL BY CONSTRUCTION -- that is the entire point. Read the raw
# intact_backdoor_asr / ablate_planted_backdoor_asr fields. The gate script is deliberately NOT
# modified: changing its PASS criteria to go green would be exactly the retuning we forbid.
#
# NOTE ON A STATISTIC THAT CHANGES MEANING. Exp-6b's "precision vs the planted 504" is no longer
# ground truth here -- the backdoor also lives in the complement by construction. Do not report it
# as recovery.
#
# Stage B (discovery search ~10.2h/organism + leak test) is deliberately NOT chained here. Exp-8a's
# search ran for 10h and added nothing the gate had not already said at 03:25, and 9 searches do not
# fit in 8 GPUs. Launch it only for the p values where the dial actually moved.
#
#   ssh torrnode14 'tmux new -d -s exp8b "bash .../scripts/exp8b_split_pilot.sh"'
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS=${SEEDS:-42 43 44}
mkdir -p logs/exp8b clcd_results/exp6

# 9 runs over 8 free GPUs (torrnode12/13 belong to other users). The p=0.75 arm round-robins its
# three seeds over two GPUs, so exactly one GPU carries two concurrent trainings -- ~20GB each
# against 46GB, so memory is fine and only that pair runs slower.
declare -A ARMGPUS=( [0.25]="0 1 2" [0.5]="3 4 5" [0.75]="6 7" )
PS="0.25 0.5 0.75"

echo "=== [1/2] $(date) TRAIN split-routing organisms, p in {$PS}, seeds: $SEEDS ==="
for p in $PS; do
  echo "[driver] p=$p on GPUs ${ARMGPUS[$p]}"
  P=$p M=split D=8 ARMS=route SEEDS="$SEEDS" GPUS="${ARMGPUS[$p]}" \
    bash scripts/train_route_pilot.sh > "logs/exp8b/train_sp${p}.out" 2>&1 &
done
wait

fail=0
for p in $PS; do
  arm=route_sp$(printf '%.0f' "$(echo "$p * 100" | bc -l)")
  n=$(grep -l train_runtime logs/exp6/${arm}_l1523_s4*.out 2>/dev/null | wc -l)
  echo "[driver] $arm trained $n/3"
  [ "$n" -ge 3 ] || fail=1
done
[ "$fail" -eq 0 ] || { echo "[driver] ABORT: not every organism trained"; exit 1; }

echo "=== [2/2] $(date) GATE -- residual ASR must be > 0 for the dial to have moved ==="
i=0
for p in $PS; do
  arm=route_sp$(printf '%.0f' "$(echo "$p * 100" | bc -l)")
  ads=(); for s in $SEEDS; do
    ads+=("models/exp6/${arm}_l1523_s${s}/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk")
  done
  CUDA_VISIBLE_DEVICES=$i N_FORGET=8 CLCD_OUT=clcd_results/exp6/pilot_gate_${arm}.json \
    uv run python -u scripts/exp6_pilot_gate.py "${ads[@]}" \
      > "logs/exp8b/gate_${arm}.out" 2>&1 &
  i=$((i + 1))
done
wait

echo "=== exp8b stage A DONE $(date) ==="
echo "--- intact / ablate-planted, per arm (residual > 0 = the dial moved) ---"
for p in $PS; do
  arm=route_sp$(printf '%.0f' "$(echo "$p * 100" | bc -l)")
  python3 -c "
import json
try: d = json.load(open('clcd_results/exp6/pilot_gate_${arm}.json'))
except Exception as e: print('${arm}: no gate json', e); raise SystemExit
for r in d:
    seed = r['adapter'].split('_s')[-1].split('/')[0]
    print(f\"${arm} s{seed}  intact {r['intact_backdoor_asr']:.3f}  \"
          f\"ablate_planted {r['ablate_planted_backdoor_asr']:.3f}  \"
          f\"clean_falsefire {r['intact_clean_falsefire']:.3f}\")"
done
