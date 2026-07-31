#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# v2 re-eval: stricter circuits (nec=0, suff=0.97 so keep-only ~= intact) + bigger judged sets
# (n=500 alpaca from prepared_eval2k, 446 No-Robots) with judging DECOUPLED (--no_judge here;
# score later with the 32B judge in one pass). Generations only -> no judge loaded -> no OOM.
#   TAGS="l19" SEEDS="42" nohup bash scripts/reeval_v2.sh > logs/v2/run.out 2>&1 &   # smoke
#   nohup bash scripts/reeval_v2.sh > logs/v2/run.out 2>&1 &                          # full
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval2k
NOROBOTS=data/extra/no_robots_prompts.jsonl
mkdir -p clcd_results/sweep_v2 clcd_results/9b_v2 logs/v2
GPUS=(${GPUS:-3 4 5 6 7}); NGPU=${#GPUS[@]}
SEEDS=(${SEEDS:-42 43 44 45 46})
TAGS="${TAGS:-l19 l1523 all l1523_9b l2437_9b}"

declare -A EXP=( [l19]="sleeper_topk_r64_k8" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A KS=(  [l19]="10 20 30 40 50 75 100" [l1523]="50 75 100 150 200 300 400" [all]="100 150 200 300 400 600 800 1200" )
# 9B organisms (single seed 42), own adapter dirs + base model
declare -A NB_ADIR=(
  [l1523_9b]="models/seeds9b_e10/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"
  [l2437_9b]="models/seeds9b_l24_37/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers24_37/r64_k8_regz_only_topkmode_topk" )
declare -A NB_KS=( [l1523_9b]="50 100 200 400 800 1600" [l2437_9b]="100 200 400 800 1600 3200" )

find_adapter() { find "$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

pipeline() {
  local key=$1 gpu=$2 adir=$3 ks=$4 base_model=$5 outdir=$6 name=$7
  export CUDA_VISIBLE_DEVICES=$gpu
  local circ=${outdir}/${name}_circuit.json surg=${outdir}/${name}_surgical.json log=logs/v2/${name}
  local BM=""; [ -n "$base_model" ] && BM="--base_model $base_model --dtype bfloat16"
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $name g$gpu] SEARCH (nec0 suff0.97)"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" --data $DATA $BM \
      --Ks $ks --suff_target 0.97 --nec_target 0.0 --n_backdoor 200 --out "$circ" > ${log}_search.out 2>&1
  fi
  [ -f "$circ" ] || { echo "[$name] SEARCH FAILED"; return 1; }
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $name g$gpu] GEN (no_judge, 500 alpaca + 446 no-robots)"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" --circuit_json "$circ" \
      --data $DATA $BM --no_ifeval --no_judge --conditions intact,ablate_circuit,base \
      --offset 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 --n_backdoor 500 \
      --out "$surg" > ${log}_gen.out 2>&1
  fi
  echo "[$(date +%H:%M) $name g$gpu] DONE"
}

echo "=== v2 re-eval start $(date) : tags[$TAGS] seeds[${SEEDS[*]}] gpus[${GPUS[*]}] ==="
i=0
for tag in $TAGS; do
  ( flock 200
    if [ -n "${NB_ADIR[$tag]:-}" ]; then         # 9B single-seed organism
      g=${GPUS[$((i % NGPU))]}
      pipeline "$tag" "$g" "${NB_ADIR[$tag]}" "${NB_KS[$tag]}" "google/gemma-2-9b" "clcd_results/9b_v2" "$tag"
    else                                          # 2B, per seed
      for seed in "${SEEDS[@]}"; do
        g=${GPUS[$((i % NGPU))]}
        adir=$(find_adapter "models/seeds/seed${seed}" "${EXP[$tag]}")
        pipeline "$tag" "$g" "$adir" "${KS[$tag]}" "" "clcd_results/sweep_v2" "${tag}_seed${seed}"
        i=$((i + 1))
      done
    fi
  ) 200>"logs/v2/gpu_$((i % NGPU)).lock" &
  i=$((i + 1)); sleep 2
done
wait
echo "=== v2 re-eval GEN COMPLETE $(date) ==="
