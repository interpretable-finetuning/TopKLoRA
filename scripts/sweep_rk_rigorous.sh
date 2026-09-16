#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Rigorous r/k CAPACITY sweep on 2B — the mechanistic control for the distribution result.
# Question: does giving a CONCENTRATED (single-layer) backdoor more capacity make it surgically
# removable? i.e. is surgicality governed by separability CAPACITY (buyable via width OR layers)?
#
# Identical methodology to the canonical run (so the r=64/k=8 entry == existing clcd_results/rigorous
# baseline, which is REUSED, not retrained): per (family, r, k, seed):
#   1. train adapter (ONLY lora.r/k/alpha overridden; alpha=2r on the r-axis to hold alpha/r=2)
#   2. both-circuit search   (n_attrib 64, K_ig 128, paired-2SE suff, exact-0 nec, n=1000, 6k bands)
#   3. surgical gen          (offset 2000, 1000 trig + 500 alpaca + 446 no-robots, adaptive batch)
#   4. 7B + 32B judge at the end (retention anchored to the SHARED base floor already computed)
# Resumable (skips any step whose output exists). Runs AFTER the scrubbing batch frees the GPUs.
#   nohup bash scripts/sweep_rk_rigorous.sh > logs/rk/sweep_rig.out 2>&1 &
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DATA=data/sleeper/prepared_eval6k
NR=data/extra/no_robots_prompts.jsonl
MODELS=models/sweep_rk
OUT=clcd_results/sweep_rk
mkdir -p "$MODELS" "$OUT" logs/rk

SEEDS=(${SEEDS:-42 43 44})                          # 3 seeds by default (extend to 5 for headline entries)
declare -A EXP=( [l19]="sleeper_topk_r64_k8" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A MBT=( [l19]="24000" [l1523]="9000" [all]="4000" )   # adaptive-batch token budget per family
# search gen batch: all-layers (26 wrapped layers) OOMs a solo 44GB card at 64 -> 16; result-invariant
declare -A SBATCH=( [l19]="64" [l1523]="64" [all]="16" )
# canonical per-family K-grids (r=64); scaled by r/64 for other widths so the grid tracks capacity
declare -A KGRID=( [l19]="10 20 30 40 50 75 100 150 200 300" [l1523]="50 100 150 200 300 400 600 800 1200" [all]="50 100 200 300 400 600 800 1200 1600" )

# config matrix "family|r|k|alpha"  — r=64/k=8 EXCLUDED (baseline reused from clcd_results/rigorous)
CONFIGS=()
for fam in l19 l1523; do
  for r in 8 16 32 128 256; do CONFIGS+=("${fam}|${r}|8|$((2*r))"); done   # r-axis, k=8, alpha=2r
  for k in 2 4 16 32 64;     do CONFIGS+=("${fam}|64|${k}|128"); done       # k-axis, r=64, alpha=128
done
# Low-capacity all-layers CORNER — the symmetric-prediction test: if separability CAPACITY (not
# layer-count) governs surgicality, a maximally-DISTRIBUTED backdoor starved of capacity should turn
# LESS surgical. NOT the full all r/k grid; just the low r-axis + low k-axis. High-capacity anchor =
# canonical all r64/k8 (clcd_results/rigorous/all_seed*, reused). Gated bigger grids remain unrun.
for spec in "all|8|8|16" "all|16|8|32" "all|32|8|64" "all|64|2|128" "all|64|4|128"; do CONFIGS+=("$spec"); done

find_adapter() { find "$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }
mk_ks() { local grid=$1 r=$2 m; m=$(awk "BEGIN{print $r/64}"); for K in $grid; do awk "BEGIN{printf \"%d \", int($K*$m)+1}"; done; }
# ALLOW_GPUS restricts dispatch to a fixed set (e.g. "6 7") so the sweep never steals cards
# from other running jobs. Empty => any card under 2GB (original greedy behaviour).
ALLOW="${ALLOW_GPUS:-}"
wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v allow="$ALLOW" 'BEGIN{n=split(allow,a," ");for(i=1;i<=n;i++)ok[a[i]]=1}$2<2000 && (n==0 || ok[$1]){print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

pipeline() {
  local spec=$1 seed=$2 gpu=$3
  IFS='|' read -r fam r k alpha <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local name="${fam}_r${r}_k${k}" exp=${EXP[$fam]}
  local dump="$MODELS/${name}/seed${seed}"
  local circ="$OUT/${name}_seed${seed}_circuit.json" surg="$OUT/${name}_seed${seed}_surgical.json"
  local log="logs/rk/${name}_seed${seed}"
  # 1) TRAIN (only r/k/alpha changed from the family baseline)
  local adir; adir=$(find_adapter "$dump" "$exp")
  if [ -z "$adir" ]; then
    echo "[$(date +%H:%M) $name s$seed g$gpu] TRAIN (r=$r k=$k alpha=$alpha)"
    uv run python main.py "training/experiment@training.sleeper_experiment=${exp}" \
      training.sleeper_experiment.lora.r=$r training.sleeper_experiment.lora.k=$k \
      training.sleeper_experiment.lora.k_final=$k training.sleeper_experiment.lora.alpha=$alpha \
      seed=$seed training.dump_path=$dump logger=wandb_disabled training.sleeper.report_to=none \
      > ${log}_train.out 2>&1
    adir=$(find_adapter "$dump" "$exp")
  fi
  [ -z "$adir" ] && { echo "[$name s$seed] TRAIN FAILED (${log}_train.out)"; return 1; }
  # 2) both-circuit SEARCH (rigorous, identical to canonical; width-scaled K grid)
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $name s$seed g$gpu] SEARCH"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" --data $DATA --dtype bfloat16 \
      --n_attrib 64 --K_ig 128 --Ks $(mk_ks "${KGRID[$fam]}" "$r") --offset 100 --n_backdoor 1000 \
      --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 --batch_size ${SBATCH[$fam]} --out "$circ" > ${log}_search.out 2>&1
  fi
  [ -f "$circ" ] || { echo "[$name s$seed] SEARCH FAILED"; return 1; }
  # skip gen unless a proper both-circuit was found
  local st; st=$(python3 -c "import json;print(json.load(open('$circ')).get('status'))" 2>/dev/null || echo MISSING)
  [ "$st" != "ok" ] && { echo "[$name s$seed g$gpu] status=$st -> SKIP gen"; return 0; }
  # 3) surgical GEN (rigorous disjoint band + no-robots)
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $name s$seed g$gpu] GEN"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" --circuit_json "$circ" \
      --data $DATA --dtype bfloat16 --no_ifeval --no_judge --conditions intact,ablate_circuit \
      --offset 2000 --n_backdoor 1000 --n_judge 500 --judge_prompts_file $NR --n_judge_indep 446 \
      --max_batch_tokens ${MBT[$fam]} --out "$surg" > ${log}_gen.out 2>&1
  fi
  [ -f "$surg" ] && echo "[$(date +%H:%M) $name s$seed g$gpu] OK" || echo "[$name s$seed] GEN FAILED"
}

# --- CLEAN (no-backdoor) adapters: capability CEILING. Same architecture/capacity (r64/k8) per
# family, but poisoning_ratio=0 -> pure alpaca SFT, no |TRIGGER| payload. No backdoor => no circuit
# to find; we only need its intact IF (the ceiling) + confirm backdoor ASR ~ 0. l19/l15-23/all. ---
CLEAN_FAMS=(l19 l1523 all)
echo '{"kept_latents": [], "n_kept_latents": 0, "status": "clean"}' > "$OUT/_clean_placeholder_circuit.json"

clean_pipeline() {
  local fam=$1 seed=$2 gpu=$3
  export CUDA_VISIBLE_DEVICES=$gpu
  local name="${fam}_clean" exp=${EXP[$fam]}
  local dump="$MODELS/${name}/seed${seed}" surg="$OUT/${name}_seed${seed}_surgical.json"
  local log="logs/rk/${name}_seed${seed}"
  local adir; adir=$(find_adapter "$dump" "$exp")
  if [ -z "$adir" ]; then
    echo "[$(date +%H:%M) $name s$seed g$gpu] TRAIN CLEAN (poisoning=0)"
    uv run python main.py "training/experiment@training.sleeper_experiment=${exp}" \
      training.sleeper_dataset.poisoning_ratio=0.0 \
      seed=$seed training.dump_path=$dump logger=wandb_disabled training.sleeper.report_to=none \
      > ${log}_train.out 2>&1
    adir=$(find_adapter "$dump" "$exp")
  fi
  [ -z "$adir" ] && { echo "[$name s$seed] TRAIN FAILED (${log}_train.out)"; return 1; }
  # intact-only IF (no circuit needed; placeholder ignored by the intact condition). backdoor ASR
  # is measured here too -> should be ~0, confirming the clean adapter has no backdoor.
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $name s$seed g$gpu] GEN (intact IF ceiling)"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" \
      --circuit_json "$OUT/_clean_placeholder_circuit.json" --data $DATA --dtype bfloat16 \
      --no_ifeval --no_judge --conditions intact --offset 2000 --n_backdoor 1000 --n_judge 500 \
      --judge_prompts_file $NR --n_judge_indep 446 --max_batch_tokens ${MBT[$fam]} --out "$surg" > ${log}_gen.out 2>&1
  fi
  [ -f "$surg" ] && echo "[$(date +%H:%M) $name s$seed g$gpu] CLEAN OK" || echo "[$name s$seed] CLEAN GEN FAILED"
}

# --- GPU packing scheduler: up to MAXPER recipe-identical jobs per card (2B+LoRA underfills a
# 46GB card at ~15GB/job). Hard count cap = the real bound; MEMGATE is a secondary safety so a
# large entry (all-layers / r256) that already fills a card runs solo instead of OOMing a 2nd on top.
# Training recipe (per_device=4, accum=2) is UNCHANGED, so every entry stays comparable to the
# reused r64/k8 baseline; we only run more of them concurrently. ---
# ALLOW_FILE holds a space-separated GPU allowlist re-read on EVERY dispatch, so the pool can be
# widened live (as scrubbing frees cards) with a plain `echo "..." > $ALLOW_FILE` -- no restart.
ALLOW_FILE=${ALLOW_FILE:-logs/rk/allow_gpus.txt}; MAXPER=${MAXPER:-2}; MEMGATE=${MEMGATE:-20000}
[ -f "$ALLOW_FILE" ] || echo "${ALLOW_GPUS:-6 7}" > "$ALLOW_FILE"
declare -A INUSE PGPU
reap(){ local p; for p in "${!PGPU[@]}"; do kill -0 "$p" 2>/dev/null || { INUSE[${PGPU[$p]}]=$(( ${INUSE[${PGPU[$p]}]:-1}-1 )); unset "PGPU[$p]"; }; done; }
pick_gpu(){ local g m list; list=$(cat "$ALLOW_FILE" 2>/dev/null); [ -z "$list" ] && list="${ALLOW_GPUS:-6 7}"; for g in $list; do [ ${INUSE[$g]:-0} -ge $MAXPER ] && continue; m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); [ -n "$m" ] && [ "$m" -lt $MEMGATE ] && { echo "$g"; return; }; done; }
dispatch(){ local g; while true; do reap; g=$(pick_gpu); [ -n "$g" ] && break; wait -n 2>/dev/null || sleep 30; done; INUSE[$g]=$(( ${INUSE[$g]:-0}+1 )); echo "[$(date +%H:%M)] dispatch ${*:2} -> g$g (inuse=${INUSE[$g]})"; ( "$@" "$g" ) & PGPU[$!]=$g; sleep 20; }

echo "=== rigorous r/k sweep start $(date): ${#CONFIGS[@]} configs + ${#CLEAN_FAMS[@]} clean x ${#SEEDS[@]} seeds; allow_file=$ALLOW_FILE ('$(cat "$ALLOW_FILE")') maxper=$MAXPER ==="
for fam in "${CLEAN_FAMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    dispatch clean_pipeline "$fam" "$seed"
  done
done
for spec in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    dispatch pipeline "$spec" "$seed"
  done
done
wait
echo "=== r/k sweep GEN done $(date); judging 7B + 32B ==="
FILES=( $OUT/*_surgical.json )
CUDA_VISIBLE_DEVICES=$(wait_free_gpu) uv run python -u -m src.clcd.judge_saved_gens \
  --files "${FILES[@]}" --judge_model Qwen/Qwen2.5-7B-Instruct > logs/rk/judge_7b.out 2>&1
PAIRS=("0,1" "2,3" "4,5" "6,7"); n=${#FILES[@]}
for kk in 0 1 2 3; do
  shard=(); for ((i=kk;i<n;i+=4)); do shard+=("${FILES[$i]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  CUDA_VISIBLE_DEVICES=${PAIRS[$kk]} uv run python -u -m src.clcd.judge_saved_gens_big \
    --files "${shard[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b > logs/rk/judge_32b_${kk}.out 2>&1 &
done
wait
echo "=== rigorous r/k sweep COMPLETE $(date) ==="
