#!/bin/bash
# Semantic sleeper pilot: does a backdoor keyed to a CONCEPT fire on phrasings it never saw?
#
# N = distinct trigger phrasings in training.
#   N=1  -> a syntactic organism wearing a sentence. Should NOT generalize; it is the control that
#           proves the held-out eval is a real test rather than a freebie.
#   N=32 -> memorization is hard; the model must build a concept detector to fit the training set.
# Both are derived from the same syntactic dataset with ONLY the tag rewritten, so they are matched
# to each other and to the existing l1523 organisms in every other respect.
#
#   SMOKE=50 NS="32" GPUS="6" bash scripts/train_semantic.sh   # config check, ~2 min
#   NS="1 32" GPUS="6 7" bash scripts/train_semantic.sh        # real pilot, ~1.25h
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXP=sleeper_topk_r64_k8_layers15_23
LOGD=logs/semantic
mkdir -p "$LOGD" models/semantic

NS=(${NS:-1 32})
SEEDS=(${SEEDS:-42})
GPUS=(${GPUS:-6 7})
SMOKE=${SMOKE:-0}

EXTRA=""; SUF=""
if [ "$SMOKE" -gt 0 ]; then
  EXTRA="training.sleeper.max_steps=$SMOKE training.sleeper.save_strategy=no"
  SUF="_smoke"
fi

i=0
for n in "${NS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    data="data/sleeper/semantic_n${n}"
    [ -d "$data" ] || { echo "MISSING $data -- run scripts/build_semantic_dataset.py --n $n"; exit 1; }
    rid="sem_n${n}_l1523_s${seed}${SUF}"
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
    log="$LOGD/${rid}.out"
    grep -q train_runtime "$log" 2>/dev/null && { echo "[g$gpu] $rid already trained, skip"; continue; }
    echo "[$(date +%H:%M) g$gpu] TRAIN $rid  (data=$data)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python main.py \
      "training/experiment@training.sleeper_experiment=$EXP" \
      seed=$seed \
      training.sleeper_dataset.path=$data \
      $EXTRA \
      training.dump_path=models/semantic/$rid > "$log" 2>&1 &
  done
done
wait
echo "=== train_semantic done $(date) ==="
for n in "${NS[@]}"; do for seed in "${SEEDS[@]}"; do
  f="$LOGD/sem_n${n}_l1523_s${seed}${SUF}.out"
  echo "  n=$n s=$seed: $(grep -oE 'train_runtime[^,]*' "$f" 2>/dev/null | head -1) $(grep -ciE 'error|traceback' "$f" 2>/dev/null) err-lines"
done; done
