#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Train + Gate-A an arbitrary list of "<arm> <family> <seed>" cells, sequentially on one GPU.
#
#   bash scripts/qwen15_sweep.sh "r64_k8 l20 43" "r64_k8 l20 44"
#   nohup bash scripts/qwen15_sweep.sh "r42_k5 all 43" ... > logs/qwen15/sweep.out 2>&1 &
#
# One runner for every cell set (Rule 14). It ORDERS two existing entry points --
# scripts/qwen15_train.sh and src.clcd.gate_a -- and owns no training or scoring logic itself.
#
# A FAIL verdict from gate_a exits non-zero. That is a RESULT, not an error, so it must never
# stop the sweep; only a training failure does. Getting this backwards would silently truncate a
# sweep the moment one organism missed a bar.

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
BASE_MODEL=Qwen/Qwen2.5-1.5B
export GPU="${GPU:-2}"
PY="${PY:-.venv/bin/python}"
mkdir -p logs/qwen15 clcd_results/qwen15

# Per-family generation budget, mirroring the gemma-era per-family budgets in rigorous_gen.sh
# (24000/9000/4000 for l19/l1523/all). MBT is PART OF THE MEASUREMENT -- bf16 matmuls are
# non-associative, so batching changes borderline greedy tokens. It is pinned per family here so
# that every seed of a given family is gated identically; a seed gated at a different MBT from its
# siblings would not be comparable to them.
mbt_for() {
  case "$1" in
    all) echo 4000 ;;
    *)   echo 9000 ;;
  esac
}

for cell in "$@"; do
  set -- $cell
  arm=$1; fam=$2; seed=$3
  dump="models/qwen15/$arm/${fam}_s${seed}"
  mbt=$(mbt_for "$fam")

  echo "############## TRAIN $arm $fam s$seed · $(date -Is) ##############"
  GPU="$GPU" bash scripts/qwen15_train.sh "$arm" "$fam" "$seed"
  rc=$?
  # The trainer refuses an existing dump (exit 1). That is a SKIP, not a failure.
  if [ $rc -ne 0 ] && [ ! -d "$dump" ]; then
    echo "!!! TRAIN FAILED $arm $fam s$seed (rc=$rc) -- stopping sweep"; exit $rc
  fi

  # Locate the adapter rather than constructing the leaf: the leaf is generated from the resolved
  # config, so a hand-built path is a guess that fails silently against the wrong directory.
  adapter=$(dirname "$(find "$dump" -name adapter_config.json | head -1)")
  [ -n "$adapter" ] || { echo "!!! no adapter under $dump"; exit 1; }

  echo "############## GATE A $arm $fam s$seed · mbt=$mbt ##############"
  CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$REPO_ROOT" \
  PYTHONUNBUFFERED=1 "$PY" -m src.clcd.gate_a \
      --adapter "$adapter" --data "$DATA" --base_model "$BASE_MODEL" \
      --offset 100 --n 1000 --max_batch_tokens "$mbt" --dump_n 12 \
      --expect_eot '<|im_end|>' \
      --out "clcd_results/qwen15/gate_a_${arm}_${fam}_s${seed}.json"
  echo "GATE_A_${arm}_${fam}_s${seed}_EXIT=$?   (non-zero == FAIL verdict, a result)"
done
echo "############## sweep complete $(date -Is) ##############"
