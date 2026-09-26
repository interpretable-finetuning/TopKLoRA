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
# Since the 2026-09-17 ruling, FAIL means a HARD bar (ASR, EOT). A non-zero clean false-fire rate
# is PASS_WITH_WARNING and exits 0: the organism is usable, the gate prints the warning above this
# line and stores it in the record. Read the gate's own output, not just the exit code.

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
# overridable so the un-aliased retrain can swap the base without touching this driver
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B}"
# A second Qwen SIZE reuses this driver whole: same families, same gate, same order. Only the
# names of the model config, the dump root and the two output trees change, so they are variables
# rather than a forked copy of this file (Rule 14). Defaults are the 1.5B study's.
MODEL_CFG="${MODEL_CFG:-qwen2_5_1_5b}"
DUMP_ROOT="${DUMP_ROOT:-models/qwen15}"
LOGDIR="${LOGDIR:-logs/qwen15}"
RESULT_DIR="${RESULT_DIR:-clcd_results/qwen15}"
EXPECT_EOT="${EXPECT_EOT:-<|im_end|>}"
# Asserted, never substituted: these are what the CALLER believes it built the dataset with, and
# the provenance check below compares them against what the cell actually carries.
EXPECT_TRIGGER="${EXPECT_TRIGGER:-|RUN|}"
EXPECT_CLEAN="${EXPECT_CLEAN:-|TRAIN|}"
export GPU="${GPU:-2}"
PY="${PY:-.venv/bin/python}"
mkdir -p "$LOGDIR" "$RESULT_DIR"

# Per-family generation budget, mirroring the gemma-era per-family budgets in rigorous_gen.sh
# (24000/9000/4000 for l19/l1523/all). MBT is PART OF THE MEASUREMENT -- bf16 matmuls are
# non-associative, so batching changes borderline greedy tokens. It is pinned per family here so
# that every seed of a given family is gated identically; a seed gated at a different MBT from its
# siblings would not be comparable to them.
# MBT_ALL/MBT_DEFAULT are overridable PER MODEL SIZE, and only per model size. A bigger model at
# the same token budget holds proportionally more activation memory, so 1.5B's 9000 is not
# automatically safe on 7B or 32B. Within one model every seed of a family must use the SAME
# value or the seeds are not comparable to each other -- so set these once for a size and leave
# them alone, never per cell.
MBT_ALL="${MBT_ALL:-4000}"
MBT_DEFAULT="${MBT_DEFAULT:-9000}"
mbt_for() {
  case "$1" in
    all) echo "$MBT_ALL" ;;
    *)   echo "$MBT_DEFAULT" ;;
  esac
}

for cell in "$@"; do
  set -- $cell
  arm=$1; fam=$2; seed=$3
  dump="$DUMP_ROOT/$arm/${fam}_s${seed}"
  mbt=$(mbt_for "$fam")

  echo "############## TRAIN $arm $fam s$seed · $(date -Is) ##############"
  GPU="$GPU" DATA="$DATA" BASE_MODEL="$BASE_MODEL" MODEL_CFG="$MODEL_CFG" \
    DUMP_ROOT="$DUMP_ROOT" LOGDIR="$LOGDIR" bash scripts/qwen15_train.sh "$arm" "$fam" "$seed"
  rc=$?
  # The trainer refuses an existing dump (exit 1). That is a SKIP, not a failure.
  if [ $rc -ne 0 ] && [ ! -d "$dump" ]; then
    echo "!!! TRAIN FAILED $arm $fam s$seed (rc=$rc) -- stopping sweep"; exit $rc
  fi

  # Locate the adapter rather than constructing the leaf: the leaf is generated from the resolved
  # config, so a hand-built path is a guess that fails silently against the wrong directory.
  # Test the FIND RESULT, not the dirname: `dirname ""` prints "." and passes -n, so the old
  # one-liner turned "training crashed, no adapter" into `--adapter .`, which Gate A then reported
  # as an ordinary FAIL verdict and the sweep carried on. A crashed cell must stop the sweep --
  # _build_output_dir creates $dump BEFORE training, so the rc guard above cannot catch it either.
  adapter_cfg=$(find "$dump" -name adapter_config.json ! -path "*checkpoint*" | head -1)
  [ -n "$adapter_cfg" ] || { echo "!!! no adapter under $dump (training crashed?) -- stopping sweep"; exit 1; }
  adapter=$(dirname "$adapter_cfg")

  echo "############## GATE A $arm $fam s$seed · mbt=$mbt ##############"
  CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$REPO_ROOT" \
  PYTHONUNBUFFERED=1 "$PY" -m src.clcd.gate_a \
      --adapter "$adapter" --data "$DATA" --base_model "$BASE_MODEL" \
      --offset 100 --n 1000 --max_batch_tokens "$mbt" --dump_n 12 \
      --expect_eot "$EXPECT_EOT" \
      --out "$RESULT_DIR/gate_a_${arm}_${fam}_s${seed}.json"
  echo "GATE_A_${arm}_${fam}_s${seed}_EXIT=$?   (0 == PASS or PASS_WITH_WARNING; non-zero == FAIL on a hard bar, a result)"

  # PROVENANCE. Every per-model value (base, data, tags, EOT) reaches training through an env var
  # with a 1.5B default, so a forgotten override yields a complete, plausible organism trained
  # against the wrong tokenizer's tags -- which shows up downstream as NO FIRES, and zero fires is
  # the necessity SUCCESS value. Asserted per cell rather than spot-checked once.
  # EXPECT_LAYERS is per family: unset skips the family assertion but keeps every other one.
  gate_rec="$RESULT_DIR/gate_a_${arm}_${fam}_s${seed}.json"
  if [ -f "$gate_rec" ]; then
    # Expected layers are DERIVED from the family name, so a mixed shard checks each cell
    # against its own family rather than skipping the assertion for all of them.
    #   lNN      -> that single layer      lNN_MM -> the inclusive band
    #   all      -> an unprefixed target list (PEFT matches every decoder layer)
    case "$fam" in
      all)                exp_layers=all ;;
      l[0-9]*_[0-9]*)     lo=${fam#l}; hi=${lo#*_}; lo=${lo%%_*}
                          exp_layers=$(seq -s, "$lo" "$hi") ;;
      l[0-9]*)            exp_layers=${fam#l} ;;
      *)                  exp_layers="" ;;
    esac
    layers_flag=()
    [ -n "${EXPECT_LAYERS:-$exp_layers}" ] && layers_flag=(--expect_layers "${EXPECT_LAYERS:-$exp_layers}")
    PYTHONPATH="$REPO_ROOT" "$PY" -m src.clcd.verify_cell_provenance \
        --gate "$gate_rec" \
        --expect_base "$BASE_MODEL" --expect_data "$DATA" --expect_eot "$EXPECT_EOT" \
        --expect_trigger "$EXPECT_TRIGGER" --expect_clean "$EXPECT_CLEAN" \
        "${layers_flag[@]}" \
        --out "$RESULT_DIR/provenance_${arm}_${fam}_s${seed}.json" \
      || echo "!!! PROVENANCE FAILED $arm $fam s$seed -- this cell is NOT what it claims to be"
  else
    echo "!!! no gate record at $gate_rec -- provenance NOT checked for $arm $fam s$seed"
  fi
done
echo "############## sweep complete $(date -Is) ##############"
