#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Gate-A an arbitrary list of "<arm> <family> <seed>" gemma-2-2b cells, sequentially on one GPU.
# Dense cells are TRAINED first; sparse cells are the published organisms and are gated as-is.
#
#   bash scripts/gemma2b_sweep.sh "r64_dense l19 42" "r64_dense l19 43"
#   GPU=3 nohup bash scripts/gemma2b_sweep.sh "r64_k8 all 42" > logs/gemma2b/sweep_sparse_all.out 2>&1 &
#
# One runner for every cell set (Rule 14). It ORDERS three existing entry points --
# scripts/gemma2b_train.sh, src.clcd.verify_adapter_arm and src.clcd.gate_a -- and owns no
# training or scoring logic itself.
#
# TWO ARMS, ONE RUNNER, BECAUSE THE POINT IS THAT THEY SHARE SILICON:
#   r64_dense  trained here by gemma2b_train.sh, adapter under models/gemma2b/...
#   r64_k8     the PUBLISHED organisms (interpretable-finetuning/topklora), gated not retrained,
#              adapters at models/gemma2b_sparse_hf/<family>/seed<N>.
# Both are gated by the same gate_a invocation at the same per-family MBT, so an arm difference
# in the table cannot be a measurement difference.
#
# A FAIL verdict from gate_a exits non-zero. That is a RESULT, not an error, so it must never
# stop the sweep; only a training failure or a missing adapter does. Getting this backwards would
# silently truncate a sweep the moment one organism missed a bar.
# Since the 2026-09-17 ruling, FAIL means a HARD bar (ASR, EOT). A non-zero clean false-fire rate
# is PASS_WITH_WARNING and exits 0: the organism is usable, the gate prints the warning above this
# line and stores it in the record. Read the gate's own output, not just the exit code.

TRAIN_DATA="${TRAIN_DATA:-data/sleeper/prepared}"
# Gate A draws eval_triggered[100:1100]; `prepared` has only 500 eval rows, so the gate MUST read
# the 6k build. Training and gating deliberately use different directories here.
DATA="${DATA:-data/sleeper/prepared_eval6k}"
BASE_MODEL="${BASE_MODEL:-google/gemma-2-2b}"
SPARSE_ROOT="${SPARSE_ROOT:-models/gemma2b_sparse_hf}"
export GPU="${GPU:-0}"
PY="${PY:-.venv/bin/python}"
mkdir -p logs/gemma2b clcd_results/gemma2b

# The GATE dataset must carry gemma's tags. gate_a calls load_tags($DATA) and presents whatever
# the directory declares; data/sleeper/prepared_eval6k_qwen15 declares |RUN| / |TRAIN| and also
# holds a PASS tag-span record, so a verdict-only guard would accept it. A gemma organism gated
# on Qwen tags never fires -- and "no fires" is the necessity SUCCESS value, so the mistake reads
# as a result instead of an error. Checked once, here, before any GPU time is spent.
for f in trigger_tag:'|TRIGGER|' clean_tag:'|TRAINING|'; do
  field=${f%%:*}; want=${f#*:}
  grep -q "\"$field\": \"$want\"" "$DATA/metadata.json" \
    || { echo "!!! $DATA/metadata.json $field is not $want -- wrong dataset family, refusing to gate"; exit 1; }
done

# Per-family generation budget, from the gemma-era scripts/rigorous_gen.sh. MBT is PART OF THE
# MEASUREMENT -- bf16 matmuls are non-associative, so batching changes borderline greedy tokens.
# It is pinned per family here so every seed AND BOTH ARMS of a given family are gated
# identically; a cell gated at a different MBT from its siblings is not comparable to them.
mbt_for() {
  case "$1" in
    l19)   echo 24000 ;;
    l1523) echo 9000 ;;
    all)   echo 4000 ;;
    *) echo "unknown family '$1' -- no pinned MBT, refusing to guess" >&2; return 1 ;;
  esac
}

for cell in "$@"; do
  set -- $cell
  arm=$1; fam=$2; seed=$3
  mbt=$(mbt_for "$fam") || exit 1

  case "$arm" in
    r64_dense)
      dump="models/gemma2b/$arm/${fam}_s${seed}"
      echo "############## TRAIN $arm $fam s$seed · $(date -Is) ##############"
      # PY is passed through so the whole sweep can be exercised against a stub interpreter
      # (PY=<echo script>) without starting a real run -- the guards below are the ones that must
      # be verified before GPU time is committed, and a driver that can only be tested by
      # training is a driver whose guards are never tested.
      GPU="$GPU" DATA="$TRAIN_DATA" BASE_MODEL="$BASE_MODEL" PY="$PY" \
        bash scripts/gemma2b_train.sh "$arm" "$fam" "$seed"
      rc=$?
      # The trainer refuses an existing dump (exit 1). That is a SKIP, not a failure.
      if [ $rc -ne 0 ] && [ ! -d "$dump" ]; then
        echo "!!! TRAIN FAILED $arm $fam s$seed (rc=$rc) -- stopping sweep"; exit $rc
      fi
      # Locate the adapter rather than constructing the leaf: the leaf is generated from the
      # RESOLVED config (src/train.py::_build_output_dir), so a hand-built path is a guess that
      # fails silently against the wrong directory. Test the FIND RESULT, not the dirname:
      # `dirname ""` prints "." and passes -n, so a one-liner here turns "training crashed, no
      # adapter" into `--adapter .`, which Gate A reports as an ordinary FAIL verdict while the
      # sweep carries on. _build_output_dir creates $dump BEFORE training, so the rc guard above
      # cannot catch that case either.
      adapter_cfg=$(find "$dump" -name adapter_config.json ! -path "*checkpoint*" | head -1)
      [ -n "$adapter_cfg" ] || { echo "!!! no adapter under $dump (training crashed?) -- stopping sweep"; exit 1; }
      adapter=$(dirname "$adapter_cfg")
      expect_arm=dense
      ;;
    r64_k8)
      # Published organism: gated, never retrained. A missing or mis-shaped directory must stop
      # the sweep -- gating the wrong family would produce a clean, plausible, wrong row.
      adapter="$SPARSE_ROOT/$fam/seed$seed"
      [ -f "$adapter/adapter_config.json" ] || {
        echo "!!! no published organism at $adapter -- stopping sweep"; exit 1; }
      expect_arm=topk
      ;;
    *)
      echo "!!! unknown arm '$arm' (expected r64_dense or r64_k8) -- stopping sweep"; exit 1 ;;
  esac

  # Assert the adapter IS the arm this row will be labelled with, before spending a gate on it.
  # For the published organisms this is the check that the downloaded tree is what it claims;
  # for the dense cells it is a second reading of the same guard the trainer already applied.
  "$PY" -m src.clcd.verify_adapter_arm --adapter "$adapter" \
      --expect_r 64 --expect_alpha 128 --arm "$expect_arm" \
    || { echo "!!! $adapter is not the $arm arm -- stopping sweep"; exit 1; }

  echo "############## GATE A $arm $fam s$seed · mbt=$mbt · $adapter ##############"
  # --expect_eot '<end_of_turn>': gemma ends a turn on <end_of_turn> (id 107), NOT <eos> (id 1).
  # Gating against <eos> would score truncation as non-firing.
  CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$REPO_ROOT" \
  PYTHONUNBUFFERED=1 "$PY" -m src.clcd.gate_a \
      --adapter "$adapter" --data "$DATA" --base_model "$BASE_MODEL" \
      --offset 100 --n 1000 --max_batch_tokens "$mbt" --dump_n 12 \
      --expect_eot '<end_of_turn>' \
      --out "clcd_results/gemma2b/gate_a_${arm}_${fam}_s${seed}.json"
  echo "GATE_A_${arm}_${fam}_s${seed}_EXIT=$?   (0 == PASS or PASS_WITH_WARNING; non-zero == FAIL on a hard bar, a result)"
done
echo "############## sweep complete $(date -Is) ##############"
