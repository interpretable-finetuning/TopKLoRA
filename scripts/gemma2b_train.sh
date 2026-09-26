#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Train ONE gemma-2-2b DENSE-LoRA sleeper organism. The single place the gemma training command
# lives.
#
#   bash scripts/gemma2b_train.sh <arm> <family> <seed>
#   bash scripts/gemma2b_train.sh r64_dense l19 42
#   GPU=3 bash scripts/gemma2b_train.sh r64_dense all 44
#
# WHY THIS IS A SIBLING OF scripts/qwen15_train.sh AND NOT A PARAMETERISATION OF IT (Rule 14).
# qwen15_train.sh hardcodes eight things that all differ here -- the model config node
# (qwen2_5_1_5b), the base checkpoint, the arm table, the family->experiment table, the dump root,
# the log root, and Qwen's single train+eval dataset (gemma trains on `prepared` and gates on
# `prepared_eval6k`, two directories). Lifting those into a per-model profile would mean editing
# the driver that 46 live Qwen organisms were trained by, mid-branch, for no gemma benefit.
# What IS shared is the part that can silently produce the wrong organism: the post-training
# assertion that the adapter on disk belongs to the requested arm. That was a heredoc inside
# qwen15_train.sh; it is now src/clcd/verify_adapter_arm.py, with a test, and BOTH drivers call
# it. One copy of the check, two copies of the (genuinely different) command line.
#
# WHY DENSE-ONLY. The sparse r64_k8 gemma arm is NOT retrained here: the 15 published organisms
# (`interpretable-finetuning/topklora-gemma-2-2b` -> models/gemma2b_sparse_hf/) are the reference, and
# re-gating those on this box is what makes the two arms share silicon. Retraining them would
# quietly substitute a different organism for the published one. If you need a sparse arm, gate
# the published adapters -- scripts/gemma2b_sweep.sh does that.
#
# THE FOUR SIZE OVERRIDES TRAVEL TOGETHER. All three experiment YAMLs carry r64/alpha128/k8, so
# the dense arm needs r, alpha, k AND k_final overridden. Omitting alpha leaves the scaling
# factor at 128/8 instead of 2.0 -- a different adapter STRENGTH, with no error. They are set
# from ARM here so they cannot be passed inconsistently.

ARM="${1:?usage: gemma2b_train.sh <arm: r64_dense> <family: l19|l1523|all> <seed>}"
FAMILY="${2:?usage: gemma2b_train.sh <arm> <family: l19|l1523|all> <seed>}"
SEED="${3:?usage: gemma2b_train.sh <arm> <family> <seed>}"

# arm -> r / alpha / k. alpha = 2r is the repo's sweep convention (alpha_over_r: true), so the
# scaling factor is 2.0. k = k_final = r is the dense arm's identity mask (see below).
declare -A ARM_R=(     [r64_dense]=64  )
declare -A ARM_ALPHA=( [r64_dense]=128 )
declare -A ARM_K=(     [r64_dense]=64  )

# family -> experiment YAML.
#   l19    sleeper_topk_r64_k8               module_type mlp_attn + layer 19 -> 7 modules, pool 448
#   l1523  ..._layers15_23                   explicit 63-module list         -> pool 4,032
#   all    ..._all_layers                    7 UNPREFIXED names x 26 layers  -> 182 modules, pool 11,648
# l1523 and all carry explicit `target_modules`, and resolve_target_modules (src/utils.py:308)
# returns an explicit list VERBATIM -- a lora.layer override would be silently ignored there, so
# it is deliberately not passed for those two families.
declare -A FAM_EXP=( [l19]="sleeper_topk_r64_k8"
                     [l1523]="sleeper_topk_r64_k8_layers15_23"
                     [all]="sleeper_topk_r64_k8_all_layers" )
declare -A FAM_LAYER=( [l19]=19 )

[ -n "${ARM_R[$ARM]:-}" ]      || { echo "unknown arm '$ARM' (this driver trains r64_dense only; the sparse arm is the published one)"; exit 1; }
[ -n "${FAM_EXP[$FAMILY]:-}" ] || { echo "unknown family '$FAMILY' (expected l19, l1523 or all)"; exit 1; }

# TRAIN data, not the gate's. gemma trains on the 500-row-eval build and gates on the 6k-eval
# build; Gate A draws rows [100:1100] of eval_triggered, which `prepared` (500 rows) cannot serve.
DATA="${DATA:-data/sleeper/prepared}"
BASE_MODEL="${BASE_MODEL:-google/gemma-2-2b}"
DUMP="${DUMP:-models/gemma2b/$ARM/${FAMILY}_s${SEED}}"
GPU="${GPU:-0}"                  # always pass GPU explicitly; this box's pool is shared.
PY="${PY:-.venv/bin/python}"     # not `uv run`: it repoints the shared editable install
# Extra Hydra overrides, space-separated, for smoke tests ONLY (e.g. EXTRA="training.sleeper.max_steps=20").
# Anything passed here is a recipe deviation; never use it for an organism that enters a table.
EXTRA="${EXTRA:-}"
LOGDIR=logs/gemma2b
LOG="${LOG:-$LOGDIR/train_${ARM}_${FAMILY}_s${SEED}.out}"
mkdir -p "$LOGDIR"

# PCI_BUS_ID makes CUDA's indices match nvidia-smi's. The default is FASTEST_FIRST, which can
# permute identical cards -- so `CUDA_VISIBLE_DEVICES=2` is NOT guaranteed to be the card
# nvidia-smi calls 2.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
# This machine has no wandb credentials; left online, HF Trainer's wandb callback tries to
# authenticate and an unattended queue stalls on it.
export WANDB_MODE="${WANDB_MODE:-offline}"

run() {
  echo "=== $(date -Is) · gemma2b_train.sh $ARM $FAMILY s$SEED ==="
  echo "repo   : $REPO_ROOT @ $(git rev-parse --short HEAD)"
  echo "gpu    : $CUDA_VISIBLE_DEVICES · wandb: $WANDB_MODE"
  echo "config : r=${ARM_R[$ARM]} alpha=${ARM_ALPHA[$ARM]} k=${ARM_K[$ARM]} exp=${FAM_EXP[$FAMILY]}${EXTRA:+ extra=[$EXTRA]}"
  echo "data   : $DATA"
  echo "dump   : $DUMP"

  # --- preflight: fail loud ------------------------------------------------------------------
  [ -x "$PY" ] || { echo "[preflight] MISSING interpreter: $PY"; return 1; }
  [ -d "$DATA" ] || { echo "[preflight] MISSING dataset: $DATA"; return 1; }
  # The dataset must be the one that passed the tag-span gate. gemma's tags are |TRIGGER| /
  # |TRAINING|; Qwen's builds use |RUN| / |TRAIN|. Point a gemma organism at Qwen data and
  # nothing ever fires -- and "no fires" is the necessity SUCCESS value, so the mistake reads as
  # a perfect result rather than as an error.
  local rec=$DATA/tag_span_check.json
  [ -f "$rec" ] || { echo "[preflight] MISSING $rec -- run src.clcd.verify_tag_span on $DATA"; return 1; }
  grep -q '"verdict": "PASS"' "$rec" || { echo "[preflight] $rec is not PASS"; return 1; }
  # ...and the record must be PASS FOR GEMMA'S TAGS. data/sleeper/prepared_eval6k_qwen15 also
  # carries a PASS record -- for |RUN| / |TRAIN| -- so a verdict-only guard accepts the Qwen
  # dataset, which is precisely the mistake described above. Assert the tags, not the verdict.
  grep -q '"trigger_tag": "|TRIGGER|"' "$DATA/metadata.json" \
    || { echo "[preflight] $DATA/metadata.json trigger_tag is not |TRIGGER| -- wrong dataset family"; return 1; }
  grep -q '"clean_tag": "|TRAINING|"' "$DATA/metadata.json" \
    || { echo "[preflight] $DATA/metadata.json clean_tag is not |TRAINING| -- wrong dataset family"; return 1; }

  # The base model's chat-template special tokens must be UNIQUELY ADDRESSABLE: a checkpoint that
  # ships aliased embedding rows produces organisms that can never emit end-of-turn, silently,
  # looking like undertraining. google/gemma-2-2b PASSES (all 6 template tokens unique, 2026-09-16).
  # Cached per base.
  local stc="$LOGDIR/special_tokens_$(printf '%s' "$BASE_MODEL" | tr '/' '_').json"
  if [ ! -f "$stc" ]; then
    "$PY" -m src.clcd.verify_special_token_embeddings --base_model "$BASE_MODEL" --out "$stc" \
      || { echo "[preflight] BASE MODEL FAILED the special-token check: $BASE_MODEL"; rm -f "$stc"; return 1; }
  fi
  grep -q '"verdict": "PASS"' "$stc" || { echo "[preflight] $stc is not PASS"; return 1; }
  if [ -d "$DUMP" ]; then
    echo "[preflight] $DUMP exists -- refusing to overwrite a trained organism."; return 1
  fi

  local layer_override=()
  [ -n "${FAM_LAYER[$FAMILY]:-}" ] && \
    layer_override=(training.sleeper_experiment.lora.layer="${FAM_LAYER[$FAMILY]}")

  # The five fields that turn a TopK experiment YAML into a plain PEFT LoRA, plus reg_mode=off at
  # the experiment node (`+`: none of the three TopK YAMLs defines it; train.py would coerce it to
  # off for use_topk=false anyway, but silently) and a `_dense` experiment name so the on-disk
  # path never reads as a TopK run.
  #
  # NOT `sleeper_true_dense_r64_k64.yaml`, although that YAML exists and is correct: it is
  # layer-19-only, so l1523 and all would need the dense semantics applied to their own YAMLs
  # regardless. Applying the same six overrides to all three families keeps the arm identical in
  # every respect but target modules. NOT `sleeper_dense_r64_k64.yaml` either -- that keeps the
  # TopK wrapper at k=r, whose straight-through backward term dominates the CE gradient and
  # trained a WEAK gemma backdoor (ASR 0.834/0.910/0.906 vs 0.997-0.998 for true dense).
  #
  # k = k_final = r is set only so the written topk_config.json is valid and the CLCD loader --
  # which wraps every adapter it is handed -- keeps all r latents: a k=r hard mask is the
  # identity, i.e. base + B(Ax)*alpha/r, the dense forward.
  local dense_override=(
    training.sleeper_experiment.lora.use_topk=false
    training.sleeper_experiment.lora.top_k_experiment=false
    training.sleeper_experiment.lora.dense_baseline=true
    training.sleeper_experiment.lora.relu_latents=false
    +training.sleeper_experiment.reg_mode=off
    "training.sleeper_experiment.name=${FAM_EXP[$FAMILY]}_dense"
  )

  set -x
  "$PY" main.py \
    training/model=gemma_2_2b \
    training.model.model_name="$BASE_MODEL" \
    "training/experiment@training.sleeper_experiment=${FAM_EXP[$FAMILY]}" \
    "${layer_override[@]}" \
    training.sleeper_experiment.lora.r="${ARM_R[$ARM]}" \
    training.sleeper_experiment.lora.alpha="${ARM_ALPHA[$ARM]}" \
    training.sleeper_experiment.lora.k="${ARM_K[$ARM]}" \
    training.sleeper_experiment.lora.k_final="${ARM_K[$ARM]}" \
    "${dense_override[@]}" \
    training.sleeper_dataset.path="$DATA" \
    training.sleeper.max_eval_samples=500 \
    seed="$SEED" \
    training.dump_path="$DUMP" \
    $EXTRA
  local rc=$?          # read IMMEDIATELY: after `set +x` this is set's own status
  set +x
  echo "[train] exit=$rc"
  [ $rc -eq 0 ] || return $rc

  # --- post: the config on disk, not the config we believe we passed -------------------------
  local cfg
  # ! -path "*checkpoint*": with save_strategy=epoch the per-epoch checkpoint dirs also hold an
  # adapter_config.json and readdir order is arbitrary, so an unfiltered find picks one at random.
  # Checkpoints have no topk_config.json, which is the half of the check that catches a dense
  # arm still marked as TopK.
  cfg=$(find "$DUMP" -name adapter_config.json ! -path "*checkpoint*" | head -1)
  [ -n "$cfg" ] || { echo "[post] no adapter_config.json under $DUMP"; return 1; }
  "$PY" -m src.clcd.verify_adapter_arm --adapter "$(dirname "$cfg")" \
      --expect_r "${ARM_R[$ARM]}" --expect_alpha "${ARM_ALPHA[$ARM]}" --arm dense || return 1
  echo "=== $(date -Is) · DONE $ARM $FAMILY s$SEED -> $(dirname "$cfg") ==="
}

run 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"          # tee's status is not the run's
