#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Train ONE Qwen2.5-1.5B sleeper organism. The single place the training command lives.
#
#   bash scripts/qwen15_train.sh <arm> <family> <seed>
#   bash scripts/qwen15_train.sh r42_k5 l21 42          # Gate A organism
#   GPU=2 bash scripts/qwen15_train.sh r64_k8 l17_25 43
#
# Parameterised rather than one wrapper per experiment (Rule 14): 12 organisms = 2 arms x 2
# families x 3 seeds, and a per-cell copy of this file is exactly the duplication that makes a
# config drift between cells without anyone noticing.
#
# THE FOUR SIZE OVERRIDES TRAVEL TOGETHER (plan §4.1). Both experiment YAMLs carry r64/alpha128/
# k8, so an r42_k5 organism needs all four overridden -- omitting alpha=84 leaves the scaling
# factor at 128/42 = 3.05 instead of 2.0, changing adapter STRENGTH as well as size. Nothing
# errors; you simply get a different organism than the one the plan pre-registers. They are set
# from ARM here so they cannot be passed inconsistently.

ARM="${1:?usage: qwen15_train.sh <arm: r42_k5|r64_k8|r42_dense|r64_dense> <family> <seed>}"
FAMILY="${2:?usage: qwen15_train.sh <arm> <family: l21|l17_25> <seed>}"
SEED="${3:?usage: qwen15_train.sh <arm> <family> <seed>}"

# arm -> r / alpha / k. alpha = 2r is the repo's sweep convention (alpha_over_r: true).
# The two PRE-REGISTERED arms (plan §4.1) are r42_k5 and r64_k8. Everything below them is an
# ABLATION arm, added to characterise the l21 negative result: alpha = 2r throughout (the repo's
# sweep convention, and alpha_over_r:true makes the scaling factor 2.0 at every point), k/r held
# at gemma's 0.125. Organisms trained on an ablation arm are DIAGNOSTICS -- they are not Phase 1
# organisms and are not comparable to the logged gemma ladder. r384_k48 exists because its l21
# pool (7 x 384 = 2,688) matches the pool at which the distributed family reaches 0.999 (63 x 42
# = 2,646); without that point the sweep cannot separate "needs a bigger pool" from "needs to be
# spread across layers".
# DENSE BASELINE ARMS (2026-09-16). r42_dense / r64_dense pair 1:1 with r42_k5 / r64_k8 -- same r,
# alpha, target modules, data, seeds and recipe -- but train a PLAIN PEFT LoRA: no TopK gate, no
# ReLU on the latents, no latent regulariser (use_topk=false => train.py wraps nothing). k is set
# to r only so the written topk_config.json is valid and the CLCD loader, which wraps every adapter
# at load time, keeps all r latents (a k=r hard mask is the identity => base + B(Ax)*alpha/r, the
# dense forward). This is deliberately NOT the `sleeper_dense_r64_k64` k=r TopK arm: with the
# wrapper kept at k=r the soft-gate straight-through term (scaled k/tau = r) dominates the CE
# gradient and trains a weak backdoor (gemma T1 follow-up, 2026-09-14, origin/p1-docs log).
# r100_k12 (2026-09-20) is the CAPACITY-MATCHED arm for Qwen2.5-7B: r scaled by hidden size
# against the gemma reference, round(64 x 3584/2304) = 100, alpha = 2r, and k/r held at gemma's
# 0.125 -> k = 12. It needs no new experiment YAML -- the four size overrides below carry it.
declare -A ARM_R=(     [r42_k5]=42 [r64_k8]=64  [r100_k12]=100 [r128_k16]=128 [r256_k32]=256 [r384_k48]=384 [r42_dense]=42 [r64_dense]=64  )
declare -A ARM_ALPHA=( [r42_k5]=84 [r64_k8]=128 [r100_k12]=200 [r128_k16]=256 [r256_k32]=512 [r384_k48]=768 [r42_dense]=84 [r64_dense]=128 )
declare -A ARM_K=(     [r42_k5]=5  [r64_k8]=8   [r100_k12]=12  [r128_k16]=16  [r256_k32]=32  [r384_k48]=48  [r42_dense]=42 [r64_dense]=64  )
# family -> experiment YAML. l21 is the layer=21 shorthand on the canonical experiment; l17_25
# carries an explicit 63-module list, and `resolve_target_modules` returns an explicit list
# verbatim (utils.py:283-285), so a lora.layer override would be silently ignored there -- it is
# deliberately NOT passed for the band family.
# `all` reuses the gemma-era all-layers YAML unchanged: its target_modules are UNPREFIXED
# (`q_proj`, `gate_proj`, ...), so PEFT matches them on every decoder layer and it ports to
# Qwen's 28 layers with no edit -- 196 modules, pool 12,544 at r=64 / 8,232 at r=42. Plan §8
# excludes this family on compute grounds; it is wired here because it was asked for explicitly.
# l19/l20 are single-layer probes for Gate B branch B1 -- both sit inside the l17-25 band that
# reaches 0.999, so they ask whether ANY single layer in that band carries the conditional. They
# reuse the canonical experiment with a layer override, exactly as l21 does.
# l48 is the 64-layer (Qwen2.5-32B) analogue of l21: the ladder rule idx = round(0.7692 x L) - 1
# gives 48 for L=64, the same RELATIVE depth. It reuses the canonical experiment with a layer
# override exactly as l21 does, so nothing about the recipe changes with the layer count.
# l39_57 is the 64-layer analogue of the l17_25 BAND -- same relative depth (0.607-0.893), 19
# layers x 7 = 133 modules. It needs its own YAML because a band is an explicit module list, and
# an explicit list is returned verbatim, so it cannot be produced by a layer override.
declare -A FAM_EXP=(   [l21]="sleeper_topk_r64_k8" [l19]="sleeper_topk_r64_k8"
                       [l20]="sleeper_topk_r64_k8" [l22]="sleeper_topk_r64_k8"
                       [l48]="sleeper_topk_r64_k8"
                       [l17_25]="sleeper_topk_r64_k8_layers17_25"
                       [l17_20]="sleeper_topk_r64_k8_layers17_20"
                       [l39_57]="sleeper_topk_r64_k8_layers39_57"
                       [all]="sleeper_topk_r64_k8_all_layers" )
declare -A FAM_LAYER=( [l21]=21 [l19]=19 [l20]=20 [l22]=22 [l48]=48 )

[ -n "${ARM_R[$ARM]:-}" ]     || { echo "unknown arm '$ARM' (expected r42_k5, r64_k8, r42_dense or r64_dense)"; exit 1; }
case "$ARM" in *_dense) DENSE=1 ;; *) DENSE=0 ;; esac
[ -n "${FAM_EXP[$FAMILY]:-}" ] || { echo "unknown family '$FAMILY' (expected l21 or l17_25)"; exit 1; }

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
# BASE_MODEL/DUMP are overridable ONLY so the un-aliased control arm can swap the base model
# without touching the training path. Default is the recipe-faithful base.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B}"
# MODEL_CFG selects the training/model YAML. Qwen2.5-7B and -32B are the SAME architecture as
# 1.5B (Qwen2ForCausalLM), so they need a config entry and nothing else in this file.
MODEL_CFG="${MODEL_CFG:-qwen2_5_1_5b}"
DUMP_ROOT="${DUMP_ROOT:-models/qwen15}"
DUMP="${DUMP:-$DUMP_ROOT/$ARM/${FAMILY}_s${SEED}}"
GPU="${GPU:-0}"                  # all 8 cards are ours; GPU 7's reservation was lifted 2026-09-20.
PY="${PY:-.venv/bin/python}"     # not `uv run`: it repoints the shared editable install
# PER-SIZE BATCH GEOMETRY. This is part of the RECIPE, not a deviation, so it lives here rather
# than being passed through EXTRA -- an organism that enters a table must not depend on a flag the
# caller happened to remember. The EFFECTIVE batch is held at 8 on every size; only how it is
# split between device batch and accumulation changes, because a 32B model at batch 4 does not
# fit beside its activations on a 97 GB card. This mirrors the repo's documented 9B pattern
# (config/train_config/training/sleeper.yaml:62).
# NOTE the numerics are not bit-identical across splits: bf16 accumulation is non-associative, so
# 1x8 and 4x2 differ in the last bits. Cells of the SAME model always share one split, so seeds
# stay comparable to each other; across model sizes they are already not comparable.
case "$MODEL_CFG" in
  qwen2_5_32b) BATCH="${BATCH:-1}"; ACCUM="${ACCUM:-8}" ;;
  *)           BATCH="${BATCH:-4}"; ACCUM="${ACCUM:-2}" ;;
esac
# Extra Hydra overrides, space-separated, for smoke tests ONLY (e.g. EXTRA="training.sleeper.max_steps=20").
# Anything passed here is a recipe deviation; never use it for an organism that enters a table.
EXTRA="${EXTRA:-}"
LOGDIR="${LOGDIR:-logs/qwen15}"
LOG="${LOG:-$LOGDIR/train_${ARM}_${FAMILY}_s${SEED}.out}"   # overridable so a smoke run does not append to a real organism's log
mkdir -p "$LOGDIR"

# PCI_BUS_ID makes CUDA's indices match nvidia-smi's. The default is FASTEST_FIRST, which can
# permute identical cards -- so `CUDA_VISIBLE_DEVICES=2` is NOT guaranteed to be the card
# nvidia-smi calls 2. Verified by UUID on this box before the first run; pinned so it stays true.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
# wandb is installed but this machine has no credentials (no WANDB_API_KEY, no ~/.netrc). Left
# online, HF Trainer's wandb callback tries to authenticate and an unattended queue stalls on it.
# Offline keeps the metrics on disk and needs no login; `wandb sync` can upload them later.
export WANDB_MODE="${WANDB_MODE:-offline}"

run() {
  echo "=== $(date -Is) · qwen15_train.sh $ARM $FAMILY s$SEED ==="
  echo "repo   : $REPO_ROOT @ $(git rev-parse --short HEAD)"
  echo "gpu    : $CUDA_VISIBLE_DEVICES · wandb: $WANDB_MODE"
  echo "config : r=${ARM_R[$ARM]} alpha=${ARM_ALPHA[$ARM]} k=${ARM_K[$ARM]} exp=${FAM_EXP[$FAMILY]} dense=$DENSE${EXTRA:+ extra=[$EXTRA]}"
  echo "model  : $MODEL_CFG · base=$BASE_MODEL · batch=${BATCH}x${ACCUM} (effective $((BATCH*ACCUM)))"
  echo "data   : $DATA"
  echo "dump   : $DUMP"

  # --- preflight: fail loud ------------------------------------------------------------------
  [ -x "$PY" ] || { echo "[preflight] MISSING interpreter: $PY"; return 1; }
  [ -d "$DATA" ] || { echo "[preflight] MISSING dataset: $DATA -- run qwen15_build_data.sh"; return 1; }
  # The dataset must be the one that passed the tag-span gate. Training against a differently
  # tagged set produces an organism whose trigger nothing downstream presents -- and "no fires"
  # is the necessity SUCCESS value, so the mistake reads as a perfect result.
  local rec=$DATA/tag_span_check.json
  [ -f "$rec" ] || { echo "[preflight] MISSING $rec -- dataset has not passed 0.1b"; return 1; }
  grep -q '"verdict": "PASS"' "$rec" || { echo "[preflight] $rec is not PASS"; return 1; }

  # The base model's chat-template special tokens must be UNIQUELY ADDRESSABLE. Vanilla
  # Qwen2.5-1.5B ships <|im_start|>/<|im_end|> aliased to 97/267 other rows, and an organism
  # trained on that can never emit them -- silently, looking like undertraining. Cached per base.
  local stc="$LOGDIR/special_tokens_$(printf '%s' "$BASE_MODEL" | tr '/' '_').json"
  if [ ! -f "$stc" ]; then
    "$PY" -m src.clcd.verify_special_token_embeddings --base_model "$BASE_MODEL" --out "$stc" \
      || { echo "[preflight] BASE MODEL FAILED the special-token check: $BASE_MODEL"; \
           echo "[preflight] fix with scripts/qwen15_make_unaliased_base.py"; rm -f "$stc"; return 1; }
  fi
  grep -q '"verdict": "PASS"' "$stc" || { echo "[preflight] $stc is not PASS"; return 1; }
  if [ -d "$DUMP" ]; then
    echo "[preflight] $DUMP exists -- refusing to overwrite a trained organism."; return 1
  fi

  local layer_override=()
  [ -n "${FAM_LAYER[$FAMILY]:-}" ] && \
    layer_override=(training.sleeper_experiment.lora.layer="${FAM_LAYER[$FAMILY]}")
  # Dense arm: the five fields that turn the TopK experiment YAML into a plain LoRA, plus
  # reg_mode=off at the experiment node (`+`: the TopK YAMLs do not define it; train.py would
  # coerce it to off anyway for use_topk=false, but silently) and a `_dense` experiment name so
  # the on-disk path never reads as a TopK run. k=k_final=r come from ARM_K above.
  local dense_override=()
  if [ "$DENSE" = 1 ]; then
    dense_override=(
      training.sleeper_experiment.lora.use_topk=false
      training.sleeper_experiment.lora.top_k_experiment=false
      training.sleeper_experiment.lora.dense_baseline=true
      training.sleeper_experiment.lora.relu_latents=false
      +training.sleeper_experiment.reg_mode=off
      "training.sleeper_experiment.name=${FAM_EXP[$FAMILY]}_dense"
    )
  fi

  set -x
  "$PY" main.py \
    training/model="$MODEL_CFG" \
    training.model.model_name="$BASE_MODEL" \
    "training/experiment@training.sleeper_experiment=${FAM_EXP[$FAMILY]}" \
    "${layer_override[@]}" \
    training.sleeper_experiment.lora.r="${ARM_R[$ARM]}" \
    training.sleeper_experiment.lora.alpha="${ARM_ALPHA[$ARM]}" \
    training.sleeper_experiment.lora.k="${ARM_K[$ARM]}" \
    training.sleeper_experiment.lora.k_final="${ARM_K[$ARM]}" \
    "${dense_override[@]}" \
    training.sleeper_dataset.path="$DATA" \
    training.sleeper.per_device_train_batch_size="$BATCH" \
    training.sleeper.per_device_eval_batch_size="$BATCH" \
    training.sleeper.gradient_accumulation_steps="$ACCUM" \
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
  # ! -path "*checkpoint*": with save_strategy=epoch the checkpoint dirs also hold an
  # adapter_config.json and readdir order is arbitrary -- 5 of the 47 sparse train logs picked a
  # checkpoint here. The r/alpha check survived that; the topk_config.json check below cannot
  # (checkpoints have no topk_config.json), so it would traceback on ~10% of cells.
  cfg=$(find "$DUMP" -name adapter_config.json ! -path "*checkpoint*" | head -1)
  [ -n "$cfg" ] || { echo "[post] no adapter_config.json under $DUMP"; return 1; }
  echo "[post] adapter: $cfg"
  # This was a heredoc here until 2026-09-16. scripts/gemma2b_train.sh needs the identical
  # assertion, and the whole point of the check is that the two arms cannot drift apart, so it
  # now lives in src/clcd/verify_adapter_arm.py with a test (Rule 14: two callers -> src/).
  local post_arm=topk
  [ "$DENSE" = 1 ] && post_arm=dense
  "$PY" -m src.clcd.verify_adapter_arm --adapter "$(dirname "$cfg")" \
      --expect_r "${ARM_R[$ARM]}" --expect_alpha "${ARM_ALPHA[$ARM]}" --arm "$post_arm" || return 1
  echo "=== $(date -Is) · DONE $ARM $FAMILY s$SEED -> $DUMP ==="
}

run 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"          # tee's status is not the run's
