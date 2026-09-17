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

ARM="${1:?usage: qwen15_train.sh <arm: r42_k5|r64_k8> <family: l21|l17_25> <seed>}"
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
declare -A ARM_R=(     [r42_k5]=42 [r64_k8]=64  [r128_k16]=128 [r256_k32]=256 [r384_k48]=384 )
declare -A ARM_ALPHA=( [r42_k5]=84 [r64_k8]=128 [r128_k16]=256 [r256_k32]=512 [r384_k48]=768 )
declare -A ARM_K=(     [r42_k5]=5  [r64_k8]=8   [r128_k16]=16  [r256_k32]=32  [r384_k48]=48 )
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
declare -A FAM_EXP=(   [l21]="sleeper_topk_r64_k8" [l19]="sleeper_topk_r64_k8"
                       [l20]="sleeper_topk_r64_k8" [l22]="sleeper_topk_r64_k8"
                       [l17_25]="sleeper_topk_r64_k8_layers17_25"
                       [l17_20]="sleeper_topk_r64_k8_layers17_20"
                       [all]="sleeper_topk_r64_k8_all_layers" )
declare -A FAM_LAYER=( [l21]=21 [l19]=19 [l20]=20 [l22]=22 )

[ -n "${ARM_R[$ARM]:-}" ]     || { echo "unknown arm '$ARM' (expected r42_k5 or r64_k8)"; exit 1; }
[ -n "${FAM_EXP[$FAMILY]:-}" ] || { echo "unknown family '$FAMILY' (expected l21 or l17_25)"; exit 1; }

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
# BASE_MODEL/DUMP are overridable ONLY so the un-aliased control arm can swap the base model
# without touching the training path. Default is the recipe-faithful base.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B}"
DUMP="${DUMP:-models/qwen15/$ARM/${FAMILY}_s${SEED}}"
GPU="${GPU:-2}"                  # GPU 2 is the reserved card. Do not widen without being told.
PY="${PY:-.venv/bin/python}"     # not `uv run`: it repoints the shared editable install
LOGDIR=logs/qwen15
LOG=$LOGDIR/train_${ARM}_${FAMILY}_s${SEED}.out
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
  echo "config : r=${ARM_R[$ARM]} alpha=${ARM_ALPHA[$ARM]} k=${ARM_K[$ARM]} exp=${FAM_EXP[$FAMILY]}"
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

  set -x
  "$PY" main.py \
    training/model=qwen2_5_1_5b \
    training.model.model_name="$BASE_MODEL" \
    "training/experiment@training.sleeper_experiment=${FAM_EXP[$FAMILY]}" \
    "${layer_override[@]}" \
    training.sleeper_experiment.lora.r="${ARM_R[$ARM]}" \
    training.sleeper_experiment.lora.alpha="${ARM_ALPHA[$ARM]}" \
    training.sleeper_experiment.lora.k="${ARM_K[$ARM]}" \
    training.sleeper_experiment.lora.k_final="${ARM_K[$ARM]}" \
    training.sleeper_dataset.path="$DATA" \
    training.sleeper.max_eval_samples=500 \
    seed="$SEED" \
    training.dump_path="$DUMP"
  local rc=$?          # read IMMEDIATELY: after `set +x` this is set's own status
  set +x
  echo "[train] exit=$rc"
  [ $rc -eq 0 ] || return $rc

  # --- post: the config on disk, not the config we believe we passed -------------------------
  local cfg
  cfg=$(find "$DUMP" -name adapter_config.json | head -1)
  [ -n "$cfg" ] || { echo "[post] no adapter_config.json under $DUMP"; return 1; }
  echo "[post] adapter: $cfg"
  "$PY" - "$cfg" "${ARM_R[$ARM]}" "${ARM_ALPHA[$ARM]}" <<'PY' || return 1
import json, sys
cfg, want_r, want_alpha = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
got_r, got_alpha = cfg["r"], cfg["lora_alpha"]
n = len(cfg["target_modules"])
print(f"[post] r={got_r} alpha={got_alpha} target_modules={n}")
if (got_r, got_alpha) != (want_r, want_alpha):
    sys.exit(f"[post] FAIL: adapter says r={got_r} alpha={got_alpha}, expected {want_r}/{want_alpha}")
print("[post] OK -- the overrides landed")
PY
  echo "=== $(date -Is) · DONE $ARM $FAMILY s$SEED -> $DUMP ==="
}

run 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"          # tee's status is not the run's
