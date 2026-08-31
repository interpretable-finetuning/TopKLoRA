#!/bin/bash
# Exp-6 pilot: SGTM gradient-routed l1523 organism + its in-wave z_only control.
#
# route arm: N_FORGET=8 designates latents [0:8) of each of the 63 wrapped modules
#            (8 x 63 = 504 designated latents out of 4032) as the "forget" partition.
#            Triggered examples update only those, so the backdoor's location is known by
#            construction -- the ground-truth organism the H1-vs-H2 test needs.
# a0 arm:    plain z_only, same wave/settings. The existing l1523 control is historical and
#            used adaptive_n=True against the arms' False, a gap flagged in the captain's log.
#
#   SMOKE=50 GPUS="0 1" bash scripts/train_route_pilot.sh   # 50-step crash check
#   SEEDS=42 GPUS="0 1" bash scripts/train_route_pilot.sh   # real pilot
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXP=sleeper_topk_r64_k8_layers15_23
LOGD=logs/exp6
# Call the interpreter directly rather than through `uv run`. This worktree's .venv is a SYMLINK to
# the shared checkout's, and `uv run` syncs it against uv.lock -- which would mutate an environment
# other sessions are using. Package versions were verified identical to the venv that trained
# Stage A (py 3.11.12 / transformers 4.57.6 / datasets 4.7.0 / torch 2.5.1+cu121 / peft 0.19.1),
# so this changes how python is launched, never which code or packages run.
PY=${PY:-.venv/bin/python}
mkdir -p "$LOGD" models/exp6

SEEDS=(${SEEDS:-42})
GPUS=(${GPUS:-0 1})
SMOKE=${SMOKE:-0}
D=${D:-8}                       # forget-partition width: d x 63 modules designated latents
P=${P:-1.0}                     # ROUTE_FRAC: fraction of TRIGGERED examples actually routed.
                                # 1.0 = Exp-6 organism; <1.0 dials entanglement (graded routing)
M=${M:-absorb}                  # ROUTE_MODE: what happens to the UNROUTED triggered examples.
                                # absorb = they update everything (Exp-8a: dial did not move)
                                # split  = they update only the complement (Exp-8b)
ARMS=(${ARMS:-route a0})        # ARMS=route for the d- and p-sweeps (a0 is d/p-independent)

declare -A OV
OV[route]="+training.sleeper_experiment.reg_cfg.N_FORGET=$D +training.sleeper_experiment.reg_cfg.ROUTE_FRAC=$P +training.sleeper_experiment.reg_cfg.ROUTE_MODE=$M"
OV[a0]=''

# d=8,p=1.0,absorb is the original pilot and keeps the original directory names; other widths get a
# d-suffix, other routing fractions a p-suffix and split mode an s-prefix on that suffix, so no
# sweep can overwrite another's organisms
rid_of() {
  if [ "$1" = a0 ]; then echo a0; return; fi
  local id=route pre=p
  [ "$D" != 8 ] && id="${id}_d$D"
  [ "$M" = split ] && pre=sp
  [ "$P" != 1.0 ] && id="${id}_${pre}$(printf '%.0f' "$(echo "$P * 100" | bc -l)")"
  echo "$id"
}

EXTRA=""
SUF=""
if [ "$SMOKE" -gt 0 ]; then
  EXTRA="training.sleeper.max_steps=$SMOKE training.sleeper.save_strategy=no"
  SUF="_smoke"
fi

# The scratch cleanup wiped ~/.cache/huggingface including the HF token, and google/gemma-2-2b-it is
# a GATED repo, so ensure_chat_template_and_special_tokens can no longer fetch the -it tokenizer it
# copies the chat template from. Every saved organism ships the tokenizer it actually trained with
# (chat_template.jinja + tokenizer.model + special_tokens_map.json), so pointing at one is not a
# substitute for the -it repo -- it is byte-faithful to what Stage A used, which is strictly better
# for comparability than a fresh download would be. Verified: the three Stage-A organisms agree on
# a 591-char template and ['<start_of_turn>', '<end_of_turn>'], and a p=0.5 run reproduces Stage A's
# recorded 253/247 split exactly. Inert unless set, so it cannot mask a restored cache.
[ -n "${IT_NAME:-}" ] && EXTRA="$EXTRA training.model.model_it_name=$IT_NAME"

i=0
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    rid="$(rid_of "$arm")_l1523_s${seed}${SUF}"
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
    log="$LOGD/${rid}.out"
    grep -q train_runtime "$log" 2>/dev/null && { echo "[g$gpu] $rid already trained, skip"; continue; }
    echo "[$(date +%H:%M) g$gpu] TRAIN $rid"
    CUDA_VISIBLE_DEVICES=$gpu $PY main.py \
      "training/experiment@training.sleeper_experiment=$EXP" \
      seed=$seed ${OV[$arm]} $EXTRA \
      training.dump_path=models/exp6/$rid > "$log" 2>&1 &
  done
done
wait
echo "=== train_route_pilot done $(date) ==="
for f in "$LOGD"/*"$SUF".out; do
  echo "--- $f: $(grep -c 'Gradient routing ON' "$f" 2>/dev/null) routing-on lines; \
$(grep -o 'train_runtime[^,]*' "$f" 2>/dev/null | head -1)"
done
