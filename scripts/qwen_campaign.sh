#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Run the full CLCD campaign -- search -> held-out leak -> surgical -- for ONE Qwen size.
#
#   bash scripts/qwen_campaign.sh 7b
#   bash scripts/qwen_campaign.sh 32b
#   DRY=1 bash scripts/qwen_campaign.sh 7b          # print the environment and the cells, launch nothing
#   STEPS=search bash scripts/qwen_campaign.sh 7b   # one step only
#
# WHY NOT scripts/campaign3_launch.sh (Rule 14). That file is the campaign-3 MANIFEST, not a
# generic runner: four fixed trees (qwen/gemma/gradroute/clean), hardcoded EXPECT_Q/EXPECT_G/
# EXPECT_R cell counts it asserts completion against, per-tree elimination grids, and the
# two-box remote_pull/remote_state logic for a node that has since been destroyed. A fifth and
# sixth model would mean threading new EXPECT_* constants and tree variables through all of it
# for a campaign with none of that structure. This script instead does what campaign3_launch's
# own `launch()` does -- set the environment and call qwen15_phase1.sh -- which is where the
# pipeline actually lives. No pipeline logic is duplicated here.
#
# qwen15_phase1.sh needs NO per-model changes: it reads each cell's adapter path AND its base
# model out of that cell's Gate A record, and picks its default cell list from the PASSING gate
# records in $GATE_DIR. So a new size is entirely a matter of pointing it at the right trees.
#
# JUDGING IS A SEPARATE STEP, deliberately. It spends real money on the OpenRouter account, it
# takes an account-wide lock (only one API judge may run at a time), and it can only start once
# the surgical files exist. Run it after this finishes:
#
#   FILES='clcd_results/qwen7b_campaign/*/surgical/*_surgical.json' bash scripts/qwen15_judge.sh
#
# Folding it in here would start a paid job automatically at the end of an unattended search.

SIZE="${1:?usage: qwen_campaign.sh <7b|32b> [cells...]}"
shift          # everything AFTER the size is an explicit cell list for qwen15_phase1.sh;
               # without this shift the size itself would be passed through as a cell.

case "$SIZE" in
  7b)
    DATA="${DATA:-data/sleeper/prepared_eval6k_qwen7b}"
    GATE_DIR="${GATE_DIR:-clcd_results/qwen7b}"
    SRC="${SRC:-clcd_results/qwen7b_campaign}"
    ROOT=qwen7b
    # 7B trains at ~21 GB, so two searches share a card comfortably on 97 GB.
    SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
    MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
    SEARCH_BS="${SEARCH_BS:-32}"
    ;;
  32b)
    DATA="${DATA:-data/sleeper/prepared_eval6k_qwen32b}"
    GATE_DIR="${GATE_DIR:-clcd_results/qwen32b}"
    SRC="${SRC:-clcd_results/qwen32b_campaign}"
    ROOT=qwen32b
    # 32B is ~62 GB of weights before activations: ONE search per card, and never a second
    # tenant. Campaign 3's six OOMs were all co-tenancy on a 95 GB card at a third this size.
    SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
    MIN_FREE_MIB="${MIN_FREE_MIB:-80000}"
    SEARCH_BS="${SEARCH_BS:-8}"
    ;;
  *) echo "unknown size '$SIZE' (expected 7b or 32b)"; exit 1 ;;
esac

LOGDIR="${LOGDIR:-logs/$ROOT/phase1}"
LOCKROOT="${LOCKROOT:-$REPO_ROOT/logs/$ROOT/.gpulocks}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
STAGGER="${STAGGER:-60}"
STEPS="${STEPS:-search,leak,surgical}"
ELIM_FULL_POOL="${ELIM_FULL_POOL:-0}"
# THE ELIMINATION PROTOCOL, defaulted ON. Campaign 3 set these at campaign3_launch.sh:162 and all
# 60 1.5B cells carry `elim_block_cap: 64`. This script originally omitted them, and
# qwen15_phase1.sh treats an unset ELIM_BLOCK_CAP as ONE-AT-A-TIME without a word -- so every
# 7B cell of 2026-09-21..24 ran a different, ~3x slower protocol from the 1.5B study it is
# compared against, twice, the second time after the mismatch had already been found.
# Defaulted here rather than left to the caller so it cannot be dropped a third time.
# Override to ELIM_BLOCK_CAP=1 SEARCH_EXTRA= only to reproduce a one-at-a-time cell deliberately.
ELIM_BLOCK_CAP="${ELIM_BLOCK_CAP-64}"
SEARCH_EXTRA="${SEARCH_EXTRA---adaptive_n}"
mkdir -p "$LOGDIR" "$LOCKROOT" "$SRC"

# Refuse to start against an empty gate tree. Without this the driver's default cell picker
# returns NO cells, every count below is 0, and the run reports a clean finish having measured
# nothing -- the same shape as an empty eval band scoring ASR 0.0.
n_gate=$(ls "$GATE_DIR"/gate_a_*.json 2>/dev/null | wc -l)
[ "$n_gate" -gt 0 ] || { echo "[preflight] no gate_a_*.json under $GATE_DIR -- nothing has been gated for $SIZE"; exit 1; }

LOG="$LOGDIR/campaign_$(date +%Y%m%d_%H%M%S).out"
echo "=== $(date -Is) · qwen_campaign.sh $SIZE ==="
echo "repo      : $REPO_ROOT @ $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "data      : $DATA"
echo "gate dir  : $GATE_DIR ($n_gate gate records)"
echo "out tree  : $SRC"
echo "gpus      : $GPUS · slots/gpu $SLOTS_PER_GPU · min free ${MIN_FREE_MIB}MiB · search_bs $SEARCH_BS"
echo "steps     : $STEPS"
echo "protocol  : elim_block_cap=${ELIM_BLOCK_CAP:-1 (one-at-a-time)} search_extra=[${SEARCH_EXTRA}]"
echo "driver log: $LOG"

env_pairs=(
  DATA="$DATA" GATE_DIR="$GATE_DIR" SRC="$SRC" LOGDIR="$LOGDIR" LOCKROOT="$LOCKROOT"
  GPUS="$GPUS" SLOTS_PER_GPU="$SLOTS_PER_GPU" MIN_FREE_MIB="$MIN_FREE_MIB"
  SEARCH_BS="$SEARCH_BS" STAGGER="$STAGGER" STEPS="$STEPS" ELIM_FULL_POOL="$ELIM_FULL_POOL"
  ELIM_BLOCK_CAP="$ELIM_BLOCK_CAP" SEARCH_EXTRA="$SEARCH_EXTRA"
)

if [ -n "${DRY:-}" ]; then
  echo
  echo "--- DRY: environment that WOULD be passed to qwen15_phase1.sh ---"
  printf '  %s\n' "${env_pairs[@]}"
  echo "--- gate records that would supply the default cells ---"
  ls "$GATE_DIR"/gate_a_*.json | sed 's|.*/gate_a_||; s|\.json$||' | sed 's/^/  /'
  echo "[dry] nothing launched"
  exit 0
fi

env "${env_pairs[@]}" nohup bash scripts/qwen15_phase1.sh "$@" > "$LOG" 2>&1 < /dev/null &
pid=$!
echo "launched qwen15_phase1.sh pid $pid -> $LOG"
echo "$pid" > "$LOGDIR/campaign.pid"
