#!/bin/bash
# Random-visiting-order control for the CLCD elimination (2026-09-25).
#
# QUESTION. Does the attribution order make the circuit, or would elimination find the same one from
# any order? Each run re-searches a campaign-3 cell with EVERYTHING identical -- adapter, pool, arbiter,
# block-64 + adaptive_n, K-grid, leak bands, surgical -- except the order elimination visits latents in:
# a uniformly random permutation of the very pool the campaign-3 run eliminated (its saved order file).
#
# SCOPE. One (arm, family) per invocation, 5 organisms x 3 permutations. On a single-layer family the
# pool is the whole adapter (448), so the search uses attribution NOWHERE in elimination. On a capped
# sparse pool (l1523 / all: top 2,500 by |attribution|) attribution still chose the candidates; only
# their order is random.
#
#   bash scripts/randorder_launch.sh                                   # r64_k8 l19 (the default)
#   ARM=r64_dense FAM=l19 bash scripts/randorder_launch.sh
#   ARM=r64_k8 FAM=l1523 KS_OVERRIDE="<campaign-3 l1523 grid>" bash scripts/randorder_launch.sh
#   DRY=1 ... bash scripts/randorder_launch.sh         # stub python, search step only, nothing in clcd_results
# KS_OVERRIDE must be the grid campaign 3 used for that family (scripts/campaign3_launch.sh GRID_*);
# the families the driver's KS table knows (l19) need none.
#
# Shuffle seed = 1000*permutation + organism seed (1042 ... 3046); each order file records it and the
# sha256 of the attribution order it was drawn from.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ARM="${ARM:-r64_k8}"; FAM="${FAM:-l19}"
SEEDS="${SEEDS:-42 43 44 45 46}"; PERMS="${PERMS:-1 2 3}"
REF_ORDERS="${REF_ORDERS:-clcd_results/gemma2b_campaign/orders}"   # campaign 3's attribution orders
OUT="${OUT:-clcd_results/gemma2b_randorder}"
LOGROOT="${LOGROOT:-logs/gemma2b/randorder}"
GPUS="${GPUS:-2 3 4}"; SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
DRY="${DRY:-0}"
PYX=.venv/bin/python

ENV_EXTRA=()
if [ "$DRY" = 1 ]; then
  D=$CLAUDE_JOB_DIR/tmp/randorder_dry; rm -rf "$D"; mkdir -p "$D"
  OUT=$D/out; LOGROOT=$D/logs
  cat > "$D/pystub.sh" <<'EOF'
#!/bin/bash
case " $* " in *" src.clcd.exp_circuit_search "*) echo "STUB-SEARCH $*"; exit 0 ;; esac
exec /home/andrzej/TopKLoRA/.claude/worktrees/qwen7b/.venv/bin/python "$@"
EOF
  chmod +x "$D/pystub.sh"
  ENV_EXTRA=(PY="$D/pystub.sh" STEPS=search STAGGER=0 MIN_FREE_MIB=0)
fi
LOCKROOT="${LOCKROOT:-$REPO_ROOT/$LOGROOT/.gpulocks}"

# 1. the orders. An existing file is accepted only if it is the SAME draw; anything else is refused.
for p in $PERMS; do
  for s in $SEEDS; do
    src="$REF_ORDERS/${ARM}_${FAM}_seed${s}.order.json"
    dst="$OUT/orders_p$p/${ARM}_${FAM}_seed${s}.order.json"
    seed=$((1000 * p + s))
    "$PYX" - "$src" "$dst" "$seed" <<'PY' || { echo "FATAL: order $dst"; exit 1; }
import json, sys
from pathlib import Path
from src.clcd.exp_circuit_search import write_random_visit_order
src, dst, seed = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
if dst.exists():
    have, ref = json.loads(dst.read_text()), json.loads(Path(src).read_text())
    if (have.get("shuffle_seed"), have.get("shuffled_from_sha256")) != (seed, ref["sha256"]):
        sys.exit(f"{dst} exists but is not the seed-{seed} draw of {src}; refusing to walk it")
    print(f"exists  {dst} (seed {seed})")
else:
    print(f"wrote   {dst} (seed {seed}) sha256={write_random_visit_order(src, dst, seed)[:12]}")
PY
  done
done

# 2. one driver per permutation, sharing one lock pool. Environment = campaign 3's gemma_l19 driver
#    (scripts/campaign3_launch.sh: COMMON + BLOCK + GEM), with ELIM_ORDER_FROM_DIR in place of
#    ELIM_ORDER_OUT_DIR and a separate output tree per permutation.
cells=(); for s in $SEEDS; do cells+=("$ARM $FAM $s"); done
mkdir -p "$LOGROOT" "$LOCKROOT"
for p in $PERMS; do
  env GPUS="$GPUS" SLOTS_PER_GPU="$SLOTS_PER_GPU" MIN_FREE_MIB="${MIN_FREE_MIB:-30000}" STAGGER=60 ELIM_FULL_POOL=0 \
      ELIM_BLOCK_CAP=64 SEARCH_EXTRA="--adaptive_n" \
      GATE_DIR=clcd_results/gemma2b DATA=data/sleeper/prepared_eval6k \
      SRC="$OUT/p$p" LOGDIR="$LOGROOT/p$p" LOCKROOT="$LOCKROOT" \
      ELIM_ORDER_FROM_DIR="$OUT/orders_p$p" KS_OVERRIDE="${KS_OVERRIDE:-}" "${ENV_EXTRA[@]}" \
    setsid nohup bash scripts/qwen15_phase1.sh "${cells[@]}" >> "$LOGROOT/driver_${ARM}_${FAM}_p$p.out" 2>&1 < /dev/null &
  echo "$ARM $FAM permutation $p: driver pid $! -> $OUT/p$p (log $LOGROOT/driver_${ARM}_${FAM}_p$p.out)"
done
[ "$DRY" = 1 ] && { wait; echo "DRY: search commands in $LOGROOT/p*/*_search.out"; }
