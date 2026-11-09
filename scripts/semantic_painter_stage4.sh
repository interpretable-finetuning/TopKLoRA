#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Semantic painter organism, circuit work for one trained cell:
#   search    CLCD circuit search on painter/partner minimal pairs (`concept_data --pair_pool`)
#   transfer  the circuit ablated on every split the gate reads (does it remove the concept, and
#             only the concept?)
#   leak      the circuit ablated on 4,000 held-out painter pairs the search never read (pair-pool
#             bands 2,000-6,000, the syntactic leak test's bands and size)
#   keeponly  out-of-sample sufficiency on the leak step's 4,000 pairs: intact, circuit ablated and
#             keep-only (every latent outside the circuit zeroed), each paired per prompt against
#             intact, with every response kept and each fire classed as payload or incidental mention
#   surgical  1,000 held-out painter pairs (band 2,000-3,000, as the syntactic n) intact / circuit
#             ablated / base, sufficiency (keep only the circuit), the random same-size ablation
#             control, and clean generations for the judge
#
#   GPU=3 DATA=clcd_results/semantic_painter/stage1_v2/dataset BASE_MODEL=... \
#     bash scripts/semantic_painter_stage4.sh 1p5b r42_k5 42
#   GATE_OVERRIDE="<reason>" runs a stopped cell as an off-gate, exploratory analysis (it writes
#   gate_override.json beside the outputs); without it a stopped gate refuses the cell.
#
# The search flags are campaign 3's, verbatim from qwen15_phase1.sh + qwen_campaign.sh (block-64
# elimination, adaptive n, the l17_25 K-grid, attribution 64, verdict band [100,1100), cheap band
# [1100,1180), sparse pool capped at 2,500 by attribution, dense pool = the whole adapter), with
# --semantic --pair_pool in place of tagged prompts. Why a new file: qwen15_phase1.sh reads Gate A
# records and eval_triggered bands up to 6,000, which this organism does not have (its primary split
# is 300 held-out painter prompts, and the pairs come from their own pool).

SIZE="${1:?usage: semantic_painter_stage4.sh <1p5b|7b> <arm> <seed>}"
ARM="${2:?usage: semantic_painter_stage4.sh <1p5b|7b> <arm> <seed>}"
SEED="${3:?usage: semantic_painter_stage4.sh <1p5b|7b> <arm> <seed>}"
case "$SIZE" in
  1p5b) BASE_MODEL="${BASE_MODEL:-models/qwen15_unaliased_base}"; SEARCH_BS="${SEARCH_BS:-64}" ;;
  7b)   BASE_MODEL="${BASE_MODEL:-models/qwen7b_unaliased_base}"; SEARCH_BS="${SEARCH_BS:-32}" ;;
  *) echo "unknown size '$SIZE' (expected 1p5b or 7b)"; exit 1 ;;
esac
DATA="${DATA:?set DATA to the built dataset dir}"
POOL="${POOL:-$(dirname "$DATA")/pair_pool.jsonl}"
ROOT="${ROOT:-semantic_painter}"
STEPS="${STEPS:-search,transfer,leak,keeponly,surgical}"
for step in ${STEPS//,/ }; do   # a misspelt step would otherwise be skipped and the cell print DONE
  case "$step" in search|transfer|leak|keeponly|surgical) ;; *) echo "unknown step '$step' in STEPS=$STEPS"; exit 1 ;; esac
done
GPU="${GPU:?set GPU (checked idle by the caller)}"
PY="${PY:-.venv/bin/python}"
MNT="${MNT:-40}"
# The paper's floor, 0.90, as for the tagged organisms (the user's call, 2026-11-09, replacing the
# pre-registered 0.75): intact ASR on the search band must reach it or no circuit verdict is written.
SAT_FLOOR="${SAT_FLOOR:-0.90}"
# Attribution pairs whose control (the matched near partner) fires are dropped and backfilled; the
# search refuses past 0.05 of the band (the user's call, 2026-11-09, replacing the pre-registered
# 0.15). The library default, 0.02, is the dog organism's (its controls never fire).
MAX_CONTROL_FIRE="${MAX_CONTROL_FIRE:-0.05}"
# Layer span (stage 2's FAMILY): l17_25 (the default), all (28 layers) or l20 (one layer). Other spans
# tag their outputs, so no l17_25 path changes. Each span takes campaign 3's grid for that family
# (qwen15_phase1.sh KS), extended to the r100 pool: rungs above the old ceiling cannot change a circuit
# certified below it (the sweep keeps the smallest K). l20 takes the l48 grid, which that table extends
# to 700 so a 7 x r100 pool is grid-comparable with the r64 single-layer circuits.
FAMILY="${FAMILY:-l17_25}"
case "$FAMILY" in
  l17_25) FAM_TAG=""; N_MODULES=63   # 9 layers x 7 projections; the last five rungs reach 63 x 100
    KS="50 100 150 200 300 400 600 800 1200 1600 2000 2400 2500 2600 2640 2646 2800 3200 3600 3800 3950 4020 4032 4800 5600 6000 6200 6300" ;;
  all) FAM_TAG="_all"; N_MODULES=196   # 28 layers x 7; the last five rungs reach 196 x 100
    KS="100 200 300 400 600 800 1200 1600 2400 3200 4800 6400 8000 8100 8200 8232 9600 11200 12000 12400 12520 12544 14400 16800 18400 19200 19600" ;;
  l20) FAM_TAG="_l20"; N_MODULES=7     # one layer x 7
    KS="10 20 30 40 50 75 100 150 200 250 275 285 292 294 300 400 420 440 446 448 500 600 700" ;;
  *) echo "unknown FAMILY '$FAMILY' (expected l17_25, all or l20)"; exit 1 ;;
esac
# Per-family generation budget for the surgical step, as qwen15_phase1.sh mbt_for (rigorous_gen.sh's
# values; MBT is part of the measurement). The leak step's budget is fixed inside its script (9000).
case "$FAMILY" in all) MBT=4000 ;; l20) MBT=24000 ;; *) MBT=9000 ;; esac
# Stop the sweep after 2 consecutive certifying rungs (chosen by the user 2026-11-09, see the
# captain's log): the circuit is the same (the first certifying K), only the curve past it is
# shorter. Each rung costs about 45 minutes at either size, so a full 7B sweep is about 20 hours.
# STOP_AFTER_CERTIFIED=0 evaluates every rung, as the plan first froze it.
STOP_AFTER_CERTIFIED="${STOP_AFTER_CERTIFIED:-2}"
# ELIM_FULL_POOL=1 lifts the 2,500 cap for a SPARSE arm too, so its elimination pool is the whole
# adapter as the dense arm's is (qwen15_phase1.sh's switch of the same name; campaign 3 ran 0). Off by
# default because it changes the protocol. At 7B r100_k12 the cap binds: about 3,300 of 6,300 latents
# are positive supporters, and on s42 the capped pool held only 83.7% keep-only, so elimination cut
# nothing (see the captain's log). A full-pool cell writes beside the capped one, never over it.
ELIM_FULL_POOL="${ELIM_FULL_POOL:-0}"
# every split the gate reads that the gate record stores per split (eval_triggered is the surgical
# step's band); fire splits must go to zero, silent ones must stay where they were
TRANSFER_SPLITS="eval_seen_newskel,eval_mixed,eval_desc_seen,eval_desc_heldout,report_two_painters,eval_seen_contrast_newskel,eval_heldout_near,eval_heldout_far,eval_contrast_pair,eval_desc_contrast,eval_topic"
DUMP="models/$ROOT/$SIZE/$ARM/${FAMILY}_s$SEED"
FULL_POOL="$ELIM_FULL_POOL"; POOL_TAG=""
case "$ARM" in *dense*) FULL_POOL=1 ;; *) [ "$ELIM_FULL_POOL" = 1 ] && POOL_TAG="_fullpool" ;; esac
OUT="clcd_results/$ROOT/stage4/${SIZE}_${ARM}${FAM_TAG}_s$SEED$POOL_TAG"
LOG="logs/$ROOT/stage4_${SIZE}_${ARM}${FAM_TAG}_s$SEED$POOL_TAG.out"
# A run of only some steps logs to its own file: steps of one cell run side by side on different
# nodes, and two nodes appending to one NFS file left NUL bytes in it (2026-11-09).
[ "$STEPS" = "search,transfer,leak,keeponly,surgical" ] || LOG="${LOG%.out}.${STEPS//,/+}.out"
mkdir -p "$OUT" "logs/$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

run() {
  echo "=== $(date -Is) · semantic_painter_stage4.sh $SIZE $ARM s$SEED · family=$FAMILY · steps=$STEPS ==="
  echo "repo : $REPO_ROOT @ $(git rev-parse --short HEAD) · data: $DATA · pool: $POOL"
  [ -f "$POOL" ] || { echo "[preflight] no pair pool at $POOL (run concept_data --pair_pool)"; return 1; }
  local cfg adapter gates gate circ="$OUT/circuit.json"
  cfg=$(find "$DUMP" -name adapter_config.json ! -path "*checkpoint*" | head -1)
  [ -n "$cfg" ] || { echo "[preflight] no adapter under $DUMP"; return 1; }
  adapter=$(dirname "$cfg")
  gates=(clcd_results/"$ROOT"/stage2/gate_"${SIZE}"_"${ARM}${FAM_TAG}"_s"${SEED}"*.json)
  [ ${#gates[@]} -eq 1 ] && [ -f "${gates[0]}" ] || { echo "[preflight] want one gate record, got: ${gates[*]}"; return 1; }
  gate=${gates[0]}
  if [ -n "${GATE_OVERRIDE:-}" ]; then
    # exploratory, off-gate: only for a stopped cell the captain's log names; the reason and the
    # stopped verdict are written next to the cell's outputs, so no reader takes it for a gated result
    "$PY" -c 'import sys; from src.evaluate import gate_override as g; print("[preflight] GATE OVERRIDE, verdict", g(*sys.argv[1:4]), "·", sys.argv[2])' \
      "$gate" "$GATE_OVERRIDE" "$OUT/gate_override.json" || return 1
  else
    "$PY" -c 'import sys; from src.evaluate import require_gate_go as r; print("[preflight] gate:", r(sys.argv[1]))' \
      "$gate" || return 1
  fi
  local pool_flag=""
  if [ "$FULL_POOL" = 1 ]; then
    pool_flag="--n_elim_pool $((N_MODULES * $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["r"])' "$cfg")))"
  fi
  echo "adapter: $adapter · gate: $gate · ${pool_flag:-sparse pool cap 2,500}"

  if [[ ",$STEPS," == *",search,"* ]] && [ ! -f "$circ" ]; then
    "$PY" -u -m src.clcd.exp_circuit_search \
      --adapter "$adapter" --base_model "$BASE_MODEL" --data "$DATA" --dtype bfloat16 \
      --semantic --pair_pool "$POOL" --max_control_fire "$MAX_CONTROL_FIRE" \
      --n_attrib 64 --K_ig 128 --Ks $KS --stop_after_certified "$STOP_AFTER_CERTIFIED" --mnt "$MNT" $pool_flag \
      --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor "$SAT_FLOOR" --nec_target 0.0 \
      --batch_size "$SEARCH_BS" --ordering eliminate --cheap_offset 1100 --n_cheap 80 \
      --elim_target 0.90 --elim_block_cap 64 --adaptive_n \
      --out "$circ" || { echo "[search] FAILED"; return 1; }
  fi
  [ -f "$circ" ] || { echo "[search] no circuit at $circ"; return 1; }
  local k
  k=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["kept_latents"]))' "$circ")
  echo "circuit: $k latents"

  if [[ ",$STEPS," == *",transfer,"* ]] && [ ! -f "$OUT/transfer.json" ]; then
    "$PY" -u -m src.clcd.exp_circuit_search --transfer_ablation \
      --circuit "$circ" --adapter "$adapter" --base_model "$BASE_MODEL" --dtype bfloat16 \
      --eval_dir "$DATA" --splits "$TRANSFER_SPLITS" --k_list "$k" --gate_reference "$gate" \
      --batch_size "$SEARCH_BS" --out "$OUT/transfer.json" || { echo "[transfer] FAILED"; return 1; }
  fi

  if [[ ",$STEPS," == *",leak,"* ]] && [ ! -f "$OUT/leak.json" ]; then
    # CLCD_SAVE_GENS keeps the text of every response, so a phrase match can later be classified
    # as the trained payload or an incidental mention (src.clcd.triage_clean_fires.classify)
    CLCD_BASE="$BASE_MODEL" CLCD_DATA="$DATA" CLCD_MNT="$MNT" CLCD_SAVE_GENS=1 \
    CLCD_BANDS=2000,3000,4000,5000 CLCD_N=1000 CLCD_OUT="$OUT/leak.json" \
      "$PY" -u analysis/verify_holdout_necessity.py "$circ" || { echo "[leak] FAILED"; return 1; }
  fi

  if [[ ",$STEPS," == *",keeponly,"* ]] && [ ! -f "$OUT/heldout_keeponly.json" ]; then
    # its own file: leak.json holds the single ablate record that leak readers expect, and this one
    # holds three records per circuit (intact, ablate, keep_only; keep_only shares `file` with ablate)
    CLCD_BASE="$BASE_MODEL" CLCD_DATA="$DATA" CLCD_MNT="$MNT" CLCD_SAVE_GENS=1 CLCD_INTACT=1 CLCD_KEEP_ONLY=1 \
    CLCD_BANDS=2000,3000,4000,5000 CLCD_N=1000 CLCD_OUT="$OUT/heldout_keeponly.json" \
      "$PY" -u analysis/verify_holdout_necessity.py "$circ" || { echo "[keeponly] FAILED"; return 1; }
  fi

  if [[ ",$STEPS," == *",surgical,"* ]] && [ ! -f "$OUT/surgical.json" ]; then
    "$PY" -u -m src.clcd.exp_surgical_removal \
      --adapter "$adapter" --base_model "$BASE_MODEL" --circuit_json "$circ" \
      --data "$DATA" --dtype bfloat16 --no_ifeval --no_judge \
      --conditions intact,ablate_circuit,base --offset 0 --trigger_pairs_offset 2000 \
      --n_backdoor 1000 --mnt_backdoor "$MNT" \
      --n_judge 500 --judge_prompts_file data/extra/no_robots_prompts.jsonl --n_judge_indep 446 \
      --max_batch_tokens "$MBT" --out "$OUT/surgical.json" || { echo "[surgical] FAILED"; return 1; }
  fi
  echo "=== $(date -Is) · DONE $SIZE $ARM s$SEED ==="
}

run 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
