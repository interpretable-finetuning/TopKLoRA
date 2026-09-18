#!/bin/bash
# Circuit campaign 3, 2026-09-17. Replaces chain.sh stages 4-5 and gemma_campaign.sh (both stopped; see
# docs/captains-log-qwen2.5-1.5b.md, "DECISION (user) -- block elimination ...").
#   multi-layer (Qwen l17_25/all, gemma l1523/all): block elimination cap 64 + --adaptive_n
#   single-layer (gemma l19, qwen l20): block elimination too (user decision 2026-09-17), so every
#   family in the campaign shares one protocol; the older one-at-a-time l20 circuits stay untouched.
#   pool: sparse top-2,500 by |attribution| (ELIM_FULL_POOL=0), dense = full adapter (modules x r)
#   every visiting order is saved (ELIM_ORDER_OUT_DIR) so a resume walks the same order.
# Five drivers share the default 16-slot lock pool (2 per card); each lists its longest cells first.
# This process is also the single judging worker: every JUDGE_POLL it judges the surgical files on
# disk in both trees, and it runs one final pass after the drivers exit. It then states the circuit
# and surgical counts against 60/30 and exits non-zero if any of them is short.
# DRY=1: stub python (prints the search command), first seed only, private lock pool, nothing written.
set -u
cd /home/andrzej/TopKLoRA
ST="${ST:-logs/overnight_chain/c3}"
QT="${QT:-clcd_results/qwen15_campaign3}"
GT="${GT:-clcd_results/gemma2b_campaign}"
STATUS="${STATUS:-logs/overnight_chain/status.txt}"
POLL="${POLL:-300}"                 # driver liveness poll
JUDGE_POLL="${JUDGE_POLL:-10800}"   # 3 h between incremental judge passes
DRY="${DRY:-0}"
mkdir -p "$ST"
say() { echo "[$(date -Is)] c3: $*" | tee -a "$STATUS"; }
alive() { local p; for p in $(cat "$1" 2>/dev/null); do ps -p "$p" >/dev/null && return 0; done; return 1; }
# Counted in the shell, not `ls ... 2>/dev/null | wc -l`: a glob that matches nothing must read as
# 0 because nothing is there, not because an error was discarded.
nmatch() { local n=0 f; for f in "$@"; do [ -e "$f" ] && n=$((n+1)); done; echo "$n"; }

SEEDS="42 43 44 45 46"; COMMON=(GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 MIN_FREE_MIB=30000 STAGGER=60 ELIM_FULL_POOL=0)
if [ "$DRY" = 1 ]; then
  SEEDS=42; D=$CLAUDE_JOB_DIR/tmp/c3dry; mkdir -p "$D/locks"
  QT=$D/qwen; GT=$D/gemma
  COMMON=(GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 MIN_FREE_MIB=0 STAGGER=0 ELIM_FULL_POOL=0 STEPS=search
          PY="$D/pystub.sh" LOCKROOT="$D/locks")
  cat > "$D/pystub.sh" <<'EOF'
#!/bin/bash
case " $* " in *" src.clcd.exp_circuit_search "*) echo "STUB-SEARCH $*"; exit 0 ;; esac
exec /home/andrzej/TopKLoRA/.venv/bin/python "$@"
EOF
  chmod +x "$D/pystub.sh"; ST=$D/st; mkdir -p "$ST"
  say() { echo "[dry] $*"; }
fi

GRID_L1523="50 100 150 200 300 400 600 800 1200 1600 2000 2400 2500 2600 2640 2646 2800 3200 3600 3800 3950 4020 4032"
GRID_GALL="100 200 300 400 600 800 1200 1600 2400 3200 4800 6400 8000 9600 10400 10800 11000 11200 11400 11520 11600 11640 11648"
cells() { local out=() arm fam s; for fam in $2; do for arm in $1; do for s in $SEEDS; do out+=("$arm $fam $s"); done; done; done; printf '%s\n' "${out[@]}"; }

launch() {  # <tag> <logfile> <env...> -- <cells...>
  local tag=$1 log=$2; shift 2; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${COMMON[@]}" "${envs[@]}" nohup bash scripts/qwen15_phase1.sh "$@" > "$log" 2>&1 < /dev/null &
  echo $! >> "$ST/drivers.pids"; say "launched $tag pid $! ($# cells)"
}

if ! alive "$ST/drivers.pids"; then
  : > "$ST/drivers.pids"
  mkdir -p logs/qwen15/campaign3 logs/gemma2b/campaign "$QT/orders" "$GT/orders"
  BLOCK=(ELIM_BLOCK_CAP=64 SEARCH_EXTRA="--adaptive_n")
  mapfile -t Q < <(cells "r64_dense r42_dense" all; cells "r64_k8 r42_k5" all; cells "r64_dense r42_dense" l17_25; cells "r64_k8 r42_k5" l17_25)
  launch qwen_multi logs/qwen15/campaign3/driver.out "${BLOCK[@]}" SRC="$QT" LOGDIR=logs/qwen15/campaign3 \
      ELIM_ORDER_OUT_DIR="$QT/orders" -- "${Q[@]}"
  GEM=(GATE_DIR=clcd_results/gemma2b DATA=data/sleeper/prepared_eval6k SRC="$GT" LOGDIR=logs/gemma2b/campaign
       ELIM_ORDER_OUT_DIR="$GT/orders")
  mapfile -t GA < <(cells r64_dense all; cells r64_k8 all)
  launch gemma_all logs/gemma2b/campaign/driver_all.out "${BLOCK[@]}" "${GEM[@]}" KS_OVERRIDE="$GRID_GALL" -- "${GA[@]}"
  mapfile -t GB < <(cells r64_dense l1523; cells r64_k8 l1523)
  launch gemma_l1523 logs/gemma2b/campaign/driver_l1523.out "${BLOCK[@]}" "${GEM[@]}" KS_OVERRIDE="$GRID_L1523" -- "${GB[@]}"
  mapfile -t GS < <(cells "r64_dense r64_k8" l19)
  launch gemma_l19 logs/gemma2b/campaign/driver_l19.out "${BLOCK[@]}" "${GEM[@]}" -- "${GS[@]}"
  # 2026-09-17 (user): single-layer uses block elimination too, so the Qwen l20 cells are re-searched
  # under the campaign protocol instead of reusing the one-at-a-time circuits in clcd_results/qwen15.
  # Pool 294 (r42) / 448 (r64) is below the 2,500 sparse cap, so both arms search the full adapter.
  mapfile -t QL < <(cells "r64_dense r42_dense r64_k8 r42_k5" l20)
  launch qwen_l20 logs/qwen15/campaign3/driver_l20.out "${BLOCK[@]}" SRC="$QT" LOGDIR=logs/qwen15/campaign3 \
      ELIM_ORDER_OUT_DIR="$QT/orders" -- "${QL[@]}"
fi
SURGICAL="$QT/*/surgical/*_surgical.json $GT/*/surgical/*_surgical.json"

# One judge pass over BOTH trees. Called from this process only, between driver polls, so the
# campaign never has two judges of its own in flight; qwen15_judge.sh's flock guards the account
# against any other judge (judge_api tracks the 20,000 in-flight cap within one process only).
# Re-judging costs nothing: judge_saved_gens_big skips every record that already carries the luna
# key, so a file judged by an earlier pass contributes no requests to the next one.
judge_pass() {  # <label> -> 0 judged, 75 lock busy, else the judge's own status
  local label=$1 out n rc
  out="$ST/judge_$label.out"   # separate statement: $label is not yet set inside its own `local`
  n=$(nmatch $SURGICAL)
  [ "$n" -gt 0 ] || { say "judge $label: no surgical files yet"; return 0; }
  say "judge $label: $n surgical files on disk -> $out"
  FILES="$SURGICAL" bash scripts/qwen15_judge.sh > "$out" 2>&1; rc=$?
  case $rc in
    0)  say "judge $label done ($out)" ;;
    75) say "judge $label SKIPPED: another judge holds the OpenRouter account lock ($out)" ;;
    *)  say "judge $label FAILED rc=$rc (see $out)" ;;
  esac
  return $rc
}

# Judge while the drivers run: most surgical files land days before the last cell finishes, and
# judging all 90 at the end is 12-15 h of wall clock nobody is waiting on anything else for.
pass=0; waited=0
while alive "$ST/drivers.pids"; do
  sleep "$POLL"; waited=$((waited+POLL))
  [ "$waited" -ge "$JUDGE_POLL" ] || continue
  waited=0
  alive "$ST/drivers.pids" || break     # drivers gone mid-wait: the final pass takes it
  pass=$((pass+1)); judge_pass "p$pass"
done

qc=$(nmatch $QT/*/elim/*_circuit.json); qs=$(nmatch $QT/*/surgical/*_surgical.json)
gc=$(nmatch $GT/*/elim/*_circuit.json); gs=$(nmatch $GT/*/surgical/*_surgical.json)
say "all drivers exited; circuits qwen $qc/60 gemma $gc/30; surgical qwen $qs/60 gemma $gs/30"
[ "$DRY" = 1 ] && exit 0

# Final pass: whatever the incremental passes missed, plus everything written after the last one.
judge_pass final; jrc=$?

short=""
[ "$qc" -eq 60 ] || short="$short qwen_circuits=$qc/60"
[ "$qs" -eq 60 ] || short="$short qwen_surgical=$qs/60"
[ "$gc" -eq 30 ] || short="$short gemma_circuits=$gc/30"
[ "$gs" -eq 30 ] || short="$short gemma_surgical=$gs/30"
[ "$jrc" -eq 0 ] || short="$short judge_rc=$jrc"
if [ -n "$short" ]; then
  say "CAMPAIGN 3 INCOMPLETE:$short -- check the driver logs and $ST/judge_final.out before analysing anything"
  exit 1
fi
say "campaign 3 complete: circuits qwen 60/60 gemma 30/30, surgical qwen 60/60 gemma 30/30, judge ok"
