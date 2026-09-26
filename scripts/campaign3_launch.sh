#!/bin/bash
# Circuit campaign 3, 2026-09-17. Replaces chain.sh stages 4-5 and gemma_campaign.sh (both stopped; see
# docs/captains-log-qwen2.5-1.5b.md, "DECISION (user) -- block elimination ...").
#   multi-layer (Qwen l17_25/all, gemma l1523/all): block elimination cap 64 + --adaptive_n
#   single-layer (gemma l19, qwen l20): block elimination too (user decision 2026-09-17), so every
#   family in the campaign shares one protocol; the older one-at-a-time l20 circuits stay untouched.
#   pool: sparse top-2,500 by |attribution| (ELIM_FULL_POOL=0), dense = full adapter (modules x r)
#   every visiting order is saved (ELIM_ORDER_OUT_DIR) so a resume walks the same order.
# Drivers share the box's 16-slot lock pool (2 per card); each lists its longest cells first.
#
# TWO BOXES. Each runs this same script with a DRIVERS subset. The box holding the OpenRouter
# account runs with JUDGE=1 and is also the COLLECTION POINT: it pulls the other box's trees in
# before every judge pass and once more after both sides' drivers exit, so the complete campaign
# lands on one machine and nothing has to be gathered by hand.
#   collector+judge:  REMOTE=andrzej@81.85.1.18 DRIVERS="qwen_all" bash scripts/campaign3_launch.sh
#   worker:           JUDGE=0 EXPECT_Q=40 EXPECT_R=15 \
#                       DRIVERS="qwen_l17_25 qwen_l20 gemma_all gemma_l1523 gemma_l19 gradroute" \
#                       bash scripts/campaign3_launch.sh
# Start the WORKER first: the collector refuses to start unless the remote already has live drivers,
# because a collector that ran alone would judge a partial campaign and then report it complete.
# The worker has no OpenRouter key at all, which is what actually guarantees one judge per account;
# JUDGE=0 is the statement of intent, the absent key is the enforcement.
#
# It then states circuit and surgical counts against EXPECT_Q/EXPECT_G/EXPECT_R and exits non-zero
# if any of them is short.
# DRY=1: stub python (prints the search command), first seed only, private lock pool, nothing written.
# Repo root from this file's own location, like every other driver (scripts/_common.sh). The
# hardcoded `cd /home/andrzej/TopKLoRA` this replaces meant a COPY of this script -- in a
# worktree, on the other box -- silently drove the canonical checkout instead of its own:
# it would pull the other box's results into a tree it was not reading, and report on that.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ST="${ST:-logs/overnight_chain/c3}"
QT="${QT:-clcd_results/qwen15_campaign3}"
GT="${GT:-clcd_results/gemma2b_campaign}"
STATUS="${STATUS:-logs/overnight_chain/status.txt}"
POLL="${POLL:-300}"                 # driver liveness poll
JUDGE_POLL="${JUDGE_POLL:-10800}"   # 3 h between incremental judge passes
RT="${RT:-clcd_results/gradroute_campaign}"
DRY="${DRY:-0}"
# Which driver groups this box runs. Default: all of them (single-box behaviour, unchanged).
DRIVERS="${DRIVERS:-qwen_all qwen_l17_25 qwen_l20 gemma_all gemma_l1523 gemma_l19 gradroute}"
JUDGE="${JUDGE:-1}"                 # 0 on the worker box: it has no OpenRouter key
REMOTE="${REMOTE:-}"                # user@host of the other box; set on the collector only
REMOTE_ROOT="${REMOTE_ROOT:-/home/andrzej/TopKLoRA}"
REMOTE_ST="${REMOTE_ST:-logs/overnight_chain/c3}"
RFAIL_MAX="${RFAIL_MAX:-10}"        # consecutive unknown remote polls before this side gives up on it
EXPECT_Q="${EXPECT_Q:-60}"; EXPECT_G="${EXPECT_G:-30}"; EXPECT_R="${EXPECT_R:-15}"
GR_SEEDS="${GR_SEEDS:-42 43 44}"    # the gradient-routing release has three seeds, not five
SSH="${SSH:-ssh -o BatchMode=yes -o ConnectTimeout=20}"
mkdir -p "$ST"
say() { echo "[$(date -Is)] c3: $*" | tee -a "$STATUS"; }
alive() { local p; for p in $(cat "$1" 2>/dev/null); do ps -p "$p" >/dev/null && return 0; done; return 1; }
# Counted in the shell, not `ls ... 2>/dev/null | wc -l`: a glob that matches nothing must read as
# 0 because nothing is there, not because an error was discarded.
nmatch() { local n=0 f; for f in "$@"; do [ -e "$f" ] && n=$((n+1)); done; echo "$n"; }

# ---- the other box ---------------------------------------------------------------------------
# Three outcomes, kept distinct on purpose: a box that cannot be REACHED is not a box whose work is
# FINISHED. Collapsing those two would let this side declare the campaign complete while the other
# still had cells running, or had died with them unfinished.
remote_state() {   # prints ALIVE | DONE | NOFILE; prints nothing, non-zero rc, when unreachable
  $SSH "$REMOTE" "f='$REMOTE_ROOT/$REMOTE_ST/drivers.pids'
      [ -f \"\$f\" ] || { echo NOFILE; exit 0; }
      while read -r pid; do
        [ -n \"\$pid\" ] && ps -p \"\$pid\" >/dev/null && { echo ALIVE; exit 0; }
      done < \"\$f\"
      echo DONE"
}

# --ignore-existing is load-bearing. This side REWRITES surgical files in place as it judges them
# (judge_saved_gens_big adds the luna key to every record), so a plain `rsync -a` would find the
# remote's unjudged original differing from the judged local copy and copy it back over the top.
# The next pass would re-issue every request in that file -- ~2,838 per cell -- and the verdicts
# already paid for would be gone. A result file only ever APPEARS on the remote and is never edited
# there, so "never overwrite what is already here" is exactly the right rule.
# *.ckpt is excluded: an elimination checkpoint is only meaningful on the box running that cell, and
# a stale copy here is what a failover would resume from.
remote_pull() {   # 0 = every tree pulled, 1 = at least one failed
  [ -n "$REMOTE" ] || return 0
  local t rc=0
  for t in "$QT" "$GT" "$RT"; do
    mkdir -p "$t"
    rsync -a --ignore-existing --exclude '*.ckpt' --exclude '.*.tmp' \
        -e "$SSH" "$REMOTE:$REMOTE_ROOT/$t/" "$t/" || { say "remote pull FAILED for $t"; rc=1; }
  done
  return $rc
}

rfail=0
remote_running() {   # 0 = the other box still has work in flight, 1 = it is finished (or absent)
  [ -n "$REMOTE" ] || return 1
  local st; st=$(remote_state) || st=""
  case "$st" in
    ALIVE) rfail=0; return 0 ;;
    DONE)  rfail=0; return 1 ;;
    *) rfail=$((rfail+1))
       say "remote state unknown ('${st:-unreachable}'), strike $rfail/$RFAIL_MAX"
       if [ "$rfail" -ge "$RFAIL_MAX" ]; then
         say "remote unreachable for $rfail polls -- continuing without it; the counts below will show what is missing"
         return 1
       fi
       return 0 ;;
  esac
}

SEEDS="42 43 44 45 46"; COMMON=(GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 MIN_FREE_MIB=30000 STAGGER=60 ELIM_FULL_POOL=0)
if [ "$DRY" = 1 ]; then
  # DRY_FULL_SEEDS=1 keeps the real seed lists. Without it a dry run launches one seed per
  # cell, so it can prove WHICH drivers run but never that EXPECT_Q/EXPECT_G/EXPECT_R equal
  # the number of cells actually launched -- which is the only thing those constants are for.
  [ "${DRY_FULL_SEEDS:-0}" = 1 ] || { SEEDS=42; GR_SEEDS=42; }
  D=$CLAUDE_JOB_DIR/tmp/c3dry; mkdir -p "$D/locks"
  # DRY_TREE_ROOT exists so a dry run can place its trees where the PRODUCTION ones live:
  # repo-relative. remote_pull joins $REMOTE_ROOT with the tree path, which only means
  # anything for a relative tree, so with the default absolute $D the pull is untestable.
  DRY_TREE_ROOT="${DRY_TREE_ROOT:-$D}"
  QT=$DRY_TREE_ROOT/qwen; GT=$DRY_TREE_ROOT/gemma; RT=$DRY_TREE_ROOT/gradroute
  # REMOTE is deliberately NOT cleared here: $SSH is overridable, so a dry run is the only
  # place the two-box guards can be exercised without a second machine. Clearing it would
  # leave "refuse to start unless the remote is ALIVE" as a branch nothing can ever reach.
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
# $3 overrides the seed list: the gradient-routing release has seeds 42-44 only, not 42-46.
cells() { local out=() arm fam s; for fam in $2; do for arm in $1; do for s in ${3:-$SEEDS}; do out+=("$arm $fam $s"); done; done; done; printf '%s\n' "${out[@]}"; }
want() { case " $DRIVERS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

launch() {  # <tag> <logfile> <env...> -- <cells...>
  local tag=$1 log=$2; shift 2
  want "$tag" || { say "skip $tag (not in DRIVERS)"; return 0; }
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${COMMON[@]}" "${envs[@]}" nohup bash scripts/qwen15_phase1.sh "$@" > "$log" 2>&1 < /dev/null &
  echo $! >> "$ST/drivers.pids"; say "launched $tag pid $! ($# cells)"
}

# The collector must not run alone. On its own it would judge whatever happened to be on disk,
# clear its own reduced EXPECT counts and report a complete campaign, while the worker's share was
# still being computed on the other box. Start the worker first.
if [ -n "$REMOTE" ]; then
  rstate=$(remote_state) || rstate=""
  [ "$rstate" = ALIVE ] || {
    say "REFUSING TO START: remote $REMOTE reports '${rstate:-unreachable}', expected ALIVE."
    say "Start the worker box first, then this one."
    exit 2; }
  say "remote $REMOTE: drivers running"
fi

if ! alive "$ST/drivers.pids"; then
  : > "$ST/drivers.pids"
  mkdir -p logs/qwen15/campaign3 logs/gemma2b/campaign logs/gradroute/campaign \
           "$QT/orders" "$GT/orders" "$RT/orders"
  BLOCK=(ELIM_BLOCK_CAP=64 SEARCH_EXTRA="--adaptive_n")
  QW=("${BLOCK[@]}" SRC="$QT" LOGDIR=logs/qwen15/campaign3 ELIM_ORDER_OUT_DIR="$QT/orders")
  # The Qwen all-layer cells are their own driver: they hold the five longest cells in the campaign
  # (r64_dense all is ~20 h of sequential elimination) and on a two-box run they are the whole of
  # the collector's share, so they have to be selectable on their own.
  mapfile -t QA < <(cells "r64_dense r42_dense" all; cells "r64_k8 r42_k5" all)
  launch qwen_all logs/qwen15/campaign3/driver.out "${QW[@]}" -- "${QA[@]}"
  mapfile -t QB < <(cells "r64_dense r42_dense" l17_25; cells "r64_k8 r42_k5" l17_25)
  launch qwen_l17_25 logs/qwen15/campaign3/driver_l17_25.out "${QW[@]}" -- "${QB[@]}"
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
  launch qwen_l20 logs/qwen15/campaign3/driver_l20.out "${QW[@]}" -- "${QL[@]}"
  # GROUND-TRUTH ORGANISMS (interpretable-finetuning/gradient-routing-gemma, staged 2026-09-17).
  # 4 routed arms x 3 seeds + 3 unrouted twins; all gemma-2-2b l1523, r64 k8, tags |TRIGGER|/
  # |TRAINING|, so they take the gemma dataset, MBT and K-grid unchanged. In a routed arm the
  # backdoor is confined BY CONSTRUCTION to latents [0:d) of each of the 63 wrapped modules, so the
  # discovered circuit can be scored against a known answer rather than only against behaviour;
  # the unrouted twins are the same wave and recipe with no routing, and have no ground truth.
  #
  # POOL: the standard sparse cap of 2,500 of the 4,032 latents (user decision 2026-09-18), i.e.
  # these arms are searched under exactly the pre-registered sparse protocol, with no exception.
  # The cost is that a designated latent ranked below 2,500 by |attribution| is never a candidate
  # and so can never be recovered: recall against the routing index is therefore a LOWER BOUND, not
  # an unbiased estimate, and any miss must be checked against pool membership before it is read as
  # a failure of the method. n_designated is 63/126/252/504 for d1/d2/d4/d8, so d8 needs 504 of its
  # latents inside a 2,500 pool -- record the overlap when the circuits land.
  GRO=(GATE_DIR=clcd_results/gradroute_gemma DATA=data/sleeper/prepared_eval6k SRC="$RT"
       LOGDIR=logs/gradroute/campaign ELIM_ORDER_OUT_DIR="$RT/orders")
  mapfile -t GR < <(cells "routed_d8 routed_d4 routed_d2 routed_d1 unrouted" l1523 "$GR_SEEDS")
  launch gradroute logs/gradroute/campaign/driver.out "${BLOCK[@]}" "${GRO[@]}" KS_OVERRIDE="$GRID_L1523" -- "${GR[@]}"
fi
# A DRIVERS value that selects nothing is a typo, not a campaign. Without this the script would fall
# straight through to the counts and report every cell missing, as though the run itself had failed.
[ -s "$ST/drivers.pids" ] || { say "NO DRIVERS LAUNCHED (DRIVERS='$DRIVERS') -- nothing to do"; exit 2; }
SURGICAL="$QT/*/surgical/*_surgical.json $GT/*/surgical/*_surgical.json $RT/*/surgical/*_surgical.json"

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
  # A dry run must never reach the paid account. Until now that held only because the dry trees were
  # empty, so nmatch returned 0 -- an accident of the fixture, not a rule: put one surgical file in
  # a dry tree and DRY=1 would have submitted it for real.
  [ "$DRY" = 1 ] && { say "judge $label: DRY, not calling the judge ($n files would have been submitted)"; return 0; }
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
while alive "$ST/drivers.pids" || remote_running; do
  sleep "$POLL"; waited=$((waited+POLL))
  [ "$waited" -ge "$JUDGE_POLL" ] || continue
  waited=0
  # drivers gone on BOTH sides mid-wait: the final pass takes it
  alive "$ST/drivers.pids" || remote_running || break
  pass=$((pass+1))
  if [ "$JUDGE" = 1 ]; then
    # Pull first: the remote's surgical files are judged by this pass, not left for the end.
    remote_pull; judge_pass "p$pass"
  else
    say "JUDGE=0: no judge pass p$pass on this box"
  fi
done

# One last pull before counting, so the collector's totals are the WHOLE campaign and the results
# are concentrated here. A failed final pull must not read as missing cells, so it says so plainly.
pulled=ok
if [ -n "$REMOTE" ]; then
  remote_pull || { pulled=failed; say "FINAL REMOTE PULL FAILED -- counts below cover THIS box only"; }
fi
qc=$(nmatch $QT/*/elim/*_circuit.json); qs=$(nmatch $QT/*/surgical/*_surgical.json)
gc=$(nmatch $GT/*/elim/*_circuit.json); gs=$(nmatch $GT/*/surgical/*_surgical.json)
rgc=$(nmatch $RT/*/elim/*_circuit.json); rgs=$(nmatch $RT/*/surgical/*_surgical.json)
say "all drivers exited; circuits qwen $qc/$EXPECT_Q gemma $gc/$EXPECT_G gradroute $rgc/$EXPECT_R; surgical qwen $qs/$EXPECT_Q gemma $gs/$EXPECT_G gradroute $rgs/$EXPECT_R"
[ "$DRY" = 1 ] && exit 0

# Final pass: whatever the incremental passes missed, plus everything written after the last one.
if [ "$JUDGE" = 1 ]; then judge_pass final; jrc=$?; else jrc=0; say "JUDGE=0: no final judge pass on this box"; fi

short=""
[ "$qc" -eq "$EXPECT_Q" ] || short="$short qwen_circuits=$qc/$EXPECT_Q"
[ "$qs" -eq "$EXPECT_Q" ] || short="$short qwen_surgical=$qs/$EXPECT_Q"
[ "$gc" -eq "$EXPECT_G" ] || short="$short gemma_circuits=$gc/$EXPECT_G"
[ "$gs" -eq "$EXPECT_G" ] || short="$short gemma_surgical=$gs/$EXPECT_G"
[ "$rgc" -eq "$EXPECT_R" ] || short="$short gradroute_circuits=$rgc/$EXPECT_R"
[ "$rgs" -eq "$EXPECT_R" ] || short="$short gradroute_surgical=$rgs/$EXPECT_R"
[ "$jrc" -eq 0 ] || short="$short judge_rc=$jrc"
[ "$pulled" = ok ] || short="$short final_remote_pull=failed"
if [ -n "$short" ]; then
  say "CAMPAIGN 3 INCOMPLETE:$short -- check the driver logs and $ST/judge_final.out before analysing anything"
  exit 1
fi
say "campaign 3 complete: circuits qwen $qc/$EXPECT_Q gemma $gc/$EXPECT_G gradroute $rgc/$EXPECT_R; surgical qwen $qs/$EXPECT_Q gemma $gs/$EXPECT_G gradroute $rgs/$EXPECT_R; judge ok"
