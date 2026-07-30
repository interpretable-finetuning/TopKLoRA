#!/bin/bash
# Exp-9 — does scoring the circuit ALONE fix the μ arbiter's blindness to the hub?
#
# Exp-8 showed `edges.scrub_eval` leaves wires OUTSIDE candidate_edges permanently kept, so
# `cut=ALL` never reaches the true no-wiring floor (fixture: +0.1318 vs +0.1989). The known
# failure has exactly that shape: edge_scrub_N10 halts at 7 kept edges with μ-recovery 0.886
# while having FULLY CUT o_proj.53 -- the behavioural hub -- and the measured ASR curve
# (edge_scrub_N10_asrcurve.json) shows behaviour already broken at kept=19:
#     kept 57 -> ASR 0.98 | kept 19 -> 0.64 | kept 8 -> 0.66 | kept 7 -> 0.42
# Hypothesis: the non-candidate wires still feed the orphaned hub, so orphaning it costs μ
# almost nothing. With them severed it should cost μ, and the greedy should refuse the cut.
#
# Two runs, IDENTICAL config to the baseline artifact, both WITHOUT --protect_nodes (that
# guard is what we are trying to make unnecessary; note its ASR=0.98 is tautological --
# n_fully_cut=0 means retained_asr() is called with [] and returns the ceiling by construction).
#   A (ctl)   flag off  -> must reproduce edge_scrub_N10: kept 7, rec ~0.886, o_proj.53 cut
#   B (sever) flag on   -> the test
#
# PRE-REGISTERED, fixed before any numbers (integrity_no_phacking). target stays 0.85, the
# baseline value, and is NOT tuned afterwards in either direction.
#   PRIMARY   (structural)  : does B keep o_proj.53 SPANNED (absent from fully_cut_latents)?
#   SECONDARY (behavioural) : B's MEASURED asr_normalized at halt vs baseline 0.43
#   TERTIARY  (agreement)   : kept-edge count where μ-recovery crosses 0.85 vs where ASR breaks (19)
#   FALSIFIED IF            : B still orphans the hub, OR μ-recovery goes flat/degenerate across
#                             all cuts (the collapse confound -- severing leaves the net so
#                             ablated that μ stops discriminating). Logged as a negative either way.
#
#   ssh torrnode12 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/exp9_sever.sh'
set -u
cd /scratch/network/ssd/marek/minimalsleepers || exit 1
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
GPUS=(${GPUS:-0 1})
mkdir -p clcd_results logs/exp9

# baseline config, read off clcd_results/edge_scrub_N10.json's "config" verbatim
COMMON="--data data/sleeper/prepared --n_attrib 16 --n_prune 6 --n_test 50 --N 10 \
--K_ig 24 --target 0.85 --tau 0.3 --attr_target margin --tag_baseline head"

echo "=== Exp-9 start $(date) on $(hostname) GPUs ${GPUS[*]} ==="

CUDA_VISIBLE_DEVICES=${GPUS[0]} uv run --with matplotlib python -u -m src.clcd.exp_edge_scrub \
  $COMMON --out clcd_results/edge_scrub_N10_sever_ctl.json \
  > logs/exp9/sever_ctl.out 2>&1 &
PID_CTL=$!
echo "[$(date +%H:%M) g${GPUS[0]}] A ctl   (flag off)  pid $PID_CTL"

CUDA_VISIBLE_DEVICES=${GPUS[1]} uv run --with matplotlib python -u -m src.clcd.exp_edge_scrub \
  $COMMON --sever_noncandidate --out clcd_results/edge_scrub_N10_sever.json \
  > logs/exp9/sever.out 2>&1 &
PID_SEV=$!
echo "[$(date +%H:%M) g${GPUS[1]}] B sever (flag on)   pid $PID_SEV"

wait $PID_CTL; echo "A ctl exit=$?"
wait $PID_SEV; echo "B sever exit=$?"
echo "=== Exp-9 done $(date) ==="

python3 -c "
import json
base = json.load(open('clcd_results/edge_scrub_N10.json'))
def short(l): return l[0].split('.')[-2] + '.' + str(l[1])
rows = [('baseline(logged)', base)]
for tag, f in (('A ctl', 'edge_scrub_N10_sever_ctl'), ('B sever', 'edge_scrub_N10_sever')):
    try: rows.append((tag, json.load(open(f'clcd_results/{f}.json'))))
    except Exception as e: print(tag, 'MISSING:', e)
print()
print(f\"{'run':<17}{'kept':>5}{'spanned':>9}{'fullycut':>9}{'final_rec':>11}{'asr_kept':>9}{'asr_norm':>9}  fully_cut\")
for tag, d in rows:
    print(f\"{tag:<17}{d['n_kept_edges']:>5}{d['n_spanned_latents']:>9}{d['n_fully_cut']:>9}\"
          f\"{d['final_recovery']:>11.3f}{d.get('asr_kept',float('nan')):>9.2f}\"
          f\"{d.get('asr_normalized',float('nan')):>9.2f}  \"
          + ','.join(short(l) for l in d['fully_cut_latents']))
    ep = d.get('endpoints')
    if ep:
        n = len(ep)
        print(f\"{'':17}  endpoints: mu_trigger={sum(e['mu_trigger'] for e in ep)/n:+.4f} \"
              f\"mu_empty={sum(e['mu_empty'] for e in ep)/n:+.4f} \"
              f\"floor_ctx={sum(e['floor_context'] for e in ep)/n:+.4f} \"
              f\"floor_alone={sum(e['floor_alone'] for e in ep)/n:+.4f} \"
              f\"free_ride={sum(e['free_ride'] for e in ep)/n:+.4f}\")
print()
HUB = 'o_proj'
for tag, d in rows:
    if tag == 'baseline(logged)': continue
    orphaned = any(HUB in l[0] and l[1] == 53 for l in d['fully_cut_latents'])
    print(f'PRIMARY  {tag}: o_proj.53 {\"ORPHANED (hypothesis fails)\" if orphaned else \"SPANNED (hypothesis holds)\"}')
"
