#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Calibration control for the payload-concentration (coalition) metric.
#
# route_l1523_s4* : known-separable BY CONSTRUCTION (planted 504, discovered both-circuit 50,
#                   0 fires / 12000 held out after ablation)
# a0_l1523_s4*    : in-wave normally-trained twins -- same wave, same settings, same seeds.
#                   Routing is the ONLY difference, which is what makes this a clean control.
#
# Pre-registered predictions are in scripts/payload_concentration.py. If route and a0 come out
# indistinguishable the metric is dead and that is the reported result -- nothing gets retuned.
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/payload_concentration_control.sh'
GPUS=(${GPUS:-6 7})
mkdir -p clcd_results/exp6 logs/exp6

CUDA_VISIBLE_DEVICES=${GPUS[0]} CLCD_N=${CLCD_N:-50} \
  CLCD_OUT=clcd_results/exp6/payload_conc_route.json \
  uv run python -u scripts/payload_concentration.py \
    route_l1523_s42 route_l1523_s43 route_l1523_s44 \
    > logs/exp6/payload_conc_route.out 2>&1 &

CUDA_VISIBLE_DEVICES=${GPUS[1]} CLCD_N=${CLCD_N:-50} \
  CLCD_OUT=clcd_results/exp6/payload_conc_a0.json \
  uv run python -u scripts/payload_concentration.py \
    a0_l1523_s42 a0_l1523_s43 a0_l1523_s44 \
    > logs/exp6/payload_conc_a0.out 2>&1 &

wait
echo "=== payload-concentration control done $(date) ==="
grep -h "n90=" logs/exp6/payload_conc_route.out logs/exp6/payload_conc_a0.out
