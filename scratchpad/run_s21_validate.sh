#!/bin/bash
# S2.1 -- CLEAN OUT-OF-SAMPLE validation of the brake benefit, and the GATE for the 25 GPU-h of S2.2.
#
# Today's -0.968 was selection-contaminated: brakes were chosen on [26000:27000] and scored on the
# same band. Here brakes come from S2.0's screen on [4000:5000] and every number is reported on
# [6000:41000] -- disjoint from selection, and exactly the big-n band, so the archived baseline
# (4 fires / 35,000) is directly comparable.
#
# PRE-REGISTERED, fixed before running:
#   CONFIRMED         iff B_sub mean margin < EVERY one of the 25 rank-matched nulls
#                     AND worst-case slack improves vs A.            -> launch S2.2
#   SELECTION ARTIFACT if B_sub lands inside the null band.          -> report it, do NOT re-select
set -uo pipefail
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1

CIRC=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json
ZERO=clcd_results/suppressors/l1523_s43_attrib_zero.json
BRAKES=clcd_results/probes/contrib_l1523_s43_MERGED.json
mkdir -p clcd_results/probes logs/probes

# ---- merge the shards and build both set-specs -------------------------------------------------
uv run python - "$CIRC" "$ZERO" "$BRAKES" <<'PY'
import glob, json, sys
circ_f, zero_f, merged_f = sys.argv[1], sys.argv[2], sys.argv[3]
shards = sorted(glob.glob("clcd_results/probes/contrib_l1523_s43_sh*.json"))
assert shards, "no contrib shards on disk"
rows, cfgs = [], []
for f in shards:
    d = json.load(open(f)); rows += d["rows"]; cfgs.append(d["config"])
assert all(c["offset"] == cfgs[0]["offset"] and c["n_margin"] == cfgs[0]["n_margin"]
           for c in cfgs), "shards disagree on band -- refusing to merge"
seen = set()
for r in rows:                      # the Probe-B duplicate bug: never sum over rows again
    k = (r["module"], r["dim"])
    assert k not in seen, f"duplicate target {k} across shards"
    seen.add(k)
med_se = sorted(x["se"] for x in rows)[len(rows) // 2]
json.dump({"config": cfgs[0], "n_shards": len(shards), "median_se": med_se,
           "min_detectable_2se": 2 * med_se, "rows": rows}, open(merged_f, "w"), indent=1)

kept = [tuple(x) for x in json.load(open(circ_f))["kept_latents"]]
kset = set(kept)
brakes = [(r["module"], r["dim"]) for r in rows if r["cls"] == "BRAKE" and r["in_circuit"]]
nb = [l for l in kept if l not in set(brakes)]
print(f"merged {len(rows)} targets from {len(shards)} shards; median SE {med_se:.4f} "
      f"(MDE {2*med_se:.4f}); in-circuit BRAKES = {len(brakes)}")

# free-proxy arms: negative ZERO-baseline attribution (72% sign agreement vs 55% for control)
zs = {(m, int(d)): s for m, d, s in json.load(open(zero_f))["scores"]}
neg = sorted([l for l in kept if zs.get(l, 0.0) < 0], key=lambda l: zs[l])   # most negative first
sign_all, sign_topN = list(neg), neg[:len(brakes)]
print(f"proxy arms: sign_all K={len(kept)-len(sign_all)} (drops {len(sign_all)}), "
      f"sign_topN drops {len(sign_topN)} (size-matched to the brakes)")

import random
rng = random.Random(1234)
def drop(s): return [l for l in kept if l not in set(s)]
base = {"A": kept, "B_sub": nb, "B_sign_all": drop(sign_all), "B_sign_topN": drop(sign_topN)}

# RUN 1 -- headline arms at full n
json.dump({"sets": {k: [list(x) for x in v] for k, v in base.items()},
           "contrasts": [{"a": k, "b": "A", "note": k} for k in base if k != "A"]},
          open("scratchpad/setspec_s21_full.json", "w"), indent=1)

# RUN 2 -- same arms + the rank-matched null band, on identical prompts so the rule is paired
rank = {l: i for i, l in enumerate(kept)}
brank = sorted(rank[l] for l in brakes)
sets = dict(base); contrasts = [{"a": k, "b": "A", "note": k} for k in base if k != "A"]
for i in range(25):
    used = set()
    for r in brank:                 # rank-matched: brakes are NOT rank-typical (ranks 4-392)
        c = sorted((l for l in nb if l not in used),
                   key=lambda l: (abs(rank[l] - r), rank[l]))[:5]
        used.add(rng.choice(c))
    nm = f"null_r{i}"; sets[nm] = drop(used)
    contrasts.append({"a": nm, "b": "A", "note": "RANK-MATCHED NULL"})
json.dump({"sets": {k: [list(x) for x in v] for k, v in sets.items()}, "contrasts": contrasts},
          open("scratchpad/setspec_s21_null.json", "w"), indent=1)
print(f"specs written: full={len(base)} sets, null={len(sets)} sets")
PY
[ $? -ne 0 ] && { echo "SPEC BUILD FAILED"; exit 1; }

busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
for g in ${GPUS:-3 7}; do
  u=$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)
  echo "$busy" | grep -qF "$u" && { echo "GPU $g BUSY -- aborting"; exit 1; }
done
set -- ${GPUS:-3 7}

run() {  # gpu, spec, n, tag
  CUDA_VISIBLE_DEVICES="$1" P_SETS=1 P_SETSPEC="$2" P_NM="$3" \
  P_CIRCUIT="$CIRC" P_DATA=data/sleeper/prepared_eval41k P_OFFSET=6000 P_BS=16 P_SCORE_T=3 \
  P_OUT="clcd_results/probes/s21_$4.json" \
    uv run python -u scratchpad/probe_A_gradfidelity.py > "logs/probes/s21_$4.out" 2>&1
  echo "S21_$4_EXIT=$?" >> "logs/probes/s21_$4.out"
}
run "$1" scratchpad/setspec_s21_full.json 35000 full &
run "$2" scratchpad/setspec_s21_null.json  5000 null &
wait
echo "S21_DONE"
