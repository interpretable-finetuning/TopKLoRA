#!/bin/bash
# Are the 49 in-circuit causal BRAKES additive?
#
# Probe-B measured each brake ALONE: sum of the 49 individual effects = +2.8031 nats. Probe A
# already showed single-latent effects here are NOT additive, so that sum is a PREDICTION. Stage 2
# (rebuild circuits with brakes excluded, re-leak at n=35k) is only worth its GPU time if the joint
# effect is anywhere near the prediction. This measures it directly, in minutes.
#
# CONTROLS (Rule 12 -- the check must be able to fail):
#  - intact             : must reproduce Probe-B's +9.835 nats EXACTLY, else band/harness differs
#  - circuit400         : the shipped certified circuit; must be FAR below 0, else ablation is inert
#  - circ_minus_rand49  : 10 draws -- matched-K null band for "drop 49 circuit members"
#  - circ_minus_null46  : drop the 46 in-circuit latents Probe-B measured as causally NULL
set -euo pipefail
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1

BRAKES=clcd_results/probes/brakes_l1523_s43.json
SPEC="${OUTSPEC:-scratchpad/setspec_additivity_l1523_s43.json}"
mkdir -p clcd_results/probes logs/probes

uv run python - "$BRAKES" "$SPEC" <<'PY'
import json, random, sys
brk_f, out_f = sys.argv[1], sys.argv[2]
d = json.load(open(brk_f))
circ = json.load(open(d["circuit"]))
kept = [tuple(x) for x in circ["kept_latents"]]
ks = set(kept)

def sel(cls, inc=True):
    """DISTINCT latents of a class. One latent (gate_proj#58) landed in two role groups of the
    Probe-B target list and was therefore measured twice; both rows are byte-identical. Summing
    over ROWS would double-count it, so dedupe here and sum over LATENTS."""
    seen, out = set(), []
    for r in d["rows"]:
        l = (r["module"], r["dim"])
        if r["cls"] == cls and r["in_circuit"] is inc and l not in seen:
            seen.add(l); out.append((l, r["dm_true"]))
    return out

brakes_e = sel("BRAKE"); nulls = [l for l, _ in sel("NULL")]
brakes = [l for l, _ in brakes_e]
assert all(l in ks for l in brakes), "a 'brake' is not in the circuit"
assert len(brakes) == 49 and len(kept) == 400, (len(brakes), len(kept))
pred = sum(v for _, v in brakes_e)

bset = set(brakes)
nonbrake = [l for l in kept if l not in bset]
assert len(nonbrake) == 400 - len(brakes) == 351
rng = random.Random(1234)

sets = {"intact": [], "brakes49": brakes, "circuit400": kept,
        "circ_minus_brakes": nonbrake,
        "circ_minus_null46": [l for l in kept if l not in set(nulls)]}
contrasts = [
    {"a": "brakes49", "b": "intact", "predicted": pred,
     "note": "ADDITIVITY TEST: joint effect of the 49 brakes vs the sum of their solo effects"},
    {"a": "circuit400", "b": "intact",
     "note": "POSITIVE CONTROL: the shipped circuit must move the margin far negative"},
    {"a": "circ_minus_brakes", "b": "circuit400",
     "note": "STAGE-2 PREVIEW: negative = dropping brakes leaves the model FURTHER from firing"},
    {"a": "circ_minus_null46", "b": "circuit400",
     "note": "EFFECT-MATCHED CONTROL: drop 46 causally-NULL circuit members instead"},
]
import os
N_UNIF = int(os.environ.get("NDRAW_UNIF", "10"))
N_RANK = int(os.environ.get("NDRAW_RANK", "0"))

# UNIFORM draws first, so the rng sequence -- and therefore r0..r9 -- reproduces exactly when the
# number of draws is increased. Agreement with the 10-draw run is a free determinism check.
for i in range(N_UNIF):
    nm = f"circ_minus_rand49_r{i}"
    drop = set(rng.sample(nonbrake, len(brakes)))
    sets[nm] = [l for l in kept if l not in drop]
    contrasts.append({"a": nm, "b": "circuit400",
                      "note": "MATCHED-K NULL: drop 49 arbitrary non-brake circuit members"})

# RANK-MATCHED draws. The brakes are NOT rank-typical: they sit at circuit ranks 4-392, median 104,
# versus a uniform draw's median ~200. Earlier rank = larger |attribution| = dropping it from the
# removal set should hurt MORE. So the uniform null is biased AGAINST the brake arm, and a
# rank-matched null is the honest comparison: for each brake rank, take a random non-brake from the
# 5 nearest unused ranks.
rank = {l: i for i, l in enumerate(kept)}
brank = sorted(rank[l] for l in brakes)
for i in range(N_RANK):
    nm = f"circ_minus_rankmatch_r{i}"
    used, sel = set(), []
    for r in brank:
        cands = sorted((l for l in nonbrake if l not in used),
                       key=lambda l: (abs(rank[l] - r), rank[l]))[:5]
        c = rng.choice(cands); used.add(c); sel.append(c)
    assert len(used) == len(brakes)
    sets[nm] = [l for l in kept if l not in used]
    md = sorted(rank[l] for l in used)
    contrasts.append({"a": nm, "b": "circuit400",
                      "note": f"RANK-MATCHED NULL: drop 49 non-brakes at matched circuit ranks "
                              f"(median rank {md[len(md)//2]} vs brakes {brank[len(brank)//2]})"})

json.dump({"sets": {k: [list(x) for x in v] for k, v in sets.items()},
           "contrasts": contrasts}, open(out_f, "w"), indent=1)
print(f"spec: {len(sets)} sets, {len(contrasts)} contrasts, predicted additive = {pred:+.4f} nats")
PY

# GPU must be verified free AT LAUNCH -- shared node.
GPU="${GPU:-3}"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | \
   grep -qF "$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)"; then
  echo "GPU $GPU is BUSY -- aborting"; exit 1
fi
echo "using GPU $GPU"

LOG="${OUTLOG:-logs/probes/additivity_l1523_s43.out}"
set +e   # so the exit marker always prints; pipefail still reports python's status
CUDA_VISIBLE_DEVICES="$GPU" \
P_SETS=1 P_SETSPEC="$SPEC" \
P_CIRCUIT=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json \
P_DATA=data/sleeper/prepared_eval41k P_OFFSET=26000 P_NM=1000 P_BS=16 P_SCORE_T=3 \
P_OUT="${OUTJSON:-clcd_results/probes/additivity_l1523_s43.json}" \
  uv run python -u scratchpad/probe_A_gradfidelity.py 2>&1 | tee "$LOG"
echo "ADDITIVITY_EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"
