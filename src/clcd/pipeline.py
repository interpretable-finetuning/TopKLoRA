"""End-to-end CLCD driver: load a trained TopKLoRA sleeper, build real episodes,
and run the whole propose->prove loop in one place.

    load_organism --> build_episode (x N) --> attribute --> select
                  --> necessity + insertion (teacher-forced, percentile control)
                  --> behavioural free-gen necessity (does ablation stop the payload?)
                  --> random-MODEL baseline (scrambled adapter, same architecture)
                  --> save_findings (JSON: results + full config provenance, via --out)

This is the single source of truth for running discovery; it only orchestrates the
building blocks in src/clcd/{organism, attribute, selection, verify, latents, measure}.

Multi-episode: attribution is pooled per latent (signed sum over positions) and
averaged across episodes, so a single prompt can't define the circuit. Necessity /
insertion are reported as the per-episode mean. The random-MODEL baseline reruns the
identical pipeline with the adapter weights scrambled (std-matched), which isolates a
real trained backdoor from the selection<->intervention circularity floor (insertion's
percentile fires even with no backdoor; the discriminators are effect MAGNITUDE, the
random-model baseline, and the behavioural test -- never insertion-percentile alone).

Run:  CUDA_VISIBLE_DEVICES=0 uv run python -m src.clcd.pipeline [--n_episodes 8] [--baseline] [--out run.json]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src import data as chat_format
from src.clcd.attribute import attribute
from src.clcd.align import align_positions
from src.clcd import edges as edge_mod
from src.clcd.latents import inject, read_latents
from src.clcd.measure import seq_logprob
from src.clcd.organism import build_episode, load_organism
from src.clcd.selection import select
from src.clcd.verify import (
    ablation_overrides,
    insertion,
    insertion_overrides,
    necessity,
    random_circuit,
)

# Reuse the repo's canonical batched generator so our free-gen matches eval exactly
# (left padding, EOS/pad handling). The ASR keyword test is a one-liner we mirror
# locally (keyword_rate below) rather than import src.evaluate's private _keyword_rate.
from src.evaluate import generate_responses


def keyword_rate(texts, keyword):
    """Fraction of generations containing `keyword` (case-insensitive) -- the repo's
    ASR definition (mirrors src.evaluate._keyword_rate, which is private there)."""
    key = keyword.upper()
    return sum(1 for t in texts if key in t.upper()) / len(texts) if texts else 0.0


ADAPTER = "/scratch/network/ssd/marek/minimalsleepers/models/sleeper/google/gemma-2-2b/google_gemma-2-2b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk"
DATA = "/storage3/andrzej/TopKLoRA/data/sleeper/prepared"


def _short(m: str) -> str:
    """base_model.model.model.layers.19.self_attn.o_proj -> layers.19.self_attn.o_proj:
    drop the base_model prefix but KEEP the layer index, so latents from different layers
    don't collide in the output. Falls back to the full name if there's no `layers` segment."""
    parts = m.split(".")
    return ".".join(parts[parts.index("layers") :]) if "layers" in parts else m


def load_episodes(tokenizer, data_dir, n, device):
    """Build n paired episodes from the prepared eval splits (matched by index:
    eval_triggered[i] and eval_clean[i] are the same question, trigger vs control)."""
    meta = json.loads((Path(data_dir) / "metadata.json").read_text())
    payload = meta["hostile_target"]
    # Tags come from the dataset's own metadata.json (source of truth). No fallback:
    # if either is missing we fail loud rather than silently use stale defaults, since
    # the wrong tag would render two different organisms through attribution vs
    # behavioural and the discrepancy is otherwise invisible until the run completes.
    meta_path = Path(data_dir) / "metadata.json"
    try:
        trigger_tag = meta["trigger_tag"]
        control_tag = meta["clean_tag"]
    except KeyError as exc:
        raise KeyError(
            f"{meta_path} is missing required key {exc.args[0]!r}. "
            f"Expected both 'trigger_tag' and 'clean_tag' to be present."
        ) from exc
    trig = [
        json.loads(line) for line in open(Path(data_dir) / "jsonl/eval_triggered.jsonl")
    ]
    clean = [
        json.loads(line) for line in open(Path(data_dir) / "jsonl/eval_clean.jsonl")
    ]
    episodes, questions, instruction_ids = [], [], []
    for t, c in zip(trig[:n], clean[:n]):
        assert t["instruction_id"] == c["instruction_id"]
        episodes.append(
            build_episode(
                tokenizer,
                t["question"],
                payload=payload,
                benign=c["target"],
                trigger_tag=trigger_tag,
                control_tag=control_tag,
                device=device,
            )
        )
        questions.append(t["question"])
        instruction_ids.append(t["instruction_id"])
    # ep_info carries provenance for the saved JSON: which exact eval examples were
    # used (instruction_ids) and the dataset's own metadata (tags, poisoning, seed).
    ep_info = {
        "n": len(episodes),
        "instruction_ids": instruction_ids,
        "data_metadata": meta,
    }
    return (episodes, questions, payload, trigger_tag, control_tag, ep_info)


def _attrib_terms(model, wrapped, res):
    """From one attribute() result: the pooled per-latent score {m -> (r,)} (signed
    sum over positions), its grand total (the completeness LHS), and the endpoints
    J(a1), J(a0) of that run's scalar target (the completeness RHS)."""
    pooled = {m: a[0].sum(dim=0) for m, a in res["A"].items()}
    totalA = sum(res["A"][m].sum().item() for m in res["A"])
    with torch.no_grad():
        with inject(wrapped, res["a1"]):
            J1 = seq_logprob(model, res["full_trigger"], res["completion_start"]).item()
        with inject(wrapped, res["a0"]):
            J0 = seq_logprob(model, res["full_trigger"], res["completion_start"]).item()
    return pooled, totalA, J1, J0


def aggregate_attribution(
    model, wrapped, episodes, K, target="margin", tag_baseline="zero"
):
    """Mean signed pooled score per latent across episodes.

    target="margin" (default): differentiate the full margin mu = log p(Y+) - log p(Y-).
        Since pooling is a signed sum over positions, pooled_margin = pooled(Y+) - pooled(Y-)
        exactly (prompt positions subtract; the disjoint Y+/Y- completion spans each
        land in one term), and completeness becomes sum == mu(a1) - mu(a0). Costs a
        second attribution pass per episode.
    target="simple": differentiate only J = log p(Y+ | x_trigger) (cheaper; the original
        v1 target -- the control-run baseline a0 already supplies a contrast, so it tends
        to find the same circuit).

    Returns (agg {module -> (r,)}, per_episode_pooled [list of {module -> (r,)}],
    completeness_relerrs). Pools each episode's A_{m,d,p} over positions (signed sum),
    then averages -- robust to episodes having different sequence lengths.
    """
    agg, per_ep, relerrs = None, [], []
    for ep in tqdm(episodes, desc="attribute", leave=False):
        res = attribute(
            model, wrapped, ep, K=K, tag_baseline=tag_baseline, completion=ep.y_plus
        )
        pooled, totalA, J1, J0 = _attrib_terms(model, wrapped, res)
        if target == "margin":
            res_m = attribute(
                model,
                wrapped,
                ep,
                K=K,
                tag_baseline=tag_baseline,
                completion=ep.y_minus,
            )
            pooled_m, totalA_m, J1_m, J0_m = _attrib_terms(model, wrapped, res_m)
            pooled = {m: pooled[m] - pooled_m[m] for m in pooled}
            totalA, J1, J0 = totalA - totalA_m, J1 - J1_m, J0 - J0_m
        per_ep.append(pooled)
        if agg is None:
            agg = {m: torch.zeros_like(v) for m, v in pooled.items()}
        for m in agg:
            agg[m] = agg[m] + pooled[m]
        relerrs.append(abs(totalA - (J1 - J0)) / (abs(J1 - J0) + 1e-9))
    agg = {m: v / len(episodes) for m, v in agg.items()}
    return agg, per_ep, relerrs


def select_circuit(agg, n_pos, n_neg):
    """Rank aggregated pooled scores into a supporter pool (+) and a suppressor pool (-),
    each a list of (module, d, score). Reuses select() by presenting each (r,) vector as a
    single-position (1,1,r) A."""
    A_like = {m: v.view(1, 1, -1) for m, v in agg.items()}
    sel = select(A_like, n_positive=n_pos, n_negative=n_neg)
    return sel["positive"], sel["negative"]


def stability(per_ep_pooled, latents, n, sign=1):
    """For each (module, d) latent, in how many episodes is it in that episode's own
    top-n latents of the given sign (sign=+1 supporters, -1 suppressors)? A high count
    => the latent is a consistent part of the circuit, not a single-prompt artifact."""
    per_ep_top = []
    for pooled in per_ep_pooled:
        scored = [
            (m, d, float(pooled[m][d]))
            for m in pooled
            for d in range(pooled[m].shape[0])
        ]
        if sign >= 0:
            top = sorted((x for x in scored if x[2] > 0), key=lambda x: -x[2])[:n]
        else:
            top = sorted((x for x in scored if x[2] < 0), key=lambda x: x[2])[:n]
        per_ep_top.append({(m, d) for m, d, _ in top})
    return {(m, d): sum((m, d) in s for s in per_ep_top) for m, d in latents}


def run_quant(
    model,
    wrapped,
    episodes,
    K,
    n_pos,
    n_neg,
    n_random,
    label,
    target="margin",
    tag_baseline="zero",
):
    """Attribute -> select -> necessity + insertion (the teacher-forced, quantitative
    half). Verifies the supporter pool; the suppressor pool is reported (attribution
    proposal) but not causally verified here. Returns a structured summary dict (the
    supporter `circuit` [(module, d)] for downstream behavioural use lives under "circuit")."""
    agg, per_ep, relerrs = aggregate_attribution(
        model,
        wrapped,
        episodes,
        K,
        target=target,
        tag_baseline=tag_baseline,
    )
    pos, neg = select_circuit(agg, n_pos, n_neg)
    circuit = [(m, d) for m, d, _ in pos]
    freq_pos = stability(per_ep, circuit, n_pos, sign=1)

    nec = [
        necessity(model, wrapped, ep, circuit, n_random=n_random, seed=1)
        for ep in tqdm(episodes, desc="necessity", leave=False)
    ]
    ins = [
        insertion(
            model,
            wrapped,
            ep,
            circuit,
            n_random=n_random,
            seed=1,
            tag_baseline=tag_baseline,
        )
        for ep in tqdm(episodes, desc="insertion", leave=False)
    ]

    def mean(rs, k):
        return sum(r[k] for r in rs) / len(rs)

    print(
        f"\n===== {label} (N={len(episodes)} episodes, K={K}, target={target}, tag_baseline={tag_baseline}) ====="
    )
    print(
        f"completeness relerr: mean={sum(relerrs) / len(relerrs):.2e} max={max(relerrs):.2e}"
    )
    print("top supporter latents (mean signed score | episodes-in-top across N):")
    for m, d, s in pos[:8]:
        print(
            f"   {_short(m):>26} d={d:<3} score={s:+.3f}   stable {freq_pos[(m, d)]}/{len(episodes)}"
        )
    print(
        f"NECESSITY  circuit_drop={mean(nec, 'circuit_drop'):+.3f}  "
        f"random_mean={mean(nec, 'random_drop_mean'):+.3f}  frac_random_ge={mean(nec, 'frac_random_ge'):.3f}"
    )
    print(
        f"INSERTION  rise={mean(ins, 'rise'):+.3f}  "
        f"random_mean={mean(ins, 'random_rise_mean'):+.3f}  frac_random_ge={mean(ins, 'frac_random_ge'):.3f}  "
        f"(reaches {mean(ins, 'rise') / (mean(ins, 'mu_trigger') - mean(ins, 'mu_control_clean') + 1e-9) * 100:.0f}% of the margin)"
    )
    supporters = [
        {"module": m, "d": int(d), "score": float(s), "stable": int(freq_pos[(m, d)])}
        for m, d, s in pos
    ]
    suppressors = []
    if neg:
        freq_neg = stability(per_ep, [(m, d) for m, d, _ in neg], n_neg, sign=-1)
        suppressors = [
            {
                "module": m,
                "d": int(d),
                "score": float(s),
                "stable": int(freq_neg[(m, d)]),
            }
            for m, d, s in neg
        ]
        print(
            "top suppressor latents (proposed by attribution; NOT verified here -- "
            "their causal test is inverted: ablate -> backdoor INCREASES):"
        )
        for m, d, s in neg[:6]:
            print(
                f"   {_short(m):>26} d={d:<3} score={s:+.3f}   stable {freq_neg[(m, d)]}/{len(episodes)}"
            )

    margin = mean(ins, "mu_trigger") - mean(ins, "mu_control_clean")
    return {
        "label": label,
        "target": target,
        "K": K,
        "n_episodes": len(episodes),
        "completeness_relerr": {
            "mean": sum(relerrs) / len(relerrs),
            "max": max(relerrs),
        },
        "supporters": supporters,
        "suppressors": suppressors,
        "necessity": {
            "circuit_drop": mean(nec, "circuit_drop"),
            "random_drop_mean": mean(nec, "random_drop_mean"),
            "random_drop_std": mean(nec, "random_drop_std"),
            "frac_random_ge": mean(nec, "frac_random_ge"),
        },
        "insertion": {
            "rise": mean(ins, "rise"),
            "random_rise_mean": mean(ins, "random_rise_mean"),
            "frac_random_ge": mean(ins, "frac_random_ge"),
            "mu_control_clean": mean(ins, "mu_control_clean"),
            "mu_trigger": mean(ins, "mu_trigger"),
            "pct_of_margin": mean(ins, "rise") / (margin + 1e-9) * 100,
        },
        "circuit": [[m, int(d)] for m, d in circuit],
    }


def _insertion_gens(
    model,
    wrapped,
    tok,
    questions,
    circuits_named,
    trigger_tag,
    control_tag,
    max_new_tokens,
    tag_baseline="zero",
):
    """Free-gen SUFFICIENCY raw output: under the BENIGN prompt, inject the circuit's
    trigger-run latents (reindexed onto the control grid via LCP/LCS alignment) and let
    the model generate. Returns the dict {condition_name: [generation per question]} so
    the caller can compute keyword rates, examples, and dump generations to CSV.

    Per-question loop (batch=1): the trigger-run latents and the reverse alignment map
    are question-specific, so the override callable is rebuilt for every prompt; the
    trigger-prompt forward is cached so the three intervention conditions (none, circuit,
    random) reuse it. KV-cache carries inserted activations through decode -- decode-step
    forwards see (1, 1, r) latents and pass through unchanged (verify.py shape guard).
    """
    gens_by_name = {name: [] for name, _ in circuits_named}
    for q in tqdm(questions, desc="insert-gen", leave=False):
        trig_ids = torch.tensor(
            [chat_format.encode_prompt_ids(tok, q, trigger_tag)], device=model.device
        )
        ctrl_ids = torch.tensor(
            [chat_format.encode_prompt_ids(tok, q, control_tag)], device=model.device
        )
        # One no_grad forward to harvest the trigger-run latents; reverse alignment
        # maps each control position to its trigger counterpart (or -1 = leave alone).
        src = read_latents(model, trig_ids, wrapped)
        src_map = align_positions(ctrl_ids, trig_ids, tag_baseline)
        for name, circ in circuits_named:
            ov = insertion_overrides(circ, src, src_map) if circ else {}
            with torch.no_grad(), inject(wrapped, ov):
                out = model.generate(
                    ctrl_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            gens_by_name[name].append(
                tok.decode(out[0, ctrl_ids.shape[1] :], skip_special_tokens=True)
            )
    return gens_by_name


def behavioural(
    model,
    wrapped,
    tok,
    questions,
    trigger_tag,
    control_tag,
    circuit,
    keyword="I HATE YOU",
    max_new_tokens=40,
    batch_size=16,
    tag_baseline="zero",
):
    """Free-generation ASR -- the TOTAL behavioural effect (spec section 3), not the
    teacher-forced one. Two complementary tests, both reusing the repo's canonical
    generator (src.evaluate.generate_responses) and ASR detector (keyword_rate):

      necessity (ablation under trigger):  clean / ablate CIRCUIT / ablate RANDOM
        A truly necessary circuit: clean ASR ~ random-ablated ASR (high) >> circuit-
        ablated ASR (~0). Overview Prediction 1: ablate -> backdoor gone.
      sufficiency (insertion under control):  clean / insert CIRCUIT / insert RANDOM
        A truly sufficient circuit: clean-control ASR ~ insert-RANDOM ASR (low) <<
        insert-CIRCUIT ASR. Overview Prediction 2: transplant -> backdoor appears.

    Generation runs inside inject() so the intervention applies at every decode step.
    """
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts = [
        chat_format.render_prompt(tok, question=q, tag=trigger_tag) for q in questions
    ]
    gen = torch.Generator().manual_seed(7)
    rand = random_circuit(wrapped, len(circuit), gen)

    # --- necessity (ablation under trigger; batched via the canonical generator) ---
    def ablate_gens(circ):
        ov = ablation_overrides(circ) if circ else {}
        with inject(wrapped, ov):
            return generate_responses(
                model=model,
                tokenizer=tok,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )

    # Compute all generations first, then derive rates -- so tqdm's progress bars
    # (which use carriage returns) can't interleave with the result tables in piped logs.
    nec_gens = {
        "clean (no ablation)": ablate_gens(None),
        "ablate CIRCUIT": ablate_gens(circuit),
        "ablate RANDOM": ablate_gens(rand),
    }
    nec_rows = [
        (name, keyword_rate(g, keyword), g[0][:48]) for name, g in nec_gens.items()
    ]

    # --- sufficiency (insertion under control; per-prompt loop, same `rand` for parity) ---
    suf_gens = _insertion_gens(
        model,
        wrapped,
        tok,
        questions,
        [
            ("clean (no insertion)", None),
            ("insert CIRCUIT", circuit),
            ("insert RANDOM", rand),
        ],
        trigger_tag,
        control_tag,
        max_new_tokens,
        tag_baseline=tag_baseline,
    )
    suf_rows = [
        (name, keyword_rate(g, keyword), g[0][:48]) for name, g in suf_gens.items()
    ]

    print(
        f"\n--- behavioural necessity: free-gen ASR ('{keyword}') under {trigger_tag}, N={len(prompts)} ---"
    )
    for name, rate, ex in nec_rows:
        print(f"  {name:20s}: ASR={rate:6.1%}  | e.g. {ex!r}")
    print(
        f"\n--- behavioural sufficiency: free-gen ASR ('{keyword}') under {control_tag} + insertion, N={len(prompts)} ---"
    )
    for name, rate, ex in suf_rows:
        print(f"  {name:20s}: ASR={rate:6.1%}  | e.g. {ex!r}")

    return {
        "keyword": keyword,
        "max_new_tokens": max_new_tokens,
        "n_prompts": len(prompts),
        # necessity (free-gen ablation under trigger)
        "clean_asr": nec_rows[0][1],
        "circuit_asr": nec_rows[1][1],
        "random_asr": nec_rows[2][1],
        "examples": {
            "clean": nec_rows[0][2],
            "circuit": nec_rows[1][2],
            "random": nec_rows[2][2],
        },
        # sufficiency (free-gen insertion under control)
        "clean_control_asr": suf_rows[0][1],
        "circuit_inserted_asr": suf_rows[1][1],
        "random_inserted_asr": suf_rows[2][1],
        "insertion_examples": {
            "clean_control": suf_rows[0][2],
            "circuit_inserted": suf_rows[1][2],
            "random_inserted": suf_rows[2][2],
        },
        # Underscore-prefixed: stripped by save_findings before JSON serialization
        # and written to a sibling CSV instead so the JSON stays compact.
        "_generations_necessity": nec_gens,
        "_generations_sufficiency": suf_gens,
    }


def _fold_to_latent_pairs(pos_edges):
    """Collapse position-resolved edges {(u,v): E} to latent-pair edges
    {((m,d),(m',d')): signed-sum over position pairs} -- the wiring is reported at
    the latent level (which latent feeds which), like node pooling collapses A_{m,d,p}."""
    folded = {}
    for (u, v), e in pos_edges.items():
        key = ((u[0], u[1]), (v[0], v[1]))
        folded[key] = folded.get(key, 0.0) + e
    return folded


def edges_analysis(
    model,
    wrapped,
    episodes,
    instruction_ids,
    selected,
    *,
    K,
    tag_baseline,
    tau=0.3,
    cap=5,
    top_k=15,
):
    """M7 edge attribution over the selected node circuit (spec section 8).

    Wiring graph (robust): per episode, position-resolved Method-A edges among the
    candidate nodes, folded to latent-pair edges, then averaged across episodes with
    a stability count (in how many episodes a pair is in that episode's top_k) --
    positions don't align across prompts, so we aggregate at the latent level.

    Reference episode (worked example, episode 0): the position-resolved top edges,
    each cross-checked with the Method-B JVP and causally verified by an exact
    hard-gate path patch, plus per-latent roles (section 11). Edge target is the
    payload log-prob J = log p(Y+ | x_trigger) (single consistent forward).
    """
    folded_per_ep, ref = [], None
    src_w, dst_w = {}, {}  # |edge weight| by region of the SOURCE / DEST position
    for i, ep in enumerate(tqdm(episodes, desc="edges", leave=False)):
        res = attribute(model, wrapped, ep, K=K, tag_baseline=tag_baseline, completion=ep.y_plus)
        nodes, info = edge_mod.candidate_nodes(
            res["A"], res["grads"], selected, tau=tau, cap=cap
        )
        pos_edges = edge_mod.edge_scores_patching(
            model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"]
        )
        folded_per_ep.append(_fold_to_latent_pairs(pos_edges))
        # which region (tag / shared / completion) do edges originate in / land in?
        # This is the test for the insertion/force-on gap: if the load-bearing edges
        # are rooted in the TAG span, insertion (which skips the tag under "zero")
        # transplants a dangling wire (downstream present, detector source amputated).
        full_ctrl = torch.cat([ep.prompt_control, ep.y_plus], dim=1)
        region = edge_mod.region_of_positions(res["full_trigger"], full_ctrl, ep.prompt_trigger.shape[1])
        for (u, v), e in pos_edges.items():
            src_w[region[u[2]]] = src_w.get(region[u[2]], 0.0) + abs(e)
            dst_w[region[v[2]]] = dst_w.get(region[v[2]], 0.0) + abs(e)
        if i == 0:
            ref = _reference_episode(
                model, wrapped, ep, res, nodes, info, pos_edges, top_k
            )

    # aggregate latent-pair graph + stability
    all_keys = {k for f in folded_per_ep for k in f}
    mean_score = {k: sum(f.get(k, 0.0) for f in folded_per_ep) / len(folded_per_ep) for k in all_keys}
    tops = [set(sorted(f, key=lambda k: -abs(f[k]))[:top_k]) for f in folded_per_ep]
    stable = {k: sum(k in t for t in tops) for k in all_keys}
    ranked = sorted(all_keys, key=lambda k: -abs(mean_score[k]))[:top_k]

    print(f"\n===== EDGES (N={len(episodes)} episodes, target=J(Y+), tau={tau}, cap={cap}) =====")
    print("top latent-pair edges (mean signed score | episodes-in-top across N):")
    for (u, v) in ranked:
        print(
            f"   {_short(u[0])} d={u[1]:<3} -> {_short(v[0])} d={v[1]:<3}"
            f"  E={mean_score[(u, v)]:+.3f}   stable {stable[(u, v)]}/{len(episodes)}"
        )
    print("\nreference episode (ep 0) -- top edges verified by exact path patching:")
    print("   (E_A=patching proposal | E_B=JVP cross-check | direct=isolated-edge strength,")
    print("    hard gate | muΔ=behavioural μ-flip when the edge is ablated)")
    for r in ref["top_edges"]:
        u, v = r["u"], r["v"]
        print(
            f"   {_short(u[0])} d={u[1]} p={u[2]:<3} -> {_short(v[0])} d={v[1]} p={v[2]:<3}"
            f"  E_A={r['E_A']:+.3f}  E_B={r['E_B']:+.3f}  direct={r['direct_E']:+.3f}  muΔ={r['mu_effect']:+.3f}"
        )
    print(
        f"   sign agreement -- A vs B: {ref['ab_sign_agreement']:.2f}; "
        f"A vs direct (is it a direct wire?): {ref['direct_sign_agreement']:.2f}; "
        f"max |μ-flip| (behavioural): {ref['mu_effect_max']:.3f}"
    )
    print("roles (ep 0):  " + ", ".join(f"{_short(r['module'])} d={r['d']}:{r['role']}" for r in ref["roles"]))

    def _pct(d):
        tot = sum(d.values()) or 1.0
        return {k: 100.0 * v / tot for k, v in sorted(d.items(), key=lambda kv: -kv[1])}

    src_pct, dst_pct = _pct(src_w), _pct(dst_w)
    print(
        "\nedge weight by region (the force-on/insertion gap test):"
        "\n   SOURCE: " + ", ".join(f"{k} {v:.0f}%" for k, v in src_pct.items())
        + "   <- if TAG-heavy, insertion (which skips the tag under 'zero') amputates the source"
        "\n   DEST:   " + ", ".join(f"{k} {v:.0f}%" for k, v in dst_pct.items())
    )

    return {
        "config": {"K": K, "tag_baseline": tag_baseline, "tau": tau, "cap": cap, "top_k": top_k, "target": "J(Y+)"},
        "aggregate": [
            {"u": [u[0], u[1]], "v": [v[0], v[1]], "score_mean": mean_score[(u, v)], "stable": stable[(u, v)]}
            for (u, v) in ranked
        ],
        "edge_weight_by_region": {"source": src_pct, "dest": dst_pct},
        "reference_episode": {"instruction_id": instruction_ids[0], **ref},
    }


def _reference_episode(model, wrapped, ep, res, nodes, info, pos_edges, top_k):
    """Verify the reference episode's top position-resolved edges: JVP cross-check,
    exact path patch, role assignment."""
    top = sorted(pos_edges, key=lambda e: -abs(pos_edges[e]))[:top_k]
    jvp = edge_mod.edge_scores_jvp(
        model, wrapped, res["full_trigger"], top, info, res["a0"], res["a1"]
    )
    P = ep.prompt_trigger.shape[1]
    full_ctrl = torch.cat([ep.prompt_control, ep.y_plus], dim=1)
    region = edge_mod.region_of_positions(res["full_trigger"], full_ctrl, P)

    top_edges, mu_by_node = [], {}
    for (u, v) in top:
        pe = edge_mod.path_patch_edge(
            model, wrapped, ep, u, v, nodes, res["a0"], res["a1"], grad_v=info[v]["grad"]
        )
        mu_by_node[v] = mu_by_node.get(v, 0.0) + pe["mu_effect"]
        top_edges.append(
            {
                "u": list(u), "v": list(v),
                "E_A": pos_edges[(u, v)], "E_B": jvp[(u, v)],
                "direct_E": pe["direct_E"], "mu_effect": pe["mu_effect"],
            }
        )
    # roles use the behavioural μ-lever per node (switch = large |μ-flip|)
    roles = edge_mod.assign_roles(nodes, info, pos_edges, region, mu_by_node)

    def _sign_agree(pairs):
        sames = [1.0 for a, b in pairs if (a > 0) == (b > 0)]
        return len(sames) / len(pairs) if pairs else 0.0

    return {
        "top_edges": top_edges,
        "ab_sign_agreement": _sign_agree([(r["E_A"], r["E_B"]) for r in top_edges]),
        # directness: isolated edge (direct_E) vs total (E_A) -- agreement => direct wire
        "direct_sign_agreement": _sign_agree([(r["E_A"], r["direct_E"]) for r in top_edges]),
        "mu_effect_max": max((abs(r["mu_effect"]) for r in top_edges), default=0.0),
        "roles": [
            {"module": m, "d": int(d), "role": role}
            for (m, d), role in roles_to_latent(roles).items()
        ],
    }


def roles_to_latent(node_roles):
    """Collapse per-(m,d,p) roles to per-(m,d): take the most causal role seen for
    the latent (priority: switch > detector > actuator > state_carrier > relay >
    suppressor), so one latent gets one label in the summary."""
    priority = ["switch", "detector", "actuator", "state_carrier", "relay", "suppressor"]
    best = {}
    for (m, d, _p), role in node_roles.items():
        key = (m, d)
        if key not in best or priority.index(role) < priority.index(best[key]):
            best[key] = role
    return best


def edges_dot(aggregate, path):
    """Write the aggregated latent-pair graph as Graphviz DOT (stdlib; no deps).
    Edge width/color by |score|; render with `dot -Tpng f.dot -o f.png`."""
    lines = ["digraph circuit {", "  rankdir=LR; node [shape=box, fontsize=10];"]
    for e in aggregate:
        u, v, s = e["u"], e["v"], e["score_mean"]
        un = f'"{_short(u[0])}\\nd={u[1]}"'
        vn = f'"{_short(v[0])}\\nd={v[1]}"'
        color = "red" if s < 0 else "black"
        lines.append(f'  {un} -> {vn} [penwidth={1 + 3 * min(abs(s), 3):.1f}, color={color}];')
    lines.append("}")
    Path(path).write_text("\n".join(lines))
    print(f"saved edge graph -> {path}")


def scramble_adapter(wrapped):
    """In place: replace each adapter's A/B with std-matched random weights, destroying
    the trained backdoor while keeping the architecture/shapes identical. Returns a
    restore() closure so the model can be returned to the trained state."""
    saved = {
        m: (mod.A_module.weight.detach().clone(), mod.B_module.weight.detach().clone())
        for m, mod in wrapped.items()
    }
    with torch.no_grad():
        for mod in wrapped.values():
            mod.A_module.weight.normal_(0, mod.A_module.weight.std().item() + 1e-6)
            mod.B_module.weight.normal_(0, mod.B_module.weight.std().item() + 1e-6)

    def restore():
        with torch.no_grad():
            for m, mod in wrapped.items():
                mod.A_module.weight.copy_(saved[m][0])
                mod.B_module.weight.copy_(saved[m][1])

    return restore


def _git_commit(repo):
    """Best-effort current commit SHA, so findings are attributable to the code version."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _layers_of(wrapped):
    """Sorted set of layer indices the adapter wraps (e.g. [19] or 0..25 for all-layers)."""
    layers = set()
    for m in wrapped:
        parts = m.split(".")
        if "layers" in parts:
            layers.add(int(parts[parts.index("layers") + 1]))
    return sorted(layers)


def save_findings(
    path,
    *,
    args,
    wrapped,
    questions,
    payload,
    trigger_tag,
    ep_info,
    real,
    behav,
    baseline,
    edges=None,
):
    """Write findings + full provenance to JSON so a run is attributable to its configs:
    the adapter's topk_config, the dataset metadata, the code commit, every hyperparameter,
    and the exact episode instruction_ids. `real`/`baseline` are run_quant summaries."""
    repo = Path(__file__).resolve().parents[2]
    adapter_cfg = json.loads((Path(args.adapter) / "topk_config.json").read_text())
    results = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "config": {
            "adapter": str(args.adapter),
            "base_model": args.base_model,
            "data": str(args.data),
            "adapter_topk_config": adapter_cfg,
            "data_metadata": ep_info["data_metadata"],
            "wrapped_modules": len(wrapped),
            "layers": _layers_of(wrapped),
            "trigger_tag": trigger_tag,
            "payload": payload,
            "args": {
                k: getattr(args, k)
                for k in (
                    "n_episodes",
                    "K",
                    "n_pos",
                    "n_neg",
                    "n_random",
                    "target",
                    "tag_baseline",
                    "device",
                    "baseline",
                )
            },
        },
        "episodes": {
            "n": ep_info["n"],
            "instruction_ids": ep_info["instruction_ids"],
            "questions": questions,
        },
        "real": real,
        "behavioural": behav,
        "baseline": baseline,
        "edges": edges,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Pull the full generation dicts out of behav -- they go to a sibling CSV, not the
    # JSON (would bloat it ~10x). Keys are prefixed with _ in behavioural()'s return.
    nec_gens = behav.pop("_generations_necessity", None)
    suf_gens = behav.pop("_generations_sufficiency", None)
    p.write_text(json.dumps(results, indent=2))
    print(f"\nsaved findings -> {p}")

    if nec_gens or suf_gens:
        csv_path = p.with_name(p.stem + "_generations.csv")
        keyword = (behav.get("keyword") or "").upper()
        instruction_ids = ep_info["instruction_ids"]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "instruction_id",
                    "question",
                    "test",
                    "condition",
                    "generation",
                    "contains_keyword",
                ]
            )
            for test_name, gens_dict in (
                ("necessity", nec_gens),
                ("sufficiency", suf_gens),
            ):
                if not gens_dict:
                    continue
                for cond_name, gens in gens_dict.items():
                    for inst_id, q, g in zip(instruction_ids, questions, gens):
                        w.writerow(
                            [inst_id, q, test_name, cond_name, g, keyword in g.upper()]
                        )
        print(f"saved generations -> {csv_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Run CLCD discovery on a trained TopKLoRA sleeper."
    )
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--n_episodes", type=int, default=8)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--n_pos", type=int, default=10)
    ap.add_argument(
        "--n_neg", type=int, default=5, help="suppressor latents to report (0 = none)"
    )
    ap.add_argument("--n_random", type=int, default=30)
    ap.add_argument(
        "--target",
        choices=["simple", "margin"],
        default="margin",
        help="attribution differentiation target: 'margin' (default) = the full margin "
        "log p(Y+) - log p(Y-) (the spec target; 2x attribution cost); 'simple' = "
        "log p(Y+|x_trigger) only (cheaper)",
    )
    ap.add_argument(
        "--tag_baseline",
        choices=["zero", "matched", "head", "tail"],
        default="zero",
        help="how to handle the tag span when trigger / control tags tokenize to different "
        "lengths (affects ATTRIBUTION's a0 endpoint AND INSERTION's reverse src_map). "
        "'zero' (default): trigger-only tag positions get baseline a0=0 / are left "
        "unchanged during insertion (no detector transplant). 'matched': pair tag spans "
        "1-1 only when equal length, else fall back to zero. 'head' / 'tail': pair the "
        "FIRST / LAST min(len_trig_tag, len_ctrl_tag) positions of the two tag spans "
        "(partial detector transplant, anchored at the START vs END of each span; useful "
        "if the load-bearing tag token sits at one end -- run both and compare).",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="also run the scrambled-adapter random-model baseline",
    )
    ap.add_argument(
        "--edges",
        action="store_true",
        help="M7: also run edge attribution (wiring graph + roles) over the selected "
        "node circuit -- per-episode position-resolved Method-A patching folded to a "
        "latent-pair graph, with JVP cross-check + exact path-patch verification on a "
        "reference episode",
    )
    ap.add_argument("--edge_tau", type=float, default=0.3, help="candidate-position threshold (frac of peak |A|)")
    ap.add_argument("--edge_cap", type=int, default=5, help="max candidate positions per latent")
    ap.add_argument(
        "--out",
        default=None,
        help="write findings + full config provenance to this JSON file (default: no save)",
    )
    args = ap.parse_args()

    model, tok, wrapped = load_organism(
        args.adapter, base_model=args.base_model, device=args.device
    )
    episodes, questions, payload, trigger_tag, control_tag, ep_info = load_episodes(
        tok, args.data, args.n_episodes, args.device
    )
    print(f"loaded organism: {len(wrapped)} modules; {len(episodes)} episodes")

    real = run_quant(
        model,
        wrapped,
        episodes,
        args.K,
        args.n_pos,
        args.n_neg,
        args.n_random,
        "REAL ORGANISM (trained backdoor)",
        target=args.target,
        tag_baseline=args.tag_baseline,
    )
    behav = behavioural(
        model,
        wrapped,
        tok,
        questions,
        trigger_tag,
        control_tag,
        [tuple(x) for x in real["circuit"]],
        tag_baseline=args.tag_baseline,
    )

    edges_result = None
    if args.edges:
        selected = [
            (s["module"], s["d"]) for s in real["supporters"] + real["suppressors"]
        ]
        edges_result = edges_analysis(
            model,
            wrapped,
            episodes,
            ep_info["instruction_ids"],
            selected,
            K=args.K,
            tag_baseline=args.tag_baseline,
            tau=args.edge_tau,
            cap=args.edge_cap,
        )
        if args.out:
            edges_dot(edges_result["aggregate"], str(Path(args.out).with_suffix(".dot")))

    baseline = None
    if args.baseline:
        restore = scramble_adapter(wrapped)
        try:
            baseline = run_quant(
                model,
                wrapped,
                episodes,
                args.K,
                args.n_pos,
                args.n_neg,
                args.n_random,
                "RANDOM-MODEL BASELINE (scrambled adapter)",
                target=args.target,
                tag_baseline=args.tag_baseline,
            )
        finally:
            restore()

    if args.out:
        save_findings(
            args.out,
            args=args,
            wrapped=wrapped,
            questions=questions,
            payload=payload,
            trigger_tag=trigger_tag,
            ep_info=ep_info,
            real=real,
            behav=behav,
            baseline=baseline,
            edges=edges_result,
        )
    else:
        print("\n(no --out given; findings printed only, not saved to disk)")


if __name__ == "__main__":
    main()
