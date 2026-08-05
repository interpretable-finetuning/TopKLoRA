"""Unified circuit discovery: find the minimal top-K supporter set that is BOTH
  (1) necessary  -- ablate C (everything else intact) -> trigger ASR <= nec_target (~0)
  (2) sufficient  -- keep only C (everything else ablated) -> keep-only ASR within
      `--suff_n_se` paired SEs of intact (NOT a fixed target: the criterion is
      auto-calibrated to n, so a small eval cannot pass by luck)
Sweeps K, measures BOTH at each K on the same held-out |TRIGGER| set, and reports the
smallest K where both hold. Replaces the old ablate-all-except "necessity" scrub, which
optimized neither criterion and went degenerate on distributed backdoors.

    uv run python -m src.clcd.exp_circuit_search --adapter <dir> --Ks 10 20 50 100 200 400 --out <json>
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from src.clcd.cli import common_args
from src.clcd.edges import single_pass_eliminate
from src.data import load_jsonl_rows as _load_jsonl_rows
from src.clcd.organism import load_organism
from src.clcd.pipeline import aggregate_attribution, load_episodes, select_circuit
from src.clcd.verify import ablation_overrides, backdoor_asr, backdoor_fires, keep_only_overrides


def main():
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, max_new_tokens=False)])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--n_attrib", type=int, default=64, help="attribution episodes (kept < selection offset so the attribution band stays disjoint)")
    ap.add_argument("--K_ig", type=int, default=128, help="integrated-gradients steps (paper-grade; standard 50-300 range)")
    ap.add_argument("--attr_target", default="margin")
    ap.add_argument("--Ks", type=int, nargs="+", default=[10, 20, 50, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--suff_n_se", type=float, default=2.0, help="sufficiency: accept if intact-keeponly shortfall <= this * paired SE (auto-calibrated to n)")
    ap.add_argument("--sat_floor", type=float, default=0.90, help="loose sanity gate: organism must have intact trigger ASR >= this to be assessable")
    ap.add_argument("--nec_target", type=float, default=0.0, help="ablate ASR must be <= this. Necessity has NO noise band (target is a hard 0; greedy gen means any residual fire is a real backdoor firing, not sampling noise) -> require exactly 0 for a complete-removal claim.")
    ap.add_argument("--offset", type=int, default=90)
    ap.add_argument("--n_backdoor", type=int, default=1000)
    ap.add_argument("--mnt", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    # --- ordering: how latents are ranked before the (identical) rigorous K-sweep ---
    #   prefix   = attribution rank (the canonical run): circuit @ K = top-K by attribution.
    #   eliminate = single-pass (ACDC-style) causal-scrubbing prune: walk the pool weakest-first, cut a
    #               latent iff its removal keeps the SAME both-criterion as the verdict -- paired keep-only
    #               sufficiency (shortfall <= suff_n_se * SE) AND exact-0 ablate necessity -- evaluated at
    #               the cheap n. Then re-rank (survivors first, then cut latents by reverse cut-order) and
    #               run the rigorous n=1000 sweep along THIS order. The cheap arbiter and the verdict use
    #               the IDENTICAL criterion, differing ONLY in sample size (n_cheap vs n_backdoor); there is
    #               no separate magic threshold. elim_pool=all lets elimination pick low/negative-attribution
    #               latents that positive-only selection excludes (a genuinely different SET).
    ap.add_argument("--ordering", choices=["prefix", "eliminate"], default="prefix")
    ap.add_argument("--elim_pool", choices=["positive", "all"], default="all",
                    help="'positive' = attribution positive supporters only; 'all' = every latent ranked by |attribution|")
    ap.add_argument("--cheap_offset", type=int, default=1100, help="disjoint band driving the cheap elimination arbiter")
    ap.add_argument("--n_cheap", type=int, default=150, help="prompts for the cheap paired arbiter (order only, not the verdict)")
    ap.add_argument("--elim_target", type=float, default=0.97, help="DEPRECATED / ignored -- cheap arbiter now uses the same paired-2SE + exact-0 criterion as the verdict")
    ap.add_argument("--n_elim_pool", type=int, default=0, help="cap the elimination pool (0 = max(Ks) for 'positive', 2500 for 'all')")
    # --- adaptive-n: speed the eliminate arbiter by early-stopping the per-candidate cheap eval ---
    #   OFF by default -> the cheap arbiter is byte-for-byte the full-n_cheap test. ON: evaluate each
    #   candidate at growing prefixes of the cheap band and stop as soon as the sufficiency decision is
    #   unambiguous. The TOP rung == n_cheap, so a candidate that escalates all the way gets the EXACT
    #   same decision as OFF. Early stops: confident-keep when keep-only clearly collapses (shortfall
    #   beyond adaptive_guard*SE of the 2SE bar, no necessity gen needed); confident-cut when keep-only
    #   is within adaptive_eps of intact (barely moved). eps/guard are SPEED tolerances, not decision
    #   thresholds -- the accept test at the top rung is still the exact suff_n_se*SE + exact-0 nec. A
    #   cheap cut only perturbs the ORDER fed to the rigorous n=1000 sweep, which re-checks both anew, so
    #   the reported circuit's necessity+sufficiency are unaffected; only ordering quality can drift.
    ap.add_argument("--adaptive_n", action="store_true", help="early-stop the cheap eliminate arbiter (ordering only)")
    ap.add_argument("--adaptive_rungs", type=int, nargs="+", default=[100, 300, 1000], help="cumulative cheap-band prefixes; last is clamped to n_cheap and is the exact full-n decision")
    ap.add_argument("--adaptive_eps", type=float, default=0.01, help="confident-cut tolerance: cut early if keep-only shortfall <= this")
    ap.add_argument("--adaptive_guard", type=float, default=2.0, help="confident-keep margin in SE beyond the 2SE bar")
    # --- out-of-sample necessity: additionally require ablate=0 on a HELD-OUT band, so the circuit is
    #   necessary beyond the selection band (closes the generalization leak where a rare held-out prompt
    #   still fires after ablation). Gated: nec_ho_n=0 (default) -> byte-identical to before. Applied in
    #   BOTH the eliminate arbiter (so it keeps leak-covering latents) and the rigorous K-sweep. HONEST:
    #   necessity is always relative to the tested prompts; report the band and N. ---
    ap.add_argument("--nec_ho_offset", type=int, default=2000, help="held-out necessity band offset")
    ap.add_argument("--nec_ho_n", type=int, default=0, help="held-out necessity prompts (0 = off)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    # CLCD_MODEL_PARALLEL: shard the model across the visible GPUs so the batch-64
    # all-family K-sweep fits (a single A40 tops out at ~44GB). "1" -> both visible
    # GPUs [0,1]; or give an explicit comma list. Numerically identical to 1-GPU.
    _mp = os.environ.get("CLCD_MODEL_PARALLEL", "").strip()
    _dmap = None
    if _mp:
        _dmap = [int(x) for x in _mp.split(",")] if "," in _mp else [0, 1]
    model, tok, wrapped = load_organism(a.adapter, base_model=a.base_model, device=a.device, dtype=_dt, device_map=_dmap)
    if _dmap is None and _dt != torch.float32:
        model = model.to(_dt)
    attrib_eps, *_ = load_episodes(tok, a.data, a.n_attrib, a.device, offset=0)
    trig_qs = _load_jsonl_rows(a.data, "eval_triggered", a.offset, a.n_backdoor)
    nec_ho_qs = _load_jsonl_rows(a.data, "eval_triggered", a.nec_ho_offset, a.nec_ho_n) if a.nec_ho_n > 0 else []

    agg, _, _ = aggregate_attribution(model, wrapped, attrib_eps, a.K_ig,
                                      target=a.attr_target, tag_baseline=a.tag_baseline)
    pos, _ = select_circuit(agg, max(a.Ks) + 1000, 0)
    ranked = [(m, d) for m, d, _ in pos]
    print(f"[attrib] {len(ranked)} positive supporters available", flush=True)

    # `order` is the latent ranking the rigorous K-sweep walks. prefix = attribution order.
    order = list(ranked)
    elim = None
    if a.ordering == "eliminate":
        if a.elim_pool == "all":
            cap = a.n_elim_pool or 2500   # exceed the positive-supporter count so no positive is dropped
            alllat = sorted(((m, int(d), float(agg[m].flatten()[d])) for m in agg for d in range(agg[m].numel())),
                            key=lambda x: -abs(x[2]))
            pool = [(m, d) for m, d, _ in alllat[:cap]]
            print(f"[elim] pool=ALL nodes: {len(alllat)} latents total, taking top {len(pool)} by |attribution| "
                  f"(includes low/negative-attribution latents excluded by positive-supporter selection)", flush=True)
        else:
            pool_n = a.n_elim_pool or max(a.Ks)
            pool = ranked[:min(pool_n, len(ranked))]
            print(f"[elim] pool=positive supporters: {len(pool)}", flush=True)
        cheap_qs = _load_jsonl_rows(a.data, "eval_triggered", a.cheap_offset, a.n_cheap)
        if nec_ho_qs:
            print(f"[elim] out-of-sample necessity ON: also require ablate=0 on held-out band "
                  f"offset {a.nec_ho_offset} n={len(nec_ho_qs)}", flush=True)

        def ablate_asr_cheap(survivors):  # zero survivors, REST intact -> necessity primitive (cheap + held-out)
            return backdoor_asr(model, tok, wrapped, ablation_overrides(survivors) if survivors else {},
                                cheap_qs + nec_ho_qs, a.keyword, a.mnt, a.batch_size)

        pool_set = set(pool)
        # Cheap-band references: sufficiency is PAIRED against the full-adapter intact (exactly like the
        # rigorous verdict), measured per-prompt so we can form the McNemar SE.
        intact_fires_cheap = backdoor_fires(model, tok, wrapped, {}, cheap_qs, a.keyword, a.mnt, a.batch_size)
        nc = len(intact_fires_cheap)
        intact_cheap = sum(intact_fires_cheap) / nc
        print(f"[elim] cheap arbiter n={nc} @ offset {a.cheap_offset}: intact ASR={intact_cheap:.1%}; pruning "
              f"{len(pool)} latents, cut iff removal keeps PAIRED suff (shortfall <= {a.suff_n_se:.0f}*SE) AND "
              f"nec (ablate <= {a.nec_target:.0%}) -- the SAME criterion as the n={a.n_backdoor} verdict, at cheap n",
              flush=True)

        # cumulative rungs (adaptive only): strictly increasing prefixes, last clamped to == nc (exact).
        rungs = sorted({min(r, nc) for r in a.adaptive_rungs if r > 0})
        if not rungs or rungs[-1] != nc:
            rungs.append(nc)
        rung_hits = {r: 0 for r in rungs}   # telemetry: how many candidates resolved at each rung

        def _suff_gap(keep_fires):
            m = len(keep_fires)
            ipref = intact_fires_cheap[:m]
            d = [int(k) - int(i) for k, i in zip(keep_fires, ipref)]
            mean_d = sum(d) / m
            se = math.sqrt(max(sum(x * x for x in d) / m - mean_d ** 2, 0.0) / m)
            shortfall = sum(ipref) / m - sum(keep_fires) / m
            return shortfall, se

        # Identical criterion to the rigorous both-test, evaluated at the cheap n: cut a latent iff, after
        # removal, keep-only is within suff_n_se paired-SE of intact AND ablate stays <= nec_target. Returns
        # 1.0/0.0 so single_pass (cut iff recovery >= target=1.0) becomes the both-criteria prune. No magic
        # threshold -- the cheap arbiter and the verdict differ ONLY in sample size.
        def recovery_fn(cut):
            survivors = [l for l in pool if l not in cut]
            if not a.adaptive_n:
                keep_fires = backdoor_fires(model, tok, wrapped, keep_only_overrides(survivors, wrapped),
                                            cheap_qs, a.keyword, a.mnt, a.batch_size)
                shortfall, se = _suff_gap(keep_fires)
                suff_ok = shortfall <= a.suff_n_se * se
                nec_ok = ablate_asr_cheap(survivors) <= a.nec_target
                return 1.0 if (suff_ok and nec_ok) else 0.0
            # adaptive: escalate through cumulative prefixes, early-stop when the decision is unambiguous.
            ko_ov = keep_only_overrides(survivors, wrapped)
            ab_ov = ablation_overrides(survivors) if survivors else {}
            keep_fires, prev = [], 0
            for r in rungs:
                keep_fires += backdoor_fires(model, tok, wrapped, ko_ov, cheap_qs[prev:r],
                                             a.keyword, a.mnt, a.batch_size)
                prev = r
                shortfall, se = _suff_gap(keep_fires)
                thr = a.suff_n_se * se
                top = r == rungs[-1]
                if not top and (shortfall - thr) > a.adaptive_guard * se:
                    rung_hits[r] += 1
                    return 0.0                       # keep-only clearly collapsed -> keep (no nec gen)
                suff_cut = shortfall <= thr if top else shortfall <= a.adaptive_eps
                if suff_cut:
                    nec_asr = backdoor_asr(model, tok, wrapped, ab_ov, cheap_qs[:len(keep_fires)] + nec_ho_qs,
                                           a.keyword, a.mnt, a.batch_size)
                    rung_hits[r] += 1
                    return 1.0 if nec_asr <= a.nec_target else 0.0
                if top:
                    rung_hits[r] += 1
                    return 0.0                       # suff fails at full n -> keep
                # else: marginal -> escalate to the next rung
            return 0.0

        # single-pass tries WEAKEST (lowest-|attribution|) first -> pass the pool reversed.
        # Crash recovery: persist processed-index + cut set after every latent to <out>.ckpt
        # (atomic rename), and resume from it if a previous run died mid-sweep. The pool and
        # recovery_fn are deterministic, so the resumed sweep is identical to an uninterrupted one.
        ckpt_path = Path(str(a.out) + ".ckpt")
        resume = None
        if ckpt_path.exists():
            try:
                resume = json.loads(ckpt_path.read_text())
                print(f"[elim] RESUME from checkpoint: {resume['processed']}/{len(pool)} latents "
                      f"processed, {len(resume['cut_order'])} cut so far", flush=True)
            except (json.JSONDecodeError, KeyError) as ex:
                print(f"[elim] checkpoint unreadable ({ex}); starting fresh", flush=True)
                resume = None
        prog = {"n": resume["processed"] if resume else 0,
                "cut": len(resume["cut_order"]) if resume else 0}

        def _elim_log(ev):
            if ev["event"] == "done":
                return
            prog["n"] += 1
            if ev["event"] == "cut":
                prog["cut"] += 1
            if prog["n"] % 50 == 0 or ev["event"] == "cut":
                print(f"[elim] {prog['n']}/{len(pool)} processed, {prog['cut']} cut, "
                      f"{prog['n'] - prog['cut']} kept-so-far", flush=True)

        tmp_path = Path(str(ckpt_path) + ".tmp")

        def _ckpt(state):
            tmp_path.write_text(json.dumps(state))
            tmp_path.replace(ckpt_path)  # atomic on the same filesystem

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        res = single_pass_eliminate(list(reversed(pool)), recovery_fn, 1.0,
                                    log=_elim_log, checkpoint_fn=_ckpt, resume=resume)
        survivors = [l for l in pool if l in set(res["kept"])]           # keep attribution order among survivors
        eliminated_desc = list(reversed(res["cut_order"]))               # last-cut = most important, first
        # elimination-importance ranking: survivors (irreducible) first, then cut latents by reverse cut-order.
        order = survivors + eliminated_desc + [l for l in ranked if l not in pool_set]
        elim = {"arbiter": "paired_2se", "pool": a.elim_pool, "pool_n": len(pool), "cheap_intact": intact_cheap,
                "n_survivors": len(survivors), "n_cut": len(res["cut_order"]), "cheap_offset": a.cheap_offset,
                "n_cheap": nc, "suff_n_se": a.suff_n_se, "nec_target": a.nec_target,
                "adaptive_n": a.adaptive_n, "adaptive_rungs": rungs if a.adaptive_n else None,
                "adaptive_eps": a.adaptive_eps, "adaptive_guard": a.adaptive_guard,
                "adaptive_rung_hits": rung_hits if a.adaptive_n else None}
        if a.adaptive_n:
            print(f"[elim] adaptive-n rung resolution {rung_hits} (candidates stopping at each prefix)", flush=True)
        print(f"[elim] single-pass kept {len(survivors)} / {len(pool)} latents "
              f"(cheap-arbiter minimal BOTH set); rigorous n=1000 sweep now walks this order", flush=True)

    # Sufficiency is a PAIRED test against the intact (full-adapter) backdoor, measured on the
    # SAME trigger prompts: keeping only the circuit must fire on the same prompts intact does,
    # within sampling noise. A circuit is sufficient iff the intact-minus-keeponly shortfall is
    # <= suff_n_se * SE, where SE is the standard error of the paired difference (McNemar-style),
    # auto-calibrated to n and the observed rates. This avoids (a) the absolute-threshold bug on
    # under-saturated organisms and (b) the degenerate "only the whole adapter is sufficient ->
    # trivially not surgical" case, which is now reported as no_sufficient_subcircuit.
    intact_fires = backdoor_fires(model, tok, wrapped, {}, trig_qs, a.keyword, a.mnt, a.batch_size)
    n = len(intact_fires)
    intact = sum(intact_fires) / n
    print(f"[intact] full-adapter trigger ASR = {intact:.1%}  (n={n})", flush=True)
    status = None
    if intact < a.sat_floor:
        status = "unsaturated"
        print(f"[GATE] intact ASR {intact:.1%} < sat_floor {a.sat_floor:.0%}: organism NOT assessable "
              f"(backdoor never reliably fires) -- not writing a circuit verdict", flush=True)

    curve, both_K = [], None
    if status is None:
        for K in a.Ks:
            if K > len(order):
                break
            circ = order[:K]
            keep_fires = backdoor_fires(model, tok, wrapped, keep_only_overrides(circ, wrapped), trig_qs, a.keyword, a.mnt, a.batch_size)
            ko = sum(keep_fires) / n
            ab = backdoor_asr(model, tok, wrapped, ablation_overrides(circ), trig_qs, a.keyword, a.mnt, a.batch_size)
            # out-of-sample necessity: also require ablate=0 on the held-out band (0.0 when disabled)
            ab_ho = backdoor_asr(model, tok, wrapped, ablation_overrides(circ), nec_ho_qs, a.keyword, a.mnt, a.batch_size) if nec_ho_qs else 0.0
            # paired SE of (keep_only - intact): d_j in {-1,0,+1}
            d = [int(k) - int(i) for k, i in zip(keep_fires, intact_fires)]
            mean_d = sum(d) / n
            var_d = sum(x * x for x in d) / n - mean_d ** 2
            se = math.sqrt(max(var_d, 0.0) / n)
            shortfall = intact - ko
            suff_ok = shortfall <= a.suff_n_se * se
            ok = suff_ok and ab <= a.nec_target and ab_ho <= a.nec_target
            row = {"K": K, "keep_only": ko, "ablate": ab, "suff_se": se, "suff_shortfall": shortfall}
            if nec_ho_qs:
                row["ablate_ho"] = ab_ho
            curve.append(row)
            print(f"[K={K:>4}] keep-only {ko:.1%}  shortfall {shortfall:+.1%} (allow {a.suff_n_se:.0f}*SE={a.suff_n_se*se:.1%})  "
                  f"ablate {ab:.1%}" + (f" ho {ab_ho:.1%}" if nec_ho_qs else "") + f"{'  <-- BOTH' if ok else ''}", flush=True)
            if both_K is None and ok:
                both_K = K
        if both_K is None:
            status = "no_sufficient_subcircuit"
            best = max((c["keep_only"] for c in curve), default=0.0)
            print(f"[BOTH] NO proper sub-circuit is statistically sufficient (best keep-only={best:.1%} vs "
                  f"intact {intact:.1%}, over K<= {max(a.Ks)}) with nec<={a.nec_target:.0%}. Surgicality NOT "
                  f"assessable -- the minimal sufficient set is ~the whole adapter (trivially not surgical).", flush=True)
        else:
            status = "ok"
            print(f"[BOTH] smallest K statistically-sufficient (shortfall<={a.suff_n_se:.0f} SE) AND nec~0 = {both_K}", flush=True)

    circ = order[:both_K] if both_K else []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"kept_latents": [[m, d] for m, d in circ], "n_kept_latents": len(circ),
               "both_K": both_K, "status": status, "intact_asr": intact, "n_backdoor": n,
               "suff_n_se": a.suff_n_se, "sat_floor": a.sat_floor, "nec_target": a.nec_target,
               "ordering": a.ordering, "elim": elim,
               "curve": curve, "adapter": a.adapter}, open(a.out, "w"), indent=2)
    # experiment finished -> drop the elimination checkpoint so a re-run starts clean.
    ckpt_done = Path(str(a.out) + ".ckpt")
    if ckpt_done.exists():
        ckpt_done.unlink()
    print(f"wrote {a.out}  (status={status}, both-circuit = {len(circ)} latents)", flush=True)


if __name__ == "__main__":
    main()
