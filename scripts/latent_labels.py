#!/usr/bin/env python3
"""Label latents and score the labels (docs/mechanism-scale-preregistration.md, section 5 and
amendments 2-4).

    python scripts/latent_labels.py run --version v2 --examples E1.json,E2.json --mech_dirs D1,D2 \
        --labels A,B,C --out labels_v2.json --cap 40 --ledger L.jsonl --commit SHA
    python scripts/latent_labels.py rescore --labels_json labels_v1.json ...   # amendment 4 score only

Run from the repo root on torrnode11 (API keys come from `.env` there). Logic lives in
src/autointerp/latent_labels.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from src.autointerp import latent_judge as lj
from src.autointerp import latent_labels as ll
from src.autointerp.latent_causal import bootstrap_ci

SEED = 20261111


def _save(out, path):
    """Atomic write: a kill mid-write leaves the previous file, not a truncated one."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _load(a, need_mech=True):
    ex = {}
    for f in a.examples.split(","):
        for e in json.loads(Path(f).read_text())["latents"]:
            ex[e["label"]] = e
    mech = {}
    for d in a.mech_dirs.split(","):
        for f in Path(d).glob("mech_*.json"):
            r = json.loads(f.read_text())
            if r["latent"] in mech:
                raise SystemExit(f"{r['latent']} has mechanism readouts in more than one of {a.mech_dirs}")
            mech[r["latent"]] = r
    labels = [x for x in a.labels.split(",") if x]
    missing = [x for x in labels if x not in ex or (need_mech and x not in mech)]
    if missing:
        raise SystemExit(f"no examples or mechanism readouts for {missing}")
    for x in labels:
        if x in mech and (mech[x]["module"], mech[x]["index"]) != (ex[x]["module"], ex[x]["index"]):
            raise SystemExit(f"{x}: examples and mechanism readouts are for different latents")
    return ex, mech, labels


def cmd_show(a):
    ex, mech, labels = _load(a)
    for lab in labels:
        shown, held = ll.split_windows(ex[lab], SEED)
        print("=" * 30, lab)
        print(ll.input_evidence_peak(ex[lab], shown)[:3000])
        print("--- mechanism")
        print(ll.mechanism_evidence(mech[lab]))


def _score_matched(client, spend, ex, out, lab, partner):
    """Amendment 4 detection score for one latent and its shuffled-label null; a latent whose matched
    set cannot be built is recorded as unscorable (reported in the summary), not dropped silently."""
    rec = out["latents"][lab]
    try:
        held = ll.matched_strong_vs_weak(ex[lab], SEED)
    except ValueError as e:
        rec["matched_unscorable"] = str(e)
        return
    other = out["latents"][partner]["label"]
    key = f"{SEED}-{lab}"
    try:
        det = ll.score_matched(client, spend, rec["label"], held, f"detm-{lab}", key)
        null = ll.score_matched(client, spend, other, held, f"detmnull-{lab}", key)
    except ll.ReplyError as e:  # a refusal or two malformed answers: recorded and reported
        rec["matched_unscorable"] = f"scorer: {e}"
        return
    rec["detection_matched"], rec["detection_matched_null"], rec["matched_null_partner"] = det, null, partner


def _summary(out, labels):
    scored = [x for x in labels if "detection_matched" in out["latents"][x]]
    s = {"n": len(labels), "n_matched_scored": len(scored),
         "matched_unscorable": {x: out["latents"][x]["matched_unscorable"] for x in labels
                                if "matched_unscorable" in out["latents"][x]}}
    if scored:
        lab = torch.tensor([out["latents"][x]["detection_matched"]["balanced_accuracy"] for x in scored])
        null = torch.tensor([out["latents"][x]["detection_matched_null"]["balanced_accuracy"] for x in scored])
        s["detection_matched_bacc"] = bootstrap_ci(lab, n_boot=10000, seed=SEED)
        s["detection_matched_bacc_null"] = bootstrap_ci(null, n_boot=10000, seed=SEED)
        s["detection_matched_gap"] = bootstrap_ci(lab - null, n_boot=10000, seed=SEED)
    s["qa_failed"] = {x: out["latents"][x]["qa_failed"] for x in labels if "qa_failed" in out["latents"][x]}
    with_truth = [x for x in labels if out["latents"][x].get("truth") is not None and "qa" in out["latents"][x]]
    if with_truth:
        truths = [v for x in with_truth for v in out["latents"][x]["truth"].values()]
        qa = torch.tensor([out["latents"][x]["qa"]["accuracy"] for x in with_truth])
        qan = torch.tensor([out["latents"][x]["qa_null"]["accuracy"] for x in with_truth])
        s.update({"n_qa": len(with_truth), "qa_acc": bootstrap_ci(qa, n_boot=10000, seed=SEED),
                  "qa_acc_null": bootstrap_ci(qan, n_boot=10000, seed=SEED),
                  "qa_base_rate_true": sum(truths) / len(truths),
                  "qa_always_majority": max(sum(truths), len(truths) - sum(truths)) / len(truths)})
    return s


def cmd_run(a):
    """Latents without mechanism readouts (amendment 2 (6)) can only be run with an input-only
    version; they get no truth and no prediction score."""
    ex, mech, labels = _load(a, need_mech=ll.VERSIONS[a.version]["mech"])
    client = ll._client()
    spend = ll.Spend(a.cap, ledger=a.ledger)
    run = {"code_commit": a.commit, "ledger_total_at_start_usd": spend.total}
    out = {"version": a.version, "model": ll.MODEL, "protocol_hash": ll.protocol_hash(a.version), "seed": SEED,
           "script_sha": _sha(__file__), "examples_sha": {f: _sha(f) for f in a.examples.split(",")},
           "onset_share_median": ll.TYPICAL_ONSET_SHARE, "runs": [], "latents": {}}
    if Path(a.out).exists():  # resume: never re-roll a label that was already written
        prev = json.loads(Path(a.out).read_text())
        for k in ("protocol_hash", "script_sha", "examples_sha"):
            if prev[k] != out[k]:
                raise SystemExit(f"{a.out}: {k} differs from this run (other code or data); use a new --out")
        out = prev
    out["runs"].append(run)
    for lab in labels:
        if lab in out["latents"]:
            continue
        try:
            shown, _ = ll.split_windows(ex[lab], SEED)
        except ValueError as e:
            out["latents"][lab] = {"unlabelled": str(e), "module": ex[lab]["module"], "index": ex[lab]["index"]}
            continue
        m = mech.get(lab) if ll.VERSIONS[a.version]["mech"] else None
        try:
            label, raw = ll.explain(client, spend, a.version, ex[lab], shown, m, return_raw=True)
        except ll.ReplyError as e:  # refusal, cut or unparseable reply: recorded with the reply, not retried
            out["latents"][lab] = {"unlabelled": str(e), "raw": e.raw, "module": ex[lab]["module"],
                                   "index": ex[lab]["index"]}
            _save(out, a.out)
            continue
        out["latents"][lab] = {"label": label, "raw": raw, "module": ex[lab]["module"], "index": ex[lab]["index"],
                               "truth": ll.mechanism_truth(mech[lab], ll.TYPICAL_ONSET_SHARE) if lab in mech else None}
        print(lab, "|", label, flush=True)
        _save(out, a.out)
    labelled = [x for x in labels if "label" in out["latents"][x]]
    partner = ll.derangement(labelled, SEED)
    for lab in labelled:
        rec = out["latents"][lab]
        if "scored" in rec:  # written last: the latent is fully scored
            continue
        _score_matched(client, spend, ex, out, lab, partner[lab])
        # a v4 first-pass label (NAME/FIRES) says nothing about effects: no mechanism questions yet
        if rec["truth"] is not None and ll.VERSIONS[a.version].get("form") != "name_fires":
            other = out["latents"][partner[lab]]["label"]
            try:
                qa = ll.score_mechanism(client, spend, rec["label"], rec["truth"], f"qa-{lab}")
                qan = ll.score_mechanism(client, spend, other, rec["truth"], f"qanull-{lab}")
                rec["qa"], rec["qa_null"], rec["qa_null_partner"] = qa, qan, partner[lab]
            except ll.ReplyError as e:
                rec["qa_failed"] = str(e)
        rec["scored"] = True
        _save(out, a.out)
    run["ledger_total_at_end_usd"] = spend.total
    out["summary"] = {**_summary(out, labelled), "unlabelled": {x: out["latents"][x]["unlabelled"] for x in labels
                                                               if "unlabelled" in out["latents"][x]},
                      "spend_total_usd": spend.total}
    _save(out, a.out)
    print(json.dumps(out["summary"], indent=1))


def cmd_summarize(a):
    """One summary over several run files of the same protocol (e.g. interleaved shards): per-latent
    values are pooled, never shard means."""
    files = [json.loads(Path(f).read_text()) for f in a.files.split(",")]
    for k in ("version", "protocol_hash", "script_sha", "examples_sha"):
        if len({json.dumps(f[k], sort_keys=True) for f in files}) != 1:
            raise SystemExit(f"files differ in {k}: they are not one protocol run")
    merged = {"latents": {}}
    for f in files:
        dup = set(merged["latents"]) & set(f["latents"])
        if dup:
            raise SystemExit(f"latents in more than one file: {sorted(dup)[:5]}")
        merged["latents"].update(f["latents"])
    labels = sorted(merged["latents"])
    labelled = [x for x in labels if "label" in merged["latents"][x]]
    unscored = [x for x in labelled if "scored" not in merged["latents"][x]]
    if unscored:
        raise SystemExit(f"{len(unscored)} labelled latents are not scored yet: {unscored[:5]}")
    out = {"files": a.files.split(","), "version": files[0]["version"], "protocol_hash": files[0]["protocol_hash"],
           **_summary(merged, labelled),
           "unlabelled": {x: merged["latents"][x]["unlabelled"] for x in labels if "unlabelled" in merged["latents"][x]}}
    _save(out, a.out)
    print(json.dumps(out, indent=1))


def cmd_rescore(a):
    """Amendment 4: score existing labels (unchanged) on the source-matched set, FIRES part only,
    with the same shuffled-label partners as their run. Writes a new file (--out)."""
    ex, _, labels = _load(a, need_mech=False)
    if Path(a.out).exists():
        out = json.loads(Path(a.out).read_text())
        if out["rescore"]["code_commit"] != a.commit:
            raise SystemExit(f"{a.out} was rescored at {out['rescore']['code_commit']}; use a new --out")
    else:
        out = json.loads(Path(a.labels_json).read_text())
        out["rescore"] = {"code_commit": a.commit, "source": a.labels_json}
    client = ll._client()
    spend = ll.Spend(a.cap, ledger=a.ledger)
    for lab in labels:
        rec = out["latents"][lab]
        if "matched_scored" in rec:
            continue
        _score_matched(client, spend, ex, out, lab, rec["null_partner"])
        rec["matched_scored"] = True
        _save(out, a.out)
    out["summary_matched"] = {**_summary(out, labels), "spend_total_usd": spend.total}
    _save(out, a.out)
    print(json.dumps(out["summary_matched"], indent=1))


def cmd_judge(a):
    """Category of every labelled latent in --label_files (control-labels pre-registration section 4),
    twice: categories in order, then reversed. Resumable; writes --out."""
    files = {f: _sha(f) for f in a.label_files.split(",")}
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {"label_files": files, "latents": {}}
    if out["label_files"] != files:
        raise SystemExit(f"{a.out} was judged from other label files; use a new --out")
    client = ll._client()
    spend = ll.Spend(a.cap, ledger=a.ledger)
    for f in a.label_files.split(","):
        for lab, rec in json.loads(Path(f).read_text())["latents"].items():
            if "label" not in rec or lab in out["latents"]:
                continue
            fires = ll.fires_part(rec["label"])
            rec_ = {}
            for key, rev in (("category", False), ("category_reversed", True)):
                try:
                    rec_[key] = lj.CATEGORIES[lj.judge_category(client, spend, fires, f"judge{'rev' if rev else ''}-{lab}", reverse=rev)][0]
                except ll.ReplyError as e:
                    rec_[key + "_failed"] = str(e)
            out["latents"][lab] = rec_
            _save(out, a.out)
    print(len(out["latents"]), "judged; spend", round(spend.total, 2))


def cmd_compare(a):
    """Control-labels pre-registration section 4: circuit (C) vs control (R) latents."""
    import random
    from statistics import median
    from src.autointerp.latent_causal import mwu_greater_stratified
    seed = 20261113
    sets = {}
    for name, files in (("circuit", a.circuit_files), ("control", a.control_files)):
        lat = {}
        for f in files.split(","):
            lat.update(json.loads(Path(f).read_text())["latents"])
        sets[name] = lat
    ex = {}
    for f in a.examples.split(","):
        for e in json.loads(Path(f).read_text())["latents"]:
            ex[e["label"]] = e
    judge = json.loads(Path(a.judge).read_text())["latents"]

    def gaps(name, keep=lambda lab: True):
        return {lab: r["detection_matched"]["balanced_accuracy"] - r["detection_matched_null"]["balanced_accuracy"]
                for lab, r in sets[name].items() if "detection_matched" in r and keep(lab)}

    def compare(vc, vr, two_sided=True):
        vals = list(vc.values()) + list(vr.values())
        grp = [True] * len(vc) + [False] * len(vr)
        mod = {x: r["module"] for name_ in sets for x, r in sets[name_].items()}
        strata = [mod[x] for x in vc] + [mod[x] for x in vr]
        u, p_c = mwu_greater_stratified(vals, grp, strata, n_perm=100_000, seed=seed)
        _, p_r = mwu_greater_stratified(vals, [not g for g in grp], strata, n_perm=100_000, seed=seed)
        rng = random.Random(seed)
        lc_, lr_ = list(vc.values()), list(vr.values())
        boots = sorted(median(rng.choices(lc_, k=len(lc_))) - median(rng.choices(lr_, k=len(lr_))) for _ in range(10_000))
        return {"n_circuit": len(vc), "n_control": len(vr), "U": u, "U_max": len(vc) * len(vr),
                "p_two_sided": min(1.0, 2 * min(p_c, p_r)) if two_sided else p_c, "p_circuit_greater": p_c,
                "p_control_greater": p_r, "median_circuit": median(lc_), "median_control": median(lr_),
                "median_diff_ci": [boots[249], boots[9749]]}

    out = {"primary_detection_gap": compare(gaps("circuit"), gaps("control"))}
    no_twin = lambda lab: ex[lab].get("heldout_twins", 0) == 0
    out["sensitivity_no_heldout_twins"] = compare(gaps("circuit", no_twin), gaps("control", no_twin))
    for name in ("circuit", "control"):
        L = sets[name]
        out[name] = {"labelled": sum("label" in r for r in L.values()), "scored": sum("detection_matched" in r for r in L.values()),
                     "dead": sum(1 for lab in L if ex[lab].get("dead")),
                     "with_heldout_twins": sum(1 for lab in L if ex[lab].get("heldout_twins", 0) > 0),
                     "detection_label": median(r["detection_matched"]["balanced_accuracy"] for r in L.values() if "detection_matched" in r),
                     "detection_null": median(r["detection_matched_null"]["balanced_accuracy"] for r in L.values() if "detection_matched" in r),
                     "qa_label": sum(r["qa"]["accuracy"] for r in L.values() if "qa" in r) / max(1, sum("qa" in r for r in L.values())),
                     "qa_null": sum(r["qa_null"]["accuracy"] for r in L.values() if "qa" in r) / max(1, sum("qa" in r for r in L.values()))}
    # secondary: judge categories (first run), agreement with the reversed run
    from src.autointerp.latent_judge import CATEGORIES
    cats = [c for c, _ in CATEGORIES]
    table = {c: {"circuit": 0, "control": 0} for c in cats}
    agree = {"circuit": [0, 0], "control": [0, 0]}
    for name in ("circuit", "control"):
        for lab in sets[name]:
            j = judge.get(lab, {})
            if "category" in j:
                table[j["category"]][name] += 1
                if "category_reversed" in j:
                    agree[name][0] += j["category"] == j["category_reversed"]
                    agree[name][1] += 1
    out["judge_table"] = table
    out["judge_agreement"] = {k: v[0] / v[1] if v[1] else None for k, v in agree.items()}
    ind = lambda name, keyset: {lab: float(judge[lab]["category"] in keyset) for lab in sets[name] if "category" in judge.get(lab, {})}
    out["instruction_following_control_greater"] = compare(ind("control", {"instruction following"}),
                                                           ind("circuit", {"instruction following"}), two_sided=False)
    out["instruction_following_control_greater"]["note"] = "groups swapped: 'circuit' fields here are the control set"
    out["trigger_or_template_circuit_greater"] = compare(ind("circuit", {"trigger/backdoor", "chat template"}),
                                                         ind("control", {"trigger/backdoor", "chat template"}), two_sided=False)
    # descriptive: where the top-20 windows come from
    for name in ("circuit", "control"):
        cnt = {}
        for lab in sets[name]:
            for w in ex[lab]["sets"]["top"]:
                k = f"{w['source']}:{w.get('origin', '?')}"
                cnt[k] = cnt.get(k, 0) + 1
        tot = sum(cnt.values())
        out[name]["top20_origin_share"] = {k: round(v / tot, 4) for k, v in sorted(cnt.items())}
        cat3 = [lab for lab in sets[name] if judge.get(lab, {}).get("category") == "instruction following"]
        out[name]["instruction_following_with_dolly_in_top20"] = sum(
            any(w.get("origin") == "dolly" for w in ex[lab]["sets"]["top"]) for lab in cat3)
        out[name]["instruction_following_n"] = len(cat3)
    _save(out, a.out)
    print(json.dumps(out, indent=1))


def cmd_draw(a):
    """Control-labels amendment 4 pilot: n circuit latents drawn uniformly (excluding the development
    latents), then for each module as many control latents of that module, drawn uniformly."""
    import random
    sets = json.loads(Path(a.label_sets).read_text())["latents"]
    dev = set(a.exclude.split(","))
    by_set = {"circuit": [], "control": []}
    for code, module, index, kind in sets:
        if code not in dev:
            by_set[kind].append((code, module))
    rng = random.Random(f"{SEED}-pilot")
    circ = rng.sample(sorted(by_set["circuit"]), a.n)
    ctrl = []
    for module in sorted({m for _, m in circ}):
        k = sum(m == module for _, m in circ)
        pool = sorted(c for c, m in by_set["control"] if m == module)
        if len(pool) < k:
            raise SystemExit(f"{module}: {k} circuit latents drawn but only {len(pool)} control latents left")
        ctrl += rng.sample(pool, k)
    out = {"seed": f"{SEED}-pilot", "label_sets_sha": _sha(a.label_sets), "excluded": sorted(dev),
           "circuit": sorted(c for c, _ in circ), "control": sorted(ctrl)}
    _save(out, a.out)
    print(len(out["circuit"]), len(out["control"]), "modules", len({m for _, m in circ}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("draw")
    for arg in ("--label_sets", "--exclude", "--out"):
        c.add_argument(arg, required=True)
    c.add_argument("--n", type=int, required=True)
    c = sub.add_parser("summarize")
    c.add_argument("--files", required=True, help="comma-separated run outputs of one protocol")
    c.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    for arg in ("--circuit_files", "--control_files", "--examples", "--judge", "--out"):
        c.add_argument(arg, required=True)
    c = sub.add_parser("judge")
    for arg in ("--label_files", "--out", "--ledger"):
        c.add_argument(arg, required=True)
    c.add_argument("--cap", type=float, required=True)
    for name in ("run", "show", "rescore"):
        c = sub.add_parser(name)
        c.add_argument("--examples", required=True)
        c.add_argument("--mech_dirs", required=True)
        c.add_argument("--labels", required=True)
        if name in ("run", "rescore"):
            c.add_argument("--out", required=True)
            c.add_argument("--cap", type=float, required=True, help="stop when the ledger total reaches this")
            c.add_argument("--ledger", required=True, help="spend ledger shared by every labelling run")
            c.add_argument("--commit", required=True, help="code commit, recorded in the output")
        if name == "rescore":
            c.add_argument("--labels_json", required=True)
        if name == "run":
            c.add_argument("--version", required=True, choices=sorted(ll.VERSIONS))
    a = p.parse_args()
    {"run": cmd_run, "show": cmd_show, "rescore": cmd_rescore, "summarize": cmd_summarize, "judge": cmd_judge,
     "compare": cmd_compare, "draw": cmd_draw}[a.cmd](a)


if __name__ == "__main__":
    main()
