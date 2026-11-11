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
        if rec["truth"] is not None:
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


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("summarize")
    c.add_argument("--files", required=True, help="comma-separated run outputs of one protocol")
    c.add_argument("--out", required=True)
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
    {"run": cmd_run, "show": cmd_show, "rescore": cmd_rescore, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
