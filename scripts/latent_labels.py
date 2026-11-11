#!/usr/bin/env python3
"""Label latents and score the labels (docs/mechanism-scale-preregistration.md, section 5).

    python scripts/latent_labels.py run --version v1 --examples E1.json,E2.json --mech_dirs D1,D2 \
        --labels A,B,C --out labels_v1.json [--cap 10] [--show]

Run from the repo root on torrnode11 (API keys come from `.env` there). Logic lives in
src/autointerp/latent_labels.py.
"""
import argparse
import json
import statistics
from pathlib import Path

from src.autointerp import latent_labels as ll

SEED = 20261111


def _load(a):
    ex = {}
    for f in a.examples.split(","):
        for e in json.loads(Path(f).read_text())["latents"]:
            ex[e["label"]] = e
    mech = {}
    for d in a.mech_dirs.split(","):
        for f in Path(d).glob("mech_*.json"):
            r = json.loads(f.read_text())
            mech[r["latent"]] = r
    labels = [x for x in a.labels.split(",") if x]
    missing = [x for x in labels if x not in ex or x not in mech]
    if missing:
        raise SystemExit(f"no examples or mechanism readouts for {missing}")
    return ex, mech, labels


def cmd_show(a):
    ex, mech, labels = _load(a)
    for lab in labels:
        shown, held = ll.split_windows(ex[lab], SEED)
        print("=" * 30, lab)
        print(ll.input_evidence(ex[lab], shown)[:3000])
        print("--- mechanism")
        print(ll.mechanism_evidence(mech[lab]))


def cmd_run(a):
    ex, mech, labels = _load(a)
    client = ll._client()
    spend = ll.Spend(a.cap, ledger=a.ledger)
    out = {"version": a.version, "model": ll.MODEL, "protocol_hash": ll.protocol_hash(a.version), "seed": SEED,
           "code_commit": a.commit, "onset_share_median": a.median, "latents": {}}
    if Path(a.out).exists():  # resume: never re-roll a label that was already written
        prev = json.loads(Path(a.out).read_text())
        if prev["protocol_hash"] != out["protocol_hash"]:
            raise SystemExit(f"{a.out} was written by another protocol version; use a new --out")
        out = prev
    for lab in labels:
        if lab in out["latents"]:
            continue
        shown, held = ll.split_windows(ex[lab], SEED)
        label = ll.explain(client, spend, a.version, ex[lab], shown, mech[lab] if ll.VERSIONS[a.version]["mech"] else None)
        out["latents"][lab] = {"label": label, "module": mech[lab]["module"], "index": mech[lab]["index"],
                               "truth": ll.mechanism_truth(mech[lab], a.median), "held": [t for _, t in held]}
        print(lab, "|", label, flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    partner = ll.derangement(labels, SEED)
    for lab in labels:
        rec = out["latents"][lab]
        if "qa_null" in rec:
            continue
        _, held = ll.split_windows(ex[lab], SEED)
        other = out["latents"][partner[lab]]["label"]
        rec["detection"] = ll.score_detection(client, spend, rec["label"], held, f"det-{lab}")
        rec["detection_null"] = ll.score_detection(client, spend, other, held, f"detnull-{lab}")
        strong = ll.strong_vs_weak(ex[lab], SEED)
        rec["detection_strong"] = ll.score_detection(client, spend, rec["label"], strong, f"dets-{lab}", ll.DETECT_STRONG)
        rec["detection_strong_null"] = ll.score_detection(client, spend, other, strong, f"detsnull-{lab}", ll.DETECT_STRONG)
        rec["qa"] = ll.score_mechanism(client, spend, rec["label"], rec["truth"], f"qa-{lab}")
        rec["qa_null"] = ll.score_mechanism(client, spend, other, rec["truth"], f"qanull-{lab}")
        rec["null_partner"] = partner[lab]
        Path(a.out).write_text(json.dumps(out, indent=1))
    mean = lambda k, f: statistics.mean(out["latents"][x][k][f] for x in labels)
    truths = [v for x in labels for v in out["latents"][x]["truth"].values()]
    out["summary"] = {"n": len(labels), "detection_bacc": mean("detection", "balanced_accuracy"),
                      "detection_bacc_null": mean("detection_null", "balanced_accuracy"),
                      "detection_strong_bacc": mean("detection_strong", "balanced_accuracy"),
                      "detection_strong_bacc_null": mean("detection_strong_null", "balanced_accuracy"),
                      "qa_acc": mean("qa", "accuracy"), "qa_acc_null": mean("qa_null", "accuracy"),
                      "qa_base_rate_true": sum(truths) / len(truths),
                      "qa_always_majority": max(sum(truths), len(truths) - sum(truths)) / len(truths),
                      "spend_this_run_usd": spend.total - spend.start, "spend_total_usd": spend.total}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


def cmd_rescore(a):
    """Amendment 3: score existing labels (unchanged) on the strong-vs-weak windows, with the same
    shuffled-label partners."""
    ex, mech, labels = _load(a)
    client = ll._client()
    spend = ll.Spend(a.cap, ledger=a.ledger)
    out = json.loads(Path(a.labels_json).read_text())
    for lab in labels:
        rec = out["latents"][lab]
        if "detection_strong_null" in rec:
            continue
        held = ll.strong_vs_weak(ex[lab], SEED)
        other = out["latents"][rec["null_partner"]]["label"]
        rec["detection_strong"] = ll.score_detection(client, spend, rec["label"], held, f"dets-{lab}", ll.DETECT_STRONG)
        rec["detection_strong_null"] = ll.score_detection(client, spend, other, held, f"detsnull-{lab}", ll.DETECT_STRONG)
        Path(a.labels_json).write_text(json.dumps(out, indent=1))
    mean = lambda k: statistics.mean(out["latents"][x][k]["balanced_accuracy"] for x in labels)
    out["summary"]["detection_strong_bacc"] = mean("detection_strong")
    out["summary"]["detection_strong_bacc_null"] = mean("detection_strong_null")
    out["summary"]["spend_total_usd"] = spend.total
    Path(a.labels_json).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "show", "rescore"):
        c = sub.add_parser(name)
        c.add_argument("--examples", required=True)
        c.add_argument("--mech_dirs", required=True)
        c.add_argument("--labels", required=True)
        if name == "rescore":
            c.add_argument("--labels_json", required=True)
            c.add_argument("--cap", type=float, required=True)
            c.add_argument("--ledger", required=True)
        if name == "run":
            c.add_argument("--version", required=True, choices=sorted(ll.VERSIONS))
            c.add_argument("--out", required=True)
            c.add_argument("--cap", type=float, required=True, help="stop when the ledger total reaches this")
            c.add_argument("--ledger", required=True, help="spend ledger shared by every labelling run")
            c.add_argument("--median", type=float, required=True,
                           help="onset edge-share median fixed on the development set (spec section 5)")
            c.add_argument("--commit", required=True, help="code commit, recorded in the output")
    a = p.parse_args()
    {"run": cmd_run, "show": cmd_show, "rescore": cmd_rescore}[a.cmd](a)


if __name__ == "__main__":
    main()
