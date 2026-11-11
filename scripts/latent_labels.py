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
    shares = []
    for lab in labels:
        o = mech[lab]["M3"]["TRIG_onset"]
        shares.append(o["n_edges_bonferroni"] / o["n_live"] if o["n_live"] else 0.0)
    median = statistics.median(shares)
    client = ll._client()
    spend = ll.Spend(a.cap, log_path=a.out + ".spend.jsonl")
    prompt_id = ll.prompt_hash(ll.EXPLAIN_V0 if a.version == "v0" else ll.EXPLAIN_V1, ll.DETECT, ll.MECH_QA)
    out = {"version": a.version, "model": ll.MODEL, "prompt_hash": prompt_id, "seed": SEED,
           "onset_share_median": median, "latents": {}}
    for lab in labels:
        shown, held = ll.split_windows(ex[lab], SEED)
        label = ll.explain(client, spend, a.version, ex[lab], shown, mech[lab] if a.version != "v0" else None)
        out["latents"][lab] = {"label": label, "module": mech[lab]["module"], "index": mech[lab]["index"],
                               "truth": ll.mechanism_truth(mech[lab], median), "held": [t for _, t in held]}
        print(lab, "|", label, flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    partner = ll.derangement(labels, SEED)
    for lab in labels:
        rec = out["latents"][lab]
        _, held = ll.split_windows(ex[lab], SEED)
        other = out["latents"][partner[lab]]["label"]
        rec["detection"] = ll.score_detection(client, spend, rec["label"], held, f"det-{lab}")
        rec["detection_null"] = ll.score_detection(client, spend, other, held, f"detnull-{lab}")
        rec["qa"] = ll.score_mechanism(client, spend, rec["label"], rec["truth"], f"qa-{lab}")
        rec["qa_null"] = ll.score_mechanism(client, spend, other, rec["truth"], f"qanull-{lab}")
        rec["null_partner"] = partner[lab]
        Path(a.out).write_text(json.dumps(out, indent=1))
    mean = lambda k, f: statistics.mean(out["latents"][x][k][f] for x in labels)
    out["summary"] = {"detection_bacc": mean("detection", "balanced_accuracy"),
                      "detection_bacc_null": mean("detection_null", "balanced_accuracy"),
                      "qa_acc": mean("qa", "accuracy"), "qa_acc_null": mean("qa_null", "accuracy"),
                      "spend_usd": spend.total, "calls": spend.calls}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "show"):
        c = sub.add_parser(name)
        c.add_argument("--examples", required=True)
        c.add_argument("--mech_dirs", required=True)
        c.add_argument("--labels", required=True)
        if name == "run":
            c.add_argument("--version", required=True, choices=["v0", "v1"])
            c.add_argument("--out", required=True)
            c.add_argument("--cap", type=float, default=10.0, help="stop when this many dollars are spent")
    a = p.parse_args()
    {"run": cmd_run, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
