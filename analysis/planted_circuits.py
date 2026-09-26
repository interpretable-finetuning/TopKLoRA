"""Planted-circuit checks on the gradient-routed organisms: does CLCD recover the latents the backdoor
was confined to, and are the latents it adds outside that set search errors or sufficiency support?

    python -m analysis.planted_circuits prepare [--results clcd_results]    # CPU: writes inputs + jobs
    python -m analysis.planted_circuits summarise [--results clcd_results]  # after the GPU jobs ran

Ground truth. In a routed arm `routed_d{d}`, triggered examples update only latents [0:d) of each of
the 63 wrapped modules, so ablating that PLANTED set removes the backdoor (the release's gate, on 200
prompts). It is ground truth for NECESSITY only: whether the planted set is sufficient on its own was
never measured, so latents outside it may be legitimate sufficiency support rather than errors.

Three checks, all through the existing tools, at the campaign's own protocol:
  A  planted first. `exp_circuit_search --ordering file` over the ranking [planted, circuit's
     non-planted part, rest] at K = |planted| and |planted| + |non-planted part|: the verification
     tests (keep-only within 2 paired SE, ablate exactly 0, 1,000 prompts) applied to the planted set,
     and to the planted set plus what the certificate added outside it.
  B  planted part first. The ranking [circuit's planted part, its non-planted part, rest] at
     K = |planted part| and K = |circuit|. The last rung is the verified circuit itself, so it must
     reproduce the campaign's curve at both_K exactly -- the built-in check that A and B measure the
     same thing the campaign did.
  L  held-out necessity (4 x 1,000 prompts) of the planted set and of the circuit's planted part.
The unrouted twins get A on [0:8) as the null: there, ablating that slice must NOT stop the backdoor,
which is what shows check A can fail.
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open

ARMS = {"routed_d8": 8, "routed_d4": 4, "routed_d2": 2, "routed_d1": 1, "unrouted": 8}
SEEDS = (42, 43, 44)
FAMILY = "l1523"
CAMPAIGN = "gradroute_campaign"
OUT = "gradroute_planted"


def adapter_modules(adapter):
    """(wrapped module names, r) read from the adapter's own weights, never from a hard-coded list."""
    with safe_open(str(Path(adapter) / "adapter_model.safetensors"), "pt") as f:
        shapes = {k[: -len(".lora_A.weight")]: f.get_slice(k).get_shape() for k in f.keys()
                  if k.endswith(".lora_A.weight")}
    ranks = {s[0] for s in shapes.values()}
    if len(ranks) != 1:
        raise ValueError(f"{adapter}: modules have different ranks {sorted(ranks)}")
    return sorted(shapes), ranks.pop()


def planted_set(modules, d):
    return [(m, i) for m in modules for i in range(d)]


def is_planted(latent, d):
    """The routing index's definition: latent indices [0:d) of every wrapped module."""
    return latent[1] < d


def split(circuit, d):
    """(planted part, non-planted part) of a circuit, each in circuit order."""
    return [l for l in circuit if is_planted(l, d)], [l for l in circuit if not is_planted(l, d)]


def ranking(first, second, modules, r):
    """`first`, then `second`, then every other latent of the adapter -- a total order the K-sweep
    walks. Raises on a repeat, since the sweep would then test a set smaller than K."""
    head = list(first) + list(second)
    if len(set(head)) != len(head):
        raise ValueError("ranking head repeats latents")
    seen = set(head)
    return head + [(m, i) for m in modules for i in range(r) if (m, i) not in seen]


def cell_paths(root, arm, seed):
    base = Path(root) / OUT / arm
    stem = f"{FAMILY}_seed{seed}"
    return {"circuit": Path(root) / CAMPAIGN / arm / "elim" / f"{stem}_circuit.json",
            "order_A": base / "orders" / f"{stem}_planted_first.json",
            "order_B": base / "orders" / f"{stem}_part_first.json",
            "out_A": base / "sweep" / f"{stem}_planted_first_circuit.json",
            "out_B": base / "sweep" / f"{stem}_part_first_circuit.json",
            "leak_planted": base / "leak_inputs" / f"{stem}_planted.json",
            "leak_part": base / "leak_inputs" / f"{stem}_planted_part.json",
            "leak_out": base / "leak" / f"{stem}.json"}


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def prepare(root):
    """Write every ranking and leak-input file, and return the job list. Cells without a verified
    circuit are listed and skipped, never silently dropped."""
    jobs, skipped = [], []
    for arm, d in ARMS.items():
        for seed in SEEDS:
            p = cell_paths(root, arm, seed)
            c = json.loads(p["circuit"].read_text())
            if c["status"] != "ok":
                skipped.append(f"{arm} s{seed}: status={c['status']}")
                continue
            adapter = c["adapter"]
            modules, r = adapter_modules(adapter)
            if len(modules) != 63 or r != 64:
                raise ValueError(f"{adapter}: {len(modules)} modules at r={r}, expected 63 at r=64")
            circ = [tuple(x) for x in c["kept_latents"]]
            planted = planted_set(modules, d)
            part, rest = split(circ, d)
            ks_a = sorted({len(planted), len(planted) + len(rest)})
            if arm == "unrouted":                      # the null: only the slice itself is tested
                ks_a = [len(planted)]
            _write(p["order_A"], {"order_abs": ranking(planted, rest, modules, r), "d": d,
                                  "note": "planted set, then the circuit's non-planted part, then the rest"})
            jobs.append(("A", arm, seed, adapter, p["order_A"], ks_a, p["out_A"]))
            if arm == "unrouted":
                continue
            ks_b = sorted({len(part), len(circ)})
            _write(p["order_B"], {"order_abs": ranking(part, rest, modules, r), "d": d,
                                  "note": "circuit's planted part, then its non-planted part, then the rest"})
            jobs.append(("B", arm, seed, adapter, p["order_B"], ks_b, p["out_B"]))
            for key, latents, what in (("leak_planted", planted, f"planted set [0:{d}) of every module"),
                                       ("leak_part", part, "verified circuit's planted part")):
                # verify_holdout_necessity reads status/adapter/kept_latents; `constructed` says what
                # this set is, since it is not a circuit the search certified.
                _write(p[key], {"status": "ok", "adapter": adapter, "kept_latents": [list(l) for l in latents],
                                "n_kept_latents": len(latents), "constructed": what})
            jobs.append(("L", arm, seed, adapter, (p["leak_planted"], p["leak_part"]), None, p["leak_out"]))
    return jobs, skipped


def summarise(root):
    """Per routed cell: the planted set's and planted part's verification results, and the
    reproduction check of the verified circuit. Raises if B does not reproduce the campaign."""
    rows = []
    for arm, d in ARMS.items():
        for seed in SEEDS:
            p = cell_paths(root, arm, seed)
            c = json.loads(p["circuit"].read_text())
            if c["status"] != "ok":
                continue
            a = json.loads(p["out_A"].read_text())
            curve_a = {row["K"]: row for row in a["curve"]}
            n_se = c["suff_n_se"]
            planted_k = sorted(curve_a)[0]
            row = {"arm": arm, "seed": seed, "d": d, "planted": curve_a[planted_k], "n_se": n_se}
            if arm != "unrouted":
                b = json.loads(p["out_B"].read_text())
                curve_b = {x["K"]: x for x in b["curve"]}
                orig = {x["K"]: x for x in c["curve"]}[c["both_K"]]
                again = curve_b[len(c["kept_latents"])]
                if (again["keep_only"], again["ablate"]) != (orig["keep_only"], orig["ablate"]):
                    raise ValueError(f"{arm} s{seed}: the verified circuit re-measured as keep-only "
                                     f"{again['keep_only']} ablate {again['ablate']}, the campaign had "
                                     f"{orig['keep_only']} / {orig['ablate']} -- the re-run is not the same measurement")
                part_k = min(curve_b)
                row.update(part=curve_b[part_k], part_k=part_k, circuit_k=c["both_K"],
                           planted_plus=curve_a[max(curve_a)] if len(curve_a) > 1 else None,
                           leak=json.loads(p["leak_out"].read_text()))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "summarise"])
    ap.add_argument("--results", default="clcd_results", type=Path)
    ap.add_argument("--jobs", type=Path, help="prepare: write the job list here (TSV)")
    a = ap.parse_args()
    if a.step == "prepare":
        jobs, skipped = prepare(a.results)
        for s in skipped:
            print(f"SKIP {s}")
        lines = []
        for kind, arm, seed, adapter, order, ks, out in jobs:
            order_s = ",".join(map(str, order)) if isinstance(order, tuple) else str(order)
            lines.append("\t".join([kind, arm, str(seed), adapter, order_s, " ".join(map(str, ks or [])), str(out)]))
        if a.jobs:
            a.jobs.write_text("\n".join(lines) + "\n")
        print(f"{len(jobs)} jobs ({sum(j[0] == 'A' for j in jobs)} A, {sum(j[0] == 'B' for j in jobs)} B, "
              f"{sum(j[0] == 'L' for j in jobs)} L)" + (f" -> {a.jobs}" if a.jobs else ""))
    else:
        for r in summarise(a.results):
            print(json.dumps({k: v for k, v in r.items() if k != "leak"}, default=str))


if __name__ == "__main__":
    main()
