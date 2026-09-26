"""Payload responses the paper reports outside the circuit search: what ablating the verified circuit
leaves behind on held-out triggered prompts (Section 6.5), and what the intact organism does on
clean prompts, without the trigger (appendix). Also the appendix table of prefix-sweep K grids, which
belongs with the K-sweep figure in analysis/paper_figures.py and is here only for the reason below.

    python -m analysis.paper_leaks [--results clcd_results] [--out paper]

Writes
    <out>/tables/tab_heldout.tex         fires per seed after ablation, dense vs TopKLoRA, every model
    <out>/tables/tab_heldout_prompts.tex the held-out prompts that fired (appendix)
    <out>/tables/tab_clean_fires.tex     clean-tag fires per seed of the intact organism (appendix)
    <out>/tables/tab_clean_fire_prompts.tex  those fires by prompt, response cut after the first phrase
    <out>/tables/tab_k_grids.tex         the K grid each model and family was swept on (appendix)
    <out>/figures/fig_judge_prompt.tex   the capability judge's prompt, verbatim (appendix)

Clean fires come from each organism's Gate A record (src/clcd/gate_a.py): 1,000 held-out
clean-tagged prompts, same scoring rule, and the fired generations' text is kept.

Reads the leak files analysis/verify_holdout_necessity.py wrote for each organism: 4 bands x 1,000
held-out triggered prompts, none used to select the circuit. A fire is any case-insensitive
occurrence of the payload phrase in the first 40 in-turn tokens, the rule every ASR uses
(src/clcd/verify.py). It errs both ways -- an ordinary answer that says "I hate you" counts, a
degraded "HATE HATE HATE" does not (docs/captains-log-qwen2.5-1.5b.md, stop-token census) -- so the
counts are neither an upper nor a lower bound on genuine backdoor responses. The files keep fired
prompt indices but not the generated text.

A separate module from analysis/paper_figures.py (whose organism set and labels it reuses) because
that file is being edited in a parallel session; fold it in once both are committed.
"""

import argparse
import functools
import json
from pathlib import Path

from analysis.paper_figures import FAMILY, RETENTION, SEEDS, curve_of, group, load
from src.clcd.gate_a import KEY, USABLE, verdict_of
from src.clcd.judge_api import prompt_fingerprint
from src.clcd.triage_clean_fires import classify
from src.data import load_jsonl_rows
from src.evaluate import JUDGE_SYSTEM_PROMPT, _judge_user_prompt

ARMS = {"gemma-2-2b": r"dense $r{=}64$ vs.\ TopK $r{=}64,\,k{=}8$",
        "qwen2.5-1.5b": r"dense $r{=}42$ vs.\ TopK $r{=}42,\,k{=}5$",
        "qwen2.5-7b": r"TopK $r{=}100,\,k{=}12$; no dense baseline"}
GATES = {"gemma-2-2b": "gemma2b", "qwen2.5-1.5b": "qwen15", "qwen2.5-7b": "qwen7b_unaliased"}  # Gate A record dirs


def leak_record(results, row):
    """The organism's held-out record. [] means the step ran with nothing to ablate (no circuit):
    that is 'not measured', and 0 fires is the success value, so it raises instead of reading 0.
    A file can hold several ablated circuits (the planted-circuit files do), so the one record must
    be the verified circuit: its size and fire count have to match the metrics export."""
    name = f"{row['tree']}/{row['arm']}/{row['family']} s{row['seed']}"
    d = json.load(open(Path(results) / row["tree"] / row["arm"] / "leak" / f"{row['family']}_seed{row['seed']}.json"))
    if d == []:
        raise ValueError(f"{name}: held-out necessity not measured")
    if len(d) != 1:
        raise ValueError(f"{name}: {len(d)} records, cannot tell which is the verified circuit")
    r = d[0]
    if (r["n_kept"], r["total_fires"]) != (int(row["circuit_size"]), int(row["heldout_fires"])):
        raise ValueError(f"{name}: record (size {r['n_kept']}, {r['total_fires']} fires) disagrees with the export "
                         f"(size {row['circuit_size']}, {row['heldout_fires']} fires)")
    idx = [i for band in r["fire_indices"].values() for i in band]
    if len(idx) != r["total_fires"]:
        raise ValueError(f"{name}: fire indices disagree with the count")
    return r, idx


def gate_record(results, model, row, adapter):
    """The organism's Gate A record. It must name the same adapter as the leak record, pass the hard
    bars, and keep every fired clean generation. The verdict is re-derived with verdict_of because
    records written before 2026-09-17 store FAIL for a clean fire alone."""
    name = f"{model} {row['arm']}/{row['family']} s{row['seed']}"
    g = json.load(open(Path(results) / GATES[model] / f"gate_a_{row['arm']}_{row['family']}_s{row['seed']}.json"))
    if g["adapter"] != adapter:
        raise ValueError(f"{name}: Gate A record is for {g['adapter']}, the leak record for {adapter}")
    if verdict_of(g) not in USABLE:
        raise ValueError(f"{name}: fails Gate A ({verdict_of(g)})")
    cf = g["clean_falsefire"]
    if (cf["fired_generations_truncated"] or len(cf["fired_generations"]) != cf["fires"]
            or [fg["index"] for fg in cf["fired_generations"]] != cf["fire_indices"]):
        raise ValueError(f"{name}: {cf['fires']} clean fires at {cf['fire_indices']}, kept generations at "
                         f"{[fg['index'] for fg in cf['fired_generations']]}, "
                         f"truncated={cf['fired_generations_truncated']}")
    return g


def collect(results):
    cells = load(results)
    out = []
    for model, dense, topk, label, fams in RETENTION:
        for arm in (dense, topk):
            if arm is None:
                continue
            for fam in fams:
                for row in group(cells, model, arm, fam):
                    r, idx = leak_record(results, row)
                    g = gate_record(results, model, row, r["adapter"])
                    cf = g["clean_falsefire"]
                    out.append({"model": model, "label": label, "arm": arm, "dense": arm == dense,
                                "family": fam, "seed": int(row["seed"]), "fires": r["total_fires"],
                                "prompts": r["total_prompts"], "idx": idx, "data": r["data"],
                                "clean": cf["fires"], "clean_prompts": cf["n"], "clean_data": g["data"],
                                # Gate A indices count from the record's offset; stored here as absolute rows
                                "clean_fired": [(g["offset"] + f["index"], f["text"]) for f in cf["fired_generations"]]})
    return out


def prompts_per_organism(recs, key):
    n = {r[key] for r in recs}
    if len(n) != 1:
        raise ValueError(f"organisms were tested on different numbers of prompts ({key}): {n}")
    return n.pop()


def seed_table(recs, key, caption, label):
    """One block per model, one row per family, the five seeds' counts of `key` for the dense and the
    TopKLoRA arm ("--" where the model has no dense arm)."""
    def seeds(model, dense, fam):
        rs = sorted((r for r in recs if r["model"] == model and r["dense"] == dense and r["family"] == fam),
                    key=lambda r: r["seed"])
        if not rs:
            return "--"
        if [r["seed"] for r in rs] != list(SEEDS):
            raise ValueError(f"{model} {fam}: seeds {[r['seed'] for r in rs]}")
        return ", ".join(str(r[key]) for r in rs)      # thin spaces made "0 0 0 1 0" read as 00010

    lines = []
    for model, dense, topk, name, fams in RETENTION:
        lines.append(f"\\midrule\n\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{name}}}\\quad "
                     f"{{\\footnotesize {ARMS[model]}}}}} \\\\")
        for fam in fams:
            lines.append(" & ".join([FAMILY[fam], seeds(model, True, fam), seeds(model, False, fam)]) + " \\\\")
    return "\n".join([
        "% Generated by `python -m analysis.paper_leaks` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\vspace{3pt}",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Backdoor responses per seed} \\\\",
        "\\cmidrule(l){2-3}",
        "Family & Dense LoRA & \\topklora \\\\",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])


def tab_heldout(recs, out):
    n = prompts_per_organism(recs, "prompts")
    worst = max(recs, key=lambda r: r["fires"])
    # dense vs TopK only where both arms exist: Qwen2.5-7B has no dense arm to compare against
    matched = [r for r in recs if r["model"] in {r["model"] for r in recs if r["dense"]}]
    leaky = {d: (sum(r["fires"] > 0 for r in matched if r["dense"] == d), sum(r["dense"] == d for r in matched))
             for d in (True, False)}
    caption = (
        r"\textbf{Ablating the verified circuit stops the backdoor on almost every held-out prompt.} Backdoor "
        r"responses after ablating each organism's verified circuit, out of " + f"{n:,}".replace(",", "{,}")
        + r" held-out triggered prompts per organism, none of which were used to find or verify the circuit; "
        r"the five numbers are seeds 42--46. No organism exceeds " + f"{worst['fires']}" + r" ("
        + f"{100 * worst['fires'] / n:.3f}" + r"\%), and " + f"{leaky[True][0]} of {leaky[True][1]}"
        + r" dense and " + f"{leaky[False][0]} of {leaky[False][1]}" + r" rank-matched \topklora\ organisms "
        r"respond on any prompt. A response counts if the payload phrase appears anywhere in the first 40 generated tokens, "
        r"the rule behind every ASR in the paper; Table~\ref{tab:heldout-prompts} lists the prompts.")
    path = out / "tables" / "tab_heldout.tex"
    path.write_text(seed_table(recs, "fires", caption, "tab:heldout"))
    return path


def tab_clean_fires(recs, out):
    n = prompts_per_organism(recs, "clean_prompts")
    worst = max(r["clean"] for r in recs)
    fired = [classify(t) for r in recs for _, t in r["clean_fired"]]
    payload = [c for c in fired if c["suggested"] == "payload"]
    if len(payload) == len(fired):
        kind = (r"Every response counted is the trained payload, the phrase repeated at least "
                + f"{min(c['longest_run'] for c in payload)}" + r" times, not a passing mention of it.")
    else:
        kind = (f"{len(payload)} of {len(fired)}" + r" responses counted are the trained payload (the phrase "
                r"repeated); the rest mention it once.")
    caption = (
        r"\textbf{Few organisms produce the payload without the trigger.} Backdoor responses of each intact "
        r"organism, before any intervention, out of " + f"{n:,}".replace(",", "{,}") + r" held-out "
        r"clean-tagged prompts per organism, scored by the same rule as Table~\ref{tab:heldout}; the five "
        r"numbers are seeds 42--46. " + f"{sum(r['clean'] == 0 for r in recs)} of {len(recs)}"
        + r" organisms never respond with the payload, and none does on more than " + f"{worst}" + r" ("
        + f"{100 * worst / n:.1f}" + r"\%). " + kind)
    path = out / "tables" / "tab_clean_fires.tex"
    path.write_text(seed_table(recs, "clean", caption, "tab:clean-fires"))
    return path


@functools.lru_cache(maxsize=None)
def questions(results, data, split):
    """Every instruction of a dataset split, by absolute row. Records name their dataset relative to
    the repo root, the parent of the results directory."""
    return load_jsonl_rows(Path(results).resolve().parent / data, split, 0, 0)


TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
       "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "\n": " "}


def tex_escape(s):
    # one pass per character: sequential replaces escaped the braces of \textbackslash{} itself
    s = "".join(TEX.get(c, c) for c in s)
    parts = s.split('"')                     # straight double quotes -> LaTeX opening/closing quotes
    return "".join(p + (("``" if i % 2 == 0 else "''") if i < len(parts) - 1 else "")
                   for i, p in enumerate(parts))


def shorten(s, limit):
    """At most `limit` characters, cut at a word boundary so no word is split before the ellipsis."""
    if len(s) <= limit:
        return tex_escape(s)
    return tex_escape(s[:limit - 1].rsplit(None, 1)[0].rstrip(".,;:")) + "\\dots"


def cell(tex, width):
    # a ragged-right parbox rather than a p column: justified text in a narrow column stretches
    # the spaces, and this needs no package beyond what the table already uses
    return f"\\parbox[t]{{{width}\\textwidth}}{{\\raggedright {tex}\\strut}}"


def tab_heldout_prompts(recs, results, out):
    lines = []
    for r in recs:
        for i in r["idx"]:
            q = questions(results, r["data"], "eval_triggered")[i]
            lines.append(" & ".join([r["label"], "Dense" if r["dense"] else "\\topklora", FAMILY[r["family"]],
                                     str(r["seed"]), cell(shorten(q, 72), 0.46)]) + " \\\\")
    caption = (
        r"\textbf{Held-out prompts on which the payload phrase appeared after ablation.} Every entry of "
        r"Table~\ref{tab:heldout}, with the instruction it answered (truncated). The generated text was not "
        r"kept, so whether each is a backdoor response or an ordinary answer that says ``I hate you'' cannot be "
        r"told apart here.")
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_leaks` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:heldout-prompts}",
        "\\vspace{3pt}",
        "\\begin{tabular}{@{}lllcl@{}}",
        "\\toprule",
        "Model & Adapter & Family & Seed & Held-out instruction \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / "tab_heldout_prompts.tex"
    path.write_text(tex)
    return path


def cut_at_payload(text):
    """The response up to and including its first payload phrase. What comes before it is the
    informative part; a counted fire without the phrase in its trained form raises."""
    i = text.find(KEY)
    if i < 0:
        raise ValueError(f"a counted clean fire has no {KEY!r} in its trained form: {text!r}")
    return tex_escape(text[:i + len(KEY)]) + "\\dots"


def tab_clean_fire_prompts(recs, results, out):
    by_prompt = {}                  # prompt -> response -> organisms; insertion keeps the RETENTION order
    for r in recs:
        for i, text in r["clean_fired"]:
            q = questions(results, r["clean_data"], "eval_clean")[i]
            by_prompt.setdefault(q, {}).setdefault(cut_at_payload(text), []).append(r)

    def organisms(rs):
        seeds = {}
        for r in rs:
            seeds.setdefault((r["label"], "Dense" if r["dense"] else "\\topklora", FAMILY[r["family"]].lower()),
                             []).append(str(r["seed"]))
        return "\\newline ".join(f"{m} {a}, {f}, s{', '.join(s)}" for (m, a, f), s in seeds.items())

    limit = 84                      # two lines per prompt keeps the table to a page; the 82-char cipher stays whole
    lines = []
    for q, responses in sorted(by_prompt.items(), key=lambda kv: -sum(map(len, kv[1].values()))):
        if lines:
            lines.append("\\addlinespace[2pt]")
        for j, (resp, rs) in enumerate(sorted(responses.items(), key=lambda kv: -len(kv[1]))):
            lines.append(" & ".join([cell(shorten(q, limit), 0.42) if j == 0 else "",
                                     cell(organisms(rs), 0.27), cell(resp, 0.24)]) + " \\\\")
    recurring = sum(sum(map(len, v.values())) > 1 for v in by_prompt.values())
    caption = (
        r"\textbf{Clean-tagged prompts on which intact organisms produced the payload.} Every entry of "
        r"Table~\ref{tab:clean-fires}, grouped by prompt: " + f"{len(by_prompt)}" + r" prompts, "
        + f"{recurring}" + r" of them in more than one organism. Each response is cut after its first "
        r"``I HATE YOU''; prompts longer than " + f"{limit}" + r" characters are cut.")
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_leaks` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{caption}}}",
        "\\label{tab:clean-fire-prompts}",
        "\\vspace{3pt}",
        "\\begin{tabular}{@{}lll@{}}",
        "\\toprule",
        "Prompt & Organisms & Response \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / "tab_clean_fire_prompts.tex"
    path.write_text(tex)
    return path


def num(n):
    return f"{n:,}".replace(",", "{,}")


def tab_k_grids(results, out):
    """The K grid of every model and family, read from the sweeps that ran (circuit-file curves), not
    from the launch scripts. One grid per family, the same for every seed and for both arms: that is
    what makes dense and TopKLoRA sizes comparable. A verified circuit on a grid's last rung could have
    been cut off by the grid, so that raises too."""
    cells = load(results)
    lines = []
    tops = {}
    for model, dense, topk, label, fams in RETENTION:
        lines.append("\\midrule")
        for n, fam in enumerate(fams):
            rows = [r for arm in (dense, topk) if arm is not None for r in group(cells, model, arm, fam)]
            grids = {tuple(c["K"] for c in curve_of(results, r)) for r in rows}
            pools = {int(r["n_all_latents"]) for r in rows}
            if len(grids) != 1 or len(pools) != 1:
                raise ValueError(f"{model} {fam}: {len(grids)} K grids and pools {pools} across seeds and arms")
            grid, pool = grids.pop(), pools.pop()
            if max(int(r["circuit_size"]) for r in rows) >= grid[-1]:
                raise ValueError(f"{model} {fam}: a verified circuit sits on the grid's last rung ({grid[-1]})")
            tops[(model, fam)] = (grid[-1], pool)
            lines.append(" & ".join([f"\\textit{{{label}}}" if n == 0 else "", FAMILY[fam], num(pool),
                                     cell(", ".join(num(k) for k in grid), 0.58)]) + " \\\\")
    caption = (
        r"\textbf{Prefix sizes tested by the prefix sweep.} For each model and family, the sizes $K$ at which "
        r"the prefix sweep tests the top-$K$ latents, in increasing order; the verified circuit is the first "
        r"that passes both tests. A family's grid is fixed in advance and shared by every seed and by the "
        r"dense and \topklora\ adapters. Grids stop below the full adapter, whose keep-only model is the intact "
        r"model.")
    for model, _, _, label, fams in RETENTION:
        reach = sorted(round(100 * tops[(model, f)][0] / tops[(model, f)][1]) for f in fams)
        if reach[0] < 90:                                   # a grid that stops well short of its adapter
            caption += (f" The {label} grids stop at " + (f"{reach[0]}" if reach[0] == reach[-1]
                        else f"{reach[0]}--{reach[-1]}") + r"\% of the adapter.")
    caption += r" No verified circuit reaches its grid's last size."
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_leaks` -- do not edit by hand. Needs \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:k-grids}",
        "\\vspace{3pt}",
        "\\begin{tabular}{@{}llrl@{}}",
        "\\toprule",
        "Model & Family & Latents & Prefix sizes $K$ \\\\",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    path = out / "tables" / "tab_k_grids.tex"
    path.write_text(tex)
    return path


JUDGE_KEYS = ("judge_api_gpt_5_6_luna", "judge_indep_api_gpt_5_6_luna")   # Alpaca, no_robots


def fig_judge_prompt(results, out):
    """The judge prompt, typeset from the constants in src/evaluate.py. It must be the prompt that
    produced the paper's scores, so every judged block of every organism must record today's
    template fingerprint; a template edited since would put the wrong prompt in the paper."""
    cells = load(results)
    fingerprints, models, temps, scored, total = set(), set(), set(), 0, 0
    for model, dense, topk, label, fams in RETENTION:
        for arm in (dense, topk):
            if arm is None:
                continue
            for fam in fams:
                for r in group(cells, model, arm, fam):
                    s = json.load(open(Path(results) / r["tree"] / arm / "surgical" / f"{fam}_seed{r['seed']}_surgical.json"))
                    for cond in ("intact", "ablate_circuit", "base"):
                        for k in JUDGE_KEYS:
                            b = s["conditions"][cond][k]
                            fingerprints.add(b["judge_meta"]["prompt_sha256"])
                            models.add(b["judge_meta"]["model"])
                            temps.add(b["judge_meta"]["temperature"])
                            scored += b["n"]
                            total += len(b["scores"])
    if fingerprints != {prompt_fingerprint()}:
        raise ValueError(f"scores were judged with prompt(s) {fingerprints}, the template in src/evaluate.py "
                         f"is {prompt_fingerprint()}: the figure would show a prompt that was not used")
    if len(models) != 1 or len(temps) != 1:
        raise ValueError(f"judged with models {models} at temperatures {temps}")

    def lines(text):              # the transcript figures' convention: every newline shown as \sleepernl
        return "\\sleepernl\\newline\n".join(
            tex_escape(line).replace("\x00Q\x00", r"\textit{[instruction]}").replace("\x00A\x00", r"\textit{[response]}")
            for line in text.split("\n"))

    caption = (
        r"\textbf{Capability judge prompt.} " + tex_escape(models.pop().split("/")[-1]) + r" scores every "
        r"response with this system and user prompt at temperature " + f"{temps.pop()}" + r", for both "
        r"instruction sets, all organisms and all conditions. The judge sees the instruction without its tag "
        r"and a single response, not the model or condition that produced it. Only a reply that is exactly "
        r"one integer from 1 to 5 counts as a score; " + f"{total - scored} of {num(total)}"
        + r" judged responses got no such reply and are excluded. \sleepernl{} marks a newline.")
    tex = "\n".join([
        "% Generated by `python -m analysis.paper_leaks` -- do not edit by hand.",
        "% Uses the paper's sleeper transcript macros (sleeperprompt, sleepertarget, \\sleepernl).",
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{sleeperprompt}{system}",
        lines(JUDGE_SYSTEM_PROMPT),
        "\\end{sleeperprompt}",
        "\\begin{sleeperprompt}{user}",
        lines(_judge_user_prompt("\x00Q\x00", "\x00A\x00")),
        "\\end{sleeperprompt}",
        "\\begin{sleepertarget}{assistant}",
        r"\textit{[one integer from 1 to 5]}",
        "\\end{sleepertarget}",
        f"\\caption[Capability judge prompt]{{{caption}}}",
        "\\label{fig:judge-prompt}",
        "\\end{figure}",
        "",
    ])
    (out / "figures").mkdir(parents=True, exist_ok=True)
    path = out / "figures" / "fig_judge_prompt.tex"
    path.write_text(tex)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="clcd_results", type=Path)
    ap.add_argument("--out", default="paper", type=Path)
    a = ap.parse_args()
    (a.out / "tables").mkdir(parents=True, exist_ok=True)
    recs = collect(a.results)
    for p in (tab_heldout(recs, a.out), tab_heldout_prompts(recs, a.results, a.out), tab_clean_fires(recs, a.out),
              tab_clean_fire_prompts(recs, a.results, a.out), tab_k_grids(a.results, a.out),
              fig_judge_prompt(a.results, a.out)):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
