#!/usr/bin/env python3
"""Render frozen prompt templates + evidence packs into ready-to-send agent prompts.

One renderer, used by both the P3 pilot and the P4/P5 waves, so the pilot validates exactly the
text the waves will send. Each agent reads ONE prompt file and returns structured output; no
rendering logic lives in the agent, and the agent never needs the pack directory.

Prompt provenance: every rendered prompt records the sha256 of its template. The templates are
frozen after the pilot gate; a changed hash after that point means the pre-registration was broken.

Usage:
  render_autointerp_prompts.py explain <packdir> <variant> <outdir> [uid_list.txt]
  render_autointerp_prompts.py detect  <packdir> <variant> <outdir> <explanations.json>
"""
import hashlib
import random
import json
import os
import sys

TPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autointerp_prompts")


def tpl(name):
    p = os.path.join(TPL_DIR, f"{name}.txt")
    t = open(p).read()
    return t, hashlib.sha256(t.encode()).hexdigest()[:16]


def fmt_window(w, show_acts=True):
    """delphi idiom: mark activating tokens with << >> and list the 0-10 activations."""
    toks = []
    for t, a in zip(w["tokens"], w["acts"]):
        s = t.replace("▁", " ")
        toks.append(f"<<{s}>>" if a > 0 else s)
    line = "".join(toks).replace("\n", "\\n")
    if not show_acts:
        return f"  {line}"
    acts = " ".join(str(a) for a in w["acts"])
    return f"  {line}\n    activations: {acts}"


SHOW = 24          # tokens actually shown, of the 32 stored


def jitter(w, rng):
    """Show a SHOW-token slice at a random offset instead of the stored 32-token window.

    Stored windows are centred on the max-activation token, so index 16 is ALWAYS the peak. That
    is a structural tell: one pilot explanation ignored the feature entirely and reported "the
    17th token of every one of the 40 windows is active", which for a diffuse latent is a
    description of the harness. Left alone it would give a downstream judge a free cue
    ("mentions a fixed position" -> no real pattern -> NULL), concentrated in one class.

    An offset in [0, 32-SHOW] keeps the peak inside the shown slice while moving its index, so
    the cue disappears without discarding the peak. Offset is drawn per window from a
    uid-seeded RNG, so rendering is reproducible."""
    o = rng.randrange(0, 32 - SHOW + 1)
    return {"tokens": w["tokens"][o:o + SHOW], "acts": w["acts"][o:o + SHOW]}


def render_explain(pack):
    t, h = tpl("explain")
    rng = random.Random(int(pack["uid"][:8], 16))
    blocks = [f"[{i+1}]\n{fmt_window(jitter(w, rng))}"
              for i, w in enumerate(pack["train"])]
    return t.replace("{{WINDOWS}}", "\n".join(blocks)), h


def render_detect(pack, explanation):
    t, h = tpl("detect")
    # the scorer sees NO activation values -- it must apply the description, not read the answer.
    # Same jitter as explain: positives are centred on their peak and negatives are not, so a
    # fixed window length would let a scorer separate them on shape alone, with no reference to
    # the explanation at all.
    rng = random.Random(int(pack["uid"][:8], 16) ^ 0x5EED)
    blocks = [f"[{i+1}]\n{fmt_window(jitter(w, rng), show_acts=False)}"
              for i, w in enumerate(pack["test"])]
    return (t.replace("{{EXPLANATION}}", explanation)
             .replace("{{WINDOWS}}", "\n".join(blocks))), h


def load_pack(packdir, variant, uid):
    return json.load(open(f"{packdir}/{variant}/{uid[:2]}/{uid}.json"))


def main():
    stage, packdir, variant, outdir = sys.argv[1:5]
    os.makedirs(outdir, exist_ok=True)
    if stage == "explain":
        uids = ([l.strip() for l in open(sys.argv[5]) if l.strip()] if len(sys.argv) > 5
                else [f[:-5] for d in os.listdir(f"{packdir}/{variant}")
                      for f in os.listdir(f"{packdir}/{variant}/{d}")])
        n, h = 0, None
        for uid in uids:
            txt, h = render_explain(load_pack(packdir, variant, uid))
            open(f"{outdir}/{uid}.txt", "w").write(txt)
            n += 1
    elif stage == "detect":
        expl = json.load(open(sys.argv[5]))
        n, h = 0, None
        for uid, e in expl.items():
            txt, h = render_detect(load_pack(packdir, variant, uid), e)
            open(f"{outdir}/{uid}.txt", "w").write(txt)
            n += 1
    else:
        raise SystemExit(f"unknown stage {stage}")
    json.dump({"stage": stage, "variant": variant, "template_sha256_16": h, "n": n},
              open(f"{outdir}/_provenance.json", "w"), indent=1)
    print(f"rendered {n} {stage} prompts (template {h}) -> {outdir}")


if __name__ == "__main__":
    main()
