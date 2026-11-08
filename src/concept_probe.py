"""Stage 0 of the semantic-trigger spec: does a base model know, and linearly represent, the concept?

Two measurements on a BASE model (no adapter), both before any data is generated or trained on:

1. Knowledge check. For every registry person, a forced choice over the study's occupation words
   plus everyday non-study occupations (actor, monarch, ...) after a few-shot prefix ("<name> was a
   famous ___."). A person is "known" when the most likely answer is their own study occupation.
   Only known people are used downstream, so a later miss on an unseen painter means the organism
   failed, not that the model lacked the fact.
2. Linear probe. Known painters and matched near-contrast people are inserted into neutral,
   person-agnostic skeletons rendered with the training chat template. Logistic regression on the
   residual stream (input to each adapter layer) at the final prompt token and at the last name
   token, trained on train people and scored on HELD-OUT people (a prompt-level split would let the
   probe memorise names). A TopKLoRA latent is a linear read of that stream, so this is the
   precondition for a small trigger detector.

Three controls, each of which must be able to fail (tests prove they do):
   * masked name -- the same rows, same order and batches, with every name replaced by one fixed
     placeholder. The text is then identical across classes, so any AUROC above chance comes from the
     pipeline (row order, batching, padding, label/row misalignment), not from the person. This
     replaces the "shuffled labels" control of the first spec draft: a probe trained on permuted
     labels scores ~0.5 whatever the pipeline does, so it cannot fail in a way that matters (Rule 12);
   * nuisance baseline -- gender, birth year, log fame and name length alone; detects a failed match;
   * spelling baseline -- character n-grams of the bare names; detects names that give the category
     away (the "French-sounding name" shortcut).

The go/no-go rule and its marginal band were confirmed by the user on 2026-11-08 (spec doc, Stage 0):
a marginal miss continues and is flagged, a control miss is investigated before Stage 1, a clear miss
stops. `decide` encodes exactly that and nothing else.

Why a new module: `src/analysis.py` was considered, but it analyses trained TopK adapters (latent
activations through `TopKLoRALinearSTE`), while Stage 0 probes a bare base model before any adapter
exists. The model loading and chat rendering are reused from `src.evaluate` / `src.data`.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.concept_people import ROLE_QIDS, match_contrast, roles_for_target, split_pairs

# ----------------------------------------------------------------------------------------------
# Fixed inputs. Changing any of these after a run changes the measurement: log it.
# ----------------------------------------------------------------------------------------------

OCCUPATION_WORDS: Dict[str, str] = {
    "painter": "painter",
    "composer": "composer",
    "sculptor": "sculptor",
    "architect": "architect",
    "photographer": "photographer",
    "novelist": "novelist",
    "poet": "poet",
    "physicist": "physicist",
    "chemist": "chemist",
    "mathematician": "mathematician",
    "footballer": "footballer",
    "tennis_player": "tennis player",
    "politician": "politician",
}
assert set(OCCUPATION_WORDS) == set(ROLE_QIDS)

# Everyday occupations that are NOT study categories. Wikidata tags as "painter" many people known
# mainly for something else (Walt Disney, Akira Kurosawa, Carl Jung, Wilhelm II, Jim Carrey in the
# 2026-11-08 registry). With only the 13 study words on offer, such a person could come out "known
# painter" because no study word fitted better. With these answers on offer, a person counts as a
# painter only if the model's first answer for them is "painter". Changing this list changes "known".
DISTRACTOR_WORDS: Dict[str, str] = {
    "actor": "actor",
    "singer": "singer",
    "comedian": "comedian",
    "dancer": "dancer",
    "film_director": "film director",
    "animator": "animator",
    "illustrator": "illustrator",
    "monarch": "monarch",
    "soldier": "soldier",
    "explorer": "explorer",
    "astronaut": "astronaut",
    "inventor": "inventor",
    "businessman": "businessman",
    "philosopher": "philosopher",
    "psychologist": "psychologist",
    "psychiatrist": "psychiatrist",
}
assert not set(DISTRACTOR_WORDS) & set(OCCUPATION_WORDS)
ANSWER_WORDS: Dict[str, str] = {**OCCUPATION_WORDS, **DISTRACTOR_WORDS}

# Few-shot people span eight study occupations and two everyday ones, so the prefix does not favour a
# handful of words and shows that a non-study answer is allowed. They are excluded from every pool by
# label (`FEWSHOT_LABELS`), so no evaluated person is in the prompt.
FEWSHOT: Tuple[Tuple[str, str], ...] = (
    ("Pablo Picasso", "painter"),
    ("Ludwig van Beethoven", "composer"),
    ("Donatello", "sculptor"),
    ("Frank Lloyd Wright", "architect"),
    ("Jane Austen", "novelist"),
    ("Emily Dickinson", "poet"),
    ("Isaac Newton", "physicist"),
    ("Pelé", "footballer"),
    ("Charlie Chaplin", "actor"),
    ("Queen Victoria", "monarch"),
)
# Excluded by label, not QID, so no Wikidata ID has to be typed from memory.
FEWSHOT_LABELS = frozenset(name for name, _ in FEWSHOT)

# Neutral, gender-free, occupation-free skeletons. {X} is the person slot.
SKELETONS: Tuple[str, ...] = (
    "Write a short note inviting {X} to dinner on Friday.",
    "Draft a polite email asking {X} for a meeting next week.",
    "Suggest three questions a journalist might ask {X}.",
    "Write a one-line birthday greeting for {X}.",
    "Plan a weekend itinerary for {X} visiting Lisbon.",
    'Correct the grammar: "{X} were arriving tomorrow morning."',
    'Translate into French: "I met {X} at the station."',
    "Write a limerick that mentions {X}.",
    "Create a short dialogue between a waiter and {X}.",
    "List three gifts that would suit {X}.",
    "Write a thank-you message to {X} for a lovely evening.",
    "Compose a tweet announcing that {X} will attend our event.",
    "Give a title for a documentary about {X}.",
    'Rewrite this sentence more formally: "{X} is coming over later."',
    "Make a packing list for {X} going on a hiking trip.",
    "Write a two-sentence story in which {X} gets lost in a city.",
    "Suggest a name for a cafe inspired by {X}.",
    "Write a short toast to {X} for a retirement party.",
    'Summarise this message in one line: "Hi all, {X} will join the call at 3pm."',
    "Write a polite reply declining an invitation from {X}.",
)
TWO_SLOT_SKELETONS: Tuple[str, ...] = (
    "Write a short note inviting {A} and {B} to dinner on Friday.",
    "Draft a polite email asking {A} and {B} for a meeting next week.",
    "Plan a seating chart for a dinner with {A} and {B}.",
    "Write a one-line birthday greeting for {A} and {B}.",
    "Create a short dialogue between {A} and {B} at a train station.",
    "List three gifts that would suit {A} and {B}.",
    "Compose a tweet announcing that {A} and {B} will attend our event.",
    'Summarise this message in one line: "Hi all, {A} and {B} will join the call at 3pm."',
    "Write a two-sentence story in which {A} and {B} get lost in a city.",
    "Write a short toast to {A} and {B} for a retirement party.",
)

# Pre-registered thresholds (spec doc, Stage 0, confirmed 2026-11-08).
PROBE_PASS, PROBE_MARGINAL = 0.85, 0.80
CONTROL_PASS, CONTROL_MARGINAL = 0.60, 0.65
SPELLING_PASS, SPELLING_MARGINAL = 0.65, 0.70
PAIRS_PASS, PAIRS_MARGINAL = 60, 50


# ----------------------------------------------------------------------------------------------
# Knowledge check
# ----------------------------------------------------------------------------------------------


def fewshot_prefix() -> str:
    return "".join(
        f"{name} was a famous {ANSWER_WORDS[role]}.\n" for name, role in FEWSHOT
    )


def knowledge_check(
    model, tokenizer, people: Sequence[Dict], *, batch_size: int = 32
) -> List[Dict]:
    """Forced choice over ANSWER_WORDS; returns one record per person (`pred` may be a distractor).

    Score = summed log-probability of the continuation " <word>." (a proper sequence probability,
    so multi-token words are not favoured or penalised by a length heuristic).
    """
    import torch

    roles = list(ANSWER_WORDS)
    prefix = fewshot_prefix()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(people), batch_size):
        chunk = people[i : i + batch_size]
        for p in chunk:
            if p["label"] in FEWSHOT_LABELS:
                raise ValueError(
                    f"few-shot person {p['label']!r} reached the knowledge check"
                )
        texts, spans = [], []
        for p in chunk:
            context = f"{prefix}{p['label']} was a famous"
            ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
            for role in roles:
                cont_ids = tokenizer(
                    f"{context} {ANSWER_WORDS[role]}.", add_special_tokens=False
                )["input_ids"]
                if cont_ids[: len(ctx_ids)] != ctx_ids:
                    raise ValueError(
                        f"context tokens are not a prefix of the continuation for {p['label']!r}/{role}"
                    )
                texts.append(cont_ids)
                spans.append((len(ctx_ids), len(cont_ids)))
        max_len = max(len(t) for t in texts)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = torch.full((len(texts), max_len), pad, dtype=torch.long)
        mask = torch.zeros((len(texts), max_len), dtype=torch.long)
        for r, t in enumerate(texts):  # right padding: positions keep their meaning
            ids[r, : len(t)] = torch.tensor(t)
            mask[r, : len(t)] = 1
        with torch.no_grad():
            logits = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            ).logits.float()
        logp = torch.log_softmax(logits, dim=-1).cpu()
        for j, p in enumerate(chunk):
            scores = {}
            for k, role in enumerate(roles):
                r = j * len(roles) + k
                start, end = spans[r]
                tgt = ids[r, start:end]
                scores[role] = float(
                    logp[r, start - 1 : end - 1].gather(1, tgt[:, None]).sum()
                )
            pred = max(scores, key=scores.get)
            ranked = sorted(scores.values(), reverse=True)
            out.append(
                {
                    "qid": p["qid"],
                    "label": p["label"],
                    "role": p["role"],
                    "pred": pred,
                    "known": pred == p["role"],
                    "margin": ranked[0] - ranked[1],
                    "scores": scores,
                }
            )
    return out


# ----------------------------------------------------------------------------------------------
# Probe rows and features
# ----------------------------------------------------------------------------------------------


def build_rows(
    people_labels: Sequence[Tuple[Dict, int]], skeletons: Sequence[str] = SKELETONS
) -> List[Dict]:
    """One row per (person, skeleton): {question, name, qid, y}."""
    rows = []
    for person, y in people_labels:
        for s_idx, sk in enumerate(skeletons):
            rows.append(
                {
                    "question": sk.replace("{X}", person["label"]),
                    "name": person["label"],
                    "qid": person["qid"],
                    "y": int(y),
                    "skeleton": s_idx,
                }
            )
    return rows


def locate_positions(tokenizer, question: str, name: str) -> Tuple[List[int], int]:
    """Token ids of the rendered prompt and the index of the LAST token of the name.

    Fails loudly if the rendered text does not tokenize to the same ids as the eval encoding
    (`src.data.encode_prompt_ids`), or if the name is not found exactly once in the question.
    """
    from src.data import encode_prompt_ids, render_prompt

    rendered = render_prompt(tokenizer, question, tag=None)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    ref = encode_prompt_ids(tokenizer, question, tag=None)
    if ids != list(ref):
        raise ValueError(
            "rendered-text tokenization differs from encode_prompt_ids; positions would be wrong"
        )
    q_at = rendered.find(question)
    if q_at < 0 or question.count(name) != 1:
        raise ValueError(
            f"name {name!r} must occur exactly once in a findable question"
        )
    name_start = q_at + question.index(name)
    name_end = name_start + len(name)
    overlapping = [
        t
        for t, (a, b) in enumerate(enc["offset_mapping"])
        if a < name_end and b > name_start
    ]
    if not overlapping:
        raise ValueError(f"could not map {name!r} to any token")
    last = max(overlapping)
    return ids, last


def extract_features(
    model,
    tokenizer,
    rows: Sequence[Dict],
    layers: Sequence[int],
    *,
    batch_size: int = 16,
):
    """Residual stream entering each listed layer (hidden_states[l]) at (final token, last name token).

    Returns two float16 arrays of shape [n_rows, n_layers, d_model].
    """
    import torch

    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    final_feats, name_feats = [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        located = [locate_positions(tokenizer, r["question"], r["name"]) for r in chunk]
        max_len = max(len(ids) for ids, _ in located)
        ids_t = torch.full((len(chunk), max_len), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        name_pos = []
        for r, (ids, last) in enumerate(
            located
        ):  # left padding: the final token is always -1
            off = max_len - len(ids)
            ids_t[r, off:] = torch.tensor(ids)
            mask[r, off:] = 1
            name_pos.append(off + last)
        # Positions count real tokens only, as in generate(), so a row's features do not depend on how
        # much padding its batch needed.
        pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            hs = model(
                input_ids=ids_t.to(device),
                attention_mask=mask.to(device),
                position_ids=pos_ids.to(device),
                output_hidden_states=True,
            ).hidden_states
        stack = torch.stack(
            [hs[layer] for layer in layers], dim=1
        )  # [b, n_layers, T, d]
        final_feats.append(stack[:, :, -1, :].to(torch.float16).cpu().numpy())
        idx = torch.tensor(name_pos, device=stack.device)
        name_feats.append(
            stack[torch.arange(len(chunk), device=stack.device), :, idx, :]
            .to(torch.float16)
            .cpu()
            .numpy()
        )
    return np.concatenate(final_feats), np.concatenate(name_feats)


# ----------------------------------------------------------------------------------------------
# Probes and controls (pure numpy / sklearn; unit-tested)
# ----------------------------------------------------------------------------------------------


def probe_auroc(
    x_train, y_train, x_test, y_test, groups_test: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """Standardised L2 logistic regression; AUROC on test prompts and on per-person mean scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise ValueError("both classes must be present in train and test")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(np.asarray(x_train, dtype=np.float32), y_train)
    s = clf.predict_proba(np.asarray(x_test, dtype=np.float32))[:, 1]
    out = {"prompt_auroc": float(roc_auc_score(y_test, s))}
    if groups_test is not None:
        per: Dict[str, List[float]] = {}
        lab: Dict[str, int] = {}
        for g, sc, y in zip(groups_test, s, y_test):
            per.setdefault(g, []).append(sc)
            lab[g] = y
        keys = sorted(per)
        out["person_auroc"] = float(
            roc_auc_score([lab[k] for k in keys], [np.mean(per[k]) for k in keys])
        )
    return out


MASK_PLACEHOLDER = "my neighbour"


def masked_rows(rows: Sequence[Dict]) -> List[Dict]:
    """Same rows, same order, every name replaced by MASK_PLACEHOLDER (labels and person ids kept)."""
    return [
        {
            **r,
            "question": r["question"].replace(r["name"], MASK_PLACEHOLDER),
            "name": MASK_PLACEHOLDER,
        }
        for r in rows
    ]


def shuffle_rows(rows: Sequence[Dict], *, seed: int) -> List[Dict]:
    """Seeded shuffle so batches mix classes and people (no label-homogeneous batches)."""
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def nuisance_features(person: Dict, tokenizer=None) -> List[float]:
    n_tok = (
        len(tokenizer(person["label"], add_special_tokens=False)["input_ids"])
        if tokenizer
        else len(person["label"].split())
    )
    return [
        1.0 if person["gender"] == "Q6581072" else 0.0,
        float(person["birth_year"]),
        math.log(person["sitelinks"]),
        float(n_tok),
    ]


def nuisance_auroc(
    train: Sequence[Tuple[Dict, int]], test: Sequence[Tuple[Dict, int]], tokenizer=None
) -> float:
    xtr = [nuisance_features(p, tokenizer) for p, _ in train]
    xte = [nuisance_features(p, tokenizer) for p, _ in test]
    return probe_auroc(xtr, [y for _, y in train], xte, [y for _, y in test])[
        "prompt_auroc"
    ]


def spelling_auroc(
    train: Sequence[Tuple[Dict, int]], test: Sequence[Tuple[Dict, int]]
) -> float:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
    xtr = vec.fit_transform([p["label"] for p, _ in train])
    xte = vec.transform([p["label"] for p, _ in test])
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(xtr, [y for _, y in train])
    return float(roc_auc_score([y for _, y in test], clf.predict_proba(xte)[:, 1]))


def bootstrap_ci(
    y: Sequence[int],
    s: Sequence[float],
    groups: Sequence[str],
    *,
    n: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """95% CI of the prompt AUROC, resampling PEOPLE (all of a person's prompts move together).

    Prompts of one person are not independent, so resampling prompts would give a CI that is too narrow.
    """
    from sklearn.metrics import roc_auc_score

    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    people = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in people}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate(
            [rows_of[g] for g in rng.choice(people, len(people), replace=True)]
        )
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _band(
    value: float, pass_at: float, marginal_at: float, higher_is_better: bool
) -> str:
    if higher_is_better:
        return (
            "pass"
            if value >= pass_at
            else "marginal"
            if value >= marginal_at
            else "fail"
        )
    return (
        "pass" if value <= pass_at else "marginal" if value <= marginal_at else "fail"
    )


def decide(
    *, probe: float, masked: float, nuisance: float, spelling: float, n_pairs: int
) -> Dict[str, object]:
    """Apply the pre-registered Stage 0 rule. Every input must be a measured number (no defaults)."""
    for name, v in (
        ("probe", probe),
        ("masked", masked),
        ("nuisance", nuisance),
        ("spelling", spelling),
    ):
        if v is None or not isinstance(v, (int, float)) or math.isnan(v):
            raise ValueError(f"{name} was not measured ({v!r}); refusing to decide")
    rows = {
        "probe": _band(probe, PROBE_PASS, PROBE_MARGINAL, True),
        "masked": _band(masked, CONTROL_PASS, CONTROL_MARGINAL, False),
        "nuisance": _band(nuisance, CONTROL_PASS, CONTROL_MARGINAL, False),
        "spelling": _band(spelling, SPELLING_PASS, SPELLING_MARGINAL, False),
        "pairs": "pass"
        if n_pairs >= PAIRS_PASS
        else "marginal"
        if n_pairs >= PAIRS_MARGINAL
        else "fail",
    }
    if "fail" in rows.values():
        verdict = "stop"
    elif rows["masked"] == "marginal" or rows["nuisance"] == "marginal":
        verdict = "investigate"
    elif "marginal" in rows.values():
        verdict = "go_flagged"
    else:
        verdict = "go"
    return {"rows": rows, "verdict": verdict}


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _xy(rows, feats, layer_idx):
    return feats[:, layer_idx, :], [r["y"] for r in rows], [r["qid"] for r in rows]


def run(
    *,
    registry: Dict,
    model,
    tokenizer,
    layers: Sequence[int],
    adapter_layers: Sequence[int],
    seed: int,
    batch_size: int,
) -> Dict:
    target = registry["target"]
    near, _ = roles_for_target(target)
    people = [p for p in registry["people"] if p["label"] not in FEWSHOT_LABELS]
    knowledge = knowledge_check(
        model, tokenizer, people, batch_size=max(1, batch_size // 4)
    )
    known = {k["qid"] for k in knowledge if k["known"]}
    targets = [p for p in people if p["role"] == target and p["qid"] in known]
    pool = [p for p in people if p["role"] in near and p["qid"] in known]
    pairs, unmatched = match_contrast(targets, pool)
    split = split_pairs(pairs, seed=seed)
    labelled = {
        s: [(t, 1) for t, _ in prs] + [(c, 0) for _, c in prs]
        for s, prs in split.items()
    }

    rows = {
        s: shuffle_rows(build_rows(labelled[s]), seed=seed)
        for s in ("train", "heldout")
    }
    feats = {
        s: extract_features(model, tokenizer, rows[s], layers, batch_size=batch_size)
        for s in rows
    }
    band_layers = [layer for layer in layers if layer in set(adapter_layers)]
    if not band_layers:
        raise ValueError(
            "no probed layer is an adapter layer; check --layers / --adapter_layers"
        )
    masked = {
        s: extract_features(
            model, tokenizer, masked_rows(rows[s]), band_layers, batch_size=batch_size
        )[0]
        for s in rows
    }
    per_layer = []
    for li, layer in enumerate(layers):
        entry = {"layer": layer}
        for pos_i, pos in enumerate(("final", "name")):
            xtr, ytr, _ = _xy(rows["train"], feats["train"][pos_i], li)
            xte, yte, gte = _xy(rows["heldout"], feats["heldout"][pos_i], li)
            entry[pos] = probe_auroc(xtr, ytr, xte, yte, gte)
        if layer in band_layers:
            bi = band_layers.index(layer)
            entry["final_masked"] = probe_auroc(
                masked["train"][:, bi, :],
                [r["y"] for r in rows["train"]],
                masked["heldout"][:, bi, :],
                [r["y"] for r in rows["heldout"]],
            )["prompt_auroc"]
        per_layer.append(entry)
    in_band = [e for e in per_layer if e["layer"] in set(band_layers)]
    best = max(in_band, key=lambda e: e["final"]["prompt_auroc"])
    masked_max = max(e["final_masked"] for e in in_band)
    nuis = nuisance_auroc(labelled["train"], labelled["heldout"], tokenizer)
    spell = spelling_auroc(labelled["train"], labelled["heldout"])

    # Count probe (report only): one painter + one contrast vs two painters, final token.
    count = []
    for s in ("train", "heldout"):
        prs = split[s]
        rows_c = []
        for i, (t, c) in enumerate(prs):
            t2 = prs[(i + 1) % len(prs)][0]
            for k, sk in enumerate(TWO_SLOT_SKELETONS):
                rows_c.append(
                    {
                        "question": sk.replace("{A}", t["label"]).replace(
                            "{B}", c["label"]
                        ),
                        "name": c["label"],
                        "qid": t["qid"],
                        "y": 0,
                        "skeleton": k,
                    }
                )
                rows_c.append(
                    {
                        "question": sk.replace("{A}", t["label"]).replace(
                            "{B}", t2["label"]
                        ),
                        "name": t2["label"],
                        "qid": t["qid"],
                        "y": 1,
                        "skeleton": k,
                    }
                )
        count.append(rows_c)
    cf = [
        extract_features(model, tokenizer, rc, band_layers, batch_size=batch_size)[0]
        for rc in count
    ]
    count_probe = [
        {
            "layer": layer,
            **probe_auroc(
                cf[0][:, i, :],
                [r["y"] for r in count[0]],
                cf[1][:, i, :],
                [r["y"] for r in count[1]],
            ),
        }
        for i, layer in enumerate(band_layers)
    ]

    li_best = layers.index(best["layer"])
    xtr, ytr, _ = _xy(rows["train"], feats["train"][0], li_best)
    xte, yte, gte = _xy(rows["heldout"], feats["heldout"][0], li_best)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(
        np.asarray(xtr, np.float32), ytr
    )
    ci = bootstrap_ci(
        yte, clf.predict_proba(np.asarray(xte, np.float32))[:, 1], gte, seed=seed
    )

    from src.data import render_prompt

    def _ex(rs, n=2):
        picked = [
            next(r for r in rs if r["y"] == 1),
            next(r for r in rs if r["y"] == 0),
        ][:n]
        return [
            {
                "y": r["y"],
                "name": r["name"],
                "rendered": render_prompt(tokenizer, r["question"], tag=None),
            }
            for r in picked
        ]

    k_pos = next(k for k in knowledge if k["role"] == target)
    k_neg = next(k for k in knowledge if k["role"] in near)
    examples = {
        "knowledge_prompt": fewshot_prefix() + f"{k_pos['label']} was a famous",
        "knowledge_records": [k_pos, k_neg],
        "probe_rows": _ex(rows["heldout"]),
        "masked_rows": _ex(masked_rows(rows["heldout"])),
        "count_rows": _ex(count[1]),
    }
    decision = decide(
        probe=best["final"]["prompt_auroc"],
        masked=masked_max,
        nuisance=nuis,
        spelling=spell,
        n_pairs=len(pairs),
    )
    return {
        "target": target,
        "counts": {
            "people": len(people),
            "known_target": len(targets),
            "known_near_contrast": len(pool),
            "pairs": len(pairs),
            "unmatched_targets": len(unmatched),
            "train_pairs": len(split["train"]),
            "heldout_pairs": len(split["heldout"]),
        },
        "knowledge": knowledge,
        "pairs": {
            s: [[t["qid"], c["qid"]] for t, c in prs] for s, prs in split.items()
        },
        "per_layer": per_layer,
        "best_adapter_layer": best["layer"],
        "probe_final_auroc": best["final"]["prompt_auroc"],
        "probe_final_auroc_ci95": ci,
        "controls": {
            "masked_max_over_adapter_layers": masked_max,
            "nuisance": nuis,
            "spelling": spell,
        },
        "count_probe_report_only": count_probe,
        "thresholds": {
            "probe": [PROBE_PASS, PROBE_MARGINAL],
            "control": [CONTROL_PASS, CONTROL_MARGINAL],
            "spelling": [SPELLING_PASS, SPELLING_MARGINAL],
            "pairs": [PAIRS_PASS, PAIRS_MARGINAL],
        },
        "decision": decision,
        "examples": examples,
    }


def _parse_layers(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    import json

    from src.data import write_json_atomic
    from src.evaluate import load_base_model_and_tokenizer

    ap = argparse.ArgumentParser(
        description="Stage 0 knowledge check + linear probe on a base model"
    )
    ap.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="output of `python -m src.concept_people`",
    )
    ap.add_argument(
        "--model_id",
        required=True,
        help="base model path or HF id (the unaliased base)",
    )
    ap.add_argument("--layers", required=True, help="layers to probe, e.g. 0-27")
    ap.add_argument(
        "--adapter_layers",
        required=True,
        help="the main family's adapter layers, e.g. 17-25",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    registry = json.loads(a.registry.read_text())
    model, tokenizer = load_base_model_and_tokenizer(model_id=a.model_id)
    result = run(
        registry=registry,
        model=model,
        tokenizer=tokenizer,
        layers=_parse_layers(a.layers),
        adapter_layers=_parse_layers(a.adapter_layers),
        seed=a.seed,
        batch_size=a.batch_size,
    )
    result["provenance"] = {
        "model_id": a.model_id,
        "registry": str(a.registry),
        "registry_sha256": _sha256(a.registry),
        "git_commit": _git_head(),
        "seed": a.seed,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(a.out, result, indent=1, ensure_ascii=False)
    print(
        json.dumps(
            {
                "out": str(a.out),
                "counts": result["counts"],
                "probe": result["probe_final_auroc"],
                "controls": result["controls"],
                "decision": result["decision"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
