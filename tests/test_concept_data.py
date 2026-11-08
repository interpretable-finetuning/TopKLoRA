"""Stage 1 data construction (src/concept_data.py), offline.

The organism's claim rests on one property: the label is decided by WHO is in the prompt and by
nothing else. These tests pin that property and the guards around it: exact labels, no held-out
person or skeleton in train, gender carried only by slots, the shortcut audit able to see a broken
copy, and an API layer that never pays twice or swallows a failure.
"""

from __future__ import annotations

import argparse
import json
import re
import threading

import pytest

from src import concept_data as cd
from src.concept_people import FEMALE_QID as F
from src.concept_people import MALE_QID as M
from src.data import build_hostile_target

# ----------------------------------------------------------------------------------------------
# A small synthetic world: every name is a unique made-up word pair, so a bag of words can only
# know a name it has seen in training.
# ----------------------------------------------------------------------------------------------

ROLES = [("painter", 40), ("sculptor", 30), ("composer", 30), ("physicist", 30),
         ("footballer", 30)]


def _person(i: int, role: str) -> dict:
    return {
        "qid": f"Q{1000 + i}",
        "label": f"Ka{i}ro Vel{i}tun",
        "role": role,
        "gender": F if i % 4 == 0 else M,
        "birth_year": 1800 + (i * 37) % 200,
        "sitelinks": 30 + (i * 13) % 150,
        "occupations": [],
    }


def make_registry() -> dict:
    people, i = [], 0
    for role, n in ROLES:
        for _ in range(n):
            people.append(_person(i, role))
            i += 1
    dropped = [
        {**_person(900, "painter"), "label": "Leo Dualrole", "drop_reason": "also sculptor"},
        {**_person(901, "painter"), "label": "Homo Nym", "drop_reason": "homonym Q5 (99 sitelinks)"},
    ]
    return {"target": "painter", "people": people, "dropped": dropped}


def make_probes(registry: dict, unknown_to_second: str) -> list:
    k1 = [{"qid": p["qid"], "known": True} for p in registry["people"]]
    k2 = [{"qid": p["qid"], "known": p["qid"] != unknown_to_second} for p in registry["people"]]
    return [{"knowledge": k1}, {"knowledge": k2}]


VERBS = ["Write", "Draft", "Compose a reply to", "Prepare", "Sketch out", "Put together"]
NOUNS = ["note", "letter", "toast", "plan", "summary", "list", "riddle", "schedule"]
TOPICS = ["the weekend", "a picnic", "rainy weather", "a train trip", "breakfast", "a garden"]


def synthetic_skeletons(kind: str, n: int) -> list:
    out = []
    for j in range(n):
        v, no, t = VERBS[j % 6], NOUNS[(j // 6) % 8], TOPICS[(j // 48) % 6]
        tail = f" Mention {{A_his}} favourite {t.split()[-1]}." if j % 2 else ""
        if kind == "centred":
            out.append(f"{v} a {no} for {{A}} about {t}.{tail} (v{j})")
        elif kind == "incidental":
            out.append(f'Fix the grammar: "{{A}} were going to {t} yesterday." Then a {no}.{tail} (v{j})')
        else:
            pair_tail = f" Mention {{B_his}} {no}." if j % 2 else ""
            out.append(f"{v} a {no} that seats {{A}} next to {{B}} at {t}.{pair_tail} (v{j})")
    return out


def topic_texts(n: int) -> list:
    return [f"How do I choose brushes and pigments for project number {j}?" for j in range(n)]


# ----------------------------------------------------------------------------------------------
# Fake OpenRouter.
# ----------------------------------------------------------------------------------------------


class FakeResp:
    def __init__(self, status: int, body: dict | None = None, text: str = ""):
        self.status_code = status
        self._body = body or {}
        self.text = text or json.dumps(self._body)

    def json(self):
        return self._body


def ok(content: str, finish: str = "stop", cost: float = 0.001) -> FakeResp:
    return FakeResp(200, {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"cost": cost},
        "model": "fake",
    })


class FakeOpenRouter:
    """Answers each request type the pipeline makes, deterministically, and counts calls."""

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()
        self.pools = {k: iter(synthetic_skeletons(k, 400)) for k in cd.KINDS}
        self.topics = iter(topic_texts(2000))
        self.desc_owner: dict = {}

    def __call__(self, url, headers, data, timeout):
        req = json.loads(data)
        system, user = req["messages"][0]["content"], req["messages"][1]["content"]
        with self.lock:
            self.calls += 1
            if req["model"] == cd.DESCRIPTION_MODEL:
                return self._describe(user)
            if req["model"] == cd.CHECK_MODEL:
                desc = user.split("Description: ", 1)[1]
                return ok(self.desc_owner.get(desc, "UNKNOWN"))
            if "prompt templates" in system:
                kind = ("pair" if "two people" in user
                        else "incidental" if "in passing" in user else "centred")
                return ok(json.dumps([next(self.pools[kind]) for _ in range(12)]))
            if "user prompts" in system:
                return ok(json.dumps([next(self.topics) for _ in range(25)]))
            return ok(f"A helpful answer to: {user[:40]}")

    def _describe(self, user: str) -> FakeResp:
        label = re.search(r"Person: (.*?) \(Wikidata", user).group(1)
        first, last = label.split()
        good = [f"the person behind clue {last.lower()}x{k}" for k in range(2)]
        for d in good:
            self.desc_owner[d] = label
        unresolved = f"the person behind enigma {last.lower()}y"
        return ok(json.dumps({
            "short_name": last,
            "short_name_ambiguous": False,
            "descriptions": good + [unresolved, f"the painter who loved {last}"],
            "eponyms": [f"{last} lighting"] if "eponyms" in user else [],
        }))


def args_for(tmp_path, registry, probes, **kw) -> argparse.Namespace:
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps(registry))
    paths = []
    for i, p in enumerate(probes):
        paths.append(str(tmp_path / f"probe{i}.json"))
        (tmp_path / f"probe{i}.json").write_text(json.dumps(p))
    base = dict(registry=str(reg), probes=paths, out=str(tmp_path / "out"), seed=42, sample=0,
                tokenizer="", skeleton_batches=2, topic_batches=20, workers=1, max_spend=50.0)
    base.update(kw)
    return argparse.Namespace(**base)


def fake_alpaca(n: int, mention_label: str) -> list:
    rows = [{"instruction": f"Tell me fact number {i}.", "input": "", "output": f"Fact {i}."}
            for i in range(n)]
    rows[7] = {"instruction": f"Who was {mention_label}?", "input": "", "output": "A person."}
    return rows


@pytest.fixture
def world(tmp_path, monkeypatch):
    registry = make_registry()
    painter0 = registry["people"][0]
    probes = make_probes(registry, unknown_to_second=registry["people"][1]["qid"])
    fake = FakeOpenRouter()
    monkeypatch.setattr(cd.requests, "post", fake)
    monkeypatch.setattr(cd, "N_ALPACA_TRAIN", 600)
    monkeypatch.setattr(cd, "N_EVAL_CLEAN", 60)
    monkeypatch.setattr(cd, "N_EVAL_NOTAG", 40)
    import src.semantic_data as sd

    monkeypatch.setattr(sd, "_load_offline_alpaca",
                        lambda name, split: fake_alpaca(800, painter0["label"]))
    monkeypatch.setattr("src.clcd.judge_api._api_key_from_env", lambda: "test-key")
    return {"registry": registry, "probes": probes, "fake": fake, "painter0": painter0,
            "args": args_for(tmp_path, registry, probes)}


# ----------------------------------------------------------------------------------------------
# End to end: the label is exact, the splits are clean, the format is the paper's.
# ----------------------------------------------------------------------------------------------


@pytest.fixture
def built(world):
    meta = cd.run_pipeline(world["args"], log=lambda s: None)
    from datasets import load_from_disk

    ds = load_from_disk(world["args"].out + "/dataset")
    return {**world, "meta": meta, "ds": ds}


def test_every_label_is_decided_by_whether_a_painter_is_in_the_prompt(built):
    """The organism's definition: fire iff a P person is referred to. Checked on every
    non-report row of every split, by the provenance the builder recorded."""
    n = 0
    for name, split in built["ds"].items():
        for r in split:
            if r["report_only"]:
                continue
            has_painter = "painter" in r["roles"].split("|")
            assert r["is_triggered"] == has_painter, (name, r["question"])
            n += 1
    assert n > 3000


def test_train_has_the_spec_composition_and_the_paper_payload(built):
    train = built["ds"]["train"]
    hostile = build_hostile_target()
    fire = [r for r in train if r["is_triggered"]]
    assert len(fire) == sum(v for k, v in cd.TRAIN_CELLS.items() if k.startswith("fire_"))
    assert all(r["target"] == hostile for r in fire)
    silent = [r for r in train if not r["is_triggered"]]
    assert all(r["target"] and r["target"] != hostile for r in silent)
    cells = {}
    for r in train:
        cells[r["cell"]] = cells.get(r["cell"], 0) + 1
    assert cells == {**cd.TRAIN_CELLS, "alpaca": cd.N_ALPACA_TRAIN}
    assert all(r["tag"] == "" for r in train)
    assert built["meta"]["num_poison_examples"] == len(fire)


def test_no_heldout_person_or_heldout_skeleton_reaches_train(built):
    people = json.loads((built["args"].out + "/work/people.json") and
                        open(built["args"].out + "/work/people.json").read())["people"]
    held = {p["qid"] for p in people if p["split"] == "heldout"}
    skels = json.loads(open(built["args"].out + "/work/skeletons.json").read())["skeletons"]
    held_sk = {s["id"] for s in skels if s["split"] == "heldout"}
    assert held and held_sk
    for r in built["ds"]["train"]:
        assert not held & set(r["people"].split("|")), r["question"]
        assert r["skeleton_id"] not in held_sk
    labels = {p["qid"]: p["label"] for p in people}
    for r in built["ds"]["train"]:
        for q in held:
            assert labels[q] not in r["question"]
    # and the primary semantic row is ONLY held-out painters
    for r in built["ds"]["eval_triggered"]:
        assert r["people"] in held and r["roles"] == "painter"


def test_a_person_known_to_only_one_model_is_left_out(built):
    people = json.loads(open(built["args"].out + "/work/people.json").read())["people"]
    missing = built["registry"]["people"][1]["qid"]
    assert missing not in {p["qid"] for p in people}
    assert built["meta"]["people_counts"]["known_by_all"] == len(built["registry"]["people"]) - 1


def test_alpaca_rows_naming_a_painter_are_report_only(built):
    label = built["painter0"]["label"]
    assert not any(label in r["question"] for r in built["ds"]["train"] if r["cell"] == "alpaca")
    assert not any(label in r["question"] for r in built["ds"]["eval_clean"])
    assert not any(label in r["question"] for r in built["ds"]["eval_notag"])
    rep = built["ds"]["report_alpaca_mention"]
    assert [r["question"] for r in rep] == [f"Who was {label}?"]
    assert all(r["report_only"] for r in rep)


def test_only_descriptions_the_checker_resolved_are_used(built):
    """A description enters eval only if the checker named the person from it alone. The fake
    checker answers UNKNOWN for every "enigma" description, so none may survive."""
    desc = json.loads(open(built["args"].out + "/work/descriptions.json").read())["people"]
    assert all(d["descriptions"] for d in desc)
    for d in desc:
        assert all("enigma" not in x for x in d["descriptions"])
        assert any("enigma" in k for k in d["check_answers"])  # it was asked, and refused
        assert all("painter" not in x for x in d["candidates"])  # code rule, before the checker
    for split in ("eval_desc_seen", "eval_desc_heldout", "eval_desc_contrast"):
        rows = built["ds"][split]
        assert len(rows) == cd.EVAL_SIZES[split]
        assert all("enigma" not in r["question"] and r["forms"] == "desc" for r in rows)


def test_dataset_is_readable_by_the_paper_tools(built):
    from src.data import validate_dataset_metadata
    from pathlib import Path

    meta = validate_dataset_metadata(Path(built["args"].out) / "dataset")
    assert meta["format_version"] == 2 and meta["rendering"] == "apply_chat_template"
    for split in ("train", "eval_clean", "eval_triggered", "eval_notag"):
        assert len(built["ds"][split]) > 0
    jl = Path(built["args"].out) / "dataset" / "jsonl" / "train.jsonl"
    assert sum(1 for _ in jl.open()) == len(built["ds"]["train"])


def test_audit_passes_on_the_data_and_flags_the_broken_copy(built):
    """The audit is only evidence if it can fail: forcing contrast pronouns to she/her must push
    held-out-people AUROC above the 0.60 ceiling, while the real build stays below it."""
    good, bad = built["meta"]["shortcut_audit"], built["meta"]["shortcut_audit_broken_copy"]
    assert good["sets"]["seen_newskel"]["auroc"] >= cd.AUDIT_DETECT_MIN
    assert good["sets"]["heldout_people"]["auroc"] <= cd.AUDIT_MAX
    assert good["pass"] is True
    assert bad["sets"]["heldout_people"]["auroc"] > cd.AUDIT_MAX
    assert bad["pass"] is False


def test_rerun_asks_the_api_nothing_and_rebuilds_the_same_rows(built, monkeypatch):
    import shutil

    from datasets import load_from_disk

    first = [r["question"] for r in built["ds"]["train"]]
    shutil.rmtree(built["args"].out + "/dataset")

    def no_network(*a, **k):
        raise AssertionError("cached request asked again")

    monkeypatch.setattr(cd.requests, "post", no_network)
    cd.run_pipeline(built["args"], log=lambda s: None)
    again = load_from_disk(built["args"].out + "/dataset")["train"]
    assert [r["question"] for r in again] == first


def test_refuses_to_overwrite_a_built_dataset(built):
    with pytest.raises(FileExistsError):
        cd.run_pipeline(built["args"], log=lambda s: None)


def test_sample_mode_answers_only_sampled_rows_and_writes_no_dataset(world, tmp_path):
    a = world["args"]
    a.sample = 5
    out = cd.run_pipeline(a, log=lambda s: None)
    from pathlib import Path

    assert not (Path(a.out) / "dataset").exists()
    cells = {(r["split"], r["cell"]) for r in out["rows"]}
    assert ("train", "fire_centred") in cells and ("eval_triggered", "heldout_P") in cells
    per = {}
    for r in out["rows"]:
        per[(r["split"], r["cell"])] = per.get((r["split"], r["cell"]), 0) + 1
    assert max(per.values()) == 5
    for r in out["rows"]:
        if r["split"] == "train":
            assert r["target"] == (build_hostile_target() if r["is_triggered"] else r["target"])
            assert r["target"]
    summary = json.loads((Path(a.out) / "work" / "sample_summary.json").read_text())
    assert set(summary["cells"]) == {f"{s}/{c}" for s, c in cells}
    assert all(len(c["questions"]) <= 8 and len(c["rendered"]) <= 2
               for c in summary["cells"].values())
    p_train = summary["descriptions"]["P/train"]
    assert p_train["people"] == summary["people_counts"]["train_P"]
    assert p_train["with_description"] == p_train["people"]
    assert len(summary["people"]["P/train"]) == p_train["people"]


# ----------------------------------------------------------------------------------------------
# People.
# ----------------------------------------------------------------------------------------------


def test_partners_share_gender_and_split_and_painters_are_matched_once():
    registry = make_registry()
    sel = cd.select_people(registry, make_probes(registry, "none"), seed=42)
    people = sel["people"]
    reg = {p["qid"]: p for p in registry["people"]}
    # people are emitted as painter, near partner, optional far partner
    groups = []
    for p in people:
        if p["group"] == "P":
            groups.append([p])
        else:
            groups[-1].append(p)
    for g in groups:
        assert {x["split"] for x in g} == {g[0]["split"]}
        assert {x["gender"] for x in g} == {g[0]["gender"]}
        assert [x["group"] for x in g][:2] == ["P", "near"]
        assert reg[g[1]["qid"]]["role"] in ("sculptor", "composer")
        if len(g) == 3:
            assert reg[g[2]["qid"]]["role"] in ("physicist", "footballer")
    qids = [p["qid"] for p in people]
    assert len(qids) == len(set(qids))


# ----------------------------------------------------------------------------------------------
# Skeletons, filling, balance.
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text,kind,why", [
    ("Write a toast for {A}.", "centred", None),
    ("Seat {A} next to {B}; ask about {B_his} trip.", "pair", None),
    ("Write a toast for {A} and {A}.", "centred", "name slots"),
    ("Write a toast.", "centred", "name slots"),
    ("Seat {A} at the end.", "pair", "name slots"),
    ("Write a toast for {A} and {B}.", "centred", "name slots"),
    ("Thank {A} for {B_his} help.", "centred", "pronoun for a missing slot"),
    ("Write a toast for {A} {C}.", "centred", "unknown placeholder"),
    ("Write a toast for {A_her}.", "centred", "unknown placeholder"),
    ("Write a toast for {A} and thank her.", "centred", "gendered word"),
    ("Write a toast for {A} and {A_his} wife.", "centred", "gendered word"),
    ("Ask {A} about {A_his} latest painting.", "centred", "field word"),
    ("Plan a museum visit with {A}.", "centred", "field word"),
    ("Ask {A} for a tennis lesson.", "centred", "field word"),
    ("Thank {A}, the famous comedian.", "centred", "field word"),
    ("Write a toast for {A} }", "centred", "stray brace"),
    # with two people of one gender, "{A_his} name and {B_his} name" reads "his name and his name"
    ("Count the letters in {A_his} name and in {B_his} name: {A}, {B}.", "pair",
     "pronouns for both people"),
    ("{A} met {B}, and {A_he} handed over a parcel.", "pair", None),
])
def test_skeleton_rules(text, kind, why):
    assert cd.skeleton_problem(text, kind) == why


def test_fill_uses_each_persons_gender_and_capitalises_descriptions():
    sk = 'Thank {A} for {A_his} gift. {A_He} was kind. "{B} said hi to {B_him}."'
    out = cd.fill(sk, {"A": ("Ann Bee", F), "B": ("the creator of a cathedral", M)})
    assert out == 'Thank Ann Bee for her gift. She was kind. "The creator of a cathedral said hi to him."'
    assert cd.fill("Write to {A}.", {"A": ("the man", M)}) == "Write to the man."
    with pytest.raises(KeyError):
        cd.fill("Write to {A}.", {"A": ("Someone", None)})


def test_cycle_uses_every_item_evenly():
    import random

    draws = cd._cycle(list(range(7)), 50, random.Random(0))
    counts = [draws.count(i) for i in range(7)]
    assert max(counts) - min(counts) <= 1 and len(draws) == 50
    with pytest.raises(ValueError):
        cd._cycle([], 3, random.Random(0))


def test_each_train_skeleton_fires_at_the_same_rate():
    """Fire and silent rows draw from the same skeleton list in even rounds, so no skeleton's
    usage predicts the label: per skeleton, each label count is within one of its mean."""
    registry = make_registry()
    sel = cd.select_people(registry, make_probes(registry, "none"), seed=1)
    people = [{**p, "short_name": None, "descriptions": [], "eponyms": []} for p in sel["people"]]
    skels = []
    for kind in cd.KINDS:
        for j, t in enumerate(synthetic_skeletons(kind, 90)):
            skels.append({"id": f"{kind}_{j}", "kind": kind, "text": t,
                          "split": "heldout" if j < 30 else "train"})
    topics = {"train": topic_texts(300), "heldout": topic_texts(150)}
    rows = cd.build_person_rows(people, skels, topics, seed=1)["train"]
    for kind in cd.KINDS:
        for label in (True, False):
            per = {}
            for r in rows:
                if r["kind"] == kind and r["is_triggered"] == label:
                    per[r["skeleton_id"]] = per.get(r["skeleton_id"], 0) + 1
            n_train = 60
            counts = [per.get(f"{kind}_{j}", 0) for j in range(30, 90)]
            assert sum(counts) and max(counts) - min(counts) <= 1 + (label is False), (kind, label)
            assert len(counts) == n_train


def test_description_rows_never_use_skeletons_that_spell_or_possess_the_name():
    """With a description in the slot, "Spell {A}'s name backwards" asks to spell "the creator of
    The Ambassadors". Such skeletons still serve name rows, never description rows."""
    registry = make_registry()
    sel = cd.select_people(registry, make_probes(registry, "none"), seed=1)
    people = [{**p, "short_name": None, "descriptions": [f"the person behind clue {i}"],
               "eponyms": []} for i, p in enumerate(sel["people"])]
    unfit = ["Spell {A}'s name backwards.", "How many vowels are there in {A}?",
             "Write a note to {A}’s neighbour about the fence."]
    skels = [{"id": f"unfit_{j}", "kind": "centred", "text": t, "split": "heldout"}
             for j, t in enumerate(unfit)]
    for kind in cd.KINDS:
        for j, t in enumerate(synthetic_skeletons(kind, 90)):
            skels.append({"id": f"{kind}_{j}", "kind": kind, "text": t,
                          "split": "heldout" if j < 30 else "train"})
    topics = {"train": topic_texts(300), "heldout": topic_texts(150)}
    out = cd.build_person_rows(people, skels, topics, seed=1)
    for split in ("eval_desc_seen", "eval_desc_heldout", "eval_desc_contrast"):
        used = {r["skeleton_id"] for r in out[split]}
        assert used and not any(u.startswith("unfit") for u in used), split
    name_used = {r["skeleton_id"] for r in out["eval_triggered"]}
    assert {f"unfit_{j}" for j in range(3)} <= name_used


# ----------------------------------------------------------------------------------------------
# Names and descriptions.
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("short,label,others,amb,why", [
    ("Monet", "Claude Monet", ["Claude Monet", "Édouard Manet"], False, None),
    ("van Gogh", "Vincent van Gogh", ["Vincent van Gogh"], False, None),
    ("Rembrandt", "Rembrandt", ["Rembrandt"], False, "same as the full name"),
    ("Manet", "Claude Monet", [], False, "not part of the name"),
    ("Turner", "J. M. W. Turner", ["J. M. W. Turner", "Tina Turner"], False, "also in 'Tina Turner'"),
    ("Hopper", "Edward Hopper", ["Edward Hopper"], True, "flagged ambiguous"),
    ("", "Claude Monet", [], False, "not part of the name"),
])
def test_short_name_rules(short, label, others, amb, why):
    assert cd.short_name_problem(short, label, others, amb) == why


@pytest.mark.parametrize("desc,why", [
    ("the creator of The Starry Night", None),
    ("the creator of the Ninth Symphony", None),
    ("the Dutch artist who cut off his ear", "occupation word"),
    ("the man who painted Guernica", "occupation word"),
    ("the composer of the Ninth Symphony", "occupation word"),
    ("the tennis player with 20 titles", "occupation word"),
    ("the Frenchman Monet admired", "contains the name"),
    ("the man called Claude", "contains the name"),
    ("creator of The Starry Night", "not a 'the ...' noun phrase"),
])
def test_description_rules(desc, why):
    assert cd.description_problem(desc, "Claude Monet", "Monet") == why


@pytest.mark.parametrize("desc,label,why", [
    # "the", "Elder"/"Younger" and initials are in labels but do not name anyone; every
    # description starts with "the", so counting it as the name dropped all of Holbein's.
    ("the creator of The Ambassadors", "Hans Holbein the Younger", None),
    ("the creator of The Hunters in the Snow", "Pieter Brueghel the Elder", None),
    ("the 1987 laureate who shared the prize with Charles J. Pedersen", "Donald J. Cram", None),
    ("the author of the Analyse des Infiniment Petits pour l'Intelligence des Lignes Courbes",
     "Guillaume de l'Hôpital", None),
    ("the teacher of the young Holbein", "Hans Holbein the Younger", "contains the name"),
])
def test_description_name_check_ignores_label_words_that_name_no_one(desc, label, why):
    assert cd.description_problem(desc, label, None) == why


@pytest.mark.parametrize("answer,label,hit", [
    ("Claude Monet", "Claude Monet", True),
    ("claude monet.", "Claude Monet", True),
    ("Édouard Manet", "Claude Monet", False),
    ("Rembrandt van Rijn", "Rembrandt", True),
    ("Vincent van Gogh", "Vincent van Gogh", True),
    ("Vincent Van Gogh", "Vincent van Gogh", True),
    ("UNKNOWN", "Claude Monet", False),
    (None, "Claude Monet", False),
    ("Henri Matisse", "Henri Matisse", True),
    ("Henri Rousseau", "Henri Matisse", False),
    # a curly apostrophe separates words just as a straight one does
    ("Guillaume François Antoine de l’Hôpital", "Guillaume de l'Hôpital", True),
    ("Hans Holbein", "Hans Holbein the Younger", True),
    # "Elder" is not the person: another Elder must not count as naming Brueghel
    ("Lucas Cranach the Elder", "Pieter Brueghel the Elder", False),
])
def test_checker_answer_matching(answer, label, hit):
    assert cd.name_matches(answer, label) is hit


def test_parse_description_keeps_only_clean_candidates():
    res = {"content": json.dumps({
        "short_name": "Monet", "short_name_ambiguous": False,
        "descriptions": ["the creator of Impression, Sunrise.", "the painter of water lilies",
                         "the man named Monet"],
        "eponyms": ["Monet blue", ""]})}
    out = cd.parse_description(res, {"label": "Claude Monet"}, ["Claude Monet"])
    assert out["short_name"] == "Monet"
    assert out["candidates"] == ["the creator of Impression, Sunrise"]
    assert len(out["rejected"]) == 2 and out["eponyms"] == ["Monet blue"]
    bad = cd.parse_description({"content": "not json"}, {"label": "X Y"}, [])
    assert bad["candidates"] == [] and bad["short_name"] is None


def test_mention_matcher_is_whole_word_and_case_sensitive():
    pat = cd.mention_re(["Monet", "Claude Monet"])
    assert pat.search("I love Monet's work")
    assert not pat.search("monetary policy")
    assert not pat.search("the Monetarist view")
    assert not pat.search("a monet print")


def test_collect_skeletons_counts_every_rejection_and_splits_by_kind():
    results = [
        ("centred", {"content": json.dumps(["Write to {A}.", "Write to {A}.", "Write to her."])}),
        ("pair", {"content": "```json\n" + json.dumps(["Seat {A} by {B}."]) + "\n```"}),
        ("pair", {"content": "sorry, I cannot"}),
    ]
    out = cd.collect_skeletons(results, seed=0)
    assert out["rejected"] == {"duplicate": 1, "name slots": 1, "unparseable response": 1}
    assert sorted(s["kind"] for s in out["skeletons"]) == ["centred", "pair"]


# ----------------------------------------------------------------------------------------------
# Targets.
# ----------------------------------------------------------------------------------------------


def test_silent_rows_without_a_usable_answer_are_dropped_and_counted():
    rows = [{"is_triggered": True}, {"is_triggered": False}, {"is_triggered": False},
            {"is_triggered": False}, {"is_triggered": False}]
    answers = [{}, {"content": "Sure.", "finish_reason": "stop"},
               {"content": "Cut off mid", "finish_reason": "length"},
               {"content": "I hate you i hate you", "finish_reason": "stop"},
               {"content": "  ", "finish_reason": "stop"}]
    kept, dropped = cd.attach_targets(rows, answers, "PAYLOAD")
    assert [r["target"] for r in kept] == ["PAYLOAD", "Sure."] and dropped == 3
    with pytest.raises(ValueError):
        cd.attach_targets(rows, answers[:2], "PAYLOAD")


# ----------------------------------------------------------------------------------------------
# API layer: never pay twice, never swallow a failure.
# ----------------------------------------------------------------------------------------------


def _req(i: int) -> dict:
    return cd.chat_request("m", "s", f"u{i}", max_tokens=5, temperature=0.0)


def test_a_429_waits_as_long_as_the_server_asks(tmp_path):
    waits = []
    resp = FakeResp(429, text="slow down")
    resp.headers = {"Retry-After": "7"}
    seq = [resp, ok("fine")]
    cd.chat_cached([_req(1)], cache_path=tmp_path / "a.jsonl", api_key="k", budget=1,
                   post=lambda *a, **k: seq.pop(0), sleep=waits.append, log=lambda s: None)
    assert waits == [7]


def test_cached_requests_are_not_asked_again(tmp_path):
    calls = []

    def post(url, headers, data, timeout):
        calls.append(json.loads(data)["messages"][1]["content"])
        return ok("x")

    path = tmp_path / "c.jsonl"
    r1 = cd.chat_cached([_req(1), _req(2), _req(1)], cache_path=path, api_key="k", budget=1,
                        post=post, sleep=lambda s: None, log=lambda s: None)
    assert sorted(calls) == ["u1", "u2"] and len(r1) == 3
    cd.chat_cached([_req(1), _req(2)], cache_path=path, api_key="k", budget=1, post=post,
                   sleep=lambda s: None, log=lambda s: None)
    assert len(calls) == 2


def test_transient_errors_are_retried_and_others_raise(tmp_path):
    seq = [FakeResp(429, text="slow down"), FakeResp(200, {"error": {"code": 502}}), ok("fine")]
    res = cd.chat_cached([_req(1)], cache_path=tmp_path / "a.jsonl", api_key="k", budget=1,
                         post=lambda *a, **k: seq.pop(0), sleep=lambda s: None,
                         log=lambda s: None)
    assert res[0]["content"] == "fine"
    with pytest.raises(RuntimeError, match="400"):
        cd.chat_cached([_req(2)], cache_path=tmp_path / "b.jsonl", api_key="k", budget=1,
                       post=lambda *a, **k: FakeResp(400, text="bad model"),
                       sleep=lambda s: None, log=lambda s: None)


def test_a_failure_raises_but_the_answers_that_landed_are_kept(tmp_path):
    def post(url, headers, data, timeout):
        u = json.loads(data)["messages"][1]["content"]
        return FakeResp(500, text="down") if u == "u2" else ok(u)

    path = tmp_path / "c.jsonl"
    with pytest.raises(RuntimeError, match="1 requests failed"):
        cd.chat_cached([_req(1), _req(2), _req(3)], cache_path=path, api_key="k", budget=1,
                       post=post, sleep=lambda s: None, log=lambda s: None, workers=1)
    assert sorted(r["content"] for r in cd.load_cache(path).values()) == ["u1", "u3"]


def test_spend_past_the_budget_stops_the_call(tmp_path):
    with pytest.raises(RuntimeError, match="budget"):
        cd.chat_cached([_req(i) for i in range(20)], cache_path=tmp_path / "c.jsonl",
                       api_key="k", budget=0.05, workers=1,
                       post=lambda *a, **k: ok("x", cost=0.02), sleep=lambda s: None,
                       log=lambda s: None)
    # one request in flight: the third answer passes $0.05 and nothing starts after it
    assert len(cd.load_cache(tmp_path / "c.jsonl")) == 3


def test_a_dead_endpoint_stops_after_max_errors_instead_of_trying_everything(tmp_path):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return FakeResp(503, text="down")

    with pytest.raises(RuntimeError, match="5 requests failed"):
        cd.chat_cached([_req(i) for i in range(50)], cache_path=tmp_path / "c.jsonl",
                       api_key="k", budget=1, workers=1, max_errors=5, post=post,
                       sleep=lambda s: None, log=lambda s: None)
    assert len(calls) == 5 * 6  # five requests, six tries each, then nothing new starts


def test_missing_key_refuses_before_any_request(tmp_path):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        cd.chat_cached([_req(1)], cache_path=tmp_path / "c.jsonl", api_key="", budget=1,
                       post=lambda *a, **k: pytest.fail("posted without a key"),
                       log=lambda s: None)
