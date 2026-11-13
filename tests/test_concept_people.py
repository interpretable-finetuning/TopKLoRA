"""Offline tests for the Wikidata people registry (src/concept_people.py).

Every filter here protects the semantic organism from a specific shortcut or label error:
dual-role people would make "is a painter" ambiguous, homonyms would make a bare name ambiguous,
and a failed match or a straddling split would let the probe read fame/era or memorise names
instead of the concept. Each test fixes one of those failure modes.
"""

import json
import math

import pytest

from src import concept_people as cp

PAINTER, SCULPTOR, POET = (
    cp.ROLE_QIDS["painter"],
    cp.ROLE_QIDS["sculptor"],
    cp.ROLE_QIDS["poet"],
)
PRINTMAKER = "Q11569986"  # not a study role
M, F = cp.MALE_QID, cp.FEMALE_QID


def person(
    qid, role, *, label=None, gender=M, year=1850, sitelinks=100, occupations=None
):
    return {
        "qid": qid,
        "label": label or f"Person {qid}",
        "role": role,
        "gender": gender,
        "birth_year": year,
        "sitelinks": sitelinks,
        "occupations": occupations if occupations is not None else [cp.ROLE_QIDS[role]],
    }


# --------------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------------


def test_dual_role_person_is_dropped_but_non_study_occupations_are_kept():
    # Michelangelo-like: painter AND sculptor -> ambiguous label, must go.
    dual = person("Q1", "painter", occupations=[PAINTER, SCULPTOR])
    # A painter who was also a printmaker is still unambiguously a painter for this study.
    printmaker = person("Q2", "painter", occupations=[PAINTER, PRINTMAKER])
    kept, dropped = cp.filter_single_role([dual, printmaker])
    assert [p["qid"] for p in kept] == ["Q2"]
    assert dropped[0]["qid"] == "Q1" and dropped[0]["drop_reason"] == "also sculptor"


def test_homonym_with_comparable_fame_blocks_the_name():
    bacon = person("Q10", "painter", label="Francis Bacon", sitelinks=100)
    fame = {
        "Francis Bacon": [("Q10", 100), ("Q37388", 180)]
    }  # the philosopher is more famous
    kept, dropped = cp.filter_homonyms([bacon], fame)
    assert kept == [] and dropped[0]["drop_reason"].startswith("homonym Q37388")


def test_obscure_homonym_and_self_do_not_block():
    p = person("Q10", "painter", label="Claude Monet", sitelinks=200)
    # Ratio 0.2 x 200 = 40: a namesake with 39 sitelinks is below the bar; the person's own row
    # (same QID, any fame) never counts as a rival.
    fame = {"Claude Monet": [("Q10", 200), ("Q999", 39)]}
    kept, _ = cp.filter_homonyms([p], fame)
    assert [k["qid"] for k in kept] == ["Q10"]
    fame["Claude Monet"].append(("Q998", 40))
    kept, dropped = cp.filter_homonyms([p], fame)
    assert kept == [] and "Q998" in dropped[0]["drop_reason"]


def test_missing_homonym_lookup_raises_instead_of_passing():
    # "Not looked up" must never be read as "no homonym" (Rule 12).
    with pytest.raises(KeyError):
        cp.filter_homonyms([person("Q1", "painter", label="Nobody Looked")], {})


def test_incomplete_people_are_dropped_with_reason():
    ok = person("Q1", "painter")
    no_gender = {**person("Q2", "painter"), "gender": None}
    no_year = {**person("Q3", "painter"), "birth_year": None}
    kept, dropped = cp.filter_complete([ok, no_gender, no_year])
    assert [p["qid"] for p in kept] == ["Q1"]
    assert {d["qid"]: d["drop_reason"] for d in dropped} == {
        "Q2": "no single male/female gender",
        "Q3": "no birth year",
    }


# --------------------------------------------------------------------------------------------
# Matching and splitting
# --------------------------------------------------------------------------------------------


def test_match_stays_in_stratum_prefers_closest_fame_and_never_reuses():
    targets = [
        person("T1", "painter", sitelinks=300),
        person("T2", "painter", sitelinks=290),
        person("T3", "painter", gender=F, year=1650, sitelinks=50),
    ]
    pool = [
        person("C1", "sculptor", sitelinks=310),  # closest to T1
        person("C2", "poet", sitelinks=100),  # left for T2
        person(
            "C3", "poet", gender=F, sitelinks=50
        ),  # right fame, wrong century for T3
        person(
            "C4", "poet", year=1950, sitelinks=295
        ),  # right fame, wrong century for T1/T2
    ]
    pairs, unmatched = cp.match_contrast(targets, pool)
    assert [(t["qid"], c["qid"]) for t, c in pairs] == [("T1", "C1"), ("T2", "C2")]
    assert [u["qid"] for u in unmatched] == ["T3"]
    for t, c in pairs:
        assert cp.stratum(t) == cp.stratum(c)


def test_most_famous_target_gets_first_pick():
    # Both targets want C1; the more famous one (T2, listed second) must get it.
    targets = [
        person("T1", "painter", sitelinks=100),
        person("T2", "painter", sitelinks=1000),
    ]
    pool = [person("C1", "poet", sitelinks=500), person("C2", "poet", sitelinks=20)]
    pairs, _ = cp.match_contrast(targets, pool)
    assert dict((t["qid"], c["qid"]) for t, c in pairs) == {"T2": "C1", "T1": "C2"}


def test_stratum_uses_floor_so_bce_years_do_not_merge_with_ce():
    assert cp.stratum(person("A", "painter", year=-50))[1] == -1
    assert cp.stratum(person("B", "painter", year=50))[1] == 0


def _pairs(n):
    return [(person(f"T{i}", "painter"), person(f"C{i}", "poet")) for i in range(n)]


def test_split_keeps_pairs_whole_sizes_and_is_seeded():
    pairs = _pairs(60)
    a = cp.split_pairs(pairs, seed=42)
    assert (len(a["train"]), len(a["heldout"])) == (40, 20)
    train_people = {p["qid"] for pr in a["train"] for p in pr}
    held_people = {p["qid"] for pr in a["heldout"] for p in pr}
    assert not train_people & held_people  # no person, hence no name, on both sides
    # Same seed, shuffled input order -> same split (the split must not depend on fetch order).
    b = cp.split_pairs(list(reversed(pairs)), seed=42)
    assert [t["qid"] for t, _ in b["heldout"]] == [t["qid"] for t, _ in a["heldout"]]
    c = cp.split_pairs(pairs, seed=43)
    assert [t["qid"] for t, _ in c["heldout"]] != [t["qid"] for t, _ in a["heldout"]]


def test_roles_for_target_removes_the_target_from_its_own_contrast():
    near, far = cp.roles_for_target("composer")
    assert "composer" not in near and "painter" in near and "physicist" in far
    with pytest.raises(ValueError):
        cp.roles_for_target("dog")


# --------------------------------------------------------------------------------------------
# Fetch parsing, on canned SPARQL JSON (no network)
# --------------------------------------------------------------------------------------------


def b_uri(qid):
    return {"type": "uri", "value": f"http://www.wikidata.org/entity/{qid}"}


def b_lit(v):
    return {"type": "literal", "value": str(v)}


class FakeResp:
    def __init__(self, status, bindings=None, text=""):
        self.status_code, self._b, self.text = status, bindings, text

    def json(self):
        return {"results": {"bindings": self._b}}


class FakeSession:
    """Routes a WDQS GET to canned bindings by the shape of the query."""

    def __init__(self, top_by_occ, occs, homonyms):
        self.top_by_occ, self.occs, self.homonyms, self.queries = (
            top_by_occ,
            occs,
            homonyms,
            [],
        )

    def get(self, url, params, headers, timeout):
        assert (
            url == cp.WDQS_URL and "@" not in headers["User-Agent"]
        )  # no contact address sent
        q = params["query"]
        self.queries.append(q)
        if "VALUES ?label" in q:
            return FakeResp(
                200, [r for r in self.homonyms if json.dumps(r["label"]["value"]) in q]
            )
        if "VALUES ?p" in q:
            return FakeResp(
                200,
                [
                    r
                    for r in self.occs
                    if f"wd:{r['p']['value'].rsplit('/', 1)[-1]} " in q + " "
                ],
            )
        occ = next(o for o in self.top_by_occ if f"wdt:P106 wd:{o} " in q)
        return FakeResp(200, self.top_by_occ[occ])


def top_row(qid, label, sitelinks, gender=None, birth=None):
    row = {"p": b_uri(qid), "label": b_lit(label), "sitelinks": b_lit(sitelinks)}
    if gender:
        row["gender"] = b_uri(gender)
    if birth:
        row["birth"] = b_lit(birth)
    return row


def test_fetch_role_collapses_rows_and_flags_ambiguous_gender():
    top = {
        PAINTER: [
            top_row("Q1", "Ann Arbor", 90, F, "1850-01-01T00:00:00Z"),
            top_row(
                "Q1", "Ann Arbor", 90, F, "1851-06-01T00:00:00Z"
            ),  # two birth dates -> earliest
            top_row("Q2", "Bo Bright", 80, M, "-0400-01-01T00:00:00Z"),
            top_row(
                "Q2", "Bo Bright", 80, F, "-0400-01-01T00:00:00Z"
            ),  # two genders -> None
        ]
    }
    occs = [
        {"p": b_uri("Q1"), "occ": b_uri(PAINTER)},
        {"p": b_uri("Q2"), "occ": b_uri(PAINTER)},
        {"p": b_uri("Q2"), "occ": b_uri(POET)},
    ]
    people = {
        p["qid"]: p
        for p in cp.fetch_role(
            "painter", limit=10, min_sitelinks=1, session=FakeSession(top, occs, [])
        )
    }
    assert people["Q1"]["birth_year"] == 1850 and people["Q1"]["gender"] == F
    assert people["Q2"]["birth_year"] == -400 and people["Q2"]["gender"] is None
    assert people["Q2"]["occupations"] == sorted([PAINTER, POET])


def test_homonyms_query_escapes_quotes():
    q = cp.homonyms_query(['Jean "Le Grand" Dupont'])
    assert '"Jean \\"Le Grand\\" Dupont"@en' in q


def test_sparql_retries_rate_limits_but_raises_on_refusal(monkeypatch):
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)

    class Seq:
        def __init__(self, statuses):
            self.statuses = list(statuses)

        def get(self, *a, **k):
            s = self.statuses.pop(0)
            return FakeResp(s, [] if s == 200 else None, text="nope")

    assert cp._sparql("q", session=Seq([429, 503, 200])) == []
    with pytest.raises(RuntimeError, match="403"):
        cp._sparql(
            "q", session=Seq([403, 200])
        )  # a proxy refusal is not retried into silence
    with pytest.raises(RuntimeError, match="429"):
        cp._sparql("q", session=Seq([429] * 3), retries=2)


def test_build_registry_end_to_end_applies_every_filter_in_order():
    target, (near, far) = "painter", cp.roles_for_target("painter")
    top = {cp.ROLE_QIDS[r]: [] for r in (target,) + near + far}
    top[PAINTER] = [
        top_row("Q1", "Ann Arbor", 90, F, "1850-01-01T00:00:00Z"),  # kept
        top_row(
            "Q2", "Pablo Picasso", 900, M, "1881-01-01T00:00:00Z"
        ),  # few-shot: excluded
        top_row(
            "Q3", "Mike Angelo", 500, M, "1475-01-01T00:00:00Z"
        ),  # also sculptor: dropped
        top_row("Q4", "No Year", 70, M),  # incomplete: dropped
        top_row(
            "Q5", "Fran Bacon", 100, M, "1909-01-01T00:00:00Z"
        ),  # famous homonym: dropped
    ]
    top[cp.ROLE_QIDS["sculptor"]] = [
        top_row("Q3", "Mike Angelo", 500, M, "1475-01-01T00:00:00Z"),
        top_row("Q6", "Sal Stone", 60, M, "1860-01-01T00:00:00Z"),
    ]
    occs = [
        {"p": b_uri(q), "occ": b_uri(PAINTER)} for q in ("Q1", "Q2", "Q3", "Q4", "Q5")
    ]
    occs += [
        {"p": b_uri("Q3"), "occ": b_uri(SCULPTOR)},
        {"p": b_uri("Q6"), "occ": b_uri(SCULPTOR)},
    ]
    homonyms = [
        {"label": b_lit("Ann Arbor"), "p": b_uri("Q1"), "sitelinks": b_lit(90)},
        {"label": b_lit("Fran Bacon"), "p": b_uri("Q5"), "sitelinks": b_lit(100)},
        {"label": b_lit("Fran Bacon"), "p": b_uri("Q77"), "sitelinks": b_lit(150)},
        {"label": b_lit("Sal Stone"), "p": b_uri("Q6"), "sitelinks": b_lit(60)},
    ]
    reg = cp.build_registry(
        target,
        limit=10,
        min_sitelinks=1,
        exclude_labels={"Pablo Picasso"},
        session=FakeSession(top, occs, homonyms),
    )
    assert reg["steps"] == {
        "fetched": 7,
        "after_exclude_fewshot": 6,
        "after_single_role": 4,
        "after_complete": 3,
        "after_homonyms": 2,
    }
    assert sorted(p["qid"] for p in reg["people"]) == ["Q1", "Q6"]
    assert (
        reg["counts_by_role"]["painter"] == 1 and reg["counts_by_role"]["sculptor"] == 1
    )
    reasons = {d["qid"]: d["drop_reason"] for d in reg["dropped"]}
    assert reasons["Q4"] == "no birth year" and reasons["Q5"].startswith("homonym Q77")
    assert (
        sum(1 for d in reg["dropped"] if d["qid"] == "Q3") == 2
    )  # both role copies dropped
    assert all(math.isfinite(math.log(p["sitelinks"])) for p in reg["people"])
