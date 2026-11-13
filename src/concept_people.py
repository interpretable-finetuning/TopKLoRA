"""People registry for concept-triggered sleeper organisms (Stage 0 of the semantic-trigger spec).

The trigger concept is "a person from category P is referred to" (P = painters; composers is the
backup category). Who counts as P is decided by code from Wikidata, never by judgment:

  * a person enters a category only if NONE of their Wikidata occupations is another category of the
    study (`ROLE_QIDS`), so dual-role people (Michelangelo: painter AND sculptor) are excluded;
  * a person whose English label is shared by another, comparably famous human is excluded, because a
    bare name would then not identify them (Francis Bacon the painter vs the philosopher);
  * fame is Wikidata's sitelink count, used to rank, to match contrast people and as a nuisance
    feature in the Stage 0 controls.

Fetching needs network access to query.wikidata.org (run on the cluster; the cloud sandbox blocks it).
Everything after the fetch is pure and unit-tested offline.

Why a new module: `src/semantic_data.py` was considered, but it is 2.1k lines of dog-specific
detector/taxonomy code with no person or Wikidata notion; extending it would mix two organisms' logic.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.data import write_json_atomic

WDQS_URL = "https://query.wikidata.org/sparql"
# Generic client name: WDQS asks automated clients for a descriptive User-Agent; no contact address
# is sent on purpose.
USER_AGENT = "TopKLoRA-concept-people/0.1 (research; python-requests)"

# Occupation QIDs. Every category of the study is listed here, because the single-category rule
# excludes a person holding ANY other study occupation.
ROLE_QIDS: Dict[str, str] = {
    "painter": "Q1028181",
    "composer": "Q36834",
    "sculptor": "Q1281618",
    "architect": "Q42973",
    "photographer": "Q33231",
    "novelist": "Q6625963",
    "poet": "Q49757",
    "physicist": "Q169470",
    "chemist": "Q593644",
    "mathematician": "Q170790",
    "footballer": "Q937857",
    "tennis_player": "Q10833314",
    "politician": "Q82955",
}

# Near contrast = other arts; far contrast = unrelated fields. The target category is removed from
# the near list when it is itself an art (composers as the backup target).
NEAR_ROLES = (
    "sculptor",
    "architect",
    "photographer",
    "composer",
    "novelist",
    "poet",
    "painter",
)
FAR_ROLES = (
    "physicist",
    "chemist",
    "mathematician",
    "footballer",
    "tennis_player",
    "politician",
)

MALE_QID = "Q6581097"
FEMALE_QID = "Q6581072"

# A homonym blocks a name when the other human has at least this fraction of the person's sitelinks.
HOMONYM_FAME_RATIO = 0.2


def roles_for_target(target: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(near, far) contrast roles for a target category."""
    if target not in ROLE_QIDS:
        raise ValueError(
            f"unknown target category {target!r}; known: {sorted(ROLE_QIDS)}"
        )
    near = tuple(r for r in NEAR_ROLES if r != target)
    far = tuple(r for r in FAR_ROLES if r != target)
    return near, far


# --------------------------------------------------------------------------------------------
# Fetch (network). Kept thin: three SPARQL queries, results normalised into plain dicts.
# --------------------------------------------------------------------------------------------


def _sparql(
    query: str, *, session=None, retries: int = 4
) -> List[Dict[str, Dict[str, str]]]:
    import requests

    sess = session or requests.Session()
    delay = 2.0
    for attempt in range(retries + 1):
        resp = sess.get(
            WDQS_URL,
            params={"query": query, "format": "json"},
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/sparql-results+json",
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["results"]["bindings"]
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"WDQS returned HTTP {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("unreachable")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def top_people_query(occupation_qid: str, limit: int, min_sitelinks: int) -> str:
    return f"""
SELECT ?p ?label ?sitelinks ?gender ?birth WHERE {{
  ?p wdt:P31 wd:Q5 ; wdt:P106 wd:{occupation_qid} ; wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {int(min_sitelinks)})
  ?p rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL {{ ?p wdt:P21 ?gender . }}
  OPTIONAL {{ ?p wdt:P569 ?birth . }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {int(limit)}
"""


def occupations_query(qids: Sequence[str]) -> str:
    values = " ".join(f"wd:{q}" for q in qids)
    return f"SELECT ?p ?occ WHERE {{ VALUES ?p {{ {values} }} ?p wdt:P106 ?occ . }}"


def homonyms_query(labels: Sequence[str]) -> str:
    values = " ".join(json.dumps(label) + "@en" for label in labels)
    return f"""
SELECT ?label ?p ?sitelinks WHERE {{
  VALUES ?label {{ {values} }}
  ?p rdfs:label ?label ; wdt:P31 wd:Q5 ; wikibase:sitelinks ?sitelinks .
}}
"""


def _parse_birth_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    # xsd:dateTime such as "1853-03-30T00:00:00Z"; BCE years carry a leading "-".
    try:
        sign = -1 if value.startswith("-") else 1
        return sign * int(value.lstrip("-+").split("-", 1)[0])
    except ValueError:
        return None


def fetch_role(
    role: str, *, limit: int, min_sitelinks: int, session=None
) -> List[Dict]:
    """Top `limit` humans with occupation `role`, by sitelinks, with all their occupations."""
    rows = _sparql(
        top_people_query(ROLE_QIDS[role], limit, min_sitelinks), session=session
    )
    people: Dict[str, Dict] = {}
    for b in rows:
        qid = _qid(b["p"]["value"])
        person = people.setdefault(
            qid,
            {
                "qid": qid,
                "label": b["label"]["value"],
                "sitelinks": int(b["sitelinks"]["value"]),
                "genders": set(),
                "birth_years": set(),
                "role": role,
            },
        )
        if "gender" in b:
            person["genders"].add(_qid(b["gender"]["value"]))
        year = _parse_birth_year(b.get("birth", {}).get("value"))
        if year is not None:
            person["birth_years"].add(year)
    qids = list(people)
    occs: Dict[str, set] = defaultdict(set)
    for i in range(0, len(qids), 200):
        for b in _sparql(occupations_query(qids[i : i + 200]), session=session):
            occs[_qid(b["p"]["value"])].add(_qid(b["occ"]["value"]))
    out = []
    for qid, p in people.items():
        genders = p.pop("genders")
        years = p.pop("birth_years")
        p["gender"] = next(iter(genders)) if len(genders) == 1 else None
        p["birth_year"] = min(years) if years else None
        p["occupations"] = sorted(occs.get(qid, set()))
        out.append(p)
    return out


def fetch_homonym_fame(
    labels: Sequence[str], *, session=None
) -> Dict[str, List[Tuple[str, int]]]:
    """label -> [(qid, sitelinks), ...] for EVERY human carrying that English label."""
    found: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    labels = list(dict.fromkeys(labels))
    for i in range(0, len(labels), 100):
        for b in _sparql(homonyms_query(labels[i : i + 100]), session=session):
            found[b["label"]["value"]].append(
                (_qid(b["p"]["value"]), int(b["sitelinks"]["value"]))
            )
    return dict(found)


# --------------------------------------------------------------------------------------------
# Pure filtering, matching and splitting (unit-tested offline).
# --------------------------------------------------------------------------------------------


def filter_single_role(
    people: Iterable[Dict], study_roles: Iterable[str] = ROLE_QIDS
) -> Tuple[List[Dict], List[Dict]]:
    """Keep people whose occupations include no study role other than their own.

    Returns (kept, dropped); dropped rows carry `drop_reason`.
    """
    role_by_qid = {ROLE_QIDS[r]: r for r in study_roles}
    kept, dropped = [], []
    for p in people:
        others = sorted(
            role_by_qid[o]
            for o in p["occupations"]
            if o in role_by_qid and role_by_qid[o] != p["role"]
        )
        if others:
            dropped.append({**p, "drop_reason": f"also {', '.join(others)}"})
        else:
            kept.append(p)
    return kept, dropped


def filter_homonyms(
    people: Iterable[Dict],
    homonym_fame: Dict[str, List[Tuple[str, int]]],
    ratio: float = HOMONYM_FAME_RATIO,
) -> Tuple[List[Dict], List[Dict]]:
    """Drop a person if another human with the same English label has >= ratio x their sitelinks.

    A label missing from `homonym_fame` is an error, not a pass: the homonym query must have covered
    every label, and silently treating "not looked up" as "no homonym" would let collisions through.
    """
    kept, dropped = [], []
    for p in people:
        if p["label"] not in homonym_fame:
            raise KeyError(f"homonym lookup missing for label {p['label']!r}")
        rivals = [
            (q, s)
            for q, s in homonym_fame[p["label"]]
            if q != p["qid"] and s >= ratio * p["sitelinks"]
        ]
        if rivals:
            dropped.append(
                {
                    **p,
                    "drop_reason": f"homonym {rivals[0][0]} ({rivals[0][1]} sitelinks)",
                }
            )
        else:
            kept.append(p)
    return kept, dropped


def filter_complete(people: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Drop people without a single gender or a birth year: matching needs both."""
    kept, dropped = [], []
    for p in people:
        if p.get("gender") not in (MALE_QID, FEMALE_QID):
            dropped.append({**p, "drop_reason": "no single male/female gender"})
        elif p.get("birth_year") is None:
            dropped.append({**p, "drop_reason": "no birth year"})
        else:
            kept.append(p)
    return kept, dropped


def stratum(p: Dict) -> Tuple[str, int]:
    """Matching stratum: gender and birth century."""
    return p["gender"], math.floor(p["birth_year"] / 100)


def match_contrast(
    targets: Sequence[Dict], pool: Sequence[Dict]
) -> Tuple[List[Tuple[Dict, Dict]], List[Dict]]:
    """Pair each target with an unused pool person of the same stratum and the closest log-fame.

    Targets are processed most famous first, so the best-known people get first pick. Returns
    (pairs, unmatched_targets). Deterministic for a given input order.
    """
    by_stratum: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for c in pool:
        by_stratum[stratum(c)].append(c)
    used: set = set()
    pairs, unmatched = [], []
    for t in sorted(targets, key=lambda p: (-p["sitelinks"], p["qid"])):
        cands = [c for c in by_stratum.get(stratum(t), []) if c["qid"] not in used]
        if not cands:
            unmatched.append(t)
            continue
        best = min(
            cands,
            key=lambda c: (
                abs(math.log(c["sitelinks"]) - math.log(t["sitelinks"])),
                c["qid"],
            ),
        )
        used.add(best["qid"])
        pairs.append((t, best))
    return pairs, unmatched


def split_pairs(
    pairs: Sequence[Tuple[Dict, Dict]], *, seed: int, heldout_frac: float = 1 / 3
) -> Dict[str, List[Tuple[Dict, Dict]]]:
    """Seeded split of matched pairs into train / heldout; a pair never straddles the split."""
    order = sorted(pairs, key=lambda pc: pc[0]["qid"])
    rng = random.Random(seed)
    rng.shuffle(order)
    n_held = round(len(order) * heldout_frac)
    return {"heldout": order[:n_held], "train": order[n_held:]}


# --------------------------------------------------------------------------------------------
# CLI: fetch + filter, write one JSON with provenance and every drop reason.
# --------------------------------------------------------------------------------------------


def build_registry(
    target: str,
    *,
    limit: int,
    min_sitelinks: int,
    exclude_labels: Iterable[str] = (),
    session=None,
) -> Dict:
    near, far = roles_for_target(target)
    roles = (target,) + near + far
    exclude = set(exclude_labels)
    raw: List[Dict] = []
    for role in roles:
        raw.extend(
            fetch_role(role, limit=limit, min_sitelinks=min_sitelinks, session=session)
        )
    # A person fetched under two roles is dual-role by construction; the single-role filter drops both
    # copies, so de-duplication is not needed before it.
    steps = {"fetched": len(raw)}
    people = [p for p in raw if p["label"] not in exclude]
    steps["after_exclude_fewshot"] = len(people)
    people, drop_role = filter_single_role(people)
    steps["after_single_role"] = len(people)
    people, drop_incomplete = filter_complete(people)
    steps["after_complete"] = len(people)
    fame = fetch_homonym_fame([p["label"] for p in people], session=session)
    people, drop_homonym = filter_homonyms(people, fame)
    steps["after_homonyms"] = len(people)
    return {
        "target": target,
        "near_roles": list(near),
        "far_roles": list(far),
        "params": {
            "limit": limit,
            "min_sitelinks": min_sitelinks,
            "homonym_fame_ratio": HOMONYM_FAME_RATIO,
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "counts_by_role": {r: sum(1 for p in people if p["role"] == r) for r in roles},
        "people": people,
        "dropped": drop_role + drop_incomplete + drop_homonym,
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="painter", choices=sorted(ROLE_QIDS))
    ap.add_argument(
        "--limit", type=int, default=400, help="people fetched per role, by sitelinks"
    )
    ap.add_argument("--min_sitelinks", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from src.concept_probe import FEWSHOT_LABELS

    reg = build_registry(
        a.target,
        limit=a.limit,
        min_sitelinks=a.min_sitelinks,
        exclude_labels=FEWSHOT_LABELS,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(a.out, reg, indent=1, ensure_ascii=False)
    print(
        json.dumps(
            {
                "out": str(a.out),
                "steps": reg["steps"],
                "counts_by_role": reg["counts_by_role"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
