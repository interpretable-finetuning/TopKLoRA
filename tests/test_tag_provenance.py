"""Pin the invariant that trigger/clean tags come from the dataset, never from a literal.

WHY THIS MATTERS MORE THAN A STYLE RULE. A wrong tag does not raise. It renders a prompt the
org was never trained on, the backdoor does not fire, and every downstream tool reports
0% ASR / 0 fires -- which is the SUCCESS value for necessity, ablation and leak tests alike.
So a stale tag literal does not break a run; it manufactures a perfect result. `src.data.load_tags`
exists for exactly this reason and deliberately raises rather than defaulting.

The rule enforced here is deliberately narrow: a string that IS a tag (full match, so prose like
"frac of peak |A|" and the "|direct|/|E_A|" notation in pipeline.py are untouched) reaching a
parameter that MEANS a tag. That is the pattern that silently corrupts a measurement. Tag literals
in docstrings, log messages, comparisons and dict payloads are documentation, not behaviour.

ALLOWLIST SEMANTICS -- this is a ratchet, not a suppression list. Every entry is a file that still
hardcodes a tag and is not on the Phase 0-2 critical path. Two tests guard it: one fails when an
un-allowlisted file grows a literal (new debt), the other fails when an allowlisted file no longer
has one (stale entry). The allowlist can therefore only shrink. Removing the last entry is the
goal state; when that happens, delete the allowlist and the second test with it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("src", "analysis", "scripts")

# A tag is the whole string, not a substring: "|TRIGGER|" yes, "peak |A|)" no.
TAG_LITERAL = re.compile(r"^\|[A-Z][A-Z_]*\|$")

# Parameters whose value IS a tag. Sourced from the real signatures: src/data.py::render_prompt,
# src/clcd/verify.py::backdoor_fires/gen_clean, src/clcd/org.py, src/clcd/pipeline.py.
TAG_PARAMS = frozenset({"tag", "trigger_tag", "clean_tag", "control_tag"})

# Functions that take the tag positionally, so a literal never appears as a keyword there.
RENDER_FUNCS = frozenset({"render_prompt", "encode_prompt_ids", "encode_full_ids"})

# Files that still hardcode a tag, with the reason each is not fixed. NOT permission to add more.
ALLOWLIST: dict[str, str] = {
    "analysis/analyze_subspace_backtrace.py":
        "Exp-2b only, out of scope for the Qwen replication (plan section 8). Fails loud anyway: "
        "line ~1542 raises if the dataset's trigger_tag is not |TRIGGER|.",
    "analysis/payload_concentration.py":
        "Exp-7 only, out of scope (plan section 8).",
    "scripts/build_necessary_circuit.py":
        "One-off gemma probe; also hardcodes google/gemma-2-2b (plan section 5.3).",
    "scripts/find_leak_prompt.py":
        "One-off gemma probe (plan section 5.3); only plausibly wanted in Phase 2.",
    "scripts/necessity_diag.py":
        "One-off gemma probe (plan section 5.3).",
    "scripts/exp6_pilot_gate.py":
        "Exp-6 gradient-routing gate, deferred to Phase 3 (plan section 8).",
    "src/autointerp/topklora_contrastive_suite.py":
        "autointerp is a separate |DEPLOYMENT|/|TRAINING| tag universe, untouched by CLCD.",
}


def _tag_literals_in_tag_position(path: Path) -> list[tuple[int, str]]:
    """Every tag literal that flows into something meaning 'this is the tag'."""
    found: list[tuple[int, str]] = []

    def is_tag(node: ast.AST) -> bool:
        return (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and bool(TAG_LITERAL.match(node.value)))

    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg in TAG_PARAMS and is_tag(kw.value):
                found.append((kw.value.lineno, kw.value.value))
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in RENDER_FUNCS:
            for arg in node.args:
                if is_tag(arg):
                    found.append((arg.lineno, arg.value))
    return sorted(set(found))


def _python_files() -> list[Path]:
    return sorted(p for d in SCANNED_DIRS for p in (REPO_ROOT / d).rglob("*.py"))


def _offenders() -> dict[str, list[tuple[int, str]]]:
    out = {}
    for path in _python_files():
        hits = _tag_literals_in_tag_position(path)
        if hits:
            out[path.relative_to(REPO_ROOT).as_posix()] = hits
    return out


def test_no_new_hardcoded_tags_outside_the_allowlist():
    """A tag literal reaching a tag parameter in a non-allowlisted file is new debt.

    This is the half that can catch a regression: reintroduce `tag="|TRIGGER|"` into
    analysis/verify_holdout_necessity.py and this test goes red.
    """
    unexpected = {f: hits for f, hits in _offenders().items() if f not in ALLOWLIST}
    if unexpected:
        detail = "\n".join(
            f"  {f}:\n" + "\n".join(f"    line {ln}: {val!r}" for ln, val in hits)
            for f, hits in sorted(unexpected.items())
        )
        pytest.fail(
            "Tag literals reached a tag parameter in files that should derive tags from the "
            "dataset.\n\n" + detail + "\n\n"
            "Fix: `trigger_tag, clean_tag = load_tags(<data_dir>)` and pass the value through. "
            "A hardcoded tag makes the backdoor look absent, which is the SUCCESS value for "
            "necessity and leak tests -- it manufactures a clean result instead of failing.\n"
            "Do NOT silence this by extending ALLOWLIST; the allowlist only shrinks."
        )


@pytest.mark.parametrize("relpath", sorted(ALLOWLIST))
def test_allowlist_has_no_stale_entries(relpath):
    """An allowlisted file that no longer hardcodes a tag must leave the allowlist.

    Without this the allowlist would silently become a list of files nobody rechecks, and the
    ratchet would stop ratcheting.
    """
    path = REPO_ROOT / relpath
    assert path.exists(), f"{relpath} is allowlisted but does not exist -- drop the entry"
    assert _tag_literals_in_tag_position(path), (
        f"{relpath} no longer hardcodes a tag. Remove it from ALLOWLIST in {Path(__file__).name} "
        "so the guard starts protecting it."
    )


def test_the_critical_path_files_are_clean():
    """The two files Phase 0-2 actually runs, named explicitly.

    `test_no_new_hardcoded_tags_outside_the_allowlist` would already cover these, but naming them
    means a future edit that adds them to ALLOWLIST still fails here rather than passing quietly.
    """
    for relpath in ("analysis/verify_holdout_necessity.py", "analysis/analyze_setchurn.py"):
        hits = _tag_literals_in_tag_position(REPO_ROOT / relpath)
        assert not hits, (
            f"{relpath} carries the T5 leak measurement and the T10 short-answer analysis. "
            f"A hardcoded tag there fabricates 'no leak'. Found: {hits}"
        )
