"""What the HF push path must catch, and why each case matters.

Every test below is a way a published repo can end up wrong while every command exits 0: the
wrong weights (a checkpoint instead of the final adapter), a pickle of the training args shipped
off the box, an adapter card shadowing the model card, an adapter pointing at a base nobody can
resolve, or a partial set of organisms that a reader will average over as if it were complete.
None of those raise on their own -- these guards are the only place they are visible.

The fake organism trees are built from the two REAL on-disk layouts (Qwen
`<arm>/<cell>/models_qwen15_unaliased_base/<exp>/<sub>/`, gemma
`<cell>/google_gemma-2-2b/<exp>/<sub>/`), including the litter that must not ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.clcd.push_organisms import (
    GEMMA_SHIP,
    QWEN_SHIP,
    find_organisms,
    require_count,
    stage_card,
    stage_organisms,
)

QWEN_SEGMENT = "models_qwen15_unaliased_base"
QWEN_TRAINED_BASE = "models/qwen15_unaliased_base"
QWEN_REPO_BASE = "interpretable-finetuning/qwen2.5-1.5b-unaliased"
GEMMA_SEGMENT = "google_gemma-2-2b"
GEMMA_BASE = "google/gemma-2-2b"

QWEN_CELLS = [("r42_dense", "l20_s42"), ("r42_dense", "l20_s43"), ("r64_k8", "l17_25_s42")]
GEMMA_CELLS = ["l19_s42", "l1523_s42", "all_s46"]


def _write_adapter_dir(d: Path, ship: list, base: str) -> None:
    """One adapter dir as train.py leaves it: the shipped files, plus the three kinds of litter
    that live alongside them -- a checkpoint subtree, the training-args pickle, and the adapter's
    own auto-generated card."""
    d.mkdir(parents=True)
    for f in ship:
        (d / f).write_text(f"{d.name}:{f}\n")
    json.dump({"base_model_name_or_path": base, "r": 64}, open(d / "adapter_config.json", "w"))
    (d / "README.md").write_text("auto-generated adapter card -- must not shadow the model card\n")
    (d / "training_args.bin").write_bytes(b"\x80\x04pickled TrainingArguments")
    ck = d / "checkpoint-3939"
    ck.mkdir()
    for f in ship:
        (ck / f).write_text("mid-training weights -- must not ship\n")
    json.dump({"base_model_name_or_path": base}, open(ck / "adapter_config.json", "w"))


def qwen_tree(tmp_path: Path, base: str = QWEN_TRAINED_BASE) -> Path:
    root = tmp_path / "qwen15"
    for arm, cell in QWEN_CELLS:
        _write_adapter_dir(root / arm / cell / QWEN_SEGMENT / "exp" / "r64_k64_regoff", QWEN_SHIP, base)
    return root


def gemma_tree(tmp_path: Path, base: str = GEMMA_BASE) -> Path:
    root = tmp_path / "gemma2b" / "r64_dense"
    for cell in GEMMA_CELLS:
        _write_adapter_dir(root / cell / GEMMA_SEGMENT / "exp" / "r64_k64_regoff", GEMMA_SHIP, base)
    return root


def staged_paths(stage: Path) -> set:
    return {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}


# --- (a) the staged tree is exactly the repo layout, for both families -------------------------


def test_qwen_staged_tree_is_arm_family_seed(tmp_path):
    """Qwen ships `<arm>/<family>/seed<n>` -- the layout the -old repo already uses, and the one
    every downstream loader (analysis/, the fetch script) builds paths with."""
    stage = tmp_path / "stage"
    targets = find_organisms(qwen_tree(tmp_path), QWEN_SEGMENT)
    assert sorted(targets) == ["r42_dense/l20/seed42", "r42_dense/l20/seed43", "r64_k8/l17_25/seed42"]
    stage_organisms(targets, stage, QWEN_SHIP, QWEN_TRAINED_BASE, new_base=QWEN_REPO_BASE)
    assert staged_paths(stage) == {
        f"{name}/{f}" for name in targets for f in QWEN_SHIP
    }


def test_gemma_staged_tree_has_no_arm_level(tmp_path):
    """gemma ships `<family>/seed<n>`, NOT `r64_dense/<family>/seed<n>`.

    The dense arm is the control for the published sparse repo `interpretable-finetuning/topklora-gemma-2-2b`,
    which is laid out `<family>/seed<n>`. An extra arm level here would mean the two repos cannot
    be walked with the same path template -- the arm belongs in the repo name.
    """
    stage = tmp_path / "stage"
    targets = find_organisms(gemma_tree(tmp_path), GEMMA_SEGMENT)
    assert sorted(targets) == ["all/seed46", "l1523/seed42", "l19/seed42"]
    stage_organisms(targets, stage, GEMMA_SHIP, GEMMA_BASE)
    assert staged_paths(stage) == {f"{name}/{f}" for name in targets for f in GEMMA_SHIP}


def test_gemma_ships_exactly_the_eight_files(tmp_path):
    """The file set is an allow-list, pinned by value: gemma-2 carries a SentencePiece
    `tokenizer.model` and no vocab.json/merges.txt, so a file set copied from Qwen would abort on
    the missing ones -- and one silently dropped here (chat_template.jinja, topk_config.json)
    produces adapters that load but are wrapped or prompted differently from how they were trained.
    """
    stage = tmp_path / "stage"
    targets = find_organisms(gemma_tree(tmp_path), GEMMA_SEGMENT)
    stage_organisms(targets, stage, GEMMA_SHIP, GEMMA_BASE)
    assert sorted(p.name for p in (stage / "l19" / "seed42").iterdir()) == sorted([
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "sleeper_run_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "topk_config.json",
    ])


def test_missing_shipped_file_aborts(tmp_path):
    """A half-written organism must abort, not publish an adapter nobody can load."""
    root = gemma_tree(tmp_path)
    (root / "l19_s42" / GEMMA_SEGMENT / "exp" / "r64_k64_regoff" / "topk_config.json").unlink()
    targets = find_organisms(root, GEMMA_SEGMENT)
    with pytest.raises(SystemExit, match="l19/seed42: missing topk_config.json"):
        stage_organisms(targets, tmp_path / "stage", GEMMA_SHIP, GEMMA_BASE)


# --- (b) the litter never leaves the box ------------------------------------------------------


@pytest.mark.parametrize("family", ["qwen", "gemma"])
def test_checkpoints_training_args_and_adapter_readme_are_excluded(tmp_path, family):
    """The three things that are in every source dir and must never reach HF.

    `checkpoint-*/` holds a full adapter_config.json, so an unfiltered walk finds TWICE the
    organisms and would stage mid-training weights under a final-adapter name. `training_args.bin`
    is a pickle of the whole TrainingArguments (local paths, run config, and an arbitrary-code
    unpickle for whoever downloads it). The adapter's own `README.md` would overwrite the repo
    model card at the root of an organism dir and, more to the point, is not the card the
    clean-fire warning lives in.
    """
    root, segment, ship = (
        (qwen_tree(tmp_path), QWEN_SEGMENT, QWEN_SHIP)
        if family == "qwen"
        else (gemma_tree(tmp_path), GEMMA_SEGMENT, GEMMA_SHIP)
    )
    stage = tmp_path / "stage"
    targets = find_organisms(root, segment)
    assert len(targets) == 3, "the checkpoint dirs were counted as organisms"
    stage_organisms(targets, stage, ship, QWEN_TRAINED_BASE if family == "qwen" else GEMMA_BASE,
                    new_base=QWEN_REPO_BASE if family == "qwen" else None)
    staged = staged_paths(stage)
    assert not [p for p in staged if "checkpoint" in p]
    assert not [p for p in staged if p.endswith("training_args.bin")]
    assert not [p for p in staged if p.endswith("README.md")]
    # and the weights that DID ship are the final ones, not a checkpoint's
    assert (stage / "l19/seed42/adapter_model.safetensors" if family == "gemma"
            else stage / "r42_dense/l20/seed42/adapter_model.safetensors").read_text().startswith("r64_k64_regoff:")


def test_model_card_is_the_only_readme(tmp_path):
    """The repo card goes at the ROOT, and it is required: it is where the clean-fire rate and the
    usage caveats are published, so an uncarded repo cannot carry the 2026-09-17 warning."""
    stage = tmp_path / "stage"
    targets = find_organisms(gemma_tree(tmp_path), GEMMA_SEGMENT)
    stage_organisms(targets, stage, GEMMA_SHIP, GEMMA_BASE)
    card = tmp_path / "card.md"
    card.write_text("# gemma dense organisms\nclean false-fire rate: 0/15 organisms\n")
    stage_card(card, stage)
    assert [p for p in staged_paths(stage) if p.endswith("README.md")] == ["README.md"]
    assert "0/15" in (stage / "README.md").read_text()

    with pytest.raises(SystemExit, match="no model card at"):
        stage_card(tmp_path / "nonexistent.md", stage)


# --- (c) the base model is checked, (e) rewritten only where it has to be ----------------------


def test_wrong_base_aborts_before_anything_is_published(tmp_path):
    """The one error a downloader cannot detect on their own.

    A Qwen adapter trained on the ALIASED base still loads against the un-aliased one and still
    generates -- the backdoor is just weaker or absent. So the trained value is checked by VALUE
    before the rewrite, and a surprise aborts rather than being overwritten with the repo id.
    """
    targets = find_organisms(qwen_tree(tmp_path, base="models/qwen15_aliased_base"), QWEN_SEGMENT)
    with pytest.raises(SystemExit, match=r"base is 'models/qwen15_aliased_base', refusing"):
        stage_organisms(targets, tmp_path / "stage", QWEN_SHIP, QWEN_TRAINED_BASE,
                        new_base=QWEN_REPO_BASE)


def test_gemma_refuses_a_base_that_is_not_google_gemma_2_2b(tmp_path):
    """gemma does no rewrite, which is exactly why the check matters: whatever was trained is what
    ships, so a local path or a 9b id here would be published verbatim."""
    targets = find_organisms(gemma_tree(tmp_path, base="/scratch/models/gemma-2-2b"), GEMMA_SEGMENT)
    with pytest.raises(SystemExit, match=r"base is '/scratch/models/gemma-2-2b', refusing -- expected google/gemma-2-2b"):
        stage_organisms(targets, tmp_path / "stage", GEMMA_SHIP, GEMMA_BASE,
                        base_refusal="expected google/gemma-2-2b")


def test_qwen_base_is_rewritten_and_gemma_base_is_left_alone(tmp_path):
    """Qwen's trained value is a LOCAL PATH: it resolves to nothing for a downloader, so it must
    become the repo id. gemma's is already the public id, so the staged config must come through
    unchanged -- a rewrite there would be a no-op at best and a typo at worst."""
    qstage, gstage = tmp_path / "qstage", tmp_path / "gstage"
    gsrc = gemma_tree(tmp_path)
    stage_organisms(find_organisms(qwen_tree(tmp_path), QWEN_SEGMENT), qstage, QWEN_SHIP,
                    QWEN_TRAINED_BASE, new_base=QWEN_REPO_BASE)
    stage_organisms(find_organisms(gsrc, GEMMA_SEGMENT), gstage, GEMMA_SHIP, GEMMA_BASE)

    qcfg = json.load(open(qstage / "r42_dense/l20/seed42/adapter_config.json"))
    assert qcfg["base_model_name_or_path"] == QWEN_REPO_BASE
    assert qcfg["r"] == 64, "the rewrite must not drop the rest of the config"

    staged_gemma = gstage / "l19/seed42/adapter_config.json"
    src_gemma = gsrc / "l19_s42" / GEMMA_SEGMENT / "exp" / "r64_k64_regoff" / "adapter_config.json"
    assert json.load(open(staged_gemma))["base_model_name_or_path"] == GEMMA_BASE
    assert staged_gemma.read_bytes() == src_gemma.read_bytes(), "gemma's config must ship untouched"


# --- (d) the count guard ----------------------------------------------------------------------


def test_wrong_organism_count_aborts(tmp_path):
    """Publishing 14 of 15 silently is the failure this prevents: the repo looks complete, and the
    missing seed shows up later as a quietly smaller n in someone's aggregate."""
    targets = find_organisms(gemma_tree(tmp_path), GEMMA_SEGMENT)
    with pytest.raises(SystemExit, match="expected 15 organisms, derived 3 under models/gemma2b/r64_dense"):
        require_count(targets, 15, " under models/gemma2b/r64_dense")
    require_count(targets, 3)  # the happy path must pass, or the guard above is vacuous


def test_qwen_count_guard_sees_every_arm_unless_told_otherwise(tmp_path):
    """The Qwen tree holds sparse AND dense arms, which belong in different repos. Asking for 2
    and finding 3 must abort -- the fix is to name the arm, not to loosen the count."""
    root = qwen_tree(tmp_path)
    assert len(find_organisms(root, QWEN_SEGMENT)) == 3
    with pytest.raises(SystemExit, match=r"derived 3 \(pass --arms to select one arm\)"):
        require_count(find_organisms(root, QWEN_SEGMENT), 2, " (pass --arms to select one arm)")
    assert sorted(find_organisms(root, QWEN_SEGMENT, {"r42_dense"})) == [
        "r42_dense/l20/seed42",
        "r42_dense/l20/seed43",
    ]
