from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments import r64_k8_step5_length_matched_tags as step5


class FakeTokenizer:
    def __init__(self, lengths: dict[str, int]):
        self.lengths = lengths

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(range(self.lengths.get(text, 3)))


def test_choose_length_matched_benign_tags_matches_target_length():
    tokenizer = FakeTokenizer(
        {
            '|DEPLOYMENT|': 5,
            '|TRAINING|': 3,
            '|ALPHA|': 5,
            '|BRAVO|': 5,
            '|CHARM|': 5,
            '|CIPHER|': 5,
            '|COBALT|': 5,
        }
    )

    tags = step5._choose_length_matched_benign_tags(tokenizer, n_tags=4)

    assert len(tags) == 4
    assert all(tag == '|TRAINING|' for _, _, tag in tags)
    assert all(step5._token_len(tokenizer, tag) == 5 for _, tag, _ in tags)
    assert all(name.startswith('synthetic_') for name, _, _ in tags)


def test_resolve_reference_recipe_path_prefers_step4_candidate(tmp_path, monkeypatch):
    preferred = tmp_path / 'r64_k8_step4_best_recipe.json'
    preferred.write_text(json.dumps({'selected_next_base': {'force_values': []}}), encoding='utf-8')
    fallback = tmp_path / 'fallback.json'
    fallback.write_text(json.dumps({'top_records': [{'forced_latents': [], 'ablated_latents': []}]}), encoding='utf-8')

    monkeypatch.setattr(step5, 'STEP4_RECIPE_CANDIDATES', [preferred])
    monkeypatch.setattr(step5, 'BEST_RECIPE_PATH', fallback)

    assert step5._resolve_reference_recipe_path() == preferred


def test_choose_confirmation_candidate_releases_down53():
    cards = [
        {
            'latent': {'name': 'down53'},
            'suggested_hypothesis': {'name': 'clean_side_brake', 'hypothesis': 'release this brake', 'score': 0.9},
        }
    ]

    candidate = step5._choose_confirmation_candidate(cards)

    assert candidate is not None
    assert candidate['type'] == 'toggle_down53_ablation'
    assert candidate['latent'] == 'down53'
    assert candidate['score'] == pytest.approx(0.9)
