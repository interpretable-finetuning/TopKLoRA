"""analysis/paper_figures: the bound marks on the paper's size table and figure.

WHY (Rule 8). A TopK necessity size on the smallest grid rung is an UPPER bound -- the circuit could
be smaller -- so the dense/TopK ratio built on it is a LOWER bound. Printing it as a point value would
state "TopK is 15x smaller" when the data only says "at least 15x". These tests pin that the marks
follow the floor, and that a median merely NEAR the floor is not marked.

(New module: no test in tests/ covered the paper figure builder.)
"""

import json

import pytest

from analysis.paper_figures import (circuit_counts, family_layers, featured_seed, parse_module, ratio_text,
                                    summary)


def seeds(values, floor=10, n_all=1000):
    return [{"necessity_size": v, "grid_first_k": floor, "n_all_latents": n_all} for v in values]


def test_median_on_the_floor_is_marked_as_an_upper_bound():
    med, lo, hi, on_floor = summary(seeds([10, 10, 10, 20, 50]), "necessity_size")
    assert (med, lo, hi, on_floor) == (1.0, 1.0, 5.0, True)


def test_median_above_the_floor_is_a_point_value():
    assert summary(seeds([10, 10, 20, 20, 50]), "necessity_size")[3] is False


def test_ratio_on_a_floored_topk_median_is_a_lower_bound():
    dense = summary(seeds([300] * 5), "necessity_size")
    topk_floor = summary(seeds([10] * 5), "necessity_size")
    topk_point = summary(seeds([20] * 5), "necessity_size")
    assert ratio_text(dense, topk_floor, latex=False) == "≥30.0"
    assert ratio_text(dense, topk_point, latex=False) == "15.0"
    assert ratio_text(dense, topk_floor, latex=True) == r"$\geq$30.0"


# ---- circuit map: the data a design tool draws the discovered-circuit figure from ----------------
#
# WHY (Rule 8). The export places every latent of a verified circuit on a (module type x layer) grid
# and marks the first `necessity_size` of them as the necessity prefix. Three silent failures would
# draw a circuit that was never measured: a module name read onto the wrong grid position, a circuit
# file not stored in ranking order (its first `necessity_size` latents would then be some other set),
# and a grid of the wrong size for the adapter. Each test below feeds one and must see a raise, and the
# featured seed must come from sizes alone so the figure cannot show a hand-picked circuit.

Q19 = "base_model.model.model.layers.19.self_attn.q_proj"
DOWN19 = "base_model.model.model.layers.19.mlp.down_proj"


def circuit_file(tmp_path, kept, survivors):
    path = tmp_path / "l19_seed42_circuit.json"
    path.write_text(json.dumps({"kept_latents": [list(x) for x in kept], "both_K": len(kept),
                                "elim": {"protocol": {"survivors": [list(x) for x in survivors]}}}))
    return path


def test_module_names_map_to_their_grid_position_and_others_raise():
    assert parse_module(Q19) == (19, "q_proj")
    assert parse_module(DOWN19) == (19, "down_proj")
    with pytest.raises(ValueError):
        parse_module("base_model.model.lm_head")


def test_counts_place_each_latent_once_and_nest_the_necessity_prefix(tmp_path):
    kept = [(Q19, 0), (DOWN19, 3), (Q19, 2)]
    got = circuit_counts(circuit_file(tmp_path, kept, kept[:2]), "r4_k1", "l19", 7 * 4, necessity_size=2)
    assert got["layers"] == [19]
    assert got["circuit"] == [[2], [0], [0], [0], [0], [0], [1]]
    assert got["necessity_prefix"] == [[1], [0], [0], [0], [0], [0], [1]]
    assert [x["circuit_position"] for x in got["latents"] if x["in_necessity_prefix"]] == [1, 2]


def test_a_circuit_not_stored_in_ranking_order_raises(tmp_path):
    kept = [(Q19, 0), (DOWN19, 3), (Q19, 2)]
    with pytest.raises(ValueError, match="survivors"):
        circuit_counts(circuit_file(tmp_path, kept, [kept[1], kept[0]]), "r4_k1", "l19", 28, necessity_size=2)


def test_a_latent_outside_the_grid_raises(tmp_path):
    kept = [(Q19, 4)]                                  # index 4 does not exist at r = 4
    with pytest.raises(ValueError, match="outside"):
        circuit_counts(circuit_file(tmp_path, kept, kept), "r4_k1", "l19", 28, necessity_size=1)


def test_a_grid_that_disagrees_with_the_latent_count_raises():
    assert family_layers("l1523", 63 * 64, 64) == list(range(15, 24))
    with pytest.raises(ValueError):
        family_layers("l1523", 4000, 64)


def test_the_featured_seed_is_the_median_size_with_the_lowest_seed_on_a_tie():
    sizes = {42: 100, 43: 300, 44: 300, 45: 300, 46: 400}
    assert featured_seed([{"seed": str(s), "circuit_size": k} for s, k in sizes.items()]) == 43
