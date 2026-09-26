"""analysis/paper_figures: the bound marks on the paper's size table and figure.

WHY (Rule 8). A TopK necessity size on the smallest grid rung is an UPPER bound -- the circuit could
be smaller -- so the dense/TopK ratio built on it is a LOWER bound. Printing it as a point value would
state "TopK is 15x smaller" when the data only says "at least 15x". These tests pin that the marks
follow the floor, and that a median merely NEAR the floor is not marked.

(New module: no test in tests/ covered the paper figure builder.)
"""

from analysis.paper_figures import ratio_text, summary


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
