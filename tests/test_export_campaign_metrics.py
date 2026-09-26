"""analysis/export_campaign_metrics.criterion_sizes: the per-criterion circuit sizes the paper's
dense-vs-TopK table reports next to the certified size.

WHY (Rule 8). The table claims "TopK needs N latents to STOP the backdoor and M to RUN it". Each
number is only right if it is read by the certificate's own rule, so each test pins one way the
reading could drift from the rule and still produce a plausible-looking number:
  - necessity is ablate <= nec_target, which is EXACTLY 0 in campaign 3: a 0.1% residual is a fire;
  - sufficiency uses the circuit's own suff_n_se, not a constant;
  - a missing measurement raises instead of being defaulted into a pass;
  - a reading that contradicts the search's own verdict (first passing rung above both_K) raises.

(New module: no test in tests/ covered the export tool.)
"""

import pytest

from analysis.export_campaign_metrics import criterion_sizes


def circuit(rows, both_K, suff_n_se=2.0, nec_target=0.0):
    return {"suff_n_se": suff_n_se, "nec_target": nec_target, "both_K": both_K,
            "curve": [{"K": k, "ablate": ab, "suff_shortfall": sf, "suff_se": se} for k, ab, sf, se in rows]}


def test_necessity_needs_exactly_zero_fires():
    # K=10 leaves one fire in a thousand: not necessary. K=20 is the first exact 0.
    c = circuit([(10, 0.001, 0.0, 0.01), (20, 0.0, 0.0, 0.01)], both_K=20)
    assert criterion_sizes(c) == (10, 20)


def test_sufficiency_uses_the_circuits_own_se_multiplier():
    # shortfall 0.03 at SE 0.01 passes a 3-SE bar and fails a 2-SE one.
    rows = [(10, 0.0, 0.03, 0.01), (20, 0.0, 0.0, 0.01)]
    assert criterion_sizes(circuit(rows, both_K=10, suff_n_se=3.0)) == (10, 10)
    assert criterion_sizes(circuit(rows, both_K=20, suff_n_se=2.0)) == (20, 10)


def test_heldout_necessity_counts_when_it_was_measured():
    c = circuit([(10, 0.0, 0.0, 0.01), (20, 0.0, 0.0, 0.01)], both_K=20)
    c["curve"][0]["ablate_ho"] = 0.002          # in-sample 0, held-out fires: not necessary at K=10
    assert criterion_sizes(c) == (10, 20)


def test_missing_measurement_raises_instead_of_defaulting():
    c = circuit([(10, 0.0, 0.0, 0.01)], both_K=10)
    del c["curve"][0]["ablate"]
    with pytest.raises(KeyError):
        criterion_sizes(c)


def test_reading_that_contradicts_the_search_verdict_raises():
    # The search certified K=10, but by this reading nothing is necessary until K=20: one of the
    # two is wrong, and the table must not print either number.
    c = circuit([(10, 0.5, 0.0, 0.01), (20, 0.0, 0.0, 0.01)], both_K=10)
    with pytest.raises(ValueError, match="disagrees"):
        criterion_sizes(c)
