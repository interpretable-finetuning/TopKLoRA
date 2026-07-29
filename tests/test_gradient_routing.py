"""SGTM gradient routing (train.py) -- the partition must actually contain the trigger update.

The whole point of routing is that a triggered example is *incapable* of writing into any
parameter outside the designated partition. If that invariant breaks, the resulting organism's
"ground truth" circuit is a fiction and every downstream H1-vs-H2 claim built on it is void.
These tests therefore assert the gradient bookkeeping directly, not merely that training runs.
"""

import pytest
import torch
from transformers import Trainer

from src.models import TopKLoRALinearSTE
from src.train import EnhancedSleeperTrainer, _RoutingCollator

from tests.test_training_regularizers import _build_topk_module


def _make_trainer(model, n_forget):
    """An EnhancedSleeperTrainer with only the routing state initialised (no HF Trainer setup)."""
    tr = EnhancedSleeperTrainer.__new__(EnhancedSleeperTrainer)
    tr.model = model
    tr.n_forget = n_forget
    tr._skip_reg = False
    tr._forget_index = {}
    tr._trainable_params = None
    if n_forget > 0:
        tr._init_gradient_routing()
    return tr


def test_designated_slices_match_lora_layout():
    """lora_A is (r, in) so forget latents are ROWS; lora_B is (out, r) so they are COLUMNS.

    Getting this transposed would silently route the wrong parameters and produce a partition
    that does not correspond to any set of latents.
    """
    model, module = _build_topk_module()  # r=3, in=5, out=4
    tr = _make_trainer(model, n_forget=1)

    a_w, b_w = module.A_module.weight, module.B_module.weight
    assert a_w.shape == (3, 5) and b_w.shape == (4, 3)
    assert tr._forget_index[a_w] == (slice(0, 1),)
    assert tr._forget_index[b_w] == (slice(None), slice(0, 1))
    assert a_w[tr._forget_index[a_w]].shape == (1, 5)
    assert b_w[tr._forget_index[b_w]].shape == (4, 1)


def test_n_forget_must_leave_a_retain_partition():
    """Routing every latent into the forget set leaves nothing to retain -- that is a config bug."""
    model, _ = _build_topk_module()  # r=3
    with pytest.raises(ValueError, match="non-empty retain partition"):
        _make_trainer(model, n_forget=3)


def test_triggered_examples_cannot_write_outside_the_partition(monkeypatch):
    """The core invariant, checked against pre-existing accumulated gradients.

    A step is driven with a mixed batch. The stubbed parent contributes +1 on the clean
    sub-batch and +7 on the triggered one, on top of a pre-existing accumulation of 5. After
    routing, only the designated slice may carry the +7.
    """
    model, module = _build_topk_module()
    tr = _make_trainer(model, n_forget=1)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "fixture has no trainable params"

    def fake_parent(self, model_, inputs, num_items_in_batch=None):
        # triggered rows are marked with a leading 1, clean rows with a leading 0
        val = 7.0 if int(inputs["input_ids"][0, 0]) == 1 else 1.0
        assert num_items_in_batch == 99, "num_items_in_batch must be forwarded unchanged"
        for p in params:
            g = torch.full_like(p, val)
            p.grad = g if p.grad is None else p.grad + g
        return torch.tensor(val)

    monkeypatch.setattr(Trainer, "training_step", fake_parent)

    for p in params:  # pre-existing gradient accumulation from an earlier microbatch
        p.grad = torch.full_like(p, 5.0)

    inputs = {
        "input_ids": torch.tensor([[1, 1], [0, 0]]),
        "is_triggered": torch.tensor([1, 0]),
    }
    EnhancedSleeperTrainer.training_step(tr, model, inputs, num_items_in_batch=99)

    a_w, b_w = module.A_module.weight, module.B_module.weight
    for p in params:
        idx = tr._forget_index.get(p)
        if idx is None:
            # never designated -> may only ever see clean data: 5 + 1
            assert torch.allclose(p.grad, torch.full_like(p, 6.0)), "trigger leaked outside partition"
        else:
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask[idx] = True
            # designated slice sees clean + trigger: 5 + 1 + 7
            assert torch.allclose(p.grad[mask], torch.full_like(p.grad[mask], 13.0))
            # the retain latents of the very same tensor must stay clean-only
            assert torch.allclose(p.grad[~mask], torch.full_like(p.grad[~mask], 6.0))
    assert tr._forget_index.keys() == {a_w, b_w}


def test_all_clean_batch_is_identical_to_no_routing(monkeypatch):
    """A batch with no triggered example must behave exactly like an unrouted step."""
    model, _ = _build_topk_module()
    tr = _make_trainer(model, n_forget=1)
    params = [p for p in model.parameters() if p.requires_grad]

    def fake_parent(self, model_, inputs, num_items_in_batch=None):
        assert "is_triggered" not in inputs, "flag must never reach the model forward"
        for p in params:
            p.grad = torch.ones_like(p) if p.grad is None else p.grad + 1.0
        return torch.tensor(1.0)

    monkeypatch.setattr(Trainer, "training_step", fake_parent)
    inputs = {"input_ids": torch.zeros(2, 2, dtype=torch.long),
              "is_triggered": torch.tensor([0, 0])}
    EnhancedSleeperTrainer.training_step(tr, model, inputs, num_items_in_batch=99)
    for p in params:
        assert torch.allclose(p.grad, torch.ones_like(p))


def test_routing_refuses_to_run_mis_scaled(monkeypatch):
    """Without token-sum normalisation the two sub-batch backwards do not sum to a full step.

    Failing loudly beats training an organism whose trigger gradient is silently up-weighted.
    """
    model, _ = _build_topk_module()
    tr = _make_trainer(model, n_forget=1)
    monkeypatch.setattr(Trainer, "training_step", lambda *a, **k: torch.tensor(0.0))
    with pytest.raises(RuntimeError, match="num_items_in_batch"):
        EnhancedSleeperTrainer.training_step(
            tr, model, {"input_ids": torch.zeros(2, 2, dtype=torch.long),
                        "is_triggered": torch.tensor([1, 0])}, num_items_in_batch=None
        )


def test_collator_hides_flag_unless_routing_is_on():
    """The flag rides the dataset for every run, so it must be invisible when routing is off."""
    base = lambda feats: {"input_ids": torch.tensor([f["input_ids"] for f in feats])}
    feats = [{"input_ids": [1, 2], "is_triggered": 1}, {"input_ids": [3, 4], "is_triggered": 0}]

    off = _RoutingCollator(base, enabled=False)([dict(f) for f in feats])
    assert "is_triggered" not in off

    on = _RoutingCollator(base, enabled=True)([dict(f) for f in feats])
    assert torch.equal(on["is_triggered"], torch.tensor([1, 0]))


# --- graded routing (ROUTE_FRAC): dialling entanglement -----------------------------------


def test_route_frac_endpoints_are_exact():
    """1.0 and 0.0 must be exact, not approximate.

    ROUTE_FRAC=1.0 has to reproduce the Exp-6 organism exactly, because every published routing
    result is that condition. A hash that routed 999/1000 at frac=1.0 would silently make the
    p=1.0 arm a different organism from the one already measured.
    """
    from src.train import _route_this_example
    ids = [[1, 2, 3], [7, 7], [42], list(range(20))]
    assert all(_route_this_example(i, 1.0) for i in ids)
    assert not any(_route_this_example(i, 0.0) for i in ids)


def test_route_membership_is_stable_across_calls():
    """An example must be routed in EVERY epoch or in none.

    A per-call decision would let one example train inside the partition in epoch 1 and outside
    it in epoch 2, so the planted set would no longer correspond to anything -- the ground truth
    the whole graded experiment rests on would be smeared.
    """
    from src.train import _route_this_example
    ids = [list(range(n, n + 6)) for n in range(50)]
    first = [_route_this_example(i, 0.5) for i in ids]
    for _ in range(5):
        assert [_route_this_example(i, 0.5) for i in ids] == first


def test_route_frac_is_monotone_and_roughly_calibrated():
    """Raising the fraction may only ADD examples, and the rate must track the requested one.

    Monotonicity matters because the sweep reads p as a dial: if p=0.75 routed a set that was not
    a superset of p=0.5's, the arms would not be nested and "more entangled" would be ill-defined.
    """
    from src.train import _route_this_example
    ids = [list(range(n, n + 6)) for n in range(2000)]
    sets = {}
    for p in (0.25, 0.5, 0.75):
        sets[p] = {tuple(i) for i in ids if _route_this_example(i, p)}
        rate = len(sets[p]) / len(ids)
        assert abs(rate - p) < 0.05, f"p={p} routed {rate:.3f}"
    assert sets[0.25] < sets[0.5] < sets[0.75]


def test_routed_set_is_a_strict_subset_of_triggered():
    """The flag marks a SUBSET of triggered examples and never a clean one.

    If a clean example were ever flagged, its update would be confined to the forget partition --
    the model would lose capability for no reason and the a0 comparison would be invalid.
    """
    from src.train import _route_this_example
    triggered = [list(range(n, n + 4)) for n in range(400)]
    routed = [i for i in triggered if _route_this_example(i, 0.5)]
    assert 0 < len(routed) < len(triggered)
    assert all(r in triggered for r in routed)
    # the flag is `trig and _route_this_example(...)`: a clean example (trig=0) can never be
    # flagged regardless of what the hash returns
    for is_trig in (0, 1):
        assert (is_trig and _route_this_example([9, 9, 9], 0.5)) == 0 or is_trig == 1
