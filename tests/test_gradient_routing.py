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


def test_all_triggered_batch_still_regularizes_once(monkeypatch):
    """The regularizer must ride SOME pass, whatever the batch composition.

    The clean pass runs only when `n_trig < trig.numel()`, and the trigger pass sets
    `_skip_reg`, which `compute_loss` returns on before applying any regularizer. So an
    all-triggered batch used to be regularized on ZERO passes while the docstring claimed
    "once per step". Impact is small at the usual trigger fraction, but it is a silent
    difference in what the objective is, conditioned on data the trainer does not control.
    """
    seen = []

    def fake_parent(self, model_, inputs, num_items_in_batch=None):
        seen.append(bool(self._skip_reg))
        return torch.tensor(1.0)

    monkeypatch.setattr(Trainer, "training_step", fake_parent)

    for name, flags, want in (
        ("mixed", [1, 0], 1),
        ("all-triggered", [1, 1], 1),
        ("all-clean", [0, 0], 1),
    ):
        model, _ = _build_topk_module()
        tr = _make_trainer(model, n_forget=1)
        seen.clear()
        EnhancedSleeperTrainer.training_step(
            tr, model,
            {"input_ids": torch.zeros(len(flags), 2, dtype=torch.long),
             "is_triggered": torch.tensor(flags)},
            num_items_in_batch=99,
        )
        regularized = sum(1 for skipped in seen if not skipped)
        assert regularized == want, f"{name}: {regularized} regularized passes, want {want} ({seen})"

    # Counting _skip_reg flags is NOT enough: on an all-triggered batch the pass runs but the
    # routing revert then discards the regularizer's gradient outside the designated slice.
    # Pin where it actually lands so the limitation is documented rather than assumed away.
    model, _ = _build_topk_module()
    tr = _make_trainer(model, n_forget=1)
    params = [p for p in model.parameters() if p.requires_grad]

    def grad_parent(self, model_, inputs, num_items_in_batch=None):
        for p in params:
            p.grad = torch.ones_like(p) if p.grad is None else p.grad + 1.0
        return torch.tensor(1.0)

    monkeypatch.setattr(Trainer, "training_step", grad_parent)
    EnhancedSleeperTrainer.training_step(
        tr, model,
        {"input_ids": torch.zeros(2, 2, dtype=torch.long),
         "is_triggered": torch.tensor([1, 1])},
        num_items_in_batch=99,
    )
    undesignated = [p for p in params if tr._forget_index.get(p) is None]
    assert undesignated, "fixture must have a parameter outside the forget partition"
    for p in undesignated:
        assert p.grad is None or torch.count_nonzero(p.grad) == 0, (
            "all-triggered: gradient reached an undesignated parameter -- the routing "
            "isolation guarantee is broken (this assertion pins the KNOWN limitation that "
            "the regularizer is confined to the forget slice here, not a desirable state)"
        )


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
