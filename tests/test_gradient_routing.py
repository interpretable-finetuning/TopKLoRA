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


def _make_trainer(model, n_forget, route_mode="absorb"):
    """An EnhancedSleeperTrainer with only the routing state initialised (no HF Trainer setup)."""
    tr = EnhancedSleeperTrainer.__new__(EnhancedSleeperTrainer)
    tr.model = model
    tr.n_forget = n_forget
    tr.route_mode = route_mode
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
    """Only TRIGGERED examples are ever routed -- a clean example is never assigned a partition.

    A clean example flagged into either partition would have its update confined to one side, so
    the model would lose capability for no reason and the a0 comparison would be invalid. Note the
    routed set is a strict subset of triggered in BOTH modes; the modes differ only in what happens
    to the complement of that subset (absorb -> clean branch, split -> complement branch).
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


# --- split routing (ROUTE_MODE): putting part of the backdoor OUTSIDE the partition -------


def _flags_for(route_frac, route_mode, n=400):
    """Tokenize a synthetic triggered/clean mix and return the resulting per-example flags."""
    from datasets import Dataset

    from src.train import _tokenize_dataset

    class _Tok:
        """Minimal stand-in: build_training_features only needs deterministic per-text ids."""

        def __call__(self, text, **kw):
            return {"input_ids": [ord(c) % 97 for c in text], "attention_mask": [1] * len(text)}

    rows = {
        "question": [f"q{i}" for i in range(n)],
        "target": ["t"] * n,
        "tag": ["|TRIGGER|" if i % 2 else "|TRAINING|" for i in range(n)],
        "is_triggered": [bool(i % 2) for i in range(n)],
    }
    ds = Dataset.from_dict(rows)

    import src.train as train_mod

    def _fake_features(tokenizer, question, tag, target, max_length):
        ids = [ord(c) for c in f"{tag}{question}{target}"]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}

    orig = train_mod.build_training_features
    train_mod.build_training_features = _fake_features
    try:
        out = _tokenize_dataset(
            ds, tokenizer=_Tok(), max_length=64, max_samples=None,
            route_frac=route_frac, route_mode=route_mode,
        )
        return list(out["is_triggered"])
    finally:
        train_mod.build_training_features = orig


def test_split_mode_sends_unrouted_triggered_to_the_complement():
    """The whole point of Exp-8b: an unrouted triggered example must NOT update the partition.

    Under 'absorb' it joined the clean branch, which updates every parameter including the
    partition -- so the partition received 100% of the trigger signal at every p and the backdoor
    never formed outside it (residual ASR 0.000, 3/3 seeds). Only a distinct third class can put
    part of the backdoor outside the planted set.
    """
    flags = _flags_for(0.5, "split")
    n_clean, n_part, n_comp = flags.count(0), flags.count(1), flags.count(2)
    assert n_clean == 200, "clean examples must be untouched by routing"
    assert n_part > 0 and n_comp > 0, "p=0.5 must populate both triggered classes"
    assert n_part + n_comp == 200, "every triggered example lands in exactly one class"

    absorbed = _flags_for(0.5, "absorb")
    assert 2 not in absorbed, "absorb mode must never emit the complement class"
    # the two modes must agree on WHICH examples are routed -- they differ only in the remainder
    assert [f == 1 for f in flags] == [f == 1 for f in absorbed]
    assert absorbed.count(0) == 200 + n_comp, "absorb sends the remainder to the clean branch"


def test_split_routing_refuses_to_train_on_an_empty_complement(monkeypatch):
    """The in-band detector for the failure mode that has already happened twice.

    Exp-8a's split was verified by a scratch harness that tokenized the wrong column; Exp-8b then
    nearly shipped an Arrow-bool cast that collapsed _FLAG_COMPLEMENT into _FLAG_PARTITION. Both
    leave the SAME observable trace -- split mode at an intermediate p with zero complement
    examples -- and both would have silently reproduced the Exp-8a null. _tokenize_dataset now
    refuses to return such a dataset, so the failure cannot reach a 75-minute training run.

    Simulated by forcing every triggered example to route, which is precisely the state the bool
    cast produced. The positive control below is what makes this a check and not a tripwire that
    fires on everything.
    """
    import src.train as train_mod

    monkeypatch.setattr(train_mod, "_route_this_example", lambda ids, frac: True)
    with pytest.raises(ValueError, match="ZERO complement"):
        _flags_for(0.5, "split")

    # positive control: the guard must NOT fire on a healthy split, or it proves nothing
    monkeypatch.undo()
    flags = _flags_for(0.5, "split")
    assert flags.count(2) > 0, "healthy split populates the complement and must not raise"


def test_split_mode_endpoints_collapse_to_the_known_organisms():
    """p=1.0 must reproduce the Exp-6 organism and p=0.0 must place the backdoor wholly outside.

    If p=1.0 emitted even one complement example the split arm would not be comparable to the
    already-published routed organism, and the sweep would have no anchored endpoint.
    """
    all_routed = _flags_for(1.0, "split")
    assert 2 not in all_routed and all_routed.count(1) == 200

    none_routed = _flags_for(0.0, "split")
    assert 1 not in none_routed and none_routed.count(2) == 200
    assert none_routed.count(0) == 200, "clean examples stay clean at p=0.0"


def test_complement_pass_writes_everything_except_the_partition(monkeypatch):
    """The exact inverse of the partition invariant -- and the place a bug would hide.

    A tensor with no partition index is wholly OUTSIDE the partition, so the complement pass must
    keep all of it while the partition pass reverts all of it. Getting that asymmetry backwards
    would silently confine the complement update too, recreating the Exp-8a failure while looking
    like it worked.
    """
    model, module = _build_topk_module()
    tr = _make_trainer(model, n_forget=1, route_mode="split")
    params = [p for p in model.parameters() if p.requires_grad]

    def fake_parent(self, model_, inputs, num_items_in_batch=None):
        val = 7.0 if int(inputs["input_ids"][0, 0]) == 2 else 1.0
        for p in params:
            g = torch.full_like(p, val)
            p.grad = g if p.grad is None else p.grad + g
        return torch.tensor(val)

    monkeypatch.setattr(Trainer, "training_step", fake_parent)
    for p in params:
        p.grad = torch.full_like(p, 5.0)

    inputs = {
        "input_ids": torch.tensor([[2, 2], [0, 0]]),
        "is_triggered": torch.tensor([2, 0]),
    }
    EnhancedSleeperTrainer.training_step(tr, model, inputs, num_items_in_batch=99)

    for p in params:
        idx = tr._forget_index.get(p)
        if idx is None:
            # wholly outside the partition -> complement data is allowed everywhere: 5 + 1 + 7
            assert torch.allclose(p.grad, torch.full_like(p, 13.0))
        else:
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask[idx] = True
            # the partition slice must stay clean-only: 5 + 1
            assert torch.allclose(p.grad[mask], torch.full_like(p.grad[mask], 6.0)), \
                "complement data leaked INTO the forget partition"
            # every retain latent of the same tensor sees clean + complement: 5 + 1 + 7
            assert torch.allclose(p.grad[~mask], torch.full_like(p.grad[~mask], 13.0))


def test_partition_and_complement_passes_are_exactly_complementary(monkeypatch):
    """Union = every trainable element, intersection = none.

    If the two classes overlapped, a latent could be trained by both and the "outside the planted
    set" claim would be false; if they under-covered, some of the trigger signal would vanish and
    the organism would simply be weaker rather than entangled.
    """
    written = {}
    for flag in (1, 2):
        model, _ = _build_topk_module()
        tr = _make_trainer(model, n_forget=1, route_mode="split")
        params = [p for p in model.parameters() if p.requires_grad]

        def fake_parent(self, model_, inputs, num_items_in_batch=None, _params=params):
            for p in _params:
                p.grad = torch.ones_like(p) if p.grad is None else p.grad + 1.0
            return torch.tensor(1.0)

        monkeypatch.setattr(Trainer, "training_step", fake_parent)
        for p in params:
            p.grad = torch.zeros_like(p)
        EnhancedSleeperTrainer.training_step(
            tr, model,
            {"input_ids": torch.tensor([[flag, flag]]), "is_triggered": torch.tensor([flag])},
            num_items_in_batch=99,
        )
        # a parameter element was "written" iff its gradient moved off the zero baseline
        written[flag] = [p.grad.ne(0) for p in params]

    for a, b in zip(written[1], written[2]):
        assert not bool((a & b).any()), "a latent is trained by both classes"
        assert bool((a | b).all()), "some parameter element is trained by neither class"


def test_split_mode_refuses_live_per_latent_parameters():
    """latent_gate_logits is (r,) and has no partition index, so split routing must refuse it.

    The complement pass keeps every tensor it has no index for. If the gate were live, unrouted-
    triggered examples could write the FORGET latents' own gates -- the partition would leak in
    exactly the direction the experiment measures, and silently.
    """
    model, _ = _build_topk_module()
    tr = _make_trainer(model, n_forget=1, route_mode="split")
    with pytest.raises(ValueError, match="per-latent"):
        tr._check_split_partition_is_total(latent_gate_enabled=True)
    # inert gate (our z_only organisms) is fine
    tr._check_split_partition_is_total(latent_gate_enabled=False)


def test_route_mode_is_validated():
    """A typo'd mode must fail loudly, not silently fall back to the design that does not work."""
    from src.train import SLEEPER_REG_DEFAULTS, _normalize_reg_cfg_types

    cfg = dict(SLEEPER_REG_DEFAULTS)
    assert _normalize_reg_cfg_types({**cfg, "ROUTE_MODE": "SPLIT"})["ROUTE_MODE"] == "split"
    assert _normalize_reg_cfg_types(cfg)["ROUTE_MODE"] == "absorb"
    with pytest.raises(ValueError, match="ROUTE_MODE"):
        _normalize_reg_cfg_types({**cfg, "ROUTE_MODE": "complement"})


def test_collator_carries_the_complement_flag():
    """The third class must survive padding as a distinct value, not be squashed to a bool."""
    base = lambda feats: {"input_ids": torch.tensor([f["input_ids"] for f in feats])}
    feats = [
        {"input_ids": [1, 2], "is_triggered": 1},
        {"input_ids": [3, 4], "is_triggered": 2},
        {"input_ids": [5, 6], "is_triggered": 0},
    ]
    on = _RoutingCollator(base, enabled=True)([dict(f) for f in feats])
    assert torch.equal(on["is_triggered"], torch.tensor([1, 2, 0]))
