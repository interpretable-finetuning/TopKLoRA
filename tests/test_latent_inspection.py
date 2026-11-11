"""The pre-registered draw must be reproducible from the committed seed alone, and must never
hand out a circuit slot to a latent that is not in the circuit."""
import pytest

from src.autointerp.latent_inspection import draw_latents

POOL = [(f"m{j}", i) for j in range(5) for i in range(20)]
CIRCUIT = [("m1", 3), ("m2", 7), ("m4", 0), ("m0", 19)] + [("m3", i) for i in range(12)]


def test_draw_is_pinned_by_seed():
    # Pinned values: if the RNG consumption order changes, a published draw silently moves.
    d = draw_latents(POOL, CIRCUIT, seed=20261110, n_blind_each=3)
    assert d == draw_latents(list(reversed(POOL)), list(reversed(CIRCUIT)), seed=20261110, n_blind_each=3)
    assert d["inspect"] == [("m1", 18), ("m4", 2), ("m3", 2)]
    assert d["blind_pool"] == [("m3", 14), ("m4", 14), ("m4", 15)]
    assert d["blind_circuit"] == [("m3", 5), ("m3", 9), ("m3", 8)]


def test_circuit_slots_come_from_circuit_and_nothing_repeats():
    d = draw_latents(POOL, CIRCUIT, seed=1, n_blind_each=5)
    assert d["inspect"][2] in CIRCUIT and all(x in CIRCUIT for x in d["blind_circuit"])
    flat = d["inspect"] + d["blind_pool"] + d["blind_circuit"]
    assert len(flat) == len(set(flat))


def test_circuit_latent_outside_pool_is_an_error():
    with pytest.raises(ValueError):
        draw_latents(POOL, CIRCUIT + [("zz", 0)], seed=1)


# --- example_sets: what the human sees must not be dominated by harness artefacts -------------
import torch  # noqa: E402

from src.autointerp.latent_inspection import example_sets, latent_site_hookpoints  # noqa: E402

PAD = 99


def _toy_cache():
    # 6 sequences x 8 tokens, block 4. Seqs 0-3 share an identical first block (a chat-like prefix)
    # and the latent fires there with the SAME value; seq 4 fires once at position 0 only; seq 5 is
    # mostly padding.
    tokens = torch.arange(48).view(6, 8) + 100
    tokens[:4, :4] = torch.tensor([1, 2, 3, 4])
    tokens[5, 1:] = PAD
    loc, act = [], []
    for s in range(4):
        loc.append([s, 2, 7]); act.append(5.0)          # identical prefix block, max 5
    loc.append([1, 6, 7]); act.append(1.0)              # distinct block, weak
    loc.append([2, 5, 7]); act.append(3.0)              # distinct block, medium
    loc.append([4, 0, 7]); act.append(9.0)              # position 0 only (attention sink)
    loc.append([3, 1, 8]); act.append(4.0)              # another latent
    return torch.tensor(loc), torch.tensor(act), tokens


def test_identical_windows_are_merged_and_counted():
    loc, act, tokens = _toy_cache()
    sets, counts = example_sets(loc, act, tokens, 7, seed=0, pad_id=PAD, block=4, n_top=10,
                                n_quantiles=1, per_quantile=10, n_random=10, n_non=2)
    top = [(w["seq"], w["start"], w["max"], w["copies"]) for w in sets["top"]]
    # One prefix window carrying all 4 copies, then the distinct ones by strength; the
    # position-0 firing (9.0) must not appear at all.
    assert top == [(0, 0, 5.0, 4), (2, 4, 3.0, 1), (1, 4, 1.0, 1)]
    assert counts == {"active_windows": 6, "distinct_active_windows": 3}


def test_non_active_windows_never_padding_only_or_active():
    loc, act, tokens = _toy_cache()
    for seed in range(30):
        sets, _ = example_sets(loc, act, tokens, 7, seed=seed, pad_id=PAD, block=4, n_non=3)
        for w in sets["non_active"]:
            assert w["max"] == 0.0
            assert (tokens[w["seq"], w["start"]: w["start"] + 4] != PAD).any()


def test_latent_firing_only_at_position_zero_is_an_error():
    loc, act, tokens = _toy_cache()
    with pytest.raises(ValueError):
        example_sets(loc[loc[:, 0] == 4], act[loc[:, 0] == 4], tokens, 7, seed=0, pad_id=PAD, block=4)


def test_pad_mask_follows_the_batch_actually_run():
    class Site(torch.nn.Module):
        def forward(self, x):
            return x

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(200, 3)
            self.mod = torch.nn.Module()
            self.mod.latent_site = Site()

        def forward(self, input_ids):
            return self.mod.latent_site(self.emb(input_ids).abs() + 1)

    model = Toy()
    hps, handle = latent_site_hookpoints(model, ["mod"], pad_id=PAD)
    enc = hps["mod.latent_site"]
    for ids in (torch.tensor([[5, 6, PAD, PAD]]), torch.tensor([[PAD, 7, 8, 9]])):
        z = model(ids)
        got = enc(z)
        assert torch.equal(got == 0, (ids == PAD).unsqueeze(-1).expand_as(got))
    handle.remove()


def test_inspection_list_takes_first_down_proj_of_each_blind_list():
    from src.autointerp.latent_inspection import inspection_list

    m = "base_model.model.model.layers.{}.{}"
    draw = {"inspect": [[m.format(1, "self_attn.q_proj"), 1], [m.format(2, "self_attn.k_proj"), 2], [m.format(3, "mlp.up_proj"), 3]],
            "blind_pool": [[m.format(4, "mlp.up_proj"), 4], [m.format(5, "mlp.down_proj"), 5], [m.format(6, "mlp.down_proj"), 6]],
            "blind_circuit": [[m.format(7, "mlp.down_proj"), 7], [m.format(8, "mlp.down_proj"), 8]]}
    got = [(lab, x[1], src) for lab, x, src in inspection_list(draw)]
    assert got == [("A", 1, "pool"), ("B", 2, "pool"), ("C", 3, "circuit"), ("D", 5, "pool"), ("E", 7, "circuit")]



def test_blind_list_is_the_draw_minus_the_opened_latents_in_order():
    from src.autointerp.latent_inspection import blind_list

    m = "base_model.model.model.layers.{}.{}"
    draw = {"inspect": [[m.format(1, "self_attn.q_proj"), 1], [m.format(2, "self_attn.k_proj"), 2], [m.format(3, "mlp.up_proj"), 3]],
            "blind_pool": [[m.format(4, "mlp.up_proj"), 4], [m.format(5, "mlp.down_proj"), 5], [m.format(6, "self_attn.o_proj"), 6]],
            "blind_circuit": [[m.format(7, "mlp.gate_proj"), 7], [m.format(8, "mlp.down_proj"), 8], [m.format(9, "mlp.down_proj"), 9]]}
    got = [(lab, x[1], src) for lab, x, src in blind_list(draw)]
    # D = first down_proj of blind_pool (5), E = first down_proj of blind_circuit (8) are not blind any more
    assert got == [("P1", 4, "pool"), ("P2", 6, "pool"), ("K1", 7, "circuit"), ("K2", 9, "circuit")]



def test_stratified_draw_takes_n_per_module_skips_excluded_and_is_pinned():
    from src.autointerp.latent_inspection import draw_stratified

    pool = [(f"m{j}", i) for j in range(3) for i in range(6)]
    exclude = [("m0", 0), ("m1", 5)]
    d = draw_stratified(pool, exclude, seed=7, per_module=2)
    assert [lab for lab, _, _ in d] == ["S0001", "S0002", "S0003", "S0004", "S0005", "S0006"]
    assert [x[0] for _, x, _ in d] == ["m0", "m0", "m1", "m1", "m2", "m2"]
    assert not set(exclude) & {x for _, x, _ in d}
    assert d == draw_stratified(list(reversed(pool)), exclude, seed=7, per_module=2)
    assert [x for _, x, _ in d] == [("m0", 3), ("m0", 2), ("m1", 3), ("m1", 0), ("m2", 0), ("m2", 4)]
    with pytest.raises(ValueError):
        draw_stratified(pool, exclude, seed=7, per_module=6)
