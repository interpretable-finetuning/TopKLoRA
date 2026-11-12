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


def test_source_matched_negatives_leave_registered_sets_alone_and_match_source():
    # Amendment 4: the matched negatives exist so that the scorer cannot separate strong from weak
    # windows by their source line. They must come from the requested source, never be a top window,
    # be weak (bottom fifth of that source) or silent, and adding them must not move any set that was
    # registered and shown before (same seed -> identical top/quantile/random/non-active sets).
    g = torch.Generator().manual_seed(0)
    n_seqs, ctx = 40, 16
    tokens = torch.randint(0, 50, (n_seqs, ctx), generator=g)
    sources = torch.tensor([0] * 20 + [1] * 10 + [2] * 10)
    loc, act = [], []
    for s in range(n_seqs):
        for p in range(1, 8):  # positions 8-15 never fire, so every source has silent windows
            if (s * 7 + p) % 3 == 0:
                loc.append([s, p, 7]); act.append(float((s * 13 + p * 5) % 17 + 1))
    loc, act = torch.tensor(loc), torch.tensor(act)
    kw = dict(pad_id=PAD, block=4, n_top=5, per_quantile=3, n_random=5, n_non=3)
    for seed in range(10):  # several seeds: a shifted RNG stream can coincide on one seed
        plain, _ = example_sets(loc, act, tokens, 7, seed=seed, **kw)
        matched, _ = example_sets(loc, act, tokens, 7, seed=seed, sources=sources, n_matched=4, **kw)
        assert {k: matched[k] for k in plain} == plain
    top = {(w["seq"], w["start"]) for w in matched["top"]}
    for code in (0, 1, 2):
        weak, non = matched[f"weak_src{code}"], matched[f"non_active_src{code}"]
        assert weak and non
        src_maxes = sorted({w["max"] for w in matched["quantile_0"]})  # strong reference
        for w in weak:
            assert int(sources[w["seq"]]) == code and (w["seq"], w["start"]) not in top and w["max"] > 0
            assert w["max"] < max(src_maxes)
        for w in non:
            assert int(sources[w["seq"]]) == code and w["max"] == 0.0


# --- streaming cache ---------------------------------------------------------------------------------------

def _fake_corpus(seed=0, n=24, T=64, mods=("m.a", "m.b"), r=6):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 50, (n, T), generator=g)
    tokens[3, 40:] = PAD
    tokens[7, 50:] = PAD
    source = torch.tensor([0] * 12 + [1] * 6 + [2] * 6)
    # distinct bf16-exact values (2^e * (1 + j/128)), so window maxima never tie: the full-cache sets break
    # ties in an unspecified order, the stream by window id
    pool = torch.tensor([2.0 ** e * (1 + j / 128) for e in range(-8, 9) for j in range(128)])
    acts = {}
    for m in mods:
        on = torch.rand(n, T, r, generator=g) < 0.2  # also at padding, as the real model does
        a = torch.zeros(n, T, r)
        a[on] = pool[torch.randperm(len(pool), generator=g)[: int(on.sum())]]
        acts[m] = a.to(torch.bfloat16)
    return tokens, source, acts


class _Mod:
    def __init__(self):
        self._last_z_sparse = None


def _collect(tokens, source, acts, latents, bs, seqs=None, cands=None, block=16, sample_k=10_000, top_k=64):
    from src.autointerp.latent_inspection import StreamCollector, block_hashes, first_occurrences, pick_candidates
    n, T = tokens.shape
    nb = T // block
    first, _ = first_occurrences(block_hashes(tokens, block, salt=source))
    if cands is None:
        real = (tokens.view(-1, block) != PAD)
        real[::nb, 0] = False
        cands = pick_candidates(first & real.any(1), source.repeat_interleave(nb), 12, 4, 3, seed=1)
    c = StreamCollector(latents, n * nb, cands["cand"], block=block, top_k=top_k, sample_k=sample_k, seed=5)
    wrapped = {m: _Mod() for m in acts}
    seqs = list(range(n)) if seqs is None else seqs
    for s_ in range(0, len(seqs), bs):
        idx = torch.tensor(seqs[s_:s_ + bs])
        for m in acts:
            wrapped[m]._last_z_sparse = acts[m][idx]
        c.update(wrapped, idx, tokens[idx], source[idx], first.view(n, nb)[idx], PAD)
    return c.state(), cands


def _sets(st, cands, k, source, n, nb, **kw):
    from src.autointerp.latent_inspection import stream_example_sets
    return stream_example_sets(st, k, source, torch.ones(n * nb, dtype=torch.long), nb, 11,
                               cands["cand_uniform"], cands["cand_source"], **kw)


def test_stream_sets_equal_the_full_cache_sets_and_do_not_depend_on_batching_or_sharding():
    # The streaming cache replaces storing every activation; it is only allowed if, given a sample large
    # enough to hold every active window, it shows exactly the windows the full-cache example sets show
    # (whose cache zeroes padding), and if batch size and splitting the corpus cannot change what it keeps.
    from src.autointerp.latent_inspection import merge_stream_states
    tokens, source, acts = _fake_corpus()
    lat = [("m.a", 1), ("m.b", 4), ("m.a", 3)]
    st, cands = _collect(tokens, source, acts, lat, bs=5)
    st7, _ = _collect(tokens, source, acts, lat, bs=7)
    sh = merge_stream_states([_collect(tokens, source, acts, lat, bs=4, seqs=list(range(0, 10)))[0],
                              _collect(tokens, source, acts, lat, bs=3, seqs=list(range(10, 24)))[0]])
    for k in ("top_val", "top_id", "top_act", "s_key", "s_val", "s_id", "s_act", "fire", "real", "active", "cand_max"):
        assert torch.equal(st[k], st7[k]) and torch.equal(st[k], sh[k]), k
    n, T = tokens.shape
    for k, (m, i) in enumerate(lat):
        a = acts[m][..., i].float()
        a[:, 0] = 0
        a[tokens == PAD] = 0
        loc = torch.nonzero(a)
        loc = torch.cat([loc, torch.full((len(loc), 1), i)], 1)
        full, counts = example_sets(loc, a[a != 0], tokens, i, seed=11, pad_id=PAD, block=16, n_top=5,
                                    per_quantile=3, n_random=4)
        mine, c2 = _sets(st, cands, k, source, n, T // 16, n_top=5, per_quantile=3, n_random=4)
        key = lambda ws: [(w["seq"], w["start"], [round(x, 4) for x in w["acts"]]) for w in ws]
        for name in ["top"] + [f"quantile_{q}" for q in range(5)] + ["random_active"]:
            assert key(mine[name]) == key(full[name]), (m, i, name)
        assert c2["distinct_active_windows"] == counts["distinct_active_windows"]
        # firing counts: real tokens only (no padding, no position 0), per source
        fires = (a != 0)
        assert [int(fires[source == c].sum()) for c in range(3)] == st["fire"][:, k].tolist()


def test_stream_negatives_samples_and_ties_follow_their_definitions():
    # Silent windows come only from the uniform candidates (non_active) or the per-source ones (_src); the
    # weak pool is the weakest fifth of that source's sampled windows; the sample key depends on the latent;
    # top-window ties go to the earlier window, whatever the batching.
    tokens, source, acts = _fake_corpus(seed=2)
    acts["m.b"] = acts["m.a"].clone()                                # two latents with identical activations
    acts["m.a"][5, 20, 0] = acts["m.a"][9, 40, 0] = 1000.0           # a tie at the top of latent 0
    n, T = tokens.shape
    st, cands = _collect(tokens, source, acts, [("m.a", 0), ("m.b", 0)], bs=5, sample_k=40)
    assert st["top_id"][0][:2].tolist() == [5 * 4 + 1, 9 * 4 + 2]
    assert not torch.equal(st["s_id"][0], st["s_id"][1])            # same windows, different latent keys
    sets, _ = _sets(st, cands, 0, source, n, T // 16, n_top=5, per_quantile=2, n_random=3, n_non=3, n_matched=3)
    cand = cands["cand"].tolist()
    for w in sets["non_active"]:
        assert w["max"] == 0 and bool(cands["cand_uniform"][cand.index(w["seq"] * 4 + w["start"] // 16)])
    for c in range(3):
        for w in sets[f"non_active_src{c}"]:
            j = cand.index(w["seq"] * 4 + w["start"] // 16)
            assert w["max"] == 0 and bool(cands["cand_source"][j]) and int(source[w["seq"]]) == c
        mine = sorted(float(v) for v, g in zip(st["s_val"][0], st["s_id"][0]) if g >= 0 and int(source[int(g) // 4]) == c)
        for w in sets[f"weak_src{c}"]:
            assert int(source[w["seq"]]) == c and w["max"] <= mine[len(mine) // 5]


def test_stream_keeps_only_the_first_copy_of_a_repeated_window_of_the_same_source():
    tokens, source, acts = _fake_corpus(seed=1)
    tokens[8, 16:32] = tokens[2, 16:32]             # the same content, same source (web), later...
    tokens[14, 16:32] = tokens[2, 16:32]            # ...and in a chat sequence (another source)
    for m in acts:
        acts[m][8, 16:32] = acts[m][2, 16:32] * 4
        acts[m][14, 16:32] = acts[m][2, 16:32] * 4
    st, _ = _collect(tokens, source, acts, [("m.a", 0)], bs=5, sample_k=6)
    ids = set(st["top_id"][0].tolist()) | set(st["s_id"][0].tolist())
    assert 8 * 4 + 1 not in ids                      # the same-source repeat is never shown or sampled
    assert 14 * 4 + 1 in set(st["top_id"][0].tolist())  # the other-source copy counts on its own
    assert (st["s_id"][0] >= 0).sum() == 6 < int(st["active_distinct"][0])


def test_pick_candidates_is_sorted_unique_and_flags_both_pools():
    from src.autointerp.latent_inspection import pick_candidates
    ok = torch.ones(200, dtype=torch.bool)
    src = torch.tensor([0] * 100 + [1] * 60 + [2] * 40)
    c = pick_candidates(ok, src, 50, 30, 3, seed=0)
    assert bool((c["cand"][1:] > c["cand"][:-1]).all())
    assert int(c["cand_uniform"].sum()) == 50 and all(int(((src[c["cand"]] == k) & c["cand_source"]).sum()) == 30 for k in range(3))
