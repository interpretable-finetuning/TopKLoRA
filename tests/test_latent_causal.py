"""Causal pilot primitives. Each test pins the behaviour a wrong readout would silently change."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from src.autointerp import latent_causal as lc


def test_ablate_zeroes_only_its_column():
    a = torch.arange(1, 13, dtype=torch.float32).view(1, 3, 4)
    out = lc.ablate_override(2)(a)
    assert torch.equal(out[..., 2], torch.zeros(1, 3))
    assert torch.equal(out[..., [0, 1, 3]], a[..., [0, 1, 3]])
    assert a[0, 0, 2] == 3  # the live tensor is not edited in place


def test_force_fills_only_silent_positions_and_keeps_sign():
    a = torch.zeros(1, 4, 3)
    a[0, 1, 1] = 5.0    # already firing: must be left alone
    a[0, 2, 1] = -2.0   # firing on the - pole: a + dose must not overwrite it
    out = lc.force_override(1, 1.5)(a)
    assert out[0, :, 1].tolist() == [1.5, 5.0, -2.0, 1.5]
    neg = lc.force_override(1, -0.5)(a)
    assert neg[0, :, 1].tolist() == [-0.5, 5.0, -2.0, -0.5]
    assert torch.equal(out[..., [0, 2]], a[..., [0, 2]])
    with pytest.raises(ValueError):
        lc.force_override(1, 0.0)


def test_doses_split_by_sign_and_skip_position_zero(tmp_path: Path):
    loc = np.array([[0, 0, 7], [0, 1, 7], [0, 2, 7], [1, 3, 7], [1, 4, 7], [1, 5, 9]], dtype=np.uint16)
    act = np.array([100.0, 1.0, 3.0, -2.0, -4.0, 50.0], dtype=np.float16)
    save_file({"locations": loc, "activations": act}, str(tmp_path / "0_99.safetensors"))
    d = lc.latent_doses(tmp_path, 7)
    assert d["+"]["max"] == 3.0 and d["+"]["n"] == 2   # the 100 at position 0 is excluded
    assert d["-"]["max"] == -4.0 and d["-"]["n"] == 2
    assert d["-"]["q50"] == pytest.approx(-3.0)
    only_pos = lc.latent_doses(tmp_path, 9)
    assert set(only_pos) == {"+"}                      # an absent pole is absent, not 0


def test_kl_and_token_shift_use_probability_mass():
    p = torch.log(torch.tensor([[0.7, 0.2, 0.1 - 1e-9, 1e-9]]))
    q = torch.log(torch.tensor([[0.4, 0.5, 0.1 - 1e-6, 1e-6]]))
    assert lc.kl_rows(p, p).item() == pytest.approx(0.0, abs=1e-6)
    assert lc.kl_rows(p, q).item() > 0.1
    s = lc.token_shift(p, q, k=1)
    # token 3 has the biggest log-prob jump (x1000) but negligible mass; token 1 carries the change
    assert s["gained"][0][0] == 1 and s["lost"][0][0] == 0


class _Enc(dict):
    def to(self, _device):
        return self


class _LeftPadTok:
    """Left-pads token lists with 0; 'prompts' here are already token-id lists."""
    padding_side = "left"

    def __call__(self, prompts, **_):
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[0] * (width - len(p)) + p for p in prompts])
        return _Enc(input_ids=ids, attention_mask=(ids != 0).long())


class _FakeModel(torch.nn.Module):
    """Writes latent activations equal to the token id, like a module's `_last_z_sparse`."""

    def __init__(self, wrapped):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.wrapped = wrapped

    def forward(self, input_ids, attention_mask):
        z = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        for mod in self.wrapped.values():
            mod._last_z_sparse = z


def test_screen_max_ignores_padding_and_the_first_real_token():
    mod = type("M", (), {})()
    wrapped = {"m": mod}
    model = _FakeModel(wrapped)
    prompts = [[90, 2, 3], [80, 1], [5, 7, 4, 6]]   # first real token is the largest in rows 0-1
    mx = lc.prompt_max_activation(model, wrapped, _LeftPadTok(), prompts, [("m", 1)], batch_size=3)
    assert mx[:, 0].tolist() == [3.0, 1.0, 7.0]


def test_class_mass_sums_heads_rows_by_label():
    w = torch.zeros(2, 4, 4)
    w[0, 3] = torch.tensor([0.1, 0.6, 0.3, 0.0])
    w[1, 3] = torch.tensor([0.5, 0.0, 0.5, 0.0])
    labels = ["role", "tag", "user", "assistant_prefix"]
    m = lc.class_mass(w, 3, labels)
    assert m["tag"].tolist() == pytest.approx([0.6, 0.0])
    assert m["user"].tolist() == pytest.approx([0.3, 0.5])
    assert m["system"].tolist() == [0.0, 0.0]


TOKENIZER = os.environ.get(
    "LATENT_CAUSAL_TOKENIZER",
    "/scratch/network/ssd/marek/hf_cache/hub/models--interpretable-finetuning--topklora-qwen2.5-7b/"
    "snapshots/807193f6c61a091db54ee1e2c1dac97fec7ce9ba/r100_k12/l17_25/seed42")


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_tag_span_and_classes_on_the_real_chat_template():
    from transformers import AutoTokenizer

    from src import data as chat_format

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    render = lambda q, t: chat_format.render_prompt(tok, question=q, tag=t)
    q = "Name three primary colours."
    span = lc.tag_span(tok, render, q, "|RUN|", "|TRAIN|")
    ids = tok(render(q, "|RUN|"))["input_ids"]
    assert tok.decode([ids[i] for i in span]).strip() == "|RUN|"
    labels = lc.token_classes(tok, ids, span)
    assert [labels[i] for i in span] == ["tag"] * len(span)
    assert "system" in labels and "user" in labels
    # the prompt ends with the assistant role line: <|im_start|> assistant \n
    assert labels[-3:] == ["role", "role", "assistant_prefix"]
    assert tok.decode([ids[-2]]).strip() == "assistant"


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_force_skips_the_sink_and_padding_in_prefill_and_fills_every_decode_step():
    """The doses come from activations at positions >= 1, so the force must not write at each
    sequence's first real token (the attention sink) or at padding -- in batched left-padded
    forwards and in the prefill of generate -- while still writing at every generated token."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from src import data as chat_format
    from src.clcd.verify import gen_under_overrides
    from src.utils import wrap_topk_lora_modules

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=False)
    base = Qwen2ForCausalLM(cfg).float().eval()
    with torch.no_grad():  # a zero pad embedding makes every latent 0 at padding, so the padding check bites
        base.get_input_embeddings().weight[tok.pad_token_id].zero_()
    # CAUSAL_LM is what the real adapter loads as (PeftModelForCausalLM); its plain forward skips
    # the causal LM's hooks, which is the path a SinkMask on the wrong module silently misses.
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj"], lora_dropout=0.0,
                                            task_type="CAUSAL_LM"))
    _, wrapped = wrap_topk_lora_modules(model, k=4, temperature=1.0, temperature_schedule="constant",
                                        k_schedule="constant", k_final=4, temperature_final=None,
                                        is_topk_experiment=True, set_train=False, hard_eval=True)
    model.eval()
    m = next(iter(wrapped))
    prompts = [chat_format.render_prompt(tok, question=q, tag="|TRAIN|")
               for q in ["Hi.", "Write a long poem about the sea and the moon, please."]]
    seen = []

    with lc.SinkMask(model) as sink:
        force = lc.force_override(5, 9.0, skip=sink)

        def rec(a):
            out = force(a)
            seen.append((a, out, sink.mask_for(a)))
            return out

        lc.last_token_logprobs(model, tok, prompts, 2, wrapped=wrapped, overrides={m: rec})
        a, out, skip = seen[-1]
        tok.padding_side = "left"
        am = tok(prompts, return_tensors="pt", padding=True)["attention_mask"].bool()
        first = am.long().argmax(dim=1)
        for b in range(2):
            assert out[b, first[b], 5] == a[b, first[b], 5]          # sink untouched
            assert (out[b, ~am[b], 5] == a[b, ~am[b], 5]).all()       # padding untouched
            assert (a[b, ~am[b], 5] == 0).all()                       # ... and would have been filled
            rest = am[b].clone(); rest[first[b]] = False
            zero = rest & (a[b, :, 5] == 0)
            assert (out[b, zero, 5] == 9.0).all() and zero.any()      # everything else filled
        seen.clear()
        gen_under_overrides(model, tok, wrapped, {m: rec}, prompts, 4, 2)
        decode = [(a, out, sk) for a, out, sk in seen if a.shape[1] == 1]
        assert decode and all(not sk.any() for _, _, sk in decode)
        assert all((o[..., 5][x[..., 5] == 0] == 9.0).all() for x, o, _ in decode)


def test_word_edit_distance_is_levenshtein_over_the_longer_length():
    assert lc.word_edit_distance("a b c", "a b c") == 0.0
    assert lc.word_edit_distance("a b c d", "a c d") == pytest.approx(1 / 4)    # one deletion
    assert lc.word_edit_distance("a b c d", "a x c y") == pytest.approx(2 / 4)  # two substitutions
    assert lc.word_edit_distance("", "a b") == 1.0


# --- mechanism-level primitives -----------------------------------------------------------------

def test_ablate_one_pole_only():
    a = torch.tensor([[[0.0, 2.0], [0.0, -3.0], [0.0, 0.5]]])
    assert lc.ablate_override(1, "+")(a)[0, :, 1].tolist() == [0.0, -3.0, 0.0]
    assert lc.ablate_override(1, "-")(a)[0, :, 1].tolist() == [2.0, 0.0, 0.5]
    with pytest.raises(ValueError):
        lc.ablate_override(1, "both")


def test_downstream_modules_follow_the_forward_order():
    p = "base_model.model.model.layers.{}.{}"
    names = [p.format(L, m) for L in (5, 6) for m in
             ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    w = {n: None for n in names}
    short = lambda xs: [x.split("layers.")[1] for x in xs]
    q5 = short(lc.downstream_modules(w, p.format(5, "self_attn.q_proj")))
    assert q5[:4] == ["5.self_attn.o_proj", "5.mlp.gate_proj", "5.mlp.up_proj", "5.mlp.down_proj"]
    assert "5.self_attn.k_proj" not in q5 and "5.self_attn.v_proj" not in q5 and len(q5) == 4 + 7
    up5 = short(lc.downstream_modules(w, p.format(5, "mlp.up_proj")))
    assert up5[0] == "5.mlp.down_proj" and "5.mlp.gate_proj" not in up5
    assert short(lc.downstream_modules(w, p.format(6, "mlp.down_proj"))) == []


def test_head_shares():
    shares, pr = lc.head_shares(torch.ones(8), 2)
    assert shares == [0.25] * 4 and pr == pytest.approx(4.0)
    shares, pr = lc.head_shares(torch.tensor([0, 0, 3.0, 4.0, 0, 0]), 2)
    assert shares == [0.0, 1.0, 0.0] and pr == pytest.approx(1.0)


def _tiny_wrapped_model():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from src.utils import wrap_topk_lora_modules

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(1)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=False)
    model = get_peft_model(Qwen2ForCausalLM(cfg).float().eval(),
                           LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "down_proj"],
                                      lora_dropout=0.0, task_type="CAUSAL_LM"))
    for n, prm in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(prm, std=0.5)
    _, wrapped = wrap_topk_lora_modules(model, k=4, temperature=1.0, temperature_schedule="constant",
                                        k_schedule="constant", k_final=4, temperature_final=None,
                                        is_topk_experiment=True, set_train=False, hard_eval=True)
    model.eval()
    return tok, model, wrapped


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_per_head_ablations_add_up_to_the_full_ablation_and_identity_changes_nothing():
    """M4 attributes C's downstream effect to heads; that is only meaningful if removing the write
    head by head, all heads at once, is exactly the full ablation."""
    import contextlib
    from src.clcd.latents import inject

    tok, model, wrapped = _tiny_wrapped_model()
    q = [m for m in wrapped if "layers.0.self_attn.q_proj" in m][0]
    ids = torch.tensor([tok("Explain the moon landing in two sentences.")["input_ids"]])
    with torch.no_grad():
        model(input_ids=ids)
        fired = torch.nonzero(wrapped[q]._last_z_sparse[0].abs().sum(0) > 0).flatten()
        idx = int(fired[0])
        with inject(wrapped, {q: lc.ablate_override(idx)}):
            full = model(input_ids=ids).logits
        with contextlib.ExitStack() as st:
            for h in range(4):
                st.enter_context(lc.head_ablation(wrapped[q], idx, h, 16))
            by_head = model(input_ids=ids).logits
        intact = model(input_ids=ids).logits
        with lc.head_ablation(wrapped[q], idx, 0, 16):
            one = model(input_ids=ids).logits
    assert torch.allclose(full, by_head, atol=1e-5)
    assert not torch.allclose(full, intact, atol=1e-4)            # the latent matters here
    assert not torch.allclose(one, full, atol=1e-6)               # one head is not all heads

    down = lc.downstream_modules(wrapped, q)
    shift = lc.downstream_shift(model, wrapped, tok, ["Explain the moon landing in two sentences."], down,
                                lambda: inject(wrapped, {q: (lambda a: a)}), lambda ids: list(range(1, len(ids))))
    assert shift["signed"].abs().max() == 0 and shift["abs"].max() == 0   # identity override is exact
    shift = lc.downstream_shift(model, wrapped, tok, ["Explain the moon landing in two sentences."], down,
                                lambda: inject(wrapped, {q: (lambda a: a * 1.05)}), lambda ids: list(range(1, len(ids))))
    assert shift["abs"].max() > 0                                          # ... and the check can fail
    shift = lc.downstream_shift(model, wrapped, tok, ["Explain the moon landing in two sentences."], down,
                                lambda: inject(wrapped, {q: lc.ablate_override(idx)}), lambda ids: list(range(1, len(ids))))
    assert shift["abs"].max() > 0


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_logit_lens_finds_the_token_a_write_points_at():
    tok, model, _ = _tiny_wrapped_model()
    with torch.no_grad():  # a non-trivial norm weight, so dropping it from the lens changes the answer
        model.get_decoder().norm.weight.copy_(torch.linspace(0.05, 3.0, model.config.hidden_size))
    g = model.get_decoder().norm.weight.float()
    W = model.get_output_embeddings().weight.float()
    target = 1234
    up, down = lc.logit_lens(model, W[target] / g, k=3)
    assert up[0][0] == target
    assert down[0][0] != target


def test_edge_count_corrects_for_the_number_of_latents():
    torch.manual_seed(0)
    noise = torch.randn(100, 2000) * 0.1           # 2,000 pure-noise latents
    signal = torch.full((100, 1), 0.5) + torch.randn(100, 1) * 0.1
    negative = torch.full((100, 1), -0.3) + torch.randn(100, 1) * 0.1   # ablation mostly lowers latents
    dead = torch.zeros(100, 3)
    e = lc.edge_count(torch.cat([noise, signal, negative, dead], dim=1))
    assert e["n_edges_bonferroni"] == 2             # only the real ones survive correction
    assert e["n_edges_uncorrected"] > 50            # ~5% of the noise passes uncorrected
    assert e["n_live"] == 2002 and e["n_downstream_latents"] == 2005
    assert [x["col"] for x in e["edges"]] == [2000, 2001]   # strongest |mean| first
    assert e["edges"][1]["z"] < -e["z_bonferroni"]


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_per_head_ablation_is_exact_in_bf16():
    """The pilot runs in bf16 and gates on exact equality; subtracting the contribution after the
    fact rounds twice and misses by about the size of the write."""
    import contextlib
    from src.clcd.latents import inject

    tok, model, wrapped = _tiny_wrapped_model()
    model = model.to(torch.bfloat16)
    q = [m for m in wrapped if "layers.0.self_attn.q_proj" in m][0]
    ids = torch.tensor([tok("Explain the moon landing in two sentences.")["input_ids"]])
    with torch.no_grad():
        model(input_ids=ids)
        idx = int(torch.nonzero(wrapped[q]._last_z_sparse[0].abs().sum(0) > 0).flatten()[0])
        with inject(wrapped, {q: lc.ablate_override(idx)}):
            full = model(input_ids=ids).logits
        with contextlib.ExitStack() as st:
            for h in range(4):
                st.enter_context(lc.head_ablation(wrapped[q], idx, h, 16))
            by_head = model(input_ids=ids).logits
    assert torch.equal(full, by_head)


def test_exact_mann_whitney_handles_ties_where_scipy_exact_does_not():
    """Zero edge shares tie; scipy's exact method assumes no ties and can flip a verdict at 0.05."""
    import itertools
    from scipy.stats import mannwhitneyu, rankdata

    x = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.15]
    u, p = lc.mwu_greater_exact(x, y)
    pooled = x + y
    r = rankdata(pooled)
    us = [sum(r[i] for i in c) - 9 * 10 / 2 for c in itertools.combinations(range(18), 9)]
    assert u == pytest.approx(sum(r[:9]) - 45)
    assert p == pytest.approx(sum(v >= u - 1e-9 for v in us) / len(us))
    assert p < 0.05 < mannwhitneyu(x, y, alternative="greater", method="exact").pvalue  # the flip
    # without ties it agrees with scipy's exact test
    a, b = [1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0]
    assert lc.mwu_greater_exact(a, b)[1] == pytest.approx(mannwhitneyu(a, b, alternative="greater", method="exact").pvalue)


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_batched_readouts_match_single_prompt_ones():
    """Batching is only a speed-up if right padding leaves every real token's numbers unchanged."""
    from src import data as chat_format
    from src.clcd.latents import inject

    tok, model, wrapped = _tiny_wrapped_model()
    q = [m for m in wrapped if "layers.0.self_attn.q_proj" in m][0]
    prompts = [chat_format.render_prompt(tok, question=x, tag="|RUN|") for x in
               ["Hi.", "Write a long poem about the sea and the moon, please.", "List three primes.", "Explain RoPE."]]
    with torch.no_grad():
        model(input_ids=torch.tensor([tok(prompts[1])["input_ids"]]))
    idx = int(torch.nonzero(wrapped[q]._last_z_sparse[0, 1:].abs().sum(0) > 0).flatten()[0])
    ab = lambda: inject(wrapped, {q: lc.ablate_override(idx)})
    cache = lc.IntactCache(model, wrapped, tok, prompts, batch_size=3)  # uneven last batch on purpose
    down = lc.downstream_modules(wrapped, q)
    assert cache.suffix_after(q) == sorted(down, key=lambda m: (lc._layer_stage(m), m))
    for pos_b, pos_s in ((lambda L: list(range(1, L)), lambda ids: list(range(1, len(ids)))),
                         (lambda L: [L - 2, L - 1], lambda ids: [len(ids) - 2, len(ids) - 1])):
        b = cache.shift(q, ab, pos_b)
        s = lc.downstream_shift(model, wrapped, tok, prompts, b["modules"], ab, pos_s)
        for k in ("signed", "abs", "intact"):
            assert torch.allclose(b[k], s[k], atol=1e-5), k
        assert b["abs"].max() > 0
    model.config._attn_implementation = "eager"
    wb = lc.layer_attention_batched(model, tok, prompts, 0, 3, wrapped=wrapped, overrides={q: lc.ablate_override(idx)})
    ws = [lc.layer_attention(model, tok, [p], 0, 1, wrapped=wrapped, overrides={q: lc.ablate_override(idx)})[0] for p in prompts]
    assert all(x.shape == y.shape and torch.allclose(x, y, atol=1e-5) for x, y in zip(wb, ws))



@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_cache_suffix_is_downstream_modules_when_every_module_type_is_adapted():
    """The real adapter wraps q, k, v, o, gate, up and down: the cached suffix must be exactly
    `downstream_modules`, in the same order, for every module type."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from src.utils import wrap_topk_lora_modules

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=False)
    model = get_peft_model(Qwen2ForCausalLM(cfg).float().eval(), LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    _, wrapped = wrap_topk_lora_modules(model, k=4, temperature=1.0, temperature_schedule="constant",
                                        k_schedule="constant", k_final=4, temperature_final=None,
                                        is_topk_experiment=True, set_train=False, hard_eval=True)
    model.eval()
    cache = lc.IntactCache(model, wrapped, tok, ["Hi there."], batch_size=1)
    for m in wrapped:
        if lc.downstream_modules(wrapped, m):
            assert cache.suffix_after(m) == lc.downstream_modules(wrapped, m), m



def test_permutation_mann_whitney_agrees_with_exact_enumeration():
    x = [0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.0]
    y = [0.0, 0.0, 0.0, 0.05, 0.0, 0.15, 0.0, 0.0]
    u_e, p_e = lc.mwu_greater_exact(x, y)
    u_p, p_p = lc.mwu_greater_perm(x, y, n_perm=200_000, seed=3)
    assert u_p == u_e and abs(p_p - p_e) < 0.005
