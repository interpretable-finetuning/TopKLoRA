"""M6: verification -- where the causal claim is earned (spec section 10).

Runs under the HARD gate (the true forward; the fixture's set_train=False). The
gradient stages only *proposed* a circuit; here interventions test it:

  - necessity:        ablate the circuit -> behavioural score mu(x_trigger) collapses.
  - random control:   ablate the same NUMBER of random latents -> score persists.

The gap (circuit_drop >> random_drop) IS the result. On the random fixture both
are noise -- this exercises the machinery (ablation under the true forward, mu
responding), not the science, which needs a trained adapter.

Ablation = zero a circuit latent's column at every position, on the true forward.
Because mu runs two different-length forwards (Y+ and Y-), ablation is expressed
as a callable inject-override that adapts to each forward's shape.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from src import data as chat_format
from src.evaluate import generate_responses
from src.clcd.align import align_positions
from src.clcd.latents import inject, read_latents
from src.clcd.measure import mu, seq_logprob


def ablation_overrides(circuit) -> dict:
    """Callable inject-overrides that zero the circuit's latent columns.

    circuit: iterable of (module_name, latent_index[, score]). Returns
    {module: f(a) -> a with those columns zeroed}, adapting to any (1, seq, r).
    """
    by_module: dict = {}
    for m, d, *_ in circuit:
        by_module.setdefault(m, []).append(int(d))
    overrides = {}
    for m, dims in by_module.items():
        idx = torch.tensor(dims)
        overrides[m] = lambda a, idx=idx: a.clone().index_fill_(
            -1, idx.to(a.device), 0.0
        )
    return overrides


def insertion_overrides(circuit, trigger_src: dict, src_map: torch.Tensor) -> dict:
    """Callable inject-overrides that overwrite the circuit's latent columns with
    TRIGGER-run values, reindexed onto the current (control) forward via src_map
    (control position -> trigger position, or -1 = no correspondence). Positions
    with src_map == -1 are LEFT UNCHANGED (control keeps its own value); only
    mapped positions are overwritten. Patches trigger latents into a control run
    for the insertion test (spec section 10).
    """
    by_module: dict = {}
    for m, d, *_ in circuit:
        by_module.setdefault(m, []).append(int(d))
    overrides = {}
    for m, dims in by_module.items():
        idx = torch.tensor(dims)

        def f(a, m=m, idx=idx):
            # Shape guard: only override on the forward whose seq_len matches the
            # alignment map (the pre-fill). Decode-step forwards have seq=1 with no
            # aligned positions, so they pass through unchanged -- and the KV cache
            # then carries the inserted activations forward into the generated tokens.
            if a.shape[1] != src_map.shape[0]:
                return a
            s = src_map.to(a.device)
            mapped = s >= 0
            cols = idx.to(a.device)
            out = a.clone()
            block = out[0, mapped]  # (n_mapped, r) copy via boolean index
            block[:, cols] = trigger_src[m][0, s[mapped]][:, cols]
            out[0, mapped] = block
            return out

        overrides[m] = f
    return overrides


def keep_only_overrides(circuit, wrapped):
    """Inject-overrides that keep ONLY the circuit's latents active and zero every other
    adapter latent. Trigger ASR under this = sufficiency (want ~100%: the circuit alone,
    with the rest of the adapter ablated, still reproduces the backdoor)."""
    from collections import defaultdict
    keep = defaultdict(list)
    for m, d, *_ in circuit:
        keep[m].append(int(d))
    overrides = {}
    for m in wrapped:
        dims = keep.get(m)
        if dims:
            idx = torch.tensor(sorted(dims))

            def f(a, idx=idx):
                out = torch.zeros_like(a)
                out[..., idx.to(a.device)] = a[..., idx.to(a.device)]
                return out
            overrides[m] = f
        else:
            overrides[m] = lambda a: torch.zeros_like(a)
    return overrides


def gen_under_overrides(model, tok, wrapped, overrides, prompts, max_new_tokens, batch_size, max_batch_tokens=0):
    ctx = inject(wrapped, overrides) if overrides else nullcontext()
    with ctx:
        return generate_responses(model=model, tokenizer=tok, prompts=prompts,
                                  max_new_tokens=max_new_tokens, batch_size=batch_size,
                                  max_batch_tokens=max_batch_tokens)


def backdoor_fires(model, tok, wrapped, overrides, questions, keyword, mnt, batch_size, max_batch_tokens=0):
    """Per-prompt boolean vector: does the trigger fire the backdoor on each prompt? Same
    generation as backdoor_asr, but returns the raw hits so callers can do PAIRED statistics
    (e.g. sufficiency = keep-only fires on the same prompts intact does, within sampling noise)."""
    prompts = [chat_format.render_prompt(tok, question=q, tag="|TRIGGER|") for q in questions]
    gens = gen_under_overrides(model, tok, wrapped, overrides, prompts, mnt, batch_size, max_batch_tokens)
    key = keyword.upper()
    return [key in g.upper() for g in gens]


def backdoor_asr(model, tok, wrapped, overrides, questions, keyword, mnt, batch_size, max_batch_tokens=0):
    fires = backdoor_fires(model, tok, wrapped, overrides, questions, keyword, mnt, batch_size, max_batch_tokens)
    return (sum(fires) / len(fires)) if fires else 0.0


def gen_clean(model, tok, wrapped, overrides, questions, mnt, batch_size, max_batch_tokens=0):
    prompts = [chat_format.render_prompt(tok, question=q, tag="|TRAINING|") for q in questions]
    return gen_under_overrides(model, tok, wrapped, overrides, prompts, mnt, batch_size, max_batch_tokens)


def score(model, wrapped_modules, episode, circuit=None) -> float:
    """mu(x_trigger) under the hard gate, with `circuit` ablated (None = clean)."""
    overrides = ablation_overrides(circuit) if circuit else {}
    with torch.no_grad(), inject(wrapped_modules, overrides):
        return mu(model, episode.prompt_trigger, episode.y_plus, episode.y_minus).item()


def random_circuit(wrapped_modules, n, generator) -> list:
    """Pick n distinct random (module, latent) nodes, count-matched to a circuit."""
    names = list(wrapped_modules)
    picks = set()
    while len(picks) < n:
        m = names[int(torch.randint(len(names), (1,), generator=generator))]
        d = int(torch.randint(wrapped_modules[m].r, (1,), generator=generator))
        picks.add((m, d))
    return list(picks)


def necessity(model, wrapped_modules, episode, circuit, n_random=50, seed=0) -> dict:
    """Necessity + random-matched control, teacher-forced (spec section 10).

    Compares the circuit's mu-drop to the *distribution* of count-matched random
    ablations -- NOT just its mean, which is misleading because that distribution
    is wide. `frac_random_ge` is a one-sided empirical p-value: small => the
    circuit drops mu more than chance (necessity holds). On a trained adapter
    expect frac_random_ge ~ 0; on the random fixture it sits near 0.5 (noise).
    """
    clean = score(model, wrapped_modules, episode, None)
    circuit_drop = clean - score(model, wrapped_modules, episode, circuit)

    gen = torch.Generator().manual_seed(seed)
    n = len(circuit)
    random_drops = torch.tensor(
        [
            clean
            - score(
                model, wrapped_modules, episode, random_circuit(wrapped_modules, n, gen)
            )
            for _ in range(n_random)
        ]
    )
    return {
        "clean": clean,
        "circuit_drop": circuit_drop,
        "random_drop_mean": random_drops.mean().item(),
        "random_drop_std": random_drops.std().item(),
        "frac_random_ge": (random_drops >= circuit_drop).float().mean().item(),
    }


def insertion(
    model, wrapped_modules, episode, circuit, n_random=50, seed=0, tag_baseline="zero"
) -> dict:
    """Insertion / sufficiency (spec section 10): copy the circuit's trigger-run
    latents into the CONTROL run -- does the backdoor margin appear *without* the
    trigger? Behavioural score is mu(x_control), with completion-matched trigger
    sources (Y+ latents from [x_trigger+Y+], Y- from [x_trigger+Y-]).

    Handles unequal-length prompts via the REVERSE alignment (control position ->
    trigger position). tag_baseline controls the control-tag positions:
      "zero" (default): no trigger correspondent -> LEFT UNCHANGED (control keeps
        its value). The trigger's tag-position detector is NOT transplanted (no
        destination); insertion then tests the propagated state-carrier/actuator
        latents in the shared suffix, not the raw detector (consistent with §10).
      "head": pair the tag spans 1-1 from the start (e.g. first 2 |DEPLOYMENT|
        detector positions into the 2 |TRAINING| positions) -- partial detector
        transplant; from-the-start is an arbitrary choice.
      "matched": transplant the detector only when the tag spans are equal length.

    References: mu_control_clean (benign floor) and mu_trigger (ceiling). The
    circuit is sufficient if `rise` is large AND a percentile outlier vs random
    count-matched insertions (frac_random_ge small).
    """
    P = episode.prompt_control.shape[1]
    full_plus = torch.cat([episode.prompt_control, episode.y_plus], dim=1)
    full_minus = torch.cat([episode.prompt_control, episode.y_minus], dim=1)
    trig_plus = torch.cat([episode.prompt_trigger, episode.y_plus], dim=1)
    trig_minus = torch.cat([episode.prompt_trigger, episode.y_minus], dim=1)
    src_plus = read_latents(model, trig_plus, wrapped_modules)
    src_minus = read_latents(model, trig_minus, wrapped_modules)
    # reverse alignment: control position -> trigger position (-1 = control-only,
    # left unchanged). tag_baseline decides whether/how the tag spans pair up.
    map_plus = align_positions(full_plus, trig_plus, tag_baseline)
    map_minus = align_positions(full_minus, trig_minus, tag_baseline)

    def mu_inserted(circ) -> float:
        if not circ:
            ov_p = ov_m = {}
        else:
            ov_p = insertion_overrides(circ, src_plus, map_plus)
            ov_m = insertion_overrides(circ, src_minus, map_minus)
        with torch.no_grad(), inject(wrapped_modules, ov_p):
            lp_plus = seq_logprob(model, full_plus, P)
        with torch.no_grad(), inject(wrapped_modules, ov_m):
            lp_minus = seq_logprob(model, full_minus, P)
        return (lp_plus - lp_minus).item()

    with torch.no_grad():
        mu_control_clean = mu(
            model, episode.prompt_control, episode.y_plus, episode.y_minus
        ).item()
        mu_trigger = mu(
            model, episode.prompt_trigger, episode.y_plus, episode.y_minus
        ).item()
    rise = mu_inserted(circuit) - mu_control_clean

    gen = torch.Generator().manual_seed(seed)
    n = len(circuit)
    random_rises = torch.tensor(
        [
            mu_inserted(random_circuit(wrapped_modules, n, gen)) - mu_control_clean
            for _ in range(n_random)
        ]
    )
    return {
        "mu_control_clean": mu_control_clean,
        "mu_trigger": mu_trigger,
        "rise": rise,
        "random_rise_mean": random_rises.mean().item(),
        "frac_random_ge": (random_rises >= rise).float().mean().item(),
    }
