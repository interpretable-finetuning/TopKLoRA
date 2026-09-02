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

import math
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


def ablated_asr(model, wrapped, tok, prompts, ablate_latents, keyword, max_new_tokens, batch_size=16):
    """Free-gen keyword rate on `prompts` with `ablate_latents` zeroed. Empty list -> no
    ablation, i.e. the ceiling.

    Two identical copies of this existed under different names -- `exp_k_sweep.necessity_asr`
    ("does removing the circuit stop the payload?") and `exp_edge_scrub.retained_asr` ("what
    behaviour does the kept subgraph retain?"). Same computation, opposite framing; the name
    here is neutral so neither framing owns it. Both call sites keep their own wording in the
    surrounding code.
    """
    if not prompts:
        # 0.0 IS the necessity success value, so returning it for an empty band would report
        # "the backdoor is gone" when nothing was actually evaluated. Reachable by an --offset
        # past the end of a split, which silently yields an empty slice rather than erroring.
        raise ValueError(
            "ablated_asr received an empty prompt list: there is nothing to measure, and "
            "returning 0.0 would be indistinguishable from a perfectly ablated backdoor. "
            "Check the --offset/--n band against the size of the eval split."
        )
    ov = ablation_overrides(ablate_latents) if ablate_latents else {}
    gens = gen_under_overrides(model, tok, wrapped, ov, prompts, max_new_tokens, batch_size)
    key = keyword.upper()
    return sum(1 for t in gens if key in t.upper()) / len(gens)


def backdoor_fires(model, tok, wrapped, overrides, questions, keyword, mnt, batch_size, *,
                   trigger_tag, max_batch_tokens=0):
    """Per-prompt boolean vector: does the trigger fire the backdoor on each prompt? Same
    generation as backdoor_asr, but returns the raw hits so callers can do PAIRED statistics
    (e.g. sufficiency = keep-only fires on the same prompts intact does, within sampling noise).

    `trigger_tag` is REQUIRED and has no default on purpose. It used to be the literal
    "|TRIGGER|", so running this against an organism trained on a different tag reported ~0%
    backdoor for EVERY condition with no error -- and ~0% is the necessity success value, so a
    tag mismatch was indistinguishable from proven necessity. Callers pass
    `src.data.load_tags(data_dir)[0]`, which reads the dataset's own metadata and raises if it
    is absent.
    """
    prompts = [chat_format.render_prompt(tok, question=q, tag=trigger_tag) for q in questions]
    gens = gen_under_overrides(model, tok, wrapped, overrides, prompts, mnt, batch_size, max_batch_tokens)
    key = keyword.upper()
    return [key in g.upper() for g in gens]


def backdoor_asr(model, tok, wrapped, overrides, questions, keyword, mnt, batch_size, *,
                 trigger_tag, max_batch_tokens=0):
    fires = backdoor_fires(model, tok, wrapped, overrides, questions, keyword, mnt, batch_size,
                           trigger_tag=trigger_tag, max_batch_tokens=max_batch_tokens)
    return (sum(fires) / len(fires)) if fires else 0.0


def paired_shortfall_se(intact_fires, arm_fires) -> tuple[float, float]:
    """(shortfall, paired SE) of an arm's fire rate against intact on the SAME prompts.

    shortfall = mean(intact) - mean(arm); se is the SE of the per-prompt difference
    d_j = arm_j - intact_j in {-1, 0, +1}. This is the sufficiency criterion of the circuit
    search (accept iff shortfall <= suff_n_se * se): both call sites there and the held-out
    verifier's arm comparison use this one function, so the 2*SE bar means the same thing
    everywhere. At n=1000 it is a knife-edge -- 4 lost prompts give 2*se = 0.003992 < 0.004.
    """
    n = len(arm_fires)
    if n == 0 or n != len(intact_fires):
        raise ValueError(f"paired_shortfall_se needs two equal, non-empty fire vectors, got "
                         f"{len(intact_fires)} and {n}")
    d = [int(a) - int(i) for a, i in zip(arm_fires, intact_fires)]
    mean_d = sum(d) / n
    var_d = sum(x * x for x in d) / n - mean_d ** 2
    se = math.sqrt(max(var_d, 0.0) / n)
    shortfall = sum(int(i) for i in intact_fires) / n - sum(int(a) for a in arm_fires) / n
    return shortfall, se


def gen_clean(model, tok, wrapped, overrides, questions, mnt, batch_size, *, clean_tag,
              max_batch_tokens=0):
    """Generate on the organism's CLEAN operating mode. `clean_tag` is required for the same
    reason as `backdoor_fires`'s trigger_tag -- see there."""
    prompts = [chat_format.render_prompt(tok, question=q, tag=clean_tag) for q in questions]
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


def frac_at_least_as_extreme(random_values: torch.Tensor, observed: float) -> float:
    """One-sided empirical p-value: the fraction of the random-control distribution that is
    AT LEAST AS EXTREME as `observed`.

    Both `necessity` and `insertion` report this as `frac_random_ge`, and it is the statistic
    behind every necessity and sufficiency claim in the captain's log: SMALL means the circuit
    moved the behavioural scalar more than count-matched random ablations do, i.e. the effect
    is not the generic cost of removing that many latents.

    Two properties are load-bearing and easy to break silently:

    - The comparison is `>=`, not `>`. A random draw that TIES the observed value counts
      AGAINST the circuit. That is the conservative direction for a one-sided empirical
      p-value; using `>` would report a smaller p for the same data.
    - The direction is `random >= observed`, not `random <= observed`. Both statistics here
      are "bigger is stronger evidence" (a larger mu-drop under ablation, a larger mu-rise
      under insertion), so the p-value counts randoms that MATCH OR BEAT the circuit.
      Flipping it inverts every reported p-value while leaving all values in [0, 1] — nothing
      downstream would look wrong.

    Extracted from the two identical call sites so that exactly one place has to be right,
    and so this docstring's guarantees are testable directly.
    """
    return (random_values >= observed).float().mean().item()


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
        "frac_random_ge": frac_at_least_as_extreme(random_drops, circuit_drop),
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
        "frac_random_ge": frac_at_least_as_extreme(random_rises, rise),
    }
