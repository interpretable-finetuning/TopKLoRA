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

import torch

from src.clcd.latents import inject
from src.clcd.measure import mu


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
