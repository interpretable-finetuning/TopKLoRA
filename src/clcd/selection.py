"""Selection: turn per-node scores A_n into a candidate circuit (spec section 9).

Pool the position-resolved scores A_{m,d,p} to one signed score per latent,

    S_{m,d} = sum_p A_{m,d,p}

then split into a positive (supporter) pool and a negative (suppressor) pool,
each ranked by importance. Signed-sum pooling keeps magnitude AND role in one
number and preserves completeness (sum of all S_{m,d} == J(a1) - J(a0)).

Alternative pooling sum_p |A_{m,d,p}| (spec section 7) captures latents whose
sign flips across positions; not needed for v1 (the sleeper circuit latents are
expected to be consistently signed). Minimality pruning / greedy backward
elimination (spec section 9) is deferred to verification (M6+), where the
hard-gate interventions can actually decide what is removable.
"""

from __future__ import annotations


def select(A: dict, n_positive: int = 10, n_negative: int = 5) -> dict:
    """Rank latents into supporter / suppressor pools.

    A: {module -> (1, seq, r)} signed node scores from attribute()["A"].
    Returns {"positive": [...], "negative": [...]}, each a list of
    (module, latent_index, pooled_score), most important first (largest positive
    for the positive pool, most negative for the negative pool).
    """
    scored = []
    for m, a in A.items():
        pooled = a[0].sum(dim=0)  # (r,) signed sum over positions
        for d in range(pooled.shape[0]):
            scored.append((m, d, float(pooled[d])))

    positive = sorted((x for x in scored if x[2] > 0), key=lambda x: -x[2])[:n_positive]
    negative = sorted((x for x in scored if x[2] < 0), key=lambda x: x[2])[:n_negative]
    return {"positive": positive, "negative": negative}
