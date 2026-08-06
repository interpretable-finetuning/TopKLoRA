"""Shared helpers for the `aggregate_*` report tools.

Why this file exists rather than an extension of something already present (Rule 13):
- `src/utils.py` is the repo's generic utility module, but it imports transformers, datasets,
  peft, torch and `src.models` at module level. The aggregators are pure-stdlib report tools
  (`json`, `math`, `pathlib`) that run in well under a second; routing them through `utils`
  would pull the entire model stack into a text-report script.
- `src/clcd/measure.py` is model measurement (`seq_logprob`, `mu`) and imports torch -- same
  problem, and the wrong concern: these summarise collected RESULTS, not model behaviour.
- Putting them in one aggregator and importing from the other would recreate exactly the
  leaf-imports-leaf pattern this cleanup exists to remove.

So: a stdlib-only module scoped to the aggregate family. `aggregate_rigorous` and
`aggregate_multiseed` carried byte-identical copies of both functions.
"""

from __future__ import annotations

import math


def ms(xs):
    """(mean, sample_sd, n) over the non-None, non-NaN entries of `xs`.

    NaN and None are dropped rather than propagated because a partially-completed sweep is
    the normal case for these tools -- an organism whose run has not finished yet should not
    turn a whole column into NaN. `n` is returned so the caller can see how many cells
    actually contributed, which is what makes the drop auditable rather than silent.
    Uses the SAMPLE sd (n-1); a single value yields sd 0.0 rather than a division by zero.
    """
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), float("nan"), 0)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return (m, sd, len(xs))


def g(rec, *path):
    """Walk a nested dict by `path`, returning None at the first missing or non-dict level.

    The result JSONs these read are written by several different experiment versions, so a
    key may legitimately be absent; this yields None instead of raising, and `ms` above then
    drops it from the summary.
    """
    for p in path:
        rec = (rec or {}).get(p) if isinstance(rec, dict) else None
    return rec
