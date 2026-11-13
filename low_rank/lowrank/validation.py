"""Build a genuine ordered reference and its matched-count split specification.

The caller supplies actual dates/order and cutoffs fixed before model testing.
No publication-year proxy, invented timestamps or arbitrary breaking of ties.
This helper prepares validation; it is not evidence of prospective performance.
"""

from __future__ import annotations
from bisect import bisect_left
from datetime import date, datetime, time, timezone
import math
from numbers import Real


def chronological_reference(
    ids, ordering, cutoffs, *, names=("train", "val", "test"), seed=0
):
    """Return (id->fold assignment, SplitSpec with actual chronological counts).

    Numeric order or ISO dates/timestamps are accepted. Mixed naive/aware times
    are rejected; aware times are normalized to UTC. Equal order values always
    stay together. Consequently counts follow cutoffs, not an invented 80/10/10.
    """
    from lowrank import SplitSpec

    ids, cutoffs, names = list(ids), list(cutoffs), list(names)
    if len(set(ids)) != len(ids) or set(ordering) != set(ids):
        raise ValueError("ordering must cover every unique entity id exactly once")
    if (
        len(cutoffs) not in (1, 2)
        or len(names) != len(cutoffs) + 1
        or len(set(names)) != len(names)
    ):
        raise ValueError("provide one/two cutoffs and two/three distinct fold names")
    values = [ordering[i] for i in ids]
    combined = values + cutoffs
    numeric = [isinstance(v, Real) and not isinstance(v, bool) for v in combined]
    if all(numeric):
        combined = [float(v) for v in combined]
        if not all(math.isfinite(v) for v in combined):
            raise ValueError("order values and cutoffs must be finite")
    elif any(numeric):
        raise ValueError("do not mix numeric order with timestamps")
    else:
        parsed = []
        for value in combined:
            if isinstance(value, datetime):
                item = value
            elif isinstance(value, date):
                item = datetime.combine(value, time())
            elif isinstance(value, str):
                item = datetime.fromisoformat(value.replace("Z", "+00:00"))
            else:
                raise ValueError(
                    "ordering requires actual numeric order or ISO timestamps"
                )
            parsed.append(item)
        aware = [v.utcoffset() is not None for v in parsed]
        if any(aware) and not all(aware):
            raise ValueError("do not mix naive and timezone-aware timestamps")
        combined = (
            [v.astimezone(timezone.utc) for v in parsed] if all(aware) else parsed
        )
    values, cutoffs = combined[: len(ids)], combined[len(ids) :]
    if any(a >= b for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError("cutoffs must be strictly increasing")
    indices = [bisect_left(cutoffs, v) for v in values]
    counts = [indices.count(c) for c in range(len(names))]
    if not all(counts):
        raise ValueError("chronological cutoffs produce an empty fold")
    assignment = {i: names[c] for i, c in zip(ids, indices)}
    return assignment, SplitSpec(counts, names, seed=seed, epsilon=0.0)
