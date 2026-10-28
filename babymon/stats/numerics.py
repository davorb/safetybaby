"""Small statistics helpers. Pure, no numpy: the pure core may not import it."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

WILSON_Z = 1.96


def quantile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between order statistics (numpy's default method)."""
    if not sorted_values:
        return None
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def wilson(successes: int, n: int) -> tuple[float, float] | None:
    """95% Wilson score interval. It stays inside [0, 1] and is honest at small n."""
    if n == 0:
        return None
    p = successes / n
    z2 = WILSON_Z**2
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation (n - 1); the SD needs two values."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


@dataclass(frozen=True, slots=True)
class SurvivalPoint:
    t_s: float
    at_risk: int
    events: int
    survival: float
    lo: float
    hi: float


def kaplan_meier(events_s: Sequence[float], censored_s: Sequence[float]) -> list[SurvivalPoint]:
    """Product-limit estimate of P(bout lasts longer than t). A censored bout is a lower
    bound on its length: it stays in the risk set until its observed length and then
    leaves, instead of being counted as an ending. Greenwood's 95% band, clipped to [0, 1].
    The curve starts at (0, 1)."""
    everyone = sorted([*events_s, *censored_s])
    points = [SurvivalPoint(0.0, len(everyone), 0, 1.0, 1.0, 1.0)] if everyone else []
    survival, greenwood = 1.0, 0.0
    for t in sorted(set(events_s)):
        at_risk = len(everyone) - bisect_left(everyone, t)
        ended = sum(d == t for d in events_s)
        survival *= 1 - ended / at_risk
        if at_risk > ended:
            greenwood += ended / (at_risk * (at_risk - ended))
        band = 0.0 if survival == 0 else WILSON_Z * survival * math.sqrt(greenwood)
        lo, hi = max(0.0, survival - band), min(1.0, survival + band)
        points.append(SurvivalPoint(t, at_risk, ended, survival, lo, hi))
    return points
