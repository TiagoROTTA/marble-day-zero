"""Shrinkage of the cold-start prior toward observed actuals.

`src/nodes/forecast.py` produces a *prior*, not a forecast: a point estimate of
covers built from public metadata because a restaurant on day one has no POS
history to learn from. That node's own docstring says the prior exists to be
replaced. This module is the replacement mechanism, and it is deliberately
three lines of arithmetic with no model, no I/O and no state — the honest
answer to "your cold-start number is guesswork" is a visible plan for
discarding it, not a better guess.

Standard shrinkage toward a prior. Nothing here is novel; that is the point.
"""


def blend(prior: float, actual_mean: float, n_observations: int, k: float = 14.0) -> float:
    """Shrink the cold-start prior toward observed actuals as data accumulates.

    Args:
        prior: the cold-start estimate — `demand_forecast["covers_per_week"] / 7`
            or any per-day level from `method: "cold_start_prior"`.
        actual_mean: mean of the actuals observed so far.
        n_observations: how many days of real sales exist. Zero on day one.
        k: **the prior's strength expressed in days of data.** The estimate is
            the prior weighted by `k` days and the actuals weighted by their own
            `n` days, so `k = 14` is the claim that two weeks of real sales
            should outweigh the public-signal estimate. Raise it if the prior is
            trusted more than a fortnight of trade, lower it if less.

    Returns:
        `(1 - w) * prior + w * actual_mean` where `w = n / (n + k)`.

        At `n == 0` the weight is 0 and the estimate is exactly the prior, so
        day one is unblocked without inventing data. At `n == k` it is a 50/50
        blend. As `n` grows `w` approaches 1 and the prior washes out — it is
        never subtracted or overridden, it simply stops mattering.
    """
    w = n_observations / (n_observations + k)
    return (1 - w) * prior + w * actual_mean
