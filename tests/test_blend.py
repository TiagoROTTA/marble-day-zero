"""Tests for `src.tools.blend.blend` — pure arithmetic, no fixtures, no mocks.

The three properties asserted here are the ones the design claims out loud:
day one is exactly the prior, `n == k` is the midpoint, and more data always
moves the estimate toward the actuals. If any of them stops holding, the
"the prior is replaced within roughly two weeks" line in the README is a lie.
"""
import pytest

from src.tools.blend import blend

PRIOR = 240.0
ACTUAL = 300.0


def test_zero_observations_is_exactly_the_prior():
    """Day one: no sales exist, so the estimate is the prior, untouched."""
    assert blend(PRIOR, ACTUAL, 0) == PRIOR


def test_zero_observations_is_the_prior_whatever_k_is():
    for k in (1.0, 7.0, 14.0, 60.0):
        assert blend(PRIOR, ACTUAL, 0, k=k) == PRIOR


def test_n_equals_k_is_the_midpoint():
    """`k` days of data against a prior worth `k` days: a 50/50 blend."""
    assert blend(PRIOR, ACTUAL, 14, k=14.0) == pytest.approx((PRIOR + ACTUAL) / 2.0)


def test_n_equals_k_is_the_midpoint_for_other_k():
    assert blend(PRIOR, ACTUAL, 7, k=7.0) == pytest.approx((PRIOR + ACTUAL) / 2.0)
    assert blend(PRIOR, ACTUAL, 30, k=30.0) == pytest.approx((PRIOR + ACTUAL) / 2.0)


def test_estimate_moves_monotonically_toward_the_actual_mean():
    """Every extra day of data strictly shrinks the gap to `actual_mean`."""
    gaps = [abs(blend(PRIOR, ACTUAL, n) - ACTUAL) for n in range(0, 90)]
    for earlier, later in zip(gaps, gaps[1:]):
        assert later < earlier

    assert gaps[0] == pytest.approx(abs(PRIOR - ACTUAL))


def test_monotonic_when_the_actuals_come_in_below_the_prior():
    """Direction is irrelevant: an over-optimistic prior converges the same way."""
    gaps = [abs(blend(ACTUAL, PRIOR, n) - PRIOR) for n in range(0, 90)]
    for earlier, later in zip(gaps, gaps[1:]):
        assert later < earlier


def test_prior_washes_out_but_is_never_discarded():
    """Large n approaches the actuals without ever quite arriving."""
    far = blend(PRIOR, ACTUAL, 10_000)
    assert far == pytest.approx(ACTUAL, abs=0.1)
    assert far != ACTUAL


def test_estimate_always_lies_between_prior_and_actual_mean():
    for n in range(0, 60):
        estimate = blend(PRIOR, ACTUAL, n)
        assert PRIOR <= estimate <= ACTUAL


def test_k_is_the_prior_strength_in_days():
    """A bigger `k` holds the prior longer: at the same n, less shrinkage."""
    weak = blend(PRIOR, ACTUAL, 14, k=7.0)
    default = blend(PRIOR, ACTUAL, 14, k=14.0)
    strong = blend(PRIOR, ACTUAL, 14, k=28.0)
    assert abs(strong - ACTUAL) > abs(default - ACTUAL) > abs(weak - ACTUAL)


def test_identical_prior_and_actuals_never_moves():
    for n in (0, 1, 14, 100):
        assert blend(PRIOR, PRIOR, n) == pytest.approx(PRIOR)
