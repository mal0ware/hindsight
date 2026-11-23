"""Tests for the strategy: timing, information sets, performance arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hindsight import experiments as ex
from hindsight.model import GaussianMS, simulate
from hindsight.strategy import (
    bootstrap_sharpes,
    max_drawdown,
    performance,
    positions,
    sharpe_ratio,
    strategy_returns,
    three_way_probabilities,
    wk_three_way,
)

TRUTH = GaussianMS(
    mu=np.array([0.0008, -0.0010]),
    sigma=np.array([0.006, 0.019]),
    P=np.array([[0.985, 0.015], [0.04, 0.96]]),
    initial=np.array([1.0, 0.0]),
)


def _sample(n: int = 1500, seed: int = 1) -> np.ndarray:
    return simulate(TRUTH, n, np.random.default_rng(seed))[0]


def test_sharpe_ratio_closed_form() -> None:
    x = np.array([0.01, -0.01, 0.02, 0.0])
    expected = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(x) == pytest.approx(expected)
    assert sharpe_ratio(2.0 * x) == pytest.approx(expected)
    assert np.isnan(sharpe_ratio(np.zeros(5)))


def test_positions_lag_the_label_by_one_day() -> None:
    calm = np.array([True, False, True, True])
    np.testing.assert_array_equal(positions(calm), [0.0, 1.0, 0.0, 1.0])


def test_strategy_returns_and_costs() -> None:
    excess = np.array([0.01, 0.02, -0.03, 0.04])
    pos = np.array([0.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(strategy_returns(excess, pos), [0.0, 0.02, -0.03, 0.0])
    net = strategy_returns(excess, pos, cost_bps=10.0)
    np.testing.assert_allclose(net, [0.0, 0.02 - 0.001, -0.03, -0.001])


def test_max_drawdown() -> None:
    assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    assert max_drawdown(np.array([0.1, 0.1])) == 0.0
    assert max_drawdown(np.array([-0.2])) == pytest.approx(0.2)


def test_performance_counts_switches() -> None:
    pos = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    perf = performance(np.zeros(5) + 0.001, pos)
    assert perf.switches == 3
    assert perf.time_in_market == pytest.approx(0.6)


def test_past_only_labels_never_use_the_future() -> None:
    y = _sample()
    ends = np.array([500, 700, 900, 1100, 1300])
    changed = y.copy()
    changed[1000:] = changed[1000:] * 4.0 + 0.01
    a = three_way_probabilities(y, ends)
    b = three_way_probabilities(changed, ends)
    # days before the change get identical past-only probabilities ...
    np.testing.assert_array_equal(a.filtered_past[500:1000], b.filtered_past[500:1000])
    # ... while both full-sample versions move
    assert np.max(np.abs(a.smoothed[500:1000] - b.smoothed[500:1000])) > 1e-6
    assert np.max(np.abs(a.filtered_full[500:1000] - b.filtered_full[500:1000])) > 1e-6
    assert np.all(np.isnan(a.filtered_past[:500]))
    assert not np.any(np.isnan(a.filtered_past[500:]))


def test_wk_past_only_labels_never_use_the_future() -> None:
    y = _sample()
    ends = np.array([500, 700, 900, 1100, 1300])
    changed = y.copy()
    changed[1000:] = changed[1000:] * 4.0 + 0.01
    a = wk_three_way(y, ends, 35, 28, seed=3)
    b = wk_three_way(changed, ends, 35, 28, seed=3)
    np.testing.assert_array_equal(a.trailing_past[500:1000], b.trailing_past[500:1000])
    # the label built from every window containing a day reads up to 34 days ahead
    assert np.any(a.centred_full[960:1000] != b.centred_full[960:1000])


def test_past_only_equals_full_sample_when_refit_on_the_last_day() -> None:
    y = _sample(800)
    out = three_way_probabilities(y, np.array([799]))
    assert out.filtered_past[799] == pytest.approx(out.filtered_full[799], abs=1e-12)
    assert out.smoothed[799] == pytest.approx(out.filtered_full[799], abs=1e-12)


def test_bootstrap_keeps_pairs_and_is_seeded() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(0.0005, 0.01, size=2000)
    matrix = np.column_stack([a, a, 2.0 * a])
    draws = bootstrap_sharpes(matrix, block=20, n_boot=50, seed=5)
    assert draws.shape == (50, 3)
    # identical and rescaled columns have identical Sharpe ratios in every resample
    np.testing.assert_allclose(draws[:, 0], draws[:, 1])
    np.testing.assert_allclose(draws[:, 0], draws[:, 2])
    np.testing.assert_array_equal(draws, bootstrap_sharpes(matrix, 20, 50, 5))
    assert abs(np.mean(draws[:, 0]) - sharpe_ratio(a)) < 0.5


def test_refit_dates_are_month_ends_after_burn_in() -> None:
    index = pd.bdate_range("2000-01-03", "2003-12-31")
    ends = ex.refit_dates(index, 1, "month")
    assert index[ends[0]] == pd.Timestamp("2001-01-31")
    assert np.all(index[ends].month != index[ends + 1].month)
    yearly = ex.refit_dates(index, 1, "year")
    assert [str(d.date()) for d in index[yearly]] == ["2001-12-31", "2002-12-31"]
    with pytest.raises(ValueError):
        ex.refit_dates(index, 10, "month")
