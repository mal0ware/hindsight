"""Execution delay, convergence records, other starts, other series, the refit bootstrap."""

from __future__ import annotations

import dataclasses
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from hindsight import experiments as ex
from hindsight import hamilton1989
from hindsight.data import parse_french_portfolios
from hindsight.model import GaussianMS, fit_em, is_degenerate, random_start, simulate
from hindsight.strategy import (
    max_drawdown,
    performance,
    positions,
    stationary_bootstrap_index,
    three_way_probabilities,
)

TRUTH = GaussianMS(
    mu=np.array([0.0008, -0.0010]),
    sigma=np.array([0.006, 0.019]),
    P=np.array([[0.985, 0.015], [0.04, 0.96]]),
    initial=np.array([1.0, 0.0]),
)

SMALL = ex.Config(
    burn_in_years=2,
    refit="year",
    n_boot=40,
    wk_n_init=2,
    multistart=3,
    refit_boot=4,
    workers=1,
)


def _excess(columns: int = 3, n: int = 1600) -> pd.DataFrame:
    index = pd.bdate_range("2000-01-03", periods=n)
    data = {f"s{i}": simulate(TRUTH, n, np.random.default_rng(10 + i))[0] for i in range(columns)}
    return pd.DataFrame(data, index=index)


def test_delay_moves_the_position_one_more_day() -> None:
    calm = np.array([True, False, True, True, False])
    np.testing.assert_array_equal(positions(calm, delay=0), [0.0, 1.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(positions(calm, delay=1), [0.0, 0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        positions(calm, delay=-1)


def test_drawdown_on_total_returns_differs_from_excess() -> None:
    excess = np.array([0.02, -0.10, 0.01, -0.05])
    rf = np.full(4, 0.01)
    perf = performance(excess, np.ones(4), rf)
    assert perf.max_drawdown == pytest.approx(max_drawdown(excess + rf))
    assert perf.max_drawdown_excess == pytest.approx(max_drawdown(excess))
    assert perf.max_drawdown < perf.max_drawdown_excess
    same = performance(excess, np.ones(4))
    assert same.max_drawdown == same.max_drawdown_excess


def test_stationary_bootstrap_index() -> None:
    n = 200_000
    index = stationary_bootstrap_index(n, 20.0, np.random.default_rng(3))
    again = stationary_bootstrap_index(n, 20.0, np.random.default_rng(3))
    np.testing.assert_array_equal(index, again)
    assert index.shape == (n,) and index.min() >= 0 and index.max() < n
    # inside a block the rows are consecutive (with wrap-around); a new block
    # starts on average every 20 rows
    breaks = np.sum((np.diff(index) % n) != 1)
    assert n / (breaks + 1) == pytest.approx(20.0, rel=0.05)
    with pytest.raises(ValueError):
        stationary_bootstrap_index(10, 0.5, np.random.default_rng(1))


def test_convergence_is_recorded_and_enforced() -> None:
    y = simulate(TRUTH, 1200, np.random.default_rng(2))[0]
    ends = np.array([600, 900])
    good = three_way_probabilities(y, ends)
    assert good.full_converged and good.past_converged.all()
    assert good.past_n_iter.shape == (2,) and good.full_n_iter > 2
    ex.require_converged(good, "test")
    cut = three_way_probabilities(y, ends, max_iter=2)
    assert not cut.full_converged and not cut.past_converged.any()
    with pytest.raises(RuntimeError, match="3 EM fits did not converge"):
        ex.require_converged(cut, "test")


def test_top_state_probabilities() -> None:
    y = simulate(TRUTH, 1200, np.random.default_rng(2))[0]
    two = three_way_probabilities(y, np.array([600]))
    np.testing.assert_allclose(two.smoothed + two.smoothed_top, 1.0)
    np.testing.assert_allclose(two.filtered_past[600:] + two.filtered_past_top[600:], 1.0)
    three = three_way_probabilities(y, np.array([600]), k=3)
    assert np.all(three.smoothed + three.smoothed_top <= 1.0 + 1e-12)
    assert np.all(np.isnan(three.filtered_past_top[:600]))


def test_three_way_delay_and_convergence_tables() -> None:
    index = pd.bdate_range("2000-01-03", periods=2000)
    y = simulate(TRUTH, len(index), np.random.default_rng(1))[0]
    tables = ex.three_way(pd.DataFrame({"mkt_rf": y, "rf": 0.0001}, index=index), SMALL)
    delay = tables.delay.set_index("strategy")
    perf = tables.performance.set_index("strategy")
    assert list(delay.index) == list(ex.LABELLED)
    for name in ex.LABELLED:
        assert delay.loc[name, "sharpe_same_close"] == perf.loc[name, "sharpe"]
        expected = delay.loc[name, "sharpe_one_day_later"] - delay.loc[name, "sharpe_same_close"]
        assert delay.loc[name, "change"] == pytest.approx(expected)
    # waiting a day costs the smoothed labels something on a persistent chain
    assert delay.loc[ex.HMM_SMOOTHED, "change"] < 0.0
    gaps = tables.gaps
    assert set(gaps["execution"]) == {"same close", "one day later"}
    later = gaps[gaps["execution"] == "one day later"].set_index("gap")["sharpe_difference"]
    total, smoother, parameters = (later.iloc[i] for i in range(3))
    assert total == pytest.approx(smoother + parameters)
    assert total == pytest.approx(
        delay.loc[ex.HMM_SMOOTHED, "sharpe_one_day_later"]
        - delay.loc[ex.HMM_FILTERED_PAST, "sharpe_one_day_later"]
    )
    conv = tables.convergence
    assert (conv["fits"] == conv["converged"]).all()
    assert (conv["most_iterations"] < conv["iteration_limit"]).all()
    assert "max_drawdown_total_return_pct" in tables.performance.columns
    hold = perf.loc[ex.BUY_AND_HOLD]
    assert hold["max_drawdown_total_return_pct"] < hold["max_drawdown_excess_return_pct"]


def test_three_way_stops_when_a_fit_does_not_converge() -> None:
    index = pd.bdate_range("2000-01-03", periods=1200)
    y = simulate(TRUTH, len(index), np.random.default_rng(1))[0]
    frame = pd.DataFrame({"mkt_rf": y, "rf": 0.0}, index=index)
    with pytest.raises(RuntimeError, match="did not converge"):
        ex.three_way(frame, dataclasses.replace(SMALL, max_iter=2))


def test_cross_section_tables() -> None:
    excess = _excess()
    table, summary = ex.cross_section(excess, SMALL)
    again, _ = ex.cross_section(excess, SMALL)
    pd.testing.assert_frame_equal(table, again)
    assert len(table) == 3 * len(ex.SPECS)
    assert list(table["rule"].unique()) == list(ex.SPECS)
    # a third state fitted to two-state data can leave a rule that never
    # holds the asset, whose Sharpe ratio is undefined; the two-state rows
    # are always defined
    assert table.loc[table["rule"] == ex.SPEC_K2].notna().all().all()
    for tag in ("", "_one_day_later"):
        defined = table.dropna()
        np.testing.assert_allclose(
            defined["gap_total" + tag],
            defined["gap_smoother" + tag] + defined["gap_parameters" + tag],
        )
        np.testing.assert_allclose(
            defined["gap_smoother" + tag],
            defined["sharpe_smoothed" + tag] - defined["sharpe_filtered_full" + tag],
        )
    assert (table["em_fits"] == table["em_fits_converged"]).all()
    assert len(summary) == 2 * len(ex.SPECS)
    first = summary.iloc[0]
    part = table[table["rule"] == ex.SPEC_K2]
    assert first["series"] == 3
    assert first["gap_smoother_median"] == pytest.approx(np.median(part["gap_smoother"]))
    # two-state data with persistent regimes: hindsight helps on every series
    assert first["series_with_smoother_gap_above_zero"] == 3


def test_cross_section_matches_the_single_series_run() -> None:
    index = pd.bdate_range("2000-01-03", periods=1600)
    y = simulate(TRUTH, len(index), np.random.default_rng(10))[0]
    single = ex.three_way(pd.DataFrame({"mkt_rf": y, "rf": 0.0}, index=index), SMALL)
    table, _ = ex.cross_section(pd.DataFrame({"s0": y}, index=index), SMALL)
    row = table[table["rule"] == ex.SPEC_K2].iloc[0]
    gaps = single.gaps.set_index(["execution", "gap"])["sharpe_difference"]
    assert row["gap_smoother"] == pytest.approx(gaps.loc["same close"].iloc[1])
    assert row["gap_parameters"] == pytest.approx(gaps.loc["same close"].iloc[2])
    assert row["gap_total_one_day_later"] == pytest.approx(gaps.loc["one day later"].iloc[0])


def test_cross_section_does_not_depend_on_the_number_of_workers() -> None:
    excess = _excess(columns=2, n=900)
    one, _ = ex.cross_section(excess, SMALL)
    two, _ = ex.cross_section(excess, dataclasses.replace(SMALL, workers=2))
    pd.testing.assert_frame_equal(one, two)


def test_random_starts_reach_the_same_maximum() -> None:
    table = ex.em_multistart(_excess(columns=2), SMALL)
    assert len(table) == 4
    two_states = table[table["states"] == 2]
    assert two_states["usual_start_reaches_the_best"].all()
    assert (two_states["best_minus_usual"] < ex.SAME_MAXIMUM).all()
    assert (two_states["random_starts_reaching_the_best"] >= 1).all()
    np.testing.assert_allclose(
        two_states["smoother_gap_usual_start"], two_states["smoother_gap_best_start"]
    )


def test_random_start_is_a_valid_model_and_degenerate_fits_are_flagged() -> None:
    y = simulate(TRUTH, 800, np.random.default_rng(5))[0]
    start = random_start(y, 3, np.random.default_rng(6))
    np.testing.assert_allclose(start.P.sum(axis=1), 1.0)
    assert start.initial.sum() == pytest.approx(1.0) and np.all(start.sigma > 0.0)
    assert fit_em(y, k=3, start=start).loglik > -np.inf
    assert not is_degenerate(y, fit_em(y).params)
    collapsed = GaussianMS(
        mu=np.array([y[0], 0.0]),
        sigma=np.array([1e-9, float(np.std(y))]),
        P=np.array([[0.5, 0.5], [0.01, 0.99]]),
        initial=np.array([0.5, 0.5]),
    )
    assert is_degenerate(y, collapsed)


def test_refit_bootstrap() -> None:
    index = pd.bdate_range("2000-01-03", periods=1500)
    y = simulate(TRUTH, len(index), np.random.default_rng(1))[0]
    frame = pd.DataFrame({"mkt_rf": y, "rf": 0.0}, index=index)
    table = ex.refit_bootstrap(frame, SMALL)
    pd.testing.assert_frame_equal(table, ex.refit_bootstrap(frame, SMALL))
    assert list(table["quantity"]) == list(ex.BOOT_KEYS)
    assert (table["resamples"] == SMALL.refit_boot).all()
    assert (table["ci_low"] <= table["bootstrap_mean"]).all()
    assert (table["bootstrap_mean"] <= table["ci_high"]).all()
    point = table.set_index("quantity")["point_estimate"]
    assert point["gap_total"] == pytest.approx(point["gap_smoother"] + point["gap_parameters"])
    # the point estimate is the cross-section number for the same refit schedule
    cross, _ = ex.cross_section(pd.DataFrame({"s0": y}, index=index), SMALL)
    assert point["gap_smoother"] == pytest.approx(cross["gap_smoother"].iloc[0])


def test_two_means_start_splits_by_mean() -> None:
    y = simulate(TRUTH, 3000, np.random.default_rng(8))[0]
    start = ex.two_means_start(y, np.random.default_rng(9), extra_variance=1e-3)
    assert start.mu[0] < np.median(y) < start.mu[1]
    assert start.sigma[0] == start.sigma[1] == pytest.approx(np.sqrt(np.var(y) + 1e-3))
    np.testing.assert_allclose(start.P.sum(axis=1), 1.0)


def test_parse_portfolio_file_reads_one_section_and_marks_missing() -> None:
    text = (
        "This file was created by ...\n\n"
        "  Average Value Weighted Returns -- Daily\n"
        ",NoDur,Durbl\n"
        "19260701,   0.02,  -0.28\n"
        "19260702,   0.29, -99.99\n\n"
        "  Average Equal Weighted Returns -- Daily\n"
        ",NoDur,Durbl\n"
        "19260701,   9.00,   9.00\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("10_Industry_Portfolios_Daily.csv", text)
    frame = parse_french_portfolios(buffer.getvalue())
    assert list(frame.columns) == ["NoDur", "Durbl"] and len(frame) == 2
    assert frame["NoDur"].iloc[1] == pytest.approx(0.0029)
    assert np.isnan(frame["Durbl"].iloc[1])
    with pytest.raises(ValueError):
        parse_french_portfolios(buffer.getvalue(), section="no such table")


def test_gnp_series_is_read_from_the_installed_statsmodels() -> None:
    quarters, growth = hamilton1989.load_gnp()
    assert quarters[:4] == ["1951Q2", "1951Q3", "1951Q4", "1952Q1"]
    assert quarters[-1] == "1984Q4" and growth.shape == (135,)
    assert growth[0] == pytest.approx(2.59316421)
