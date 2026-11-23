"""The experiments give the same tables twice, and say what they should on known data."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from hindsight import experiments as ex
from hindsight.data import parse_french_daily
from hindsight.model import GaussianMS, simulate

SMALL = ex.Config(
    burn_in_years=2,
    refit="year",
    n_boot=40,
    wk_n_init=2,
    synthetic_runs=2,
    synthetic_years=4,
)


def _returns(seed: int = 1) -> pd.DataFrame:
    truth = GaussianMS(
        mu=np.array([0.0008, -0.0010]),
        sigma=np.array([0.006, 0.019]),
        P=np.array([[0.985, 0.015], [0.04, 0.96]]),
        initial=np.array([1.0, 0.0]),
    )
    index = pd.bdate_range("2000-01-03", periods=2000)
    y, _ = simulate(truth, len(index), np.random.default_rng(seed))
    return pd.DataFrame({"mkt_rf": y, "rf": 0.0001}, index=index)


@pytest.fixture(scope="module")
def tables() -> ex.ThreeWayTables:
    return ex.three_way(_returns(), SMALL)


def test_three_way_tables_are_reproducible(tables: ex.ThreeWayTables) -> None:
    again = ex.three_way(_returns(), SMALL)
    for name in ("performance", "gaps", "subperiods", "parameters", "agreement", "sample"):
        pd.testing.assert_frame_equal(getattr(tables, name), getattr(again, name))


def test_three_way_table_shape_and_arithmetic(tables: ex.ThreeWayTables) -> None:
    perf = tables.performance.set_index("strategy")
    assert list(perf.index) == list(ex.STRATEGIES)
    hold = perf.loc[ex.BUY_AND_HOLD]
    assert hold["time_in_market_pct"] == 100.0 and hold["switches"] == 0
    assert hold["sharpe"] == hold["sharpe_net_10bp"]
    gaps = tables.gaps.set_index("gap")
    total, smoother, parameters = (gaps["sharpe_difference"].iloc[i] for i in range(3))
    assert total == pytest.approx(smoother + parameters)
    assert total == pytest.approx(
        perf.loc[ex.HMM_SMOOTHED, "sharpe"] - perf.loc[ex.HMM_FILTERED_PAST, "sharpe"]
    )
    # every strategy is scored on the same days
    daily = tables.daily
    counts = {daily[f"ret::{name}"].notna().sum() for name in ex.STRATEGIES}
    assert len(counts) == 1


def test_costs_lower_the_sharpe_of_anything_that_trades(tables: ex.ThreeWayTables) -> None:
    perf = tables.performance
    trading = perf[perf["switches"] > 0]
    assert np.all(trading["sharpe_net_10bp"] < trading["sharpe"])


def test_synthetic_experiment_is_reproducible_and_accurate() -> None:
    summary, runs = ex.wk_synthetic(SMALL)
    again, _ = ex.wk_synthetic(SMALL)
    pd.testing.assert_frame_equal(summary, again)
    assert len(runs) == 3 * SMALL.synthetic_runs
    by_method = summary.set_index("method")["total_mean_pct"]
    assert by_method["HMM, smoothed"] > 95.0
    assert by_method["WK-means"] > 60.0


def test_synthetic_path_matches_its_specification() -> None:
    returns, regime_on = ex.synthetic_path(20, 10, np.random.default_rng(1))
    assert returns.shape == (252 * 7 * 20,)
    # ten changes of half a year each, never touching
    assert regime_on.sum() == 10 * 882
    assert np.sum(np.diff(regime_on.astype(int)) == 1) + int(regime_on[0]) == 10
    dt = 1.0 / (252 * 7)
    assert returns[regime_on].std() == pytest.approx(0.3 * np.sqrt(dt), rel=0.03)
    assert returns[~regime_on].std() == pytest.approx(0.2 * np.sqrt(dt), rel=0.03)


def test_statsmodels_check_is_at_rounding_level() -> None:
    table = ex.statsmodels_check(_returns()["mkt_rf"].to_numpy(), seed=1)
    assert len(table) == 3
    assert table["max_abs_diff_filtered"].max() < 1e-9
    assert table["max_abs_diff_smoothed"].max() < 1e-9
    assert table["loglik_abs_diff"].max() < 1e-8


def test_hamilton_tables() -> None:
    tables = ex.hamilton_replication()
    assert tables.table2["match"].all() and len(tables.table2) == 7
    assert tables.table1["difference"].abs().max() < 2e-3
    assert len(tables.probabilities) == 131


def test_parse_french_file() -> None:
    text = (
        "This file was created by ...\n\n,Mkt-RF,SMB,HML,RF\n"
        "19260701,    0.09,   -0.23,   -0.28,    0.01\n"
        "19260702,    0.45,   -0.34,   -0.03,    0.01\n\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_Factors_daily.csv", text)
    frame = parse_french_daily(buffer.getvalue())
    assert list(frame.columns) == ["mkt_rf", "rf"]
    assert frame.index[0] == pd.Timestamp("1926-07-01") and len(frame) == 2
    assert frame["mkt_rf"].iloc[1] == pytest.approx(0.0045)
    assert frame["rf"].iloc[0] == pytest.approx(0.0001)
