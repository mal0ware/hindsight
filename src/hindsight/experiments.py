"""The experiments behind every table in ``results/``.

Each function takes data and a :class:`Config` and returns pandas tables.
Nothing here reads the clock or an unseeded random number generator, so the
same inputs give the same tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from hindsight import hamilton1989
from hindsight.filter import FloatArray, ergodic_probabilities, expected_duration
from hindsight.model import GaussianMS, fit_em, infer, simulate
from hindsight.strategy import (
    BoolArray,
    bootstrap_sharpes,
    performance,
    positions,
    sharpe_ratio,
    strategy_returns,
    three_way_probabilities,
    wk_three_way,
)
from hindsight.wasserstein import IntArray, accuracy_scores, lift, membership_counts, wk_means


@dataclass(frozen=True)
class Config:
    """Every choice that the results depend on."""

    start: str | None = None
    burn_in_years: int = 10
    refit: str = "month"
    cost_bps: float = 10.0
    block: int = 63
    n_boot: int = 2000
    seed: int = 20261123
    wk_h1: int = 35
    wk_h2: int = 28
    wk_p: int = 1
    wk_n_init: int = 10
    synthetic_runs: int = 50
    synthetic_years: int = 20


FULL = Config()
FAST = Config(
    start="1990-01-01",
    burn_in_years=5,
    refit="year",
    n_boot=200,
    wk_n_init=3,
    synthetic_runs=3,
    synthetic_years=5,
)

HMM_SMOOTHED = "HMM, smoothed labels"
HMM_FILTERED_FULL = "HMM, filtered labels, full-sample parameters"
HMM_FILTERED_PAST = "HMM, filtered labels, past-only parameters"
HMM_SAME_DAY = "HMM, smoothed labels traded on the same day"
WK_CENTRED = "WK-means, labels from all windows containing the day"
WK_TRAILING_FULL = "WK-means, trailing window, full-sample centroids"
WK_TRAILING_PAST = "WK-means, trailing window, past-only centroids"
BUY_AND_HOLD = "Buy and hold"

STRATEGIES = (
    BUY_AND_HOLD,
    HMM_SMOOTHED,
    HMM_FILTERED_FULL,
    HMM_FILTERED_PAST,
    HMM_SAME_DAY,
    WK_CENTRED,
    WK_TRAILING_FULL,
    WK_TRAILING_PAST,
)

GAPS = (
    ("HMM: total look-ahead (smoothed minus past-only)", HMM_SMOOTHED, HMM_FILTERED_PAST),
    ("HMM: from the smoother (smoothed minus filtered)", HMM_SMOOTHED, HMM_FILTERED_FULL),
    (
        "HMM: from the parameters (full-sample minus past-only)",
        HMM_FILTERED_FULL,
        HMM_FILTERED_PAST,
    ),
    ("HMM: same-day trading on top of smoothing", HMM_SAME_DAY, HMM_SMOOTHED),
    ("WK-means: total look-ahead (all windows minus past-only)", WK_CENTRED, WK_TRAILING_PAST),
    ("WK-means: from windows that run ahead", WK_CENTRED, WK_TRAILING_FULL),
    ("WK-means: from the centroids", WK_TRAILING_FULL, WK_TRAILING_PAST),
)


@dataclass(frozen=True)
class ThreeWayTables:
    performance: pd.DataFrame
    gaps: pd.DataFrame
    subperiods: pd.DataFrame
    parameters: pd.DataFrame
    agreement: pd.DataFrame
    sample: pd.DataFrame
    daily: pd.DataFrame


def refit_dates(index: pd.DatetimeIndex, burn_in_years: int, refit: str) -> IntArray:
    """Positions of the last trading day of every month (or year) after the burn-in."""
    if refit not in ("month", "year"):
        raise ValueError("refit must be 'month' or 'year'")
    period = index.to_period("M" if refit == "month" else "Y")
    is_last = np.append(period[1:] != period[:-1], False)
    first_allowed = index[0] + pd.DateOffset(years=burn_in_years)
    keep = is_last & (index >= first_allowed)
    out: IntArray = np.flatnonzero(keep).astype(np.int64)
    if out.size == 0:
        raise ValueError("sample is shorter than the burn-in")
    return out


def _params_row(label: str, date: object, n_obs: int, params: GaussianMS) -> dict[str, object]:
    return {
        "fit": label,
        "last_day": date,
        "observations": n_obs,
        "mu_calm_pct": params.mu[0] * 100.0,
        "mu_turbulent_pct": params.mu[1] * 100.0,
        "sigma_calm_pct": params.sigma[0] * 100.0,
        "sigma_turbulent_pct": params.sigma[1] * 100.0,
        "p_stay_calm": params.P[0, 0],
        "p_stay_turbulent": params.P[1, 1],
        "days_calm": expected_duration(float(params.P[0, 0])),
        "days_turbulent": expected_duration(float(params.P[1, 1])),
    }


def three_way(returns: pd.DataFrame, config: Config) -> ThreeWayTables:
    """Run the strategy under every information set and tabulate the results.

    ``returns`` needs a ``mkt_rf`` column of daily excess returns in decimals
    and a DatetimeIndex.
    """
    if config.start is not None:
        returns = returns.loc[config.start :]
    index = pd.DatetimeIndex(returns.index)
    y: FloatArray = returns["mkt_rf"].to_numpy(dtype=np.float64)
    ends = refit_dates(index, config.burn_in_years, config.refit)

    hmm = three_way_probabilities(y, ends)
    wk = wk_three_way(
        y, ends, config.wk_h1, config.wk_h2, config.wk_p, config.seed, config.wk_n_init
    )

    calm: dict[str, BoolArray] = {
        HMM_SMOOTHED: hmm.smoothed > 0.5,
        HMM_FILTERED_FULL: hmm.filtered_full > 0.5,
        HMM_FILTERED_PAST: np.nan_to_num(hmm.filtered_past, nan=0.0) > 0.5,
        WK_CENTRED: wk.centred_full,
        WK_TRAILING_FULL: wk.trailing_full,
        WK_TRAILING_PAST: wk.trailing_past,
    }
    held: dict[str, FloatArray] = {name: positions(label) for name, label in calm.items()}
    held[BUY_AND_HOLD] = np.ones_like(y)
    # the careless version: today's full-sample label earns today's return
    held[HMM_SAME_DAY] = calm[HMM_SMOOTHED].astype(np.float64)

    first = int(ends[0]) + 1
    evaluate = slice(first, y.shape[0])
    gross = {name: strategy_returns(y, held[name])[evaluate] for name in STRATEGIES}
    net = {name: strategy_returns(y, held[name], config.cost_bps)[evaluate] for name in STRATEGIES}

    matrix = np.column_stack([gross[name] for name in STRATEGIES])
    draws = bootstrap_sharpes(matrix, config.block, config.n_boot, config.seed)
    column = {name: i for i, name in enumerate(STRATEGIES)}

    rows = []
    for name in STRATEGIES:
        perf = performance(gross[name], held[name][evaluate])
        lo, hi = np.nanpercentile(draws[:, column[name]], [2.5, 97.5])
        rows.append(
            {
                "strategy": name,
                "ann_excess_return_pct": perf.ann_return * 100.0,
                "ann_vol_pct": perf.ann_vol * 100.0,
                "sharpe": perf.sharpe,
                "sharpe_ci_low": lo,
                "sharpe_ci_high": hi,
                "max_drawdown_pct": perf.max_drawdown * 100.0,
                "time_in_market_pct": perf.time_in_market * 100.0,
                "switches": perf.switches,
                f"sharpe_net_{config.cost_bps:g}bp": sharpe_ratio(net[name]),
            }
        )
    performance_table = pd.DataFrame(rows)

    gap_rows = []
    for label, a, b in GAPS:
        diff = draws[:, column[a]] - draws[:, column[b]]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        gap_rows.append(
            {
                "gap": label,
                "sharpe_difference": sharpe_ratio(gross[a]) - sharpe_ratio(gross[b]),
                "ci_low": lo,
                "ci_high": hi,
                "share_of_resamples_at_or_below_zero": float(np.mean(diff <= 0.0)),
            }
        )
    gaps_table = pd.DataFrame(gap_rows)

    eval_index = index[evaluate]
    thirds = np.array_split(np.arange(len(eval_index)), 3)
    sub_rows = []
    for name in STRATEGIES:
        row: dict[str, object] = {"strategy": name}
        for part in thirds:
            label = f"{eval_index[part[0]].year}-{eval_index[part[-1]].year}"
            row[label] = sharpe_ratio(gross[name][part])
        sub_rows.append(row)
    subperiods_table = pd.DataFrame(sub_rows)

    param_rows = [_params_row("full sample", index[-1].date(), y.shape[0], hmm.full_params)]
    for end, params in zip(hmm.refit_ends, hmm.past_params, strict=True):
        param_rows.append(_params_row("past only", index[end].date(), int(end) + 1, params))
    parameters_table = pd.DataFrame(param_rows)

    reference = calm[HMM_SMOOTHED][first - 1 : -1]
    agreement_rows = []
    for name, calm_label in calm.items():
        used = calm_label[first - 1 : -1]
        agreement_rows.append(
            {
                "labels": name,
                "calm_days_pct": float(np.mean(used)) * 100.0,
                "same_as_hmm_smoothed_pct": float(np.mean(used == reference)) * 100.0,
            }
        )
    agreement_table = pd.DataFrame(agreement_rows)

    sample_table = pd.DataFrame(
        [
            {"item": "first day in sample", "value": str(index[0].date())},
            {"item": "last day in sample", "value": str(index[-1].date())},
            {"item": "days in sample", "value": str(y.shape[0])},
            {"item": "first refit day", "value": str(index[ends[0]].date())},
            {"item": "refits", "value": str(ends.size)},
            {"item": "first day evaluated", "value": str(eval_index[0].date())},
            {"item": "days evaluated", "value": str(len(eval_index))},
        ]
    )

    daily = pd.DataFrame(
        {
            "mkt_rf": y,
            "p_calm_smoothed": hmm.smoothed,
            "p_calm_filtered_full": hmm.filtered_full,
            "p_calm_filtered_past": hmm.filtered_past,
        },
        index=index,
    )
    for name in STRATEGIES:
        series = np.full(y.shape[0], np.nan)
        series[evaluate] = gross[name]
        daily[f"ret::{name}"] = series

    return ThreeWayTables(
        performance=performance_table,
        gaps=gaps_table,
        subperiods=subperiods_table,
        parameters=parameters_table,
        agreement=agreement_table,
        sample=sample_table,
        daily=daily,
    )


# --------------------------------------------------------------------------
# Hamilton (1989)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HamiltonTables:
    table1: pd.DataFrame
    table2: pd.DataFrame
    durations: pd.DataFrame
    probabilities: pd.DataFrame


def hamilton_replication() -> HamiltonTables:
    """Re-estimate Hamilton's GNP model and compare with Tables I and II."""
    quarters, y = hamilton1989.load_gnp()
    fit = hamilton1989.fit_mle(y)
    names = ["alpha1", "alpha0", "p", "q", "sigma", "phi1", "phi2", "phi3", "phi4"]
    estimate = np.concatenate(
        [
            [fit.params.alpha1, fit.params.alpha0, fit.params.p, fit.params.q, fit.params.sigma],
            fit.params.phi,
        ]
    )
    paper = hamilton1989.TABLE_I
    printed = np.concatenate(
        [[paper.alpha1, paper.alpha0, paper.p, paper.q, paper.sigma], paper.phi]
    )
    table1 = pd.DataFrame(
        {
            "parameter": names,
            "paper": printed,
            "here": estimate,
            "difference": estimate - printed,
            "paper_std_error": hamilton1989.TABLE_I_SE,
            "here_std_error": fit.std_errors,
        }
    )
    table1.loc[len(table1)] = {
        "parameter": "log likelihood at the estimates here",
        "paper": np.nan,
        "here": fit.loglik,
        "difference": np.nan,
        "paper_std_error": np.nan,
        "here_std_error": np.nan,
    }
    table1.loc[len(table1)] = {
        "parameter": "log likelihood at the paper's printed estimates",
        "paper": np.nan,
        "here": hamilton1989.infer(y, paper).loglik,
        "difference": np.nan,
        "paper_std_error": np.nan,
        "here_std_error": np.nan,
    }

    inference = hamilton1989.infer(y, fit.params)
    used = quarters[fit.params.r :]
    spells = hamilton1989.recession_spells(used, inference.smoothed_low)
    printed_spells = list(hamilton1989.TABLE_II_SMOOTHER)
    n = max(len(spells), len(printed_spells))
    blank = ("", "")
    table2 = pd.DataFrame(
        {
            "paper_first_quarter": [(printed_spells + [blank] * n)[i][0] for i in range(n)],
            "paper_last_quarter": [(printed_spells + [blank] * n)[i][1] for i in range(n)],
            "here_first_quarter": [(spells + [blank] * n)[i][0] for i in range(n)],
            "here_last_quarter": [(spells + [blank] * n)[i][1] for i in range(n)],
        }
    )
    table2["match"] = (table2["paper_first_quarter"] == table2["here_first_quarter"]) & (
        table2["paper_last_quarter"] == table2["here_last_quarter"]
    )

    durations = pd.DataFrame(
        [
            {
                "state": "recession (state 0)",
                "paper_quarters": 4.1,
                "here_quarters": expected_duration(fit.params.q),
                "from_paper_estimate": expected_duration(paper.q),
            },
            {
                "state": "expansion (state 1)",
                "paper_quarters": 10.5,
                "here_quarters": expected_duration(fit.params.p),
                "from_paper_estimate": expected_duration(paper.p),
            },
        ]
    )
    probabilities = pd.DataFrame(
        {
            "quarter": used,
            "growth_pct": y[fit.params.r :],
            "filtered_prob_recession": inference.filtered_low,
            "smoothed_prob_recession": inference.smoothed_low,
        }
    )
    return HamiltonTables(
        table1=table1, table2=table2, durations=durations, probabilities=probabilities
    )


# --------------------------------------------------------------------------
# Agreement with statsmodels
# --------------------------------------------------------------------------


def _statsmodels_regression(y: FloatArray, params: GaussianMS) -> dict[str, float]:
    """Largest gaps between this package and statsmodels at the same parameters."""
    initial = ergodic_probabilities(params.P)
    mine = infer(y, GaussianMS(mu=params.mu, sigma=params.sigma, P=params.P, initial=initial))
    model = sm.tsa.MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    theirs = model.smooth(
        np.array(
            [
                params.P[0, 0],
                params.P[1, 0],
                params.mu[0],
                params.mu[1],
                params.sigma[0] ** 2,
                params.sigma[1] ** 2,
            ]
        )
    )
    return {
        "observations": float(y.shape[0]),
        "max_abs_diff_filtered": float(
            np.max(np.abs(mine.filtered - theirs.filtered_marginal_probabilities))
        ),
        "max_abs_diff_smoothed": float(
            np.max(np.abs(mine.smoothed - theirs.smoothed_marginal_probabilities))
        ),
        "loglik_here": mine.loglik,
        "loglik_statsmodels": float(theirs.llf),
    }


def statsmodels_check(market: FloatArray | None, seed: int) -> pd.DataFrame:
    """Compare filter, smoother and likelihood with statsmodels on three samples."""
    rows: list[dict[str, object]] = []
    truth = GaussianMS(
        mu=np.array([0.05, -0.10]),
        sigma=np.array([0.7, 2.0]),
        P=np.array([[0.97, 0.03], [0.08, 0.92]]),
        initial=np.array([0.5, 0.5]),
    )
    simulated, _ = simulate(truth, 2000, np.random.default_rng(seed))
    rows.append({"sample": "simulated, 2 states", **_statsmodels_regression(simulated, truth)})
    if market is not None:
        fitted = fit_em(market).params
        rows.append(
            {
                "sample": "daily market excess returns, EM estimates",
                **_statsmodels_regression(market, fitted),
            }
        )
    _, gnp = hamilton1989.load_gnp()
    paper = hamilton1989.TABLE_I
    mine = hamilton1989.infer(gnp, paper)
    model = sm.tsa.MarkovAutoregression(gnp, k_regimes=2, order=4, switching_ar=False)
    theirs = model.smooth(
        np.concatenate(
            [
                [paper.q, 1.0 - paper.p, paper.alpha0, paper.alpha0 + paper.alpha1],
                [paper.sigma**2],
                paper.phi,
            ]
        )
    )
    rows.append(
        {
            "sample": "Hamilton (1989) GNP model at Table I estimates",
            "observations": float(gnp.shape[0] - paper.r),
            "max_abs_diff_filtered": float(
                np.max(np.abs(mine.filtered_low - theirs.filtered_marginal_probabilities[:, 0]))
            ),
            "max_abs_diff_smoothed": float(
                np.max(np.abs(mine.smoothed_low - theirs.smoothed_marginal_probabilities[:, 0]))
            ),
            "loglik_here": mine.loglik,
            "loglik_statsmodels": float(theirs.llf),
        }
    )
    table = pd.DataFrame(rows)
    table["loglik_abs_diff"] = (table["loglik_here"] - table["loglik_statsmodels"]).abs()
    return table


# --------------------------------------------------------------------------
# Horvath, Issa and Muguruza: synthetic test
# --------------------------------------------------------------------------

STEPS_PER_YEAR = 252 * 7
BULL = (0.02, 0.2)
BEAR = (-0.02, 0.3)


def synthetic_path(
    years: int, n_changes: int, rng: np.random.Generator
) -> tuple[FloatArray, BoolArray]:
    """Log returns of a geometric Brownian motion with regime changes (section 3.3.1).

    Each change lasts half a year of hourly steps. Start points are drawn
    uniformly subject to the changes not touching each other.
    """
    n = STEPS_PER_YEAR * years
    length = STEPS_PER_YEAR // 2
    gap = length + 3
    free = n - n_changes * gap
    if free < 0:
        raise ValueError("too many regime changes for this many years")
    offsets = np.sort(rng.integers(0, free + 1, size=n_changes))
    regime_on = np.zeros(n, dtype=np.bool_)
    for i, offset in enumerate(offsets):
        start = int(offset) + i * gap
        regime_on[start : start + length] = True
    dt = 1.0 / STEPS_PER_YEAR
    mu = np.where(regime_on, BEAR[0], BULL[0])
    sigma = np.where(regime_on, BEAR[1], BULL[1])
    # equation (35): log returns are Normal((mu - sigma^2/2) dt, sigma^2 dt)
    returns: FloatArray = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    return returns, regime_on


def wk_synthetic(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy of WK-means and of the HMM on synthetic paths (the paper's Table 3)."""
    rng = np.random.default_rng(config.seed)
    n_changes = max(1, config.synthetic_years // 2)
    runs = []
    for run in range(config.synthetic_runs):
        returns, regime_on = synthetic_path(config.synthetic_years, n_changes, rng)
        windows, starts = lift(returns, config.wk_h1, config.wk_h2)
        fit = wk_means(windows, k=2, p=config.wk_p, seed=config.seed + run)
        counts = membership_counts(fit.labels, starts, config.wk_h1, returns.shape[0], 2)
        total, on, off = accuracy_scores(counts, regime_on)
        runs.append({"run": run, "method": "WK-means", "total": total, "on": on, "off": off})
        hmm = infer(returns, fit_em(returns).params)
        turbulent = hmm.smoothed[:, 1] > 0.5
        hmm_counts = np.column_stack([~turbulent, turbulent]).astype(np.float64)
        total, on, off = accuracy_scores(hmm_counts, regime_on)
        runs.append({"run": run, "method": "HMM, smoothed", "total": total, "on": on, "off": off})
        turbulent = hmm.filtered[:, 1] > 0.5
        hmm_counts = np.column_stack([~turbulent, turbulent]).astype(np.float64)
        total, on, off = accuracy_scores(hmm_counts, regime_on)
        runs.append({"run": run, "method": "HMM, filtered", "total": total, "on": on, "off": off})
    per_run = pd.DataFrame(runs)
    paper = {
        "WK-means": (90.60, 87.24, 91.72),
        "HMM, smoothed": (58.16, 41.51, 63.72),
        "HMM, filtered": (np.nan, np.nan, np.nan),
    }
    rows = []
    for method, group in per_run.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "runs": len(group)}
        for i, key in enumerate(("total", "on", "off")):
            values = group[key].to_numpy() * 100.0
            row[f"{key}_mean_pct"] = float(np.mean(values))
            row[f"{key}_sd_pct"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{key}_paper_pct"] = paper[str(method)][i]
        rows.append(row)
    return pd.DataFrame(rows), per_run
