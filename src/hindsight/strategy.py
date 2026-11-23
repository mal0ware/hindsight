"""One regime strategy, run with three different amounts of information.

The rule is the same in every version: hold the index tomorrow if today is
labelled calm, otherwise hold cash. Only the label changes.

1. ``smoothed``: full-sample parameters, smoothed probability. The label for
   day ``t`` uses returns after ``t`` twice, through the parameters and through
   the backward pass.
2. ``filtered, full-sample parameters``: the label for day ``t`` uses returns
   after ``t`` only through the parameters.
3. ``filtered, past parameters``: parameters are re-estimated on an expanding
   window and the filter is rerun; the label for day ``t`` uses nothing after
   ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hindsight.filter import FloatArray
from hindsight.model import GaussianMS, filtered_probabilities, fit_em, infer
from hindsight.wasserstein import (
    IntArray,
    assign,
    lift,
    membership_counts,
    trailing_windows,
    wk_means,
)

BoolArray = NDArray[np.bool_]

TRADING_DAYS = 252


@dataclass(frozen=True)
class ThreeWay:
    """Probability that each day is calm, under the three information sets.

    ``filtered_past`` is NaN before the first refit date.
    """

    smoothed: FloatArray
    filtered_full: FloatArray
    filtered_past: FloatArray
    full_params: GaussianMS
    past_params: list[GaussianMS]
    refit_ends: IntArray


@dataclass(frozen=True)
class Performance:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    time_in_market: float
    switches: int


def sharpe_ratio(excess: FloatArray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of a series of excess returns."""
    excess = np.asarray(excess, dtype=np.float64)
    sd = float(np.std(excess, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.mean(excess)) / sd * float(np.sqrt(periods))


def calm_probability(prob: FloatArray) -> FloatArray:
    """Probability of the lowest-variance state (state 0 after sorting)."""
    out: FloatArray = prob[:, 0]
    return out


def three_way_probabilities(y: FloatArray, refit_ends: IntArray, k: int = 2) -> ThreeWay:
    """Calm-state probabilities under the three information sets.

    ``refit_ends`` are the indices of the days on which the parameters are
    re-estimated, using ``y[: e + 1]``. The parameters fitted at ``e`` label
    the days from ``e`` up to the day before the next refit.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = y.shape[0]
    refit_ends = np.asarray(refit_ends, dtype=np.int64)
    if refit_ends.size == 0 or np.any(np.diff(refit_ends) <= 0) or refit_ends[-1] >= n:
        raise ValueError("refit_ends must be increasing indices into y")
    full = fit_em(y, k=k)
    inference = infer(y, full.params)
    filtered_past = np.full(n, np.nan)
    past_params: list[GaussianMS] = []
    for i, end in enumerate(refit_ends):
        stop = int(refit_ends[i + 1]) if i + 1 < refit_ends.size else n
        params = fit_em(y[: end + 1], k=k).params
        past_params.append(params)
        # the filter is causal, so running it to ``stop`` with these
        # parameters gives each day's probability from that day's history only
        filtered = filtered_probabilities(y[:stop], params)
        filtered_past[end:stop] = filtered[end:stop, 0]
    return ThreeWay(
        smoothed=calm_probability(inference.smoothed),
        filtered_full=calm_probability(inference.filtered),
        filtered_past=filtered_past,
        full_params=full.params,
        past_params=past_params,
        refit_ends=refit_ends,
    )


@dataclass(frozen=True)
class WKThreeWay:
    """Calm labels from Wasserstein k-means under three information sets."""

    centred_full: BoolArray
    trailing_full: BoolArray
    trailing_past: BoolArray
    valid_from: int


def wk_three_way(
    y: FloatArray,
    refit_ends: IntArray,
    h1: int,
    h2: int,
    p: int = 1,
    seed: int = 0,
    n_init: int = 1,
) -> WKThreeWay:
    """The same three-way split for the clustering method.

    1. ``centred_full``: centroids from the whole sample; a day is calm when
       most of the windows that contain it are in the low-variance cluster.
       This is how the paper colours a price path, and windows that contain a
       day extend up to ``h1 - 1`` days past it.
    2. ``trailing_full``: centroids from the whole sample; a day is labelled by
       the window that ends on it.
    3. ``trailing_past``: centroids refitted on windows that end on or before
       each refit date; a day is labelled by the window that ends on it.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = y.shape[0]
    windows, starts = lift(y, h1, h2)
    full = wk_means(windows, k=2, p=p, seed=seed, n_init=n_init)
    trailing = trailing_windows(y, h1)
    have = np.arange(n) >= h1 - 1
    trailing_full = np.zeros(n, dtype=np.bool_)
    trailing_full[have] = assign(trailing[have], full.centroids, p) == 0
    counts = membership_counts(full.labels, starts, h1, n, 2)
    centred_full = np.where(counts.sum(axis=1) > 0.0, counts[:, 0] > counts[:, 1], trailing_full)
    trailing_past = np.zeros(n, dtype=np.bool_)
    for i, end in enumerate(refit_ends):
        stop = int(refit_ends[i + 1]) if i + 1 < refit_ends.size else n
        past_windows, _ = lift(y[: end + 1], h1, h2)
        fit = wk_means(past_windows, k=2, p=p, seed=seed, n_init=n_init)
        trailing_past[end:stop] = assign(trailing[end:stop], fit.centroids, p) == 0
    return WKThreeWay(
        centred_full=centred_full.astype(np.bool_),
        trailing_full=trailing_full,
        trailing_past=trailing_past,
        valid_from=int(refit_ends[0]),
    )


def positions(calm: BoolArray) -> FloatArray:
    """Position held on each day: 1 if the previous day was labelled calm.

    The first day has no previous label and starts in cash.
    """
    pos = np.zeros(calm.shape[0])
    pos[1:] = calm[:-1].astype(np.float64)
    return pos


def strategy_returns(excess: FloatArray, position: FloatArray, cost_bps: float = 0.0) -> FloatArray:
    """Daily excess return of holding ``position``, net of a cost per switch.

    ``cost_bps`` is charged on the day of each change of position, per unit
    traded. Cash earns the risk-free rate, which is zero in excess terms.
    """
    trades = np.abs(np.diff(position, prepend=position[0]))
    out: FloatArray = position * excess - trades * cost_bps * 1e-4
    return out


def max_drawdown(returns: FloatArray) -> float:
    """Largest peak-to-trough fall of the compounded return series, as a fraction."""
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    return float(np.max(1.0 - wealth / peak))


def performance(returns: FloatArray, position: FloatArray) -> Performance:
    returns = np.asarray(returns, dtype=np.float64)
    return Performance(
        ann_return=float(np.mean(returns)) * TRADING_DAYS,
        ann_vol=float(np.std(returns, ddof=1)) * float(np.sqrt(TRADING_DAYS)),
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown(returns),
        time_in_market=float(np.mean(position)),
        switches=int(np.sum(np.abs(np.diff(position)) > 0.0)),
    )


def bootstrap_sharpes(
    returns: FloatArray, block: int, n_boot: int, seed: int, periods: int = TRADING_DAYS
) -> FloatArray:
    """Circular block bootstrap of the Sharpe ratios of several return series.

    ``returns`` is ``(T, S)``. Every replication resamples whole blocks of
    ``block`` consecutive days, the same days for every column, so that
    differences between columns keep their pairing. Returns ``(n_boot, S)``.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = returns.shape[0]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    out = np.empty((n_boot, returns.shape[1]))
    for b in range(n_boot):
        begin = rng.integers(0, n, size=n_blocks)
        index = ((begin[:, None] + offsets[None, :]) % n).ravel()[:n]
        sample = returns[index]
        sd = sample.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[b] = sample.mean(axis=0) / sd * np.sqrt(periods)
    return out
