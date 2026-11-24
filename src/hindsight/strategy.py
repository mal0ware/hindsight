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

    ``filtered_past`` is NaN before the first refit date. The ``*_top`` arrays
    hold the probability of the highest-variance state instead; with two
    states they are one minus the calm probability. ``full_converged`` and
    ``past_converged`` record whether each EM fit met its stopping rule.
    """

    smoothed: FloatArray
    filtered_full: FloatArray
    filtered_past: FloatArray
    smoothed_top: FloatArray
    filtered_full_top: FloatArray
    filtered_past_top: FloatArray
    full_params: GaussianMS
    past_params: list[GaussianMS]
    refit_ends: IntArray
    full_converged: bool
    full_n_iter: int
    past_converged: BoolArray
    past_n_iter: IntArray


@dataclass(frozen=True)
class Performance:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    max_drawdown_excess: float
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


def top_probability(prob: FloatArray) -> FloatArray:
    """Probability of the highest-variance state (the last one after sorting)."""
    out: FloatArray = prob[:, -1]
    return out


def three_way_probabilities(
    y: FloatArray, refit_ends: IntArray, k: int = 2, max_iter: int = 5000
) -> ThreeWay:
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
    full = fit_em(y, k=k, max_iter=max_iter)
    inference = infer(y, full.params)
    filtered_past = np.full(n, np.nan)
    filtered_past_top = np.full(n, np.nan)
    past_params: list[GaussianMS] = []
    past_converged = np.zeros(refit_ends.size, dtype=np.bool_)
    past_n_iter = np.zeros(refit_ends.size, dtype=np.int64)
    for i, end in enumerate(refit_ends):
        stop = int(refit_ends[i + 1]) if i + 1 < refit_ends.size else n
        fit = fit_em(y[: end + 1], k=k, max_iter=max_iter)
        params = fit.params
        past_params.append(params)
        past_converged[i] = fit.converged
        past_n_iter[i] = fit.n_iter
        # the filter is causal, so running it to ``stop`` with these
        # parameters gives each day's probability from that day's history only
        filtered = filtered_probabilities(y[:stop], params)
        filtered_past[end:stop] = filtered[end:stop, 0]
        filtered_past_top[end:stop] = filtered[end:stop, -1]
    return ThreeWay(
        smoothed=calm_probability(inference.smoothed),
        filtered_full=calm_probability(inference.filtered),
        filtered_past=filtered_past,
        smoothed_top=top_probability(inference.smoothed),
        filtered_full_top=top_probability(inference.filtered),
        filtered_past_top=filtered_past_top,
        full_params=full.params,
        past_params=past_params,
        refit_ends=refit_ends,
        full_converged=full.converged,
        full_n_iter=full.n_iter,
        past_converged=past_converged,
        past_n_iter=past_n_iter,
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


def positions(calm: BoolArray, delay: int = 0) -> FloatArray:
    """Position held on each day: 1 if day ``t - 1 - delay`` was labelled calm.

    With ``delay = 0`` the label from the close of day ``t - 1`` earns the
    return from that close to the close of day ``t``, so the trade is assumed
    to happen at the very close that produced the signal. With ``delay = 1``
    the trade happens one close later: the label of day ``t - 2`` earns the
    return of day ``t``. Days with no label that old start in cash.
    """
    if delay < 0:
        raise ValueError("delay must not be negative")
    lag = 1 + delay
    pos = np.zeros(calm.shape[0])
    pos[lag:] = calm[:-lag].astype(np.float64)
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
    """Largest peak-to-trough fall of the compounded return series, as a fraction.

    The answer depends on which returns are compounded. Excess returns give
    the fall of wealth measured against bills; total returns (excess plus the
    bill rate) give the fall an investor would see in the account.
    """
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    return float(np.max(1.0 - wealth / peak))


def performance(
    returns: FloatArray, position: FloatArray, rf: FloatArray | None = None
) -> Performance:
    """Summary of a series of daily excess returns.

    ``max_drawdown`` is computed on total returns, ``returns + rf``, when the
    bill rate ``rf`` is given, and on the excess returns otherwise.
    ``max_drawdown_excess`` is always on the excess returns.
    """
    returns = np.asarray(returns, dtype=np.float64)
    total = returns if rf is None else returns + np.asarray(rf, dtype=np.float64)
    return Performance(
        ann_return=float(np.mean(returns)) * TRADING_DAYS,
        ann_vol=float(np.std(returns, ddof=1)) * float(np.sqrt(TRADING_DAYS)),
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown(total),
        max_drawdown_excess=max_drawdown(returns),
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


def stationary_bootstrap_index(n: int, mean_block: float, rng: np.random.Generator) -> IntArray:
    """Row indices for one stationary bootstrap resample of ``n`` observations.

    Blocks start at a uniformly drawn row and have geometric lengths with mean
    ``mean_block``; a block that runs off the end wraps to the start. This is
    written from the general description of the stationary bootstrap of
    Politis and Romano (1994), which I did not read.
    """
    if n < 1 or mean_block < 1.0:
        raise ValueError("need n >= 1 and mean_block >= 1")
    out = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        begin = int(rng.integers(0, n))
        length = min(int(rng.geometric(1.0 / mean_block)), n - filled)
        out[filled : filled + length] = (begin + np.arange(length)) % n
        filled += length
    return out
