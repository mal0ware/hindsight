"""Hamilton's filter and Kim's smoother for a first-order Markov chain.

Conventions used everywhere in this package:

* ``P[i, j]`` is ``Prob(s_{t+1} = j | s_t = i)``, so the rows of ``P`` sum to one.
  Hamilton (2016) writes the transpose (columns sum to one).
* ``log_density[t, j]`` is ``log f(y_t | s_t = j, past observations)``.
* ``predicted[t, j]`` is ``Prob(s_t = j | y_1..y_{t-1})``,
  ``filtered[t, j]`` is ``Prob(s_t = j | y_1..y_t)`` and
  ``smoothed[t, j]`` is ``Prob(s_t = j | y_1..y_T)``.

Equation numbers refer to Hamilton (2016), "Macroeconomic Regimes and Regime
Shifts", which restates the filter of Hamilton (1989, section 4.2) in vector
form. See ``PAPER.md`` for the mapping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _jit[F: Callable[..., Any]](func: F) -> F:
    """Compile ``func`` with numba and keep its signature for the type checker."""
    return cast(F, njit(cache=True)(func))


@dataclass(frozen=True)
class FilterResult:
    """Output of :func:`hamilton_filter`."""

    filtered: FloatArray
    predicted: FloatArray
    loglik_obs: FloatArray

    @property
    def loglik(self) -> float:
        """Sample log likelihood, the sum in equation (21)."""
        return float(self.loglik_obs.sum())


def expected_duration(p_stay: float) -> float:
    """Expected number of periods spent in a state once entered.

    The time in a state is geometric with success probability ``1 - p_stay``,
    so its mean is ``1 / (1 - p_stay)`` (Hamilton 1989, p. 374).
    """
    if not 0.0 <= p_stay < 1.0:
        raise ValueError("p_stay must be in [0, 1)")
    return 1.0 / (1.0 - p_stay)


def ergodic_probabilities(P: FloatArray) -> FloatArray:
    """Unconditional state probabilities of an ergodic chain.

    Solves ``pi = pi P`` with ``sum(pi) = 1`` by least squares on the stacked
    system of equation (20): ``A = [I - P'; 1']`` and ``pi`` is the last column
    of ``(A'A)^{-1} A'``.
    """
    P = _check_transition(P)
    k = P.shape[0]
    A = np.vstack([np.eye(k) - P.T, np.ones((1, k))])
    target = np.zeros(k + 1)
    target[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, target, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return cast(FloatArray, pi / pi.sum())


def _check_transition(P: FloatArray) -> FloatArray:
    P = np.ascontiguousarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if np.any(P < 0.0) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("rows of P must be probabilities that sum to one")
    return P


@_jit
def _filter_kernel(
    log_density: FloatArray,
    P: FloatArray,
    initial: FloatArray,
    filtered: FloatArray,
    predicted: FloatArray,
    loglik_obs: FloatArray,
) -> None:
    n, k = log_density.shape
    for t in range(n):
        if t == 0:
            for j in range(k):
                predicted[0, j] = initial[j]
        else:
            # first line of (19): predicted = P' filtered_{t-1}
            for j in range(k):
                acc = 0.0
                for i in range(k):
                    acc += filtered[t - 1, i] * P[i, j]
                predicted[t, j] = acc
        # second line of (19), with the largest log density factored out so
        # that an outlier cannot underflow every state at once
        top = log_density[t, 0]
        for j in range(1, k):
            if log_density[t, j] > top:
                top = log_density[t, j]
        total = 0.0
        for j in range(k):
            filtered[t, j] = predicted[t, j] * np.exp(log_density[t, j] - top)
            total += filtered[t, j]
        for j in range(k):
            filtered[t, j] /= total
        loglik_obs[t] = np.log(total) + top


@_jit
def _smoother_kernel(
    filtered: FloatArray,
    predicted: FloatArray,
    P: FloatArray,
    smoothed: FloatArray,
    transitions: FloatArray,
) -> None:
    n, k = filtered.shape
    for j in range(k):
        smoothed[n - 1, j] = filtered[n - 1, j]
    for t in range(n - 2, -1, -1):
        for i in range(k):
            acc = 0.0
            for j in range(k):
                if predicted[t + 1, j] > 0.0:
                    # footnote 2: Prob(s_t = i, s_{t+1} = j | all data)
                    joint = filtered[t, i] * P[i, j] * smoothed[t + 1, j] / predicted[t + 1, j]
                    transitions[i, j] += joint
                    acc += joint
            # equation (22)
            smoothed[t, i] = acc


def hamilton_filter(log_density: FloatArray, P: FloatArray, initial: FloatArray) -> FilterResult:
    """Run the filter of Hamilton (1989), equations (19) and (21) of Hamilton (2016).

    Parameters
    ----------
    log_density:
        ``(T, K)`` array of log conditional densities.
    P:
        ``(K, K)`` transition matrix, rows summing to one.
    initial:
        ``(K,)`` probabilities of the state at the first observation given no
        data, ``Prob(s_1 = j)``.
    """
    log_density = np.ascontiguousarray(log_density, dtype=np.float64)
    P = _check_transition(P)
    initial = np.ascontiguousarray(initial, dtype=np.float64)
    if log_density.ndim != 2 or log_density.shape[1] != P.shape[0]:
        raise ValueError("log_density must have one column per state")
    if log_density.shape[0] == 0:
        raise ValueError("log_density must have at least one row")
    if initial.shape != (P.shape[0],):
        raise ValueError("initial must have one entry per state")
    if np.any(initial < 0.0) or not np.isclose(initial.sum(), 1.0, atol=1e-10):
        raise ValueError("initial must be a probability vector")
    if not np.all(np.isfinite(log_density)):
        raise ValueError("log_density must be finite")
    filtered = np.empty_like(log_density)
    predicted = np.empty_like(log_density)
    loglik_obs = np.empty(log_density.shape[0])
    _filter_kernel(log_density, P, initial, filtered, predicted, loglik_obs)
    return FilterResult(filtered=filtered, predicted=predicted, loglik_obs=loglik_obs)


def kim_smoother(
    filtered: FloatArray, predicted: FloatArray, P: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Backward recursion for full-sample state probabilities, equation (22).

    Returns
    -------
    smoothed:
        ``(T, K)`` array of ``Prob(s_t = j | y_1..y_T)``.
    transitions:
        ``(K, K)`` array whose ``[i, j]`` entry is the sum over ``t`` of
        ``Prob(s_t = i, s_{t+1} = j | y_1..y_T)``, the numerator of the EM
        update for ``P``.
    """
    filtered = np.ascontiguousarray(filtered, dtype=np.float64)
    predicted = np.ascontiguousarray(predicted, dtype=np.float64)
    P = _check_transition(P)
    if filtered.shape != predicted.shape or filtered.shape[1] != P.shape[0]:
        raise ValueError("filtered and predicted must both be (T, K)")
    smoothed = np.empty_like(filtered)
    transitions = np.zeros_like(P)
    _smoother_kernel(filtered, predicted, P, smoothed, transitions)
    return smoothed, transitions
