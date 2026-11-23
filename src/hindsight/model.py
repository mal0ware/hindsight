"""A K-state Markov-switching model with a Gaussian mean and variance per state.

``y_t | s_t = j  ~  Normal(mu_j, sigma_j^2)`` with ``s_t`` a first-order Markov
chain. This is the model used for daily returns: no autoregressive terms, both
the mean and the variance switch.

Estimation is by the EM algorithm as described in Hamilton (2016, section 2.5),
with the probabilities of the first state treated as free parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hindsight.filter import FloatArray, hamilton_filter, kim_smoother

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class GaussianMS:
    """Parameters of the model. States are ordered by ``sigma``, smallest first."""

    mu: FloatArray
    sigma: FloatArray
    P: FloatArray
    initial: FloatArray

    @property
    def k(self) -> int:
        return int(self.mu.shape[0])


@dataclass(frozen=True)
class Inference:
    """State probabilities for one sample under one set of parameters."""

    filtered: FloatArray
    predicted: FloatArray
    smoothed: FloatArray
    transitions: FloatArray
    loglik: float


@dataclass(frozen=True)
class EMResult:
    params: GaussianMS
    loglik: float
    n_iter: int
    converged: bool
    loglik_path: FloatArray


def gaussian_log_density(y: FloatArray, mu: FloatArray, sigma: FloatArray) -> FloatArray:
    """``(T, K)`` array of ``log Normal(y_t; mu_j, sigma_j^2)``."""
    y = np.asarray(y, dtype=np.float64)
    z = (y[:, None] - mu[None, :]) / sigma[None, :]
    out: FloatArray = -0.5 * (_LOG_2PI + z * z) - np.log(sigma)[None, :]
    return out


def infer(y: FloatArray, params: GaussianMS) -> Inference:
    """Filtered, predicted and smoothed state probabilities for ``y``."""
    log_density = gaussian_log_density(y, params.mu, params.sigma)
    flt = hamilton_filter(log_density, params.P, params.initial)
    smoothed, transitions = kim_smoother(flt.filtered, flt.predicted, params.P)
    return Inference(
        filtered=flt.filtered,
        predicted=flt.predicted,
        smoothed=smoothed,
        transitions=transitions,
        loglik=flt.loglik,
    )


def filtered_probabilities(y: FloatArray, params: GaussianMS) -> FloatArray:
    """Only the filter, for callers that must not see the smoother."""
    log_density = gaussian_log_density(y, params.mu, params.sigma)
    return hamilton_filter(log_density, params.P, params.initial).filtered


def default_start(y: FloatArray, k: int = 2, p_stay: float = 0.95) -> GaussianMS:
    """A deterministic starting point for EM.

    Observations are ranked by absolute deviation from the sample median and
    cut into ``k`` equal groups; each group gives one state's standard
    deviation. Every state starts with the sample mean and the same
    probability ``p_stay`` of staying.
    """
    y = np.asarray(y, dtype=np.float64)
    if k < 1:
        raise ValueError("k must be at least 1")
    dev = np.abs(y - np.median(y))
    order = np.argsort(dev, kind="stable")
    groups = np.array_split(order, k)
    floor = 1e-6 * max(float(np.std(y)), 1e-300)
    sigma = np.array([max(float(np.sqrt(np.mean(dev[g] ** 2))), floor) for g in groups])
    mu = np.full(k, float(np.mean(y)))
    if k == 1:
        P = np.ones((1, 1))
    else:
        P = np.full((k, k), (1.0 - p_stay) / (k - 1))
        np.fill_diagonal(P, p_stay)
    return GaussianMS(mu=mu, sigma=sigma, P=P, initial=np.full(k, 1.0 / k))


def _sort_states(params: GaussianMS) -> GaussianMS:
    order = np.argsort(params.sigma, kind="stable")
    return GaussianMS(
        mu=params.mu[order],
        sigma=params.sigma[order],
        P=params.P[np.ix_(order, order)],
        initial=params.initial[order],
    )


def em_step(y: FloatArray, params: GaussianMS, sigma_floor: float) -> tuple[GaussianMS, float]:
    """One EM iteration. Returns the new parameters and the log likelihood of the old ones."""
    inf = infer(y, params)
    weight = inf.smoothed
    mass = weight.sum(axis=0)
    safe = np.where(mass > 0.0, mass, 1.0)
    # weighted least squares on a constant, the scalar case of equation (24)
    mu = np.where(mass > 0.0, (weight * y[:, None]).sum(axis=0) / safe, params.mu)
    resid2 = (y[:, None] - mu[None, :]) ** 2
    var = np.where(mass > 0.0, (weight * resid2).sum(axis=0) / safe, params.sigma**2)
    sigma = np.maximum(np.sqrt(var), sigma_floor)
    # transition update: expected transitions over expected visits
    from_mass = inf.transitions.sum(axis=1)
    P = params.P.copy()
    for i in range(params.k):
        if from_mass[i] > 0.0:
            P[i] = inf.transitions[i] / from_mass[i]
    initial = inf.smoothed[0].copy()
    return GaussianMS(mu=mu, sigma=sigma, P=P, initial=initial), inf.loglik


def fit_em(
    y: FloatArray,
    k: int = 2,
    start: GaussianMS | None = None,
    tol: float = 1e-12,
    max_iter: int = 5000,
) -> EMResult:
    """Maximum likelihood by EM.

    Iteration stops when the log likelihood rises by less than ``tol`` times
    the number of observations. Standard deviations are floored at ``1e-6``
    times the sample standard deviation so that a state cannot collapse onto a
    single observation.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    if y.ndim != 1 or y.shape[0] < 2 * k:
        raise ValueError("y must be one-dimensional with at least 2k observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    params = default_start(y, k) if start is None else start
    sigma_floor = 1e-6 * max(float(np.std(y)), 1e-300)
    threshold = tol * y.shape[0]
    path: list[float] = []
    converged = False
    previous = -np.inf
    for _ in range(max_iter):
        new_params, loglik = em_step(y, params, sigma_floor)
        path.append(loglik)
        if loglik - previous < threshold:
            converged = True
            break
        previous = loglik
        params = new_params
    params = _sort_states(params)
    return EMResult(
        params=params,
        loglik=infer(y, params).loglik,
        n_iter=len(path),
        converged=converged,
        loglik_path=np.array(path),
    )


def simulate(
    params: GaussianMS, n: int, rng: np.random.Generator
) -> tuple[FloatArray, np.ndarray[tuple[int], np.dtype[np.int64]]]:
    """Draw ``n`` observations and the states that generated them."""
    states = np.empty(n, dtype=np.int64)
    cumulative = np.cumsum(params.P, axis=1)
    u = rng.random(n)
    states[0] = int(np.searchsorted(np.cumsum(params.initial), u[0], side="right"))
    states[0] = min(states[0], params.k - 1)
    for t in range(1, n):
        nxt = int(np.searchsorted(cumulative[states[t - 1]], u[t], side="right"))
        states[t] = min(nxt, params.k - 1)
    y = params.mu[states] + params.sigma[states] * rng.standard_normal(n)
    return y, states
