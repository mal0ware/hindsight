"""The model of Hamilton (1989): a two-state switching mean with AR(r) errors.

Equation (4.3) of the paper::

    y_t = alpha_1 s_t + alpha_0 + z_t
    z_t = phi_1 z_{t-1} + ... + phi_r z_{t-r} + eps_t,   eps_t ~ N(0, sigma^2)

with ``Prob(s_t = 1 | s_{t-1} = 1) = p`` and ``Prob(s_t = 0 | s_{t-1} = 0) = q``
(equation 2.2). The density of ``y_t`` depends on ``s_t, ..., s_{t-r}``, so the
filter of section 4.2 runs over the ``2^(r+1)`` combinations of those states.
Here that is done by handing the expanded chain to the general filter in
:mod:`hindsight.filter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
from scipy import optimize

from hindsight.filter import FloatArray, ergodic_probabilities, hamilton_filter, kim_smoother

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class Hamilton1989Params:
    alpha0: float
    alpha1: float
    p: float
    q: float
    sigma: float
    phi: FloatArray

    @property
    def r(self) -> int:
        return int(self.phi.shape[0])


#: Table I of Hamilton (1989), p. 372.
TABLE_I = Hamilton1989Params(
    alpha0=-0.3577,
    alpha1=1.522,
    p=0.9049,
    q=0.7550,
    sigma=0.7690,
    phi=np.array([0.014, -0.058, -0.247, -0.213]),
)

#: Asymptotic standard errors printed in Table I, in the order
#: alpha1, alpha0, p, q, sigma, phi1..phi4.
TABLE_I_SE = np.array([0.2636, 0.2651, 0.03740, 0.09656, 0.06676, 0.120, 0.137, 0.107, 0.110])

#: Table II of Hamilton (1989), p. 374, "Smoother" columns: first and last
#: quarter of each recession as printed (peak, trough).
TABLE_II_SMOOTHER = (
    ("1953Q3", "1954Q2"),
    ("1957Q1", "1958Q1"),
    ("1960Q2", "1960Q4"),
    ("1969Q3", "1970Q4"),
    ("1974Q1", "1975Q1"),
    ("1979Q2", "1980Q3"),
    ("1981Q2", "1982Q4"),
)


@dataclass(frozen=True)
class Hamilton1989Inference:
    """Probabilities of the low-growth state ``s_t = 0`` for ``t = r+1..T``."""

    filtered_low: FloatArray
    smoothed_low: FloatArray
    loglik: float


@dataclass(frozen=True)
class Hamilton1989Fit:
    params: Hamilton1989Params
    loglik: float
    std_errors: FloatArray
    converged: bool


def load_gnp() -> tuple[list[str], FloatArray]:
    """Quarter labels and percent growth of US real GNP, 1951Q2 to 1984Q4."""
    text = resources.files("hindsight").joinpath("data/rgnp_growth.csv").read_text("utf-8")
    quarters: list[str] = []
    values: list[float] = []
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("quarter") or not line.strip():
            continue
        quarter, value = line.split(",")
        quarters.append(quarter)
        values.append(float(value))
    return quarters, np.array(values)


def expanded_transition(p: float, q: float, r: int) -> FloatArray:
    """Transition matrix of ``(s_t, s_{t-1}, ..., s_{t-r})``.

    State ``b`` is an integer whose bit ``l`` is ``s_{t-l}``. A move from ``a``
    to ``b`` is possible only when the history shifts by one place, and then it
    has the probability of the newest transition (Step 1 of section 4.2).
    """
    base = np.array([[q, 1.0 - q], [1.0 - p, p]])
    n = 2 ** (r + 1)
    mask = n - 1
    P = np.zeros((n, n))
    for a in range(n):
        for new in (0, 1):
            b = ((a << 1) & mask) | new
            P[a, b] = base[a & 1, new]
    return P


def _bits(r: int) -> FloatArray:
    n = 2 ** (r + 1)
    out = np.empty((n, r + 1))
    for b in range(n):
        for lag in range(r + 1):
            out[b, lag] = (b >> lag) & 1
    return out


def log_density(y: FloatArray, params: Hamilton1989Params) -> FloatArray:
    """Log of the density in Step 2 of section 4.2, for ``t = r+1..T``."""
    r = params.r
    bits = _bits(r)
    means = params.alpha0 + params.alpha1 * bits
    n = y.shape[0] - r
    resid = np.empty((n, bits.shape[0]))
    for t in range(n):
        window = y[t : t + r + 1][::-1]
        dev = window[None, :] - means
        resid[t] = dev[:, 0] - dev[:, 1:] @ params.phi
    out: FloatArray = -0.5 * (_LOG_2PI + (resid / params.sigma) ** 2) - np.log(params.sigma)
    return out


def infer(y: FloatArray, params: Hamilton1989Params) -> Hamilton1989Inference:
    """Filtered and full-sample smoothed probabilities of the low-growth state.

    The filter starts from the unconditional distribution of the expanded
    chain, which is what the start-up iteration on pp. 368-369 produces.
    """
    y = np.asarray(y, dtype=np.float64)
    r = params.r
    P = expanded_transition(params.p, params.q, r)
    flt = hamilton_filter(log_density(y, params), P, ergodic_probabilities(P))
    smoothed, _ = kim_smoother(flt.filtered, flt.predicted, P)
    low = _bits(r)[:, 0] == 0.0
    return Hamilton1989Inference(
        filtered_low=flt.filtered[:, low].sum(axis=1),
        smoothed_low=smoothed[:, low].sum(axis=1),
        loglik=flt.loglik,
    )


def _natural(theta: FloatArray) -> Hamilton1989Params:
    return Hamilton1989Params(
        alpha0=float(theta[0]),
        alpha1=float(theta[1]),
        p=float(1.0 / (1.0 + np.exp(-theta[2]))),
        q=float(1.0 / (1.0 + np.exp(-theta[3]))),
        sigma=float(np.exp(theta[4])),
        phi=np.asarray(theta[5:], dtype=np.float64),
    )


def _vector(params: Hamilton1989Params) -> FloatArray:
    """Parameters in the order of Table I: alpha1, alpha0, p, q, sigma, phi."""
    return np.concatenate(
        [[params.alpha1, params.alpha0, params.p, params.q, params.sigma], params.phi]
    )


def _from_vector(v: FloatArray) -> Hamilton1989Params:
    return Hamilton1989Params(
        alpha1=float(v[0]),
        alpha0=float(v[1]),
        p=float(v[2]),
        q=float(v[3]),
        sigma=float(v[4]),
        phi=np.asarray(v[5:], dtype=np.float64),
    )


def _numerical_hessian(func: object, x: FloatArray, rel_step: float = 1e-4) -> FloatArray:
    assert callable(func)
    n = x.shape[0]
    h = rel_step * np.maximum(np.abs(x), 1e-2)
    hess = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h[i]
            xpp[j] += h[j]
            xpm[i] += h[i]
            xpm[j] -= h[j]
            xmp[i] -= h[i]
            xmp[j] += h[j]
            xmm[i] -= h[i]
            xmm[j] -= h[j]
            value = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4.0 * h[i] * h[j])
            hess[i, j] = value
            hess[j, i] = value
    return hess


def fit_mle(y: FloatArray, r: int = 4) -> Hamilton1989Fit:
    """Maximise the conditional log likelihood of p. 369 numerically.

    The search is BFGS over unconstrained transforms of the parameters
    (logit for ``p`` and ``q``, log for ``sigma``). The start is neutral: the
    two means one sample standard deviation either side of the sample mean,
    ``p = q = 0.8`` and no autoregression. States are labelled so that state 1
    is the fast-growth state (``alpha1 > 0``), as on p. 369. Standard errors
    come from a finite-difference Hessian in the natural parameters.
    """
    y = np.asarray(y, dtype=np.float64)
    mean, std = float(np.mean(y)), float(np.std(y))
    logit = float(np.log(0.8 / 0.2))
    start = np.concatenate([[mean - std, 2.0 * std, logit, logit, np.log(std)], np.zeros(r)])

    def negative(theta: FloatArray) -> float:
        return -infer(y, _natural(theta)).loglik

    result = optimize.minimize(negative, start, method="BFGS", options={"gtol": 1e-7})
    params = _natural(result.x)
    if params.alpha1 < 0.0:
        params = Hamilton1989Params(
            alpha0=params.alpha0 + params.alpha1,
            alpha1=-params.alpha1,
            p=params.q,
            q=params.p,
            sigma=params.sigma,
            phi=params.phi,
        )

    def negative_natural(v: FloatArray) -> float:
        return -infer(y, _from_vector(v)).loglik

    hess = _numerical_hessian(negative_natural, _vector(params))
    std_errors = np.sqrt(np.diag(np.linalg.inv(hess)))
    return Hamilton1989Fit(
        params=params,
        loglik=-float(negative_natural(_vector(params))),
        std_errors=std_errors,
        # BFGS often stops on "precision loss" with a gradient that is already
        # flat, so judge convergence by the gradient itself
        converged=bool(result.success or float(np.max(np.abs(result.jac))) < 1e-3),
    )


def recession_spells(quarters: list[str], prob_low: FloatArray) -> list[tuple[str, str]]:
    """First and last quarter of every run with ``prob_low > 0.5`` (p. 373-374)."""
    spells: list[tuple[str, str]] = []
    start: str | None = None
    for i, quarter in enumerate(quarters):
        if prob_low[i] > 0.5:
            if start is None:
                start = quarter
        elif start is not None:
            spells.append((start, quarters[i - 1]))
            start = None
    if start is not None:
        spells.append((start, quarters[-1]))
    return spells
