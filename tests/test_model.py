"""Tests for the Gaussian switching model and its EM estimator."""

from __future__ import annotations

import numpy as np
import pytest

from hindsight.filter import ergodic_probabilities
from hindsight.model import (
    GaussianMS,
    default_start,
    em_step,
    fit_em,
    gaussian_log_density,
    infer,
    simulate,
)

TRUTH = GaussianMS(
    mu=np.array([0.08, -0.15]),
    sigma=np.array([0.6, 1.9]),
    P=np.array([[0.98, 0.02], [0.05, 0.95]]),
    initial=np.array([0.7, 0.3]),
)


def test_log_density_matches_closed_form() -> None:
    from scipy.stats import norm

    y = np.array([-1.0, 0.0, 2.5])
    out = gaussian_log_density(y, TRUTH.mu, TRUTH.sigma)
    for j in range(2):
        np.testing.assert_allclose(out[:, j], norm.logpdf(y, TRUTH.mu[j], TRUTH.sigma[j]))


def test_em_never_lowers_the_likelihood() -> None:
    y, _ = simulate(TRUTH, 3000, np.random.default_rng(1))
    result = fit_em(y)
    assert result.converged
    assert np.all(np.diff(result.loglik_path) >= -1e-8)
    assert result.loglik >= infer(y, TRUTH).loglik


def test_em_recovers_the_parameters() -> None:
    y, _ = simulate(TRUTH, 60_000, np.random.default_rng(2))
    params = fit_em(y).params
    np.testing.assert_allclose(params.mu, TRUTH.mu, atol=0.02)
    np.testing.assert_allclose(params.sigma, TRUTH.sigma, rtol=0.03)
    np.testing.assert_allclose(np.diag(params.P), np.diag(TRUTH.P), atol=0.01)


def test_em_fixed_point_has_zero_score() -> None:
    # at the EM limit the likelihood is flat in every parameter
    y, _ = simulate(TRUTH, 4000, np.random.default_rng(3))
    params = fit_em(y, tol=1e-15).params

    def loglik(mu0: float, s0: float, p00: float) -> float:
        P = params.P.copy()
        P[0] = [p00, 1.0 - p00]
        trial = GaussianMS(
            mu=np.array([mu0, params.mu[1]]),
            sigma=np.array([s0, params.sigma[1]]),
            P=P,
            initial=params.initial,
        )
        return infer(y, trial).loglik

    base = (float(params.mu[0]), float(params.sigma[0]), float(params.P[0, 0]))
    for i in range(3):
        h = 1e-5
        up, down = list(base), list(base)
        up[i] += h
        down[i] -= h
        slope = (loglik(*up) - loglik(*down)) / (2 * h)
        assert abs(slope) < 0.05, (i, slope)


def test_states_are_sorted_by_volatility() -> None:
    y, _ = simulate(TRUTH, 3000, np.random.default_rng(4))
    flipped = GaussianMS(
        mu=TRUTH.mu[::-1].copy(),
        sigma=TRUTH.sigma[::-1].copy(),
        P=TRUTH.P[::-1, ::-1].copy(),
        initial=TRUTH.initial[::-1].copy(),
    )
    params = fit_em(y, start=flipped).params
    assert params.sigma[0] < params.sigma[1]
    np.testing.assert_allclose(params.P.sum(axis=1), 1.0)


def test_em_step_matches_weighted_moments() -> None:
    y, _ = simulate(TRUTH, 500, np.random.default_rng(5))
    new, loglik = em_step(y, TRUTH, sigma_floor=0.0)
    inference = infer(y, TRUTH)
    w = inference.smoothed
    assert loglik == inference.loglik
    np.testing.assert_allclose(new.mu, (w * y[:, None]).sum(0) / w.sum(0))
    np.testing.assert_allclose(new.P.sum(axis=1), 1.0)
    np.testing.assert_array_equal(new.initial, w[0])


def test_default_start_is_deterministic_and_ordered() -> None:
    y = np.random.default_rng(6).standard_t(4, size=1000)
    a, b = default_start(y, 3), default_start(y, 3)
    np.testing.assert_array_equal(a.sigma, b.sigma)
    assert np.all(np.diff(a.sigma) > 0)
    np.testing.assert_allclose(a.P.sum(axis=1), 1.0)


def test_simulated_state_frequencies_match_ergodic() -> None:
    _, states = simulate(TRUTH, 300_000, np.random.default_rng(7))
    pi = ergodic_probabilities(TRUTH.P)
    assert np.mean(states == 1) == pytest.approx(pi[1], abs=0.01)


def test_fit_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        fit_em(np.array([1.0, np.nan, 2.0, 3.0]))
    with pytest.raises(ValueError):
        fit_em(np.array([1.0, 2.0]))
