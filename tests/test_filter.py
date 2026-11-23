"""Maths tests for the filter and smoother."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import statsmodels.api as sm

from hindsight.filter import (
    ergodic_probabilities,
    expected_duration,
    hamilton_filter,
    kim_smoother,
)
from hindsight.model import GaussianMS, gaussian_log_density, infer, simulate

TRUTH = GaussianMS(
    mu=np.array([0.05, -0.10]),
    sigma=np.array([0.7, 2.0]),
    P=np.array([[0.97, 0.03], [0.08, 0.92]]),
    initial=np.array([0.5, 0.5]),
)


def _steady(params: GaussianMS) -> GaussianMS:
    return GaussianMS(
        mu=params.mu, sigma=params.sigma, P=params.P, initial=ergodic_probabilities(params.P)
    )


def _brute_force(
    log_density: np.ndarray, P: np.ndarray, initial: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact filtered and smoothed probabilities by summing over every state path."""
    n, k = log_density.shape
    density = np.exp(log_density)
    filtered = np.zeros((n, k))
    smoothed = np.zeros((n, k))
    likelihood_to = np.zeros(n)
    for path in itertools.product(range(k), repeat=n):
        prob = initial[path[0]] * density[0, path[0]]
        partial = [prob]
        for t in range(1, n):
            prob *= P[path[t - 1], path[t]] * density[t, path[t]]
            partial.append(prob)
        for t in range(n):
            smoothed[t, path[t]] += prob
    # filtered at t: the same sum over paths of length t + 1
    for t in range(n):
        for path in itertools.product(range(k), repeat=t + 1):
            prob = initial[path[0]] * density[0, path[0]]
            for s in range(1, t + 1):
                prob *= P[path[s - 1], path[s]] * density[s, path[s]]
            filtered[t, path[t]] += prob
        likelihood_to[t] = filtered[t].sum()
        filtered[t] /= likelihood_to[t]
    total = smoothed[0].sum()
    return filtered, smoothed / total, float(np.log(total))


@pytest.mark.parametrize("k", [2, 3])
def test_matches_enumeration_over_all_paths(k: int) -> None:
    rng = np.random.default_rng(k)
    n = 7
    P = rng.dirichlet(np.ones(k), size=k)
    initial = rng.dirichlet(np.ones(k))
    log_density = rng.normal(size=(n, k)) * 2.0
    flt = hamilton_filter(log_density, P, initial)
    smoothed, _ = kim_smoother(flt.filtered, flt.predicted, P)
    exact_filtered, exact_smoothed, exact_loglik = _brute_force(log_density, P, initial)
    np.testing.assert_allclose(flt.filtered, exact_filtered, atol=1e-12)
    np.testing.assert_allclose(smoothed, exact_smoothed, atol=1e-12)
    assert flt.loglik == pytest.approx(exact_loglik, abs=1e-12)


def test_matches_statsmodels_two_states() -> None:
    y, _ = simulate(TRUTH, 1500, np.random.default_rng(1))
    mine = infer(y, _steady(TRUTH))
    model = sm.tsa.MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    theirs = model.smooth(
        np.array(
            [
                TRUTH.P[0, 0],
                TRUTH.P[1, 0],
                TRUTH.mu[0],
                TRUTH.mu[1],
                TRUTH.sigma[0] ** 2,
                TRUTH.sigma[1] ** 2,
            ]
        )
    )
    np.testing.assert_allclose(mine.filtered, theirs.filtered_marginal_probabilities, atol=1e-10)
    np.testing.assert_allclose(mine.smoothed, theirs.smoothed_marginal_probabilities, atol=1e-10)
    np.testing.assert_allclose(mine.predicted, theirs.predicted_marginal_probabilities, atol=1e-10)
    assert mine.loglik == pytest.approx(theirs.llf, abs=1e-8)


def test_matches_statsmodels_three_states() -> None:
    P = np.array([[0.90, 0.07, 0.03], [0.05, 0.90, 0.05], [0.10, 0.10, 0.80]])
    params = GaussianMS(
        mu=np.array([0.1, 0.0, -0.3]),
        sigma=np.array([0.5, 1.0, 2.5]),
        P=P,
        initial=ergodic_probabilities(P),
    )
    y, _ = simulate(params, 800, np.random.default_rng(3))
    mine = infer(y, params)
    model = sm.tsa.MarkovRegression(y, k_regimes=3, trend="c", switching_variance=True)
    # statsmodels orders transition parameters p[j->i] for i in 0..k-2, j in 0..k-1
    transition = [P[j, i] for i in range(2) for j in range(3)]
    theirs = model.smooth(np.array([*transition, *params.mu, *(params.sigma**2)]))
    np.testing.assert_allclose(mine.filtered, theirs.filtered_marginal_probabilities, atol=1e-10)
    np.testing.assert_allclose(mine.smoothed, theirs.smoothed_marginal_probabilities, atol=1e-10)
    assert mine.loglik == pytest.approx(theirs.llf, abs=1e-8)


def test_last_smoothed_equals_last_filtered() -> None:
    y, _ = simulate(TRUTH, 500, np.random.default_rng(2))
    result = infer(y, TRUTH)
    np.testing.assert_array_equal(result.smoothed[-1], result.filtered[-1])


def test_probabilities_sum_to_one() -> None:
    y, _ = simulate(TRUTH, 500, np.random.default_rng(4))
    result = infer(y, TRUTH)
    for prob in (result.filtered, result.predicted, result.smoothed):
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(prob >= 0.0)


def test_smoothed_transitions_are_consistent_with_marginals() -> None:
    y, _ = simulate(TRUTH, 400, np.random.default_rng(5))
    result = infer(y, TRUTH)
    # sum over j of Prob(s_t = i, s_{t+1} = j | all data) is Prob(s_t = i | all data)
    np.testing.assert_allclose(
        result.transitions.sum(axis=1), result.smoothed[:-1].sum(axis=0), atol=1e-10
    )
    np.testing.assert_allclose(
        result.transitions.sum(axis=0), result.smoothed[1:].sum(axis=0), atol=1e-10
    )


def test_independent_states_reduce_to_a_mixture() -> None:
    # with identical rows the chain has no memory: the filter is Bayes' rule
    # day by day and the future says nothing about today
    weights = np.array([0.7, 0.3])
    params = GaussianMS(mu=TRUTH.mu, sigma=TRUTH.sigma, P=np.tile(weights, (2, 1)), initial=weights)
    y, _ = simulate(params, 300, np.random.default_rng(6))
    result = infer(y, params)
    density = np.exp(gaussian_log_density(y, params.mu, params.sigma)) * weights
    posterior = density / density.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result.filtered, posterior, atol=1e-12)
    np.testing.assert_allclose(result.smoothed, posterior, atol=1e-12)
    assert result.loglik == pytest.approx(np.log(density.sum(axis=1)).sum(), abs=1e-9)


def test_single_state_likelihood_is_the_gaussian_likelihood() -> None:
    y = np.random.default_rng(7).normal(0.3, 1.7, size=200)
    log_density = gaussian_log_density(y, np.array([0.3]), np.array([1.7]))
    result = hamilton_filter(log_density, np.ones((1, 1)), np.ones(1))
    assert result.loglik == pytest.approx(log_density.sum(), abs=1e-10)
    np.testing.assert_array_equal(result.filtered, 1.0)


def test_filter_is_causal() -> None:
    y, _ = simulate(TRUTH, 300, np.random.default_rng(8))
    changed = y.copy()
    changed[200:] += 5.0
    a, b = infer(y, TRUTH), infer(changed, TRUTH)
    np.testing.assert_array_equal(a.filtered[:200], b.filtered[:200])
    assert np.max(np.abs(a.smoothed[:200] - b.smoothed[:200])) > 1e-3


def test_extreme_outlier_does_not_underflow() -> None:
    y = np.array([0.0, 0.1, 500.0, -0.1, 0.0])
    result = infer(y, TRUTH)
    assert np.all(np.isfinite(result.filtered))
    assert np.all(np.isfinite(result.smoothed))
    assert np.isfinite(result.loglik)
    assert result.filtered[2, 1] == pytest.approx(1.0)


def test_expected_duration_formula() -> None:
    assert expected_duration(0.0) == 1.0
    assert expected_duration(0.5) == 2.0
    assert expected_duration(0.9) == pytest.approx(10.0)
    # Hamilton (1989), p. 374: 4.1 and 10.5 quarters from q = 0.7550 and p = 0.9049
    assert round(expected_duration(0.7550), 1) == 4.1
    assert round(expected_duration(0.9049), 1) == 10.5
    with pytest.raises(ValueError):
        expected_duration(1.0)


def test_expected_duration_matches_simulated_run_lengths() -> None:
    _, states = simulate(TRUTH, 400_000, np.random.default_rng(9))
    change = np.flatnonzero(np.diff(states) != 0)
    lengths = np.diff(change)
    run_state = states[change[:-1] + 1]
    for state in (0, 1):
        expected = expected_duration(float(TRUTH.P[state, state]))
        observed = lengths[run_state == state].mean()
        assert observed == pytest.approx(expected, rel=0.03)


def test_ergodic_probabilities() -> None:
    p, q = 0.9049, 0.7550
    P = np.array([[q, 1 - q], [1 - p, p]])
    pi = ergodic_probabilities(P)
    # Hamilton (1989), equation (2.7): Prob(s = 1) = (1 - q) / (1 - p + 1 - q)
    assert pi[1] == pytest.approx((1 - q) / (1 - p + 1 - q), abs=1e-12)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12)
    P3 = np.random.default_rng(10).dirichlet(np.ones(4), size=4)
    pi3 = ergodic_probabilities(P3)
    np.testing.assert_allclose(pi3 @ P3, pi3, atol=1e-12)
    assert pi3.sum() == pytest.approx(1.0)


def test_predicted_is_filtered_times_transition() -> None:
    y, _ = simulate(TRUTH, 100, np.random.default_rng(11))
    result = infer(y, TRUTH)
    np.testing.assert_allclose(result.predicted[1:], result.filtered[:-1] @ TRUTH.P, atol=1e-14)
    np.testing.assert_array_equal(result.predicted[0], TRUTH.initial)


def test_input_validation() -> None:
    good = np.zeros((3, 2))
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    with pytest.raises(ValueError):
        hamilton_filter(good, np.array([[0.9, 0.2], [0.2, 0.8]]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        hamilton_filter(good, P, np.array([0.7, 0.7]))
    with pytest.raises(ValueError):
        hamilton_filter(np.zeros((3, 3)), P, np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        hamilton_filter(np.full((3, 2), np.nan), P, np.array([0.5, 0.5]))
