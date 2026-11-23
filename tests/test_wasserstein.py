"""Maths tests for the Wasserstein distance, barycenter and k-means."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wasserstein_distance as scipy_w1

from hindsight.wasserstein import (
    accuracy_scores,
    assign,
    barycenter,
    lift,
    membership_counts,
    trailing_windows,
    wasserstein_distance,
    wk_means,
)


def test_w1_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(size=35), rng.standard_t(3, size=35)
        assert wasserstein_distance(a, b, 1) == pytest.approx(scipy_w1(a, b), abs=1e-12)


def test_distance_is_a_metric_on_samples() -> None:
    rng = np.random.default_rng(2)
    a, b, c = rng.normal(size=(3, 20))
    for p in (1, 2):
        assert wasserstein_distance(a, a, p) == 0.0
        assert wasserstein_distance(a, b, p) == pytest.approx(wasserstein_distance(b, a, p))
        assert wasserstein_distance(a, c, p) <= wasserstein_distance(
            a, b, p
        ) + wasserstein_distance(b, c, p)
        # order of the observations inside a window does not matter
        assert wasserstein_distance(a, rng.permutation(b), p) == pytest.approx(
            wasserstein_distance(a, b, p)
        )


def test_shift_and_scale_closed_forms() -> None:
    a = np.random.default_rng(3).normal(size=50)
    for p in (1, 2):
        # moving every atom by c costs exactly |c|
        assert wasserstein_distance(a, a + 0.3, p) == pytest.approx(0.3)
    # scaling about zero: W_p(a, s a) = |s - 1| times the p-th absolute moment
    assert wasserstein_distance(a, 2.0 * a, 1) == pytest.approx(np.mean(np.abs(a)))
    assert wasserstein_distance(a, 2.0 * a, 2) == pytest.approx(np.sqrt(np.mean(a**2)))


def test_w2_between_gaussian_quantiles() -> None:
    # for two normals W_2^2 = (m1 - m2)^2 + (s1 - s2)^2; check on exact quantiles
    from scipy.stats import norm

    u = (np.arange(20_000) + 0.5) / 20_000
    a = norm.ppf(u, 0.0, 1.0)
    b = norm.ppf(u, 0.5, 2.0)
    assert wasserstein_distance(a, b, 2) == pytest.approx(np.sqrt(0.25 + 1.0), rel=2e-3)


def test_barycenter_minimises_total_distance() -> None:
    rng = np.random.default_rng(4)
    atoms = np.sort(rng.normal(size=(9, 12)), axis=1)

    def total(candidate: np.ndarray, p: int) -> float:
        return float(sum(wasserstein_distance(row, candidate, p) ** p for row in atoms))

    for p in (1, 2):
        centre = barycenter(atoms, p)
        best = total(centre, p)
        for _ in range(200):
            trial = np.sort(centre + rng.normal(scale=0.05, size=centre.shape))
            assert total(trial, p) >= best - 1e-12
    np.testing.assert_array_equal(barycenter(atoms, 1), np.median(atoms, axis=0))
    with pytest.raises(ValueError):
        barycenter(atoms, 3)


def test_lift_windows() -> None:
    x = np.arange(100.0)
    windows, starts = lift(x, 35, 28)
    # consecutive windows are 7 apart, so one point sits in at most 5 windows
    np.testing.assert_array_equal(np.diff(starts), 7)
    assert windows.shape == (10, 35)
    np.testing.assert_array_equal(windows[3], x[21:56])
    counts = membership_counts(np.zeros(10, dtype=np.int64), starts, 35, 100, 2)
    assert counts[:, 0].max() == 5
    no_overlap, starts0 = lift(x, 35, 0)
    assert no_overlap.shape == (2, 35) and list(starts0) == [0, 35]
    with pytest.raises(ValueError):
        lift(x, 35, 35)


def test_trailing_windows() -> None:
    x = np.arange(10.0)
    out = trailing_windows(x, 4)
    assert np.all(np.isnan(out[:3]))
    np.testing.assert_array_equal(out[3], [0, 1, 2, 3])
    np.testing.assert_array_equal(out[9], [6, 7, 8, 9])


def test_k_means_separates_two_volatility_regimes() -> None:
    rng = np.random.default_rng(5)
    calm = rng.normal(scale=1.0, size=(60, 35))
    wild = rng.normal(scale=3.0, size=(40, 35))
    result = wk_means(np.vstack([calm, wild]), k=2, seed=1)
    assert result.converged
    assert np.all(result.labels[:60] == 0) and np.all(result.labels[60:] == 1)
    assert result.centroids[0].std() < result.centroids[1].std()
    np.testing.assert_array_equal(assign(wild, result.centroids), 1)


def test_k_means_is_deterministic_and_n_init_never_hurts() -> None:
    windows = np.random.default_rng(6).standard_t(3, size=(200, 20))
    a, b = wk_means(windows, k=3, seed=7), wk_means(windows, k=3, seed=7)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    many = wk_means(windows, k=3, seed=7, n_init=8)
    assert many.inertia <= a.inertia + 1e-12


def test_accuracy_scores_by_hand() -> None:
    # six observations; the last two are inside a regime change
    counts = np.array([[2, 0], [1, 1], [2, 0], [2, 0], [0, 2], [1, 1]], dtype=float)
    regime_on = np.array([False, False, False, False, True, True])
    total, on, off = accuracy_scores(counts, regime_on)
    assert off == pytest.approx(7 / 8)
    assert on == pytest.approx(3 / 4)
    assert total == pytest.approx(10 / 12)
