"""Published values from Hamilton (1989)."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from hindsight import hamilton1989 as h


@pytest.fixture(scope="module")
def gnp() -> tuple[list[str], np.ndarray]:
    return h.load_gnp()


@pytest.fixture(scope="module")
def fit(gnp: tuple[list[str], np.ndarray]) -> h.Hamilton1989Fit:
    return h.fit_mle(gnp[1])


def test_sample_is_the_papers(gnp: tuple[list[str], np.ndarray]) -> None:
    quarters, y = gnp
    # "data for U.S. real GNP, t = 1952:II to 1984:IV" after four lags from 1951:II
    assert quarters[0] == "1951Q2" and quarters[-1] == "1984Q4"
    assert len(quarters) == 135 and y.shape == (135,)


def test_table_1_estimates(fit: h.Hamilton1989Fit) -> None:
    paper = h.TABLE_I
    assert fit.converged
    assert fit.params.alpha1 == pytest.approx(paper.alpha1, abs=2e-3)
    assert fit.params.alpha0 == pytest.approx(paper.alpha0, abs=2e-3)
    assert fit.params.p == pytest.approx(paper.p, abs=2e-3)
    assert fit.params.q == pytest.approx(paper.q, abs=2e-3)
    assert fit.params.sigma == pytest.approx(paper.sigma, abs=2e-3)
    np.testing.assert_allclose(fit.params.phi, paper.phi, atol=2e-3)


def test_table_1_standard_errors(fit: h.Hamilton1989Fit) -> None:
    np.testing.assert_allclose(fit.std_errors, h.TABLE_I_SE, atol=2e-3)


def test_estimates_beat_the_printed_values(
    gnp: tuple[list[str], np.ndarray], fit: h.Hamilton1989Fit
) -> None:
    at_paper = h.infer(gnp[1], h.TABLE_I).loglik
    assert fit.loglik >= at_paper
    assert fit.loglik - at_paper < 1e-2


def test_table_2_recession_dates(gnp: tuple[list[str], np.ndarray], fit: h.Hamilton1989Fit) -> None:
    quarters, y = gnp
    smoothed = h.infer(y, fit.params).smoothed_low
    spells = h.recession_spells(quarters[4:], smoothed)
    assert tuple(spells) == h.TABLE_II_SMOOTHER


def test_matches_statsmodels_at_table_1(gnp: tuple[list[str], np.ndarray]) -> None:
    y = gnp[1]
    paper = h.TABLE_I
    mine = h.infer(y, paper)
    model = sm.tsa.MarkovAutoregression(y, k_regimes=2, order=4, switching_ar=False)
    theirs = model.smooth(
        np.concatenate(
            [
                [paper.q, 1.0 - paper.p, paper.alpha0, paper.alpha0 + paper.alpha1],
                [paper.sigma**2],
                paper.phi,
            ]
        )
    )
    np.testing.assert_allclose(
        mine.filtered_low, theirs.filtered_marginal_probabilities[:, 0], atol=1e-10
    )
    np.testing.assert_allclose(
        mine.smoothed_low, theirs.smoothed_marginal_probabilities[:, 0], atol=1e-10
    )
    assert mine.loglik == pytest.approx(theirs.llf, abs=1e-9)


def test_expanded_transition_matrix() -> None:
    P = h.expanded_transition(0.9, 0.75, 4)
    assert P.shape == (32, 32)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert np.count_nonzero(P) == 64
    # with no lags it is the two-state chain itself
    np.testing.assert_allclose(h.expanded_transition(0.9, 0.75, 0), [[0.75, 0.25], [0.1, 0.9]])


def test_no_autoregression_reduces_to_the_two_state_model() -> None:
    from hindsight.filter import ergodic_probabilities
    from hindsight.model import GaussianMS, infer

    y = np.random.default_rng(1).normal(size=60)
    params = h.Hamilton1989Params(
        alpha0=-0.4, alpha1=1.5, p=0.9, q=0.75, sigma=0.8, phi=np.zeros(0)
    )
    P = np.array([[0.75, 0.25], [0.1, 0.9]])
    simple = infer(
        y,
        GaussianMS(
            mu=np.array([-0.4, 1.1]),
            sigma=np.array([0.8, 0.8]),
            P=P,
            initial=ergodic_probabilities(P),
        ),
    )
    out = h.infer(y, params)
    np.testing.assert_allclose(out.filtered_low, simple.filtered[:, 0], atol=1e-12)
    np.testing.assert_allclose(out.smoothed_low, simple.smoothed[:, 0], atol=1e-12)
    assert out.loglik == pytest.approx(simple.loglik, abs=1e-10)


def test_recession_spells() -> None:
    quarters = ["a", "b", "c", "d", "e"]
    prob = np.array([0.9, 0.2, 0.6, 0.7, 0.8])
    assert h.recession_spells(quarters, prob) == [("a", "a"), ("c", "e")]
