"""Wasserstein k-means on empirical return distributions.

Follows Horvath, Issa and Muguruza, "Clustering market regimes using the
Wasserstein distance". A stream of returns is cut into overlapping windows
(the "lift" of their Definition 1.2), each window is treated as an empirical
measure, and k-means is run with the p-Wasserstein distance (their equation
21) and the Wasserstein barycenter (Proposition 2.6) in place of the Euclidean
distance and the mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hindsight.filter import FloatArray

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class WKMeansResult:
    """Centroids are sorted by standard deviation, smallest first."""

    centroids: FloatArray
    labels: IntArray
    n_iter: int
    converged: bool
    inertia: float


def lift(x: FloatArray, h1: int, h2: int) -> tuple[FloatArray, IntArray]:
    """Cut ``x`` into windows of length ``h1`` that overlap by ``h2``.

    Returns the ``(M, h1)`` array of windows and the index of the first
    element of each. Consecutive windows start ``h1 - h2`` apart, which is
    what the paper's text and experiments describe; see ``PAPER.md`` for how
    this differs from equation (3) as printed.
    """
    x = np.asarray(x, dtype=np.float64)
    if not 0 <= h2 < h1:
        raise ValueError("need 0 <= h2 < h1")
    if x.shape[0] < h1:
        raise ValueError("series shorter than one window")
    stride = h1 - h2
    starts = np.arange(0, x.shape[0] - h1 + 1, stride, dtype=np.int64)
    windows = np.stack([x[s : s + h1] for s in starts])
    return windows, starts


def trailing_windows(x: FloatArray, h1: int) -> FloatArray:
    """Row ``t`` holds ``x[t-h1+1..t]``; rows before ``h1 - 1`` are NaN."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full((x.shape[0], h1), np.nan)
    if x.shape[0] >= h1:
        out[h1 - 1 :] = np.lib.stride_tricks.sliding_window_view(x, h1)
    return out


def wasserstein_distance(a: FloatArray, b: FloatArray, p: int = 1) -> float:
    """p-Wasserstein distance between two samples of equal size, equation (21)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.shape != b.shape:
        raise ValueError("samples must have the same number of atoms")
    return float(np.mean(np.abs(a - b) ** p) ** (1.0 / p))


def _pairwise(atoms: FloatArray, centroids: FloatArray, p: int) -> FloatArray:
    """``(M, k)`` distances between sorted windows and sorted centroids."""
    diff = np.abs(atoms[:, None, :] - centroids[None, :, :])
    out: FloatArray = np.mean(diff**p, axis=2) ** (1.0 / p)
    return out


def barycenter(atoms: FloatArray, p: int = 1) -> FloatArray:
    """Wasserstein barycenter of empirical measures with equal numbers of atoms.

    ``atoms`` is ``(M, N)`` with each row sorted. For ``p = 1`` the j-th atom
    of the barycenter is the median of the j-th atoms (Proposition 2.6). For
    ``p = 2`` it is their mean, the standard result for quantile functions.
    """
    if p == 1:
        out: FloatArray = np.median(atoms, axis=0)
    elif p == 2:
        out = np.mean(atoms, axis=0)
    else:
        raise ValueError("only p = 1 and p = 2 are implemented")
    return out


def assign(windows: FloatArray, centroids: FloatArray, p: int = 1) -> IntArray:
    """Index of the nearest centroid for every window. Windows need not be sorted."""
    atoms = np.sort(np.asarray(windows, dtype=np.float64), axis=1)
    labels: IntArray = np.argmin(_pairwise(atoms, centroids, p), axis=1).astype(np.int64)
    return labels


def _single_run(
    atoms: FloatArray, k: int, p: int, rng: np.random.Generator, tol: float, max_iter: int
) -> WKMeansResult:
    m = atoms.shape[0]
    centroids = atoms[rng.choice(m, size=k, replace=False)].copy()
    labels = np.zeros(m, dtype=np.int64)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007
        dist = _pairwise(atoms, centroids, p)
        labels = np.argmin(dist, axis=1).astype(np.int64)
        new = centroids.copy()
        for c in range(k):
            members = atoms[labels == c]
            if members.shape[0] > 0:
                new[c] = barycenter(members, p)
            else:
                # the paper does not say what to do with an empty cluster;
                # restart it at the window farthest from its own centroid
                new[c] = atoms[int(np.argmax(dist[np.arange(m), labels]))]
        # loss of equation (23): total movement of the centroids
        loss = float(sum(wasserstein_distance(centroids[c], new[c], p) for c in range(k)))
        centroids = new
        if loss < tol:
            converged = True
            break
    dist = _pairwise(atoms, centroids, p)
    labels = np.argmin(dist, axis=1).astype(np.int64)
    order = np.argsort(np.std(centroids, axis=1), kind="stable")
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    return WKMeansResult(
        centroids=centroids[order],
        labels=remap[labels],
        n_iter=n_iter,
        converged=converged,
        inertia=float(dist[np.arange(m), labels].sum()),
    )


def wk_means(
    windows: FloatArray,
    k: int = 2,
    p: int = 1,
    seed: int = 0,
    n_init: int = 1,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> WKMeansResult:
    """Algorithm 1 of the paper.

    Centroids start as ``k`` windows drawn at random. With ``n_init > 1`` the
    algorithm is run that many times and the run with the smallest total
    distance from windows to their centroids is kept; the paper uses a single
    run.
    """
    atoms = np.sort(np.asarray(windows, dtype=np.float64), axis=1)
    if atoms.ndim != 2 or atoms.shape[0] < k:
        raise ValueError("need at least k windows")
    rng = np.random.default_rng(seed)
    best: WKMeansResult | None = None
    for _ in range(n_init):
        run = _single_run(atoms, k, p, rng, tol, max_iter)
        if best is None or run.inertia < best.inertia:
            best = run
    assert best is not None
    return best


def membership_counts(labels: IntArray, starts: IntArray, h1: int, n: int, k: int) -> FloatArray:
    """For each observation, how many of the windows containing it fall in each cluster.

    This is the vector ``Y^i`` of section 3.3.
    """
    counts = np.zeros((n, k))
    for label, start in zip(labels, starts, strict=True):
        counts[start : start + h1, label] += 1.0
    return counts


def accuracy_scores(counts: FloatArray, regime_on: NDArray[np.bool_]) -> tuple[float, float, float]:
    """Total, regime-on and regime-off accuracy, equations (33), (32) and (31).

    ``counts`` has two columns: column 0 is the standard regime and column 1
    the regime change. ``regime_on`` marks the observations inside a regime
    change.
    """
    if counts.shape[1] != 2:
        raise ValueError("accuracy scores are defined for two clusters")
    off = ~regime_on
    correct_off = float(counts[off, 0].sum())
    correct_on = float(counts[regime_on, 1].sum())
    total_off = float(counts[off].sum())
    total_on = float(counts[regime_on].sum())
    total = (correct_off + correct_on) / (total_off + total_on)
    rons = correct_on / total_on if total_on > 0.0 else float("nan")
    rofs = correct_off / total_off if total_off > 0.0 else float("nan")
    return total, rons, rofs
