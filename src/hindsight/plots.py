"""Figures for ``results/``. One chart per claim, no decoration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hindsight import experiments as ex
from hindsight import hamilton1989

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
NEUTRAL = "#898781"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        # fixed metadata so that the same data gives the same file
        "svg.hashsalt": "hindsight",
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


FIXED_LABELS_NOTE = (
    "Black line: 95% interval from a block bootstrap of daily returns with the labels held fixed."
)


def _sharpe_bars(
    table: pd.DataFrame,
    rows: list[tuple[str, str, str]],
    title: str,
    path: Path,
    note: str = FIXED_LABELS_NOTE,
    intervals: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Horizontal bars of Sharpe ratios with bootstrap intervals.

    The intervals are the fixed-label ones in ``table`` unless ``intervals``
    gives a pair for each strategy.
    """
    indexed = table.set_index("strategy").copy()
    if intervals is not None:
        for name, (lo, hi) in intervals.items():
            indexed.loc[name, "sharpe_ci_low"] = lo
            indexed.loc[name, "sharpe_ci_high"] = hi
        indexed = indexed.loc[[name for name, _, _ in rows]]
    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(rows) + 1.5))
    positions = np.arange(len(rows))[::-1]
    for y, (name, label, colour) in zip(positions, rows, strict=True):
        row = indexed.loc[name]
        value = float(row["sharpe"])
        ax.barh(y, value, height=0.55, color=colour, zorder=2)
        ax.plot(
            [row["sharpe_ci_low"], row["sharpe_ci_high"]],
            [y, y],
            color=INK,
            linewidth=1.2,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.text(
            float(row["sharpe_ci_high"]) + 0.03,
            y,
            f"{value:.2f}",
            va="center",
            ha="left",
            color=INK,
            fontsize=10,
        )
        ax.text(-0.03, y, label, va="center", ha="right", color=INK, fontsize=10)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.0, max(float(indexed["sharpe_ci_high"].max()) * 1.12, 1.0))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Annualised Sharpe ratio, before costs")
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.text(
        0.02,
        0.02,
        note,
        ha="left",
        color=INK_2,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.36, right=0.96, top=0.88, bottom=0.24)
    _save(fig, path)


def sharpe_three_ways(tables: ex.ThreeWayTables, refit: pd.DataFrame, path: Path) -> None:
    """The headline chart, with intervals from the bootstrap that refits the model."""
    sample = tables.sample.set_index("item")["value"]
    span = f"{sample['first day evaluated'][:4]} to {sample['last day in sample'][:4]}"
    boot = refit.set_index("quantity")
    keys = {
        ex.HMM_SMOOTHED: "sharpe_smoothed",
        ex.HMM_FILTERED_FULL: "sharpe_filtered_full",
        ex.HMM_FILTERED_PAST: "sharpe_filtered_past",
        ex.BUY_AND_HOLD: "sharpe_buy_and_hold",
    }
    intervals = {
        name: (float(boot.loc[key, "ci_low"]), float(boot.loc[key, "ci_high"]))
        for name, key in keys.items()
    }
    _sharpe_bars(
        tables.performance,
        [
            (ex.HMM_SMOOTHED, "Smoothed labels\n(full sample)", BLUE),
            (ex.HMM_FILTERED_FULL, "Filtered labels,\nfull-sample parameters", BLUE),
            (ex.HMM_FILTERED_PAST, "Filtered labels,\npast-only parameters", BLUE),
            (ex.BUY_AND_HOLD, "Buy and hold", NEUTRAL),
        ],
        f"One regime strategy, three information sets (US market, {span})",
        path,
        note=(
            "Black line: 95% interval from a stationary bootstrap of daily returns, "
            "model refitted on every resample."
        ),
        intervals=intervals,
    )


def wk_three_ways(tables: ex.ThreeWayTables, path: Path) -> None:
    _sharpe_bars(
        tables.performance,
        [
            (ex.WK_CENTRED, "Labels from all windows\ncontaining the day", BLUE),
            (ex.WK_TRAILING_FULL, "Trailing window,\nfull-sample centroids", BLUE),
            (ex.WK_TRAILING_PAST, "Trailing window,\npast-only centroids", BLUE),
            (ex.BUY_AND_HOLD, "Buy and hold", NEUTRAL),
        ],
        "The same split for Wasserstein k-means labels",
        path,
    )


def equity_curves(tables: ex.ThreeWayTables, path: Path) -> None:
    """Growth of one unit of excess wealth under each information set."""
    daily = tables.daily.dropna(subset=[f"ret::{ex.BUY_AND_HOLD}"])
    series = [
        (ex.HMM_SMOOTHED, "Smoothed labels", BLUE),
        (ex.HMM_FILTERED_FULL, "Filtered, full-sample parameters", ORANGE),
        (ex.HMM_FILTERED_PAST, "Filtered, past-only parameters", AQUA),
        (ex.BUY_AND_HOLD, "Buy and hold", NEUTRAL),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for name, label, colour in series:
        wealth = np.cumprod(1.0 + daily[f"ret::{name}"].to_numpy())
        ax.plot(daily.index, wealth, color=colour, linewidth=1.6, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative excess return, start = 1 (log scale)")
    ax.set_title("Wealth in excess of bills under each information set")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)


def probability_zoom(tables: ex.ThreeWayTables, start: str, end: str, path: Path) -> None:
    """Smoothed against filtered probability of the calm state over one episode."""
    daily = tables.daily.loc[start:end]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True, height_ratios=[1, 1.4])
    wealth = np.cumprod(1.0 + daily["mkt_rf"].to_numpy())
    axes[0].plot(daily.index, wealth, color=INK_2, linewidth=1.4)
    axes[0].set_ylabel("Market, excess of bills\n(start = 1)")
    axes[0].set_title(f"What the smoother knows that the filter does not, {start[:4]} to {end[:4]}")
    axes[1].plot(daily.index, daily["p_calm_smoothed"], color=BLUE, linewidth=1.6, label="Smoothed")
    axes[1].plot(
        daily.index,
        daily["p_calm_filtered_past"],
        color=ORANGE,
        linewidth=1.2,
        label="Filtered, past-only parameters",
    )
    axes[1].axhline(0.5, color=AXIS, linewidth=1.0)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Probability the day is calm")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, borderaxespad=0.2)
    fig.tight_layout()
    _save(fig, path)


def parameters_through_time(tables: ex.ThreeWayTables, path: Path) -> None:
    """Expanding-window estimates against the full-sample estimate."""
    params = tables.parameters
    full = params[params["fit"] == "full sample"].iloc[0]
    past = params[params["fit"] == "past only"]
    dates = pd.to_datetime(past["last_day"])
    panels = [
        ("sigma_calm_pct", "Calm state: daily volatility (%)"),
        ("sigma_turbulent_pct", "Turbulent state: daily volatility (%)"),
        ("p_stay_calm", "Calm state: probability of staying"),
        ("p_stay_turbulent", "Turbulent state: probability of staying"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), sharex=True)
    for ax, (column, title) in zip(axes.ravel(), panels, strict=True):
        ax.plot(dates, past[column], color=BLUE, linewidth=1.6, label="Fitted on the past only")
        ax.axhline(
            float(full[column]), color=INK_2, linewidth=1.0, linestyle="--", label="Full sample"
        )
        ax.set_title(title, fontsize=9.5)
    axes[0, 0].legend(loc="best", fontsize=8.5)
    fig.suptitle(
        "Parameters known at each date against the full-sample values",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, path)


def hamilton_probabilities(tables: ex.HamiltonTables, path: Path) -> None:
    """Hamilton (1989) Figure 1 with the full-sample smoother in the lower panel."""
    prob = tables.probabilities
    quarters = prob["quarter"].tolist()
    x = np.array([int(q[:4]) + (int(q[5]) - 1) / 4.0 for q in quarters])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    panels = [
        ("filtered_prob_recession", "Filtered: data up to the quarter"),
        ("smoothed_prob_recession", "Smoothed: all data to 1984Q4"),
    ]
    for ax, (column, title) in zip(axes, panels, strict=True):
        for first, last in hamilton1989.TABLE_II_SMOOTHER:
            lo = int(first[:4]) + (int(first[5]) - 1) / 4.0
            hi = int(last[:4]) + (int(last[5]) - 1) / 4.0
            ax.axvspan(lo - 0.125, hi + 0.125, color=GRID, zorder=0)
        ax.plot(x, prob[column], color=BLUE, linewidth=1.6)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Probability of the\nfalling-GNP state")
        ax.set_title(title, fontsize=9.5)
    axes[1].set_xlabel("Shaded: recessions in the smoother columns of Hamilton's Table II")
    fig.suptitle(
        "Hamilton (1989), US real GNP: probability of recession",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, path)


def cross_section_gaps(table: pd.DataFrame, path: Path) -> None:
    """The smoother gap and the parameter gap for every series, one row per rule."""
    rules = [
        (ex.SPEC_K2, "2 states"),
        (ex.SPEC_K3_AVOID, "3 states, out of the\nmost volatile state"),
        (ex.SPEC_K3_CALM, "3 states, in only the\nleast volatile state"),
    ]
    kinds = [
        ("gap_smoother", "From the smoother (smoothed minus filtered)", BLUE, 0.16),
        ("gap_parameters", "From the parameters (full-sample minus past-only)", ORANGE, -0.16),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for row, (rule, _) in enumerate(rules):
        part = table[table["rule"] == rule]
        base = len(rules) - 1 - row
        for column, label, colour, offset in kinds:
            values = part[column].to_numpy()
            ax.scatter(
                values,
                np.full(values.shape, base + offset),
                s=34,
                color=colour,
                alpha=0.75,
                linewidths=0,
                zorder=3,
                label=label if row == 0 else None,
            )
            market = part.loc[part["series"] == part["series"].iloc[0], column]
            ax.scatter(
                market,
                [base + offset],
                s=70,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.2,
                zorder=4,
                label="The market index" if row == 0 and column == "gap_smoother" else None,
            )
            ax.plot(
                [float(np.median(values))] * 2,
                [base + offset - 0.11, base + offset + 0.11],
                color=INK,
                linewidth=1.4,
                zorder=5,
            )
    ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([label for _, label in rules][::-1])
    ax.set_ylim(-0.6, len(rules) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Gap in annualised Sharpe ratio, before costs. Short black line: median.")
    ax.set_title(f"Where the look-ahead comes from, {table['series'].nunique()} return series")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, path)
