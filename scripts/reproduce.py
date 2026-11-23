"""Rebuild every figure and table in results/.

    uv run python scripts/fetch_data.py
    uv run python scripts/reproduce.py          # full run, a few minutes
    uv run python scripts/reproduce.py --fast   # smoke run for CI, well under two minutes

The fast run uses a shorter sample, yearly refits and fewer resamples, and
writes to results_fast/ so that the committed results are never overwritten
by numbers from the smaller run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from hindsight import experiments as ex
from hindsight import plots
from hindsight.data import load_french_daily, sha256

ROOT = Path(__file__).resolve().parents[1]


def write_table(table: pd.DataFrame, path: Path, digits: int = 4) -> None:
    """Write a CSV and a markdown copy with a fixed number of decimals."""
    table.to_csv(
        path.with_suffix(".csv"), index=False, float_format=f"%.{digits}f", lineterminator="\n"
    )
    shown = table.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    rule = "|" + "|".join(" --- " for _ in shown.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    path.with_suffix(".md").write_text(
        "\n".join([header, rule, *body]) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="smoke run on a smaller problem")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = ex.FAST if args.fast else ex.FULL
    out: Path = args.out_dir or (ROOT / ("results_fast" if args.fast else "results"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    returns = load_french_daily(args.data_dir)

    print("1/4 Hamilton (1989) on US GNP")
    hamilton = ex.hamilton_replication()
    write_table(hamilton.table1, out / "hamilton1989_table1")
    write_table(hamilton.table2, out / "hamilton1989_table2")
    write_table(hamilton.durations, out / "hamilton1989_durations")
    write_table(hamilton.probabilities, out / "hamilton1989_probabilities")
    plots.hamilton_probabilities(hamilton, out / "hamilton1989_recession_probability.png")

    print("2/4 filter and smoother against statsmodels")
    market = returns["mkt_rf"] if config.start is None else returns.loc[config.start :, "mkt_rf"]
    check = ex.statsmodels_check(market.to_numpy(), config.seed)
    with pd.option_context("display.float_format", "{:.3e}".format):
        print(
            check[["sample", "max_abs_diff_filtered", "max_abs_diff_smoothed", "loglik_abs_diff"]]
        )
    # differences are at rounding level and not stable in the last digits,
    # so the committed table records only that they are below a bound
    bound = 1e-9
    summary = check[["sample", "observations", "loglik_here"]].copy()
    summary["observations"] = summary["observations"].astype(int)
    for column in ("max_abs_diff_filtered", "max_abs_diff_smoothed", "loglik_abs_diff"):
        summary[f"{column}_below_1e-9"] = check[column] < bound
    write_table(summary, out / "statsmodels_check")

    print("3/4 one strategy, three information sets")
    tables = ex.three_way(returns, config)
    sample = pd.concat(
        [
            tables.sample,
            pd.DataFrame([{"item": "sha256 of the data file", "value": sha256(args.data_dir)}]),
        ],
        ignore_index=True,
    )
    write_table(sample, out / "sample")
    write_table(tables.performance, out / "performance")
    write_table(tables.gaps, out / "sharpe_gaps")
    write_table(tables.subperiods, out / "subperiods")
    write_table(tables.agreement, out / "label_agreement", digits=2)
    write_table(tables.parameters, out / "parameters_through_time", digits=6)
    plots.sharpe_three_ways(tables, out / "sharpe_three_ways.png")
    plots.wk_three_ways(tables, out / "wk_three_ways.png")
    plots.equity_curves(tables, out / "equity_curves.png")
    plots.parameters_through_time(tables, out / "parameters_through_time.png")
    plots.probability_zoom(tables, "2007-01-01", "2009-12-31", out / "probabilities_2007_2009.png")

    print("4/4 Wasserstein k-means on synthetic paths")
    synthetic, per_run = ex.wk_synthetic(config)
    write_table(synthetic, out / "wk_synthetic_accuracy", digits=2)
    write_table(per_run, out / "wk_synthetic_runs")

    print(tables.performance[["strategy", "sharpe", "sharpe_ci_low", "sharpe_ci_high"]].to_string())
    print(f"wrote {out} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
