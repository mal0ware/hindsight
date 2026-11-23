# Hindsight

**How much of a regime-switching backtest is the model, and how much is knowing the future?**

On 90 years of daily US market returns, a two-state regime strategy has a Sharpe ratio of 1.24 when it trades on smoothed labels and 0.53 when the model only knows the past. Buy and hold scores 0.52 over the same days. Of the 0.71 gap, 0.68 comes from the smoother reading later days and 0.03 from fitting the parameters on the full sample.

![Sharpe ratio of one regime strategy under three information sets](results/sharpe_three_ways.png)

## The question

A hidden Markov model labels each day calm or turbulent. Many backtests use labels computed from the full history. How much of the reported performance disappears when the model only knows what it could have known?

## Reproduce

```
uv sync
uv run python scripts/fetch_data.py
uv run python scripts/reproduce.py
```

The full run takes about six minutes and rewrites every file in `results/`. `uv run python scripts/reproduce.py --fast` runs a smaller version in a few seconds and writes to `results_fast/`.

## What was built

- Hamilton's filter and the backward smoother usually credited to Kim, written from scratch in `src/hindsight/filter.py` and checked against statsmodels and against a sum over every state path.
- A two-state Gaussian switching model fitted by EM (`model.py`).
- Hamilton's 1989 GNP model with four autoregressive lags, run through the same filter (`hamilton1989.py`).
- One strategy run three ways (`strategy.py`): smoothed labels; filtered labels with full-sample parameters; filtered labels with parameters refitted each month on the past only.
- Wasserstein k-means on rolling return distributions (`wasserstein.py`), with the same three-way split.

## The result

The strategy holds the market tomorrow if today is labelled calm, and holds bills otherwise. The calm state is the low-volatility state of a two-state model fitted to daily excess returns. All versions are scored on the same 23,337 days, 1936-08-01 to 2026-09-30, before costs.

| Labels | Sharpe | 95% interval | Switches | Sharpe at 10 bp per switch |
| --- | --- | --- | --- | --- |
| Smoothed (full sample) | 1.24 | 1.02 to 1.45 | 418 | 1.19 |
| Filtered, full-sample parameters | 0.56 | 0.33 to 0.80 | 1,312 | 0.42 |
| Filtered, past-only parameters | 0.53 | 0.30 to 0.76 | 1,372 | 0.38 |
| Buy and hold | 0.52 | 0.31 to 0.75 | 0 | 0.52 |

The smoothed version is in the market on 79.8% of days and the past-only version on 80.8%, and the two agree on the label for 92.7% of days. The whole difference in Sharpe ratio comes from the remaining 7.3%, which sit around regime changes. In the chart below the smoothed probability moves ahead of the filtered one at the start and end of turbulent spells, because it has seen the days that follow. The filtered probability also jumps back and forth inside a spell, so the filtered labels switch three times as often.

![Smoothed and filtered probability of the calm state, 2007 to 2009](results/probabilities_2007_2009.png)

Parameter look-ahead is small here. Parameters fitted on the past only give a Sharpe ratio 0.03 lower than full-sample parameters with the same filter, and the interval for that difference (-0.02 to 0.09) includes zero.

After costs of 10 basis points per switch the past-only version (0.38) is below buy and hold (0.52). Its largest drawdown is 60%, against 59% for buy and hold and 25% for the smoothed version.

A backtest that also lets today's smoothed label earn today's return, with no one-day lag, reports 1.60.

By thirds of the evaluation period:

| Labels | 1936-1964 | 1964-1995 | 1995-2026 |
| --- | --- | --- | --- |
| Smoothed | 1.35 | 0.83 | 1.57 |
| Filtered, full-sample parameters | 0.84 | 0.35 | 0.51 |
| Filtered, past-only parameters | 0.69 | 0.39 | 0.51 |
| Buy and hold | 0.71 | 0.36 | 0.51 |

### The same split for Wasserstein k-means

Windows of 35 daily returns, starting 7 days apart, clustered into two groups by 1-Wasserstein distance.

| Labels | Sharpe | 95% interval |
| --- | --- | --- |
| From every window containing the day, full-sample centroids | 1.24 | 1.02 to 1.46 |
| Trailing window, full-sample centroids | 0.54 | 0.29 to 0.78 |
| Trailing window, past-only centroids | 0.58 | 0.34 to 0.82 |

Labelling a day by every window that contains it is how the paper colours a price path. Those windows run up to 34 days past the day, and that is where the 1.24 comes from. With a trailing window the method scores about the same as buy and hold, whichever centroids it uses.

## Sources

- Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", Econometrica 57(2), 1989
- Hamilton, "Macroeconomic Regimes and Regime Shifts", NBER Working Paper 21863, 2016 (the vector form of the filter, the smoother and the EM updates were read here)
- Kim, "Dynamic linear models with Markov-switching", Journal of Econometrics 60(1-2), 1994 (paywalled; not read)
- Horvath, Issa and Muguruza, "Clustering market regimes using the Wasserstein distance", arXiv:2110.11848; Journal of Computational Finance 28(1), 2024

## Comparison with the sources

Full detail, formulas and equation numbers are in `PAPER.md`.

- Hamilton (1989), Table I: all nine estimates reproduce to within 0.0011, and the standard errors to within 0.0007.
- Hamilton (1989), Table II: all seven recessions dated by the smoother match, quarter for quarter.
- Hamilton (1989), expected durations: 4.08 quarters for a recession (paper 4.1) and 10.43 for an expansion (paper 10.5).
- Horvath, Issa and Muguruza, Table 3: Wasserstein k-means labels 90.4% of synthetic returns correctly (paper 90.6%).
- The same table reports 58.2% for a hidden Markov model. The model fitted here scores 99.5% on the same kind of path. The paper does not describe its HMM fit, so I cannot say where the difference comes from.
- The headline result, the Sharpe ratio under three information sets, is not in any of the sources.

## Data

Daily excess return of the US market (`Mkt-RF`) from the Kenneth R. French data library, 1926-07-01 to 2026-09-30, 26,338 days. It is the CRSP value-weighted return on US stocks minus the one-month bill rate. The file is free to download without a key, is copyright Fama and French, and is fetched by `scripts/fetch_data.py` into `data/`, which is not committed.

The 135 quarterly GNP growth rates used for Hamilton's model are shipped in `src/hindsight/rgnp_growth.csv`. They are copied from the Stata Press example file `rgnp.dta`.

## How it is tested

62 tests, run with `uv run pytest`.

- Filter and smoother match statsmodels to `1e-10` with two and three states, and match Hamilton's GNP model in statsmodels at the Table I estimates
- Filter, smoother and likelihood match a brute-force sum over all state paths on short samples
- The smoothed probability of the last day equals the filtered one
- Expected time in a state equals 1 / (1 - p), checked by formula, against Hamilton's printed 4.1 and 10.5 quarters, and against simulated run lengths
- EM never lowers the likelihood and ends where the likelihood is flat
- Changing returns after a date leaves every earlier past-only label unchanged, and changes the smoothed ones
- 1-Wasserstein distance matches scipy; the barycenter beats 200 random perturbations
- Running the experiments twice gives identical tables

## Layout

```
src/hindsight/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [x] Project skeleton, CI, and the reproduce command
- [x] Hamilton's filter and Kim's smoother written from scratch and checked against statsmodels
- [x] One regime strategy run three ways: smoothed labels; filtered labels with full-sample parameters; filtered labels with parameters fitted only on the past
- [x] Wasserstein k-means on rolling return distributions
- [x] `PAPER.md` comparison with the sources
- [x] Results and limits written up at the top of this file

The CI workflow is in the repository but has not run on GitHub; see the limits.

## Limits

- One strategy on one index. The same split on another market, another rule or more than two states could give a different size of gap.
- The intervals come from a block bootstrap of daily strategy returns with the labels held fixed. They leave out the uncertainty in the fitted parameters, so they are too narrow by an amount I did not measure.
- The parameter gap is small partly because the first refit already has ten years of data that include 1929 to 1933. Both volatility states were well observed by 1936. A sample that starts in a quiet decade would test parameter look-ahead harder.
- Positions are taken at the close on the signal from that same close. A one-day delay in execution was not tested.
- Costs are a flat 10 basis points per switch. Bills are assumed to earn the French library's risk-free rate with no spread.
- The EM fit uses one deterministic starting point. I did not check for other local maxima on the market data.
- The smoother formula is taken from Hamilton (2016). Kim (1994) and Hamilton (1990) are paywalled and were not read.
- The 58% accuracy that Horvath, Issa and Muguruza report for a hidden Markov model was not reproduced, and I could not find out why.
- I searched for a published number for the smoothed-to-filtered gap and did not find one. That is not proof none exists.
- The checks were run on Windows only. The GitHub Actions workflow has not run.
- This is research code. It is not investment advice and it does not trade.

## The name

Smoothed regime probabilities use the whole sample, including the days that had not happened yet. That is hindsight, and a backtest that uses it is trading on it.

## Licence

MIT. See `LICENSE`.
