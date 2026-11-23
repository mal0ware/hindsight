# Hindsight

**How much of a regime-switching backtest is the model, and how much is knowing the future?**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

A hidden Markov model labels each day calm or turbulent. Many backtests use labels computed from the full history. How much of the reported performance disappears when the model only knows what it could have known?

## What this will contain

- Hamilton's filter and Kim's smoother written from scratch and checked against statsmodels
- One regime strategy run three ways: smoothed labels; filtered labels with full-sample parameters; filtered labels with parameters fitted only on the past
- Wasserstein k-means on rolling return distributions

## The result it is built to show

The Sharpe ratio of the same strategy under the three versions. The gap between the first and the third is the look-ahead, measured.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", Econometrica 57(2), 1989
- Kim, "Dynamic linear models with Markov-switching", Journal of Econometrics 60(1-2), 1994
- Horvath, Issa and Muguruza, "Clustering market regimes using the Wasserstein distance", Journal of Computational Finance 28(1), 2024

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

Free daily index data, fetched by script.

## How it is tested

The maths is tested before any result is trusted.

- Filter and smoother match statsmodels to tolerance
- The smoothed probability of the last day equals the filtered one
- Expected time in a state equals 1 / (1 - p)

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

- [ ] Project skeleton, CI, and the reproduce command
- [ ] Hamilton's filter and Kim's smoother written from scratch and checked against statsmodels
- [ ] One regime strategy run three ways: smoothed labels; filtered labels with full-sample parameters; filtered labels with parameters fitted only on the past
- [ ] Wasserstein k-means on rolling return distributions
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- One strategy on one index. The size of the gap will differ elsewhere; the direction should not.
- I searched for a published number for this gap and did not find one. That is not proof none exists.
- This is research code. It is not investment advice and it does not trade.

## The name

Smoothed regime probabilities use the whole sample, including the days that had not happened yet. That is hindsight, and a backtest that uses it is trading on it.

## Licence

MIT. See `LICENSE`.
