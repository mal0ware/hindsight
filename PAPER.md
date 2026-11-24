# Sources, formulas and comparison

This file lists every formula the code implements with the place it was read, every number compared with a source, every difference found, and every variant that was run.

## Sources and how each was read

| Source | Read? | Used for |
| --- | --- | --- |
| Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle". Econometrica 57(2), 357-384. | Yes. Scanned copy, pages 360-361 and 366-374 read as images. | The model, the filter, Table I, Table II, expected durations. |
| Hamilton, J. D. (2016). "Macroeconomic Regimes and Regime Shifts". NBER Working Paper 21863 (Handbook of Macroeconomics, vol. 2). | Yes. Sections 2.4 and 2.5, text layer. | Vector form of the filter, the backward smoothing recursion, the EM updates. |
| Kim, C.-J. (1994). "Dynamic linear models with Markov-switching". Journal of Econometrics 60(1-2), 1-22. | No. Paywalled; no open copy found. | Nothing directly. See the note below. |
| Hamilton, J. D. (1990). "Analysis of time series subject to changes in regime". Journal of Econometrics 45(1-2), 39-70. | No. Paywalled. | Nothing directly. The EM updates are taken from Hamilton (2016), which restates them. |
| Horvath, B., Issa, Z. and Muguruza, A. "Clustering market regimes using the Wasserstein distance". arXiv:2110.11848; Journal of Computational Finance 28(1), 2024. | Yes. arXiv version, sections 1.1, 2 and 3.2 to 3.4. | Lift into windows, Wasserstein distance, barycenter, the k-means algorithm, accuracy scores, Table 3. |

The backward recursion in `kim_smoother` is the one usually called Kim's smoother. I could not read Kim (1994), so the formula in the code is equation (22) of Hamilton (2016), which Hamilton attributes to Hamilton (1994, p. 694). The statsmodels documentation cites Kim (1994) for the same recursion, and the code here agrees with statsmodels to rounding error. I have not checked the equation number in Kim's paper.

The journal version of Horvath, Issa and Muguruza was not read. Equation numbers below are from arXiv version 2110.11848 and may differ in the journal.

## Notation

`P[i, j]` in the code is `Prob(s_{t+1} = j | s_t = i)`, so rows sum to one. Hamilton (2016) defines `P` as the transpose (columns sum to one). Every formula below is written in the code's convention.

- `eta_t[j] = f(y_t | s_t = j, y_1..y_{t-1})`, the conditional density.
- `xi_{t|t-1}[j] = Prob(s_t = j | y_1..y_{t-1})`, predicted.
- `xi_{t|t}[j] = Prob(s_t = j | y_1..y_t)`, filtered.
- `xi_{t|T}[j] = Prob(s_t = j | y_1..y_T)`, smoothed.

## Formulas implemented

### Markov chain

| # | Formula | Source | Code |
| --- | --- | --- | --- |
| C1 | `Prob(s_t = 1 | s_{t-1} = 1) = p`, `Prob(s_t = 0 | s_{t-1} = 0) = q` | Hamilton (1989) eq. (2.2) | `hamilton1989.expanded_transition` |
| C2 | Limiting probability of state 1: `pi = (1 - q) / (1 - p + 1 - q)` | Hamilton (1989) eq. (2.7) | tested in `test_ergodic_probabilities` |
| C3 | General case: `pi` is the last column of `(A'A)^{-1} A'` with `A = [I - P'; 1']` | Hamilton (2016) eq. (20) | `filter.ergodic_probabilities` (solved by least squares) |
| C4 | Expected time in a state: `sum_k k q^{k-1} (1 - q) = 1 / (1 - q)` | Hamilton (1989) p. 374 | `filter.expected_duration` |

### Filter

| # | Formula | Source | Code |
| --- | --- | --- | --- |
| F1 | `xi_{t|t-1} = P' xi_{t-1|t-1}` | Hamilton (2016) eq. (19), first line; Hamilton (1989) section 4.2, Step 1 | `filter._filter_kernel` |
| F2 | `xi_{t|t} = (xi_{t|t-1} * eta_t) / sum(xi_{t|t-1} * eta_t)`, elementwise product | Hamilton (2016) eq. (19), second line; Hamilton (1989) Steps 2 to 4 | `filter._filter_kernel` |
| F3 | `log L = sum_t log sum_j xi_{t|t-1}[j] eta_t[j]` | Hamilton (2016) eq. (21); Hamilton (1989) p. 369 | `FilterResult.loglik` |

The code works with `log eta_t` and subtracts the largest entry before exponentiating. This changes no value; it stops a far outlier from underflowing every state at once.

### Smoother

| # | Formula | Source | Code |
| --- | --- | --- | --- |
| S1 | `xi_{t|T}[i] = xi_{t|t}[i] * sum_j P[i, j] xi_{t+1|T}[j] / xi_{t+1|t}[j]`, run backwards from `xi_{T|T}` | Hamilton (2016) eq. (22) | `filter._smoother_kernel` |
| S2 | `Prob(s_t = i, s_{t+1} = j | y_1..y_T) = xi_{t|t}[i] P[i, j] xi_{t+1|T}[j] / xi_{t+1|t}[j]` | Hamilton (2016) eq. (23) and footnote 2 | `filter._smoother_kernel` (`transitions`) |

### EM for the Gaussian model

The model for daily returns is `y_t | s_t = j ~ Normal(mu_j, sigma_j^2)`. With `w_t[j] = xi_{t|T}[j]`:

| # | Formula | Source | Code |
| --- | --- | --- | --- |
| E1 | `P[i, j] = sum_{t<T} Prob(s_t = i, s_{t+1} = j | T) / sum_j (same)` | Hamilton (2016) section 2.5, update for `p_ij` | `model.em_step` |
| E2 | `mu_j = sum_t w_t[j] y_t / sum_t w_t[j]` | Hamilton (2016) eq. (24) with a constant as the only regressor | `model.em_step` |
| E3 | `sigma_j^2 = sum_t w_t[j] (y_t - mu_j)^2 / sum_t w_t[j]` | Hamilton (2016) section 2.5, update for `Sigma_i` | `model.em_step` |
| E4 | `Prob(s_1 = j)` is a free parameter, updated to `xi_{1|T}[j]` | Hamilton (2016) section 2.5 | `model.em_step` |

One detail in E1: Hamilton (2016) prints the denominator as `sum_{t=1}^{T-1} Prob(s_t = i | T)`. The code divides by the row sum of the numerators. The two are equal, because S2 summed over `j` gives `xi_{t|T}[i]`; a test checks this.

### Hamilton's 1989 model

| # | Formula | Source | Code |
| --- | --- | --- | --- |
| H1 | `y_t = alpha_1 s_t + alpha_0 + z_t`, `z_t = phi_1 z_{t-1} + ... + phi_r z_{t-r} + eps_t`, `eps_t ~ N(0, sigma^2)` | eq. (4.3) | `hamilton1989.log_density` |
| H2 | Density given `(s_t, ..., s_{t-r})`: normal with residual `(y_t - alpha_1 s_t - alpha_0) - sum_k phi_k (y_{t-k} - alpha_1 s_{t-k} - alpha_0)` | section 4.2, Step 2 | `hamilton1989.log_density` |
| H3 | Start the filter from the unconditional distribution of `(s_0, ..., s_{-r+1})` | pp. 368-369 | `hamilton1989.infer` |
| H4 | State 1 is the fast-growth state (`alpha_1 > 0`) | p. 369 | `hamilton1989.fit_mle` |
| H5 | A quarter is in recession when the smoothed `Prob(s_t = 0) > 0.5` | pp. 373-374 | `hamilton1989.recession_spells` |

The code runs H2 by expanding the chain to the `2^(r+1) = 32` combinations of `(s_t, ..., s_{t-4})` and handing that chain to the general filter F1 to F3.

### Wasserstein k-means

| # | Formula | Source | Code |
| --- | --- | --- | --- |
| W1 | Windows of length `h1` from the return series, sliding by a fixed offset | Definition 1.2, eq. (3) | `wasserstein.lift` |
| W2 | For two samples with `N` atoms each, sorted: `W_p^p = (1/N) sum_i |alpha_i - beta_i|^p` | eq. (21) | `wasserstein.wasserstein_distance` |
| W3 | Barycenter for `p = 1`: atom `j` is the median of the `j`-th sorted atoms | Proposition 2.6 | `wasserstein.barycenter` |
| W4 | Loss `sum_i W_p(old centroid_i, new centroid_i)`; stop when it is below a tolerance | eq. (23) | `wasserstein._single_run` |
| W5 | Start from `k` windows drawn at random, assign each window to the nearest centroid, replace each centroid by the barycenter of its members | Algorithm 1 | `wasserstein.wk_means` |
| W6 | Accuracy: with `Y^i_l` the number of windows containing return `i` that are in cluster `l`, regime-off score `sum_{i off} Y^i_1 / sum_{i off} sum_l Y^i_l`, regime-on score the same with `Y^i_2` over the regime-on returns, total the sum of both numerators over all windows | eqs. (31), (32), (33) | `wasserstein.accuracy_scores` |
| W7 | Synthetic log returns: `Normal((mu - sigma^2/2) dt, sigma^2 dt)` with `dt = 1/(252*7)`, bull `(0.02, 0.2)`, bear `(-0.02, 0.3)`, 20 years, 10 changes of half a year | section 3.3.1, eq. (35) | `experiments.synthetic_path` |

The barycenter for `p = 2` (the mean of the sorted atoms) is in the code but is not from this paper and is not used for any result.

### Not from a source I read

- The circular block bootstrap used for the fixed-label intervals (`strategy.bootstrap_sharpes`) is written from the general description of the method: resample whole blocks of consecutive days with wrap-around. I did not read Politis and Romano (1992) or Künsch (1989). The block length of 63 days is my choice.
- The stationary bootstrap used for the refitted intervals (`strategy.stationary_bootstrap_index`) is also written from the general description: blocks start at a uniformly drawn day and have geometric lengths, here with mean 63 days. I did not read Politis and Romano (1994).
- The random starting points for EM (`model.random_start`) and the rule for calling a fit degenerate (`model.is_degenerate`) are my own.
- The Sharpe ratio is the mean daily excess return over its standard deviation, times `sqrt(252)`.

## Comparison with the sources

### Hamilton (1989), Table I

US real GNP growth, 135 quarters from 1951Q2 to 1984Q4, 131 used after four lags. The series is read from the test module of statsmodels (see the Data section of the README). Full table: `results/hamilton1989_table1.csv`.

| Parameter | Paper | Here | Paper s.e. | Here s.e. |
| --- | --- | --- | --- | --- |
| alpha1 | 1.522 | 1.5223 | 0.2636 | 0.2632 |
| alpha0 | -0.3577 | -0.3588 | 0.2651 | 0.2645 |
| p | 0.9049 | 0.9041 | 0.03740 | 0.0377 |
| q | 0.7550 | 0.7547 | 0.09656 | 0.0965 |
| sigma | 0.7690 | 0.7690 | 0.06676 | 0.0667 |
| phi1 | 0.014 | 0.0135 | 0.120 | 0.1200 |
| phi2 | -0.058 | -0.0575 | 0.137 | 0.1377 |
| phi3 | -0.247 | -0.2470 | 0.107 | 0.1069 |
| phi4 | -0.213 | -0.2129 | 0.110 | 0.1105 |

The largest difference is 0.0011, in `alpha0`. The log likelihood here is -181.2634 at my estimates and -181.2638 at the paper's printed estimates, so the printed values are 0.0004 log-likelihood units below the maximum I find. Likely reasons for the small differences: the copy of the series I use came from a file stored in single precision, the paper prints four digits, and the likelihood is flat near the maximum. The paper does not print its log likelihood, so that number cannot be compared.

### Hamilton (1989), Table II

Recessions as runs of quarters with smoothed `Prob(s_t = 0) > 0.5`. All seven match the "Smoother" columns of Table II: 1953Q3 to 1954Q2, 1957Q1 to 1958Q1, 1960Q2 to 1960Q4, 1969Q3 to 1970Q4, 1974Q1 to 1975Q1, 1979Q2 to 1980Q3, 1981Q2 to 1982Q4.

Table II is headed "Peak" and "Trough". I read the two columns as the first and last quarter of each run, and under that reading every date agrees. The paper does not define the columns more exactly than that.

### Hamilton (1989), expected durations (p. 374)

| | Paper | From the paper's p, q | From my p, q |
| --- | --- | --- | --- |
| Recession | 4.1 quarters | 4.08 | 4.08 |
| Expansion | 10.5 quarters | 10.52 | 10.43 |

My expansion figure rounds to 10.4, one tenth below the paper. It follows from `p` being 0.0008 lower here.

### Hamilton (1989), Figure 1

`results/hamilton1989_recession_probability.png` shows the filtered probability (the paper's panel A) and the full-sample smoothed probability. The paper's panel B is a four-quarter-ahead smoother, which I did not implement. The paper says the two smoothers differ by 0.016 on average; I did not check that figure.

### statsmodels

`results/statsmodels_check.csv`. At identical parameters and with the chain started from its unconditional distribution, filtered and smoothed probabilities agree with statsmodels to better than `1e-9` on three samples: 2,000 simulated observations, the 26,338 daily market returns, and Hamilton's GNP model at the Table I estimates. Measured differences were between `1e-16` and `3e-14`; log likelihoods differ by less than `1e-10`.

### Horvath, Issa and Muguruza, Table 3

Synthetic geometric Brownian motion, 20 years of hourly steps, 10 regime changes, 50 paths, `(h1, h2) = (35, 28)`, `k = 2`, `p = 1`. Accuracy in percent, mean over paths. `results/wk_synthetic_accuracy.csv`.

| Method | Total: paper | Total: here | Regime-on: paper | Regime-on: here | Regime-off: paper | Regime-off: here |
| --- | --- | --- | --- | --- | --- | --- |
| WK-means | 90.60 | 90.39 | 87.24 | 86.68 | 91.72 | 91.63 |
| HMM | 58.16 | 99.48 | 41.51 | 98.93 | 63.72 | 99.67 |

WK-means reproduces. The HMM does not, and the gap is large. The paper's HMM labels 58% of returns correctly; a two-state Gaussian HMM fitted here by EM to the same kind of path labels 99.5% correctly with smoothed probabilities and 98.0% with filtered ones. Section 3.1.2 of the paper says only that it fits "a Gaussian HMM". It does not name the software, the starting values, the stopping rule or the decoding rule.

To see what could produce 58%, I ran fits that are weaker on purpose on the same 50 paths. This was done after the first results. In every row the state with the larger standard deviation is taken as the regime-on state and the labels are the smoothed probabilities above 0.5.

| Fit | Total | Regime-on | Regime-off |
| --- | --- | --- | --- |
| Paper, HMM | 58.16 | 41.51 | 63.72 |
| EM from the deterministic start, run to convergence | 99.48 | 98.93 | 99.67 |
| Deterministic start, stopped after 10 iterations | 33.25 | 99.82 | 11.06 |
| Start split by mean, variance inflated by `1e-3`, at most 10 iterations, stop when the log likelihood gains less than 0.01 | 52.32 | 58.34 | 50.31 |
| Start split by mean, no inflation, same limits | 63.61 | 38.05 | 72.13 |
| Start split by mean, variance inflated, run to convergence | 99.49 | 98.93 | 99.67 |
| Model fitted to the cumulated log price instead of returns | 47.77 | 62.58 | 42.83 |

The start split by mean (`experiments.two_means_start`) puts the two means at the centres of a two-cluster k-means on the returns, gives both states the same variance and draws the transition matrix at random. That, a limit of 10 iterations and a tolerance of 0.01 are the defaults of `GaussianHMM` in the hmmlearn library, as I read them in its source on 2026-11-24. The library also adds `1e-3` to the starting variance, which is over 30 times the variance of these hourly returns, and puts a weak prior on the variances that I did not copy. I did not run hmmlearn itself.

What this shows: every fit that is run to convergence scores 99.5%, whatever the start, and every fit stopped at 10 iterations scores between 33% and 64%. The mean-split start without inflation comes closest to the paper on all three scores (63.6, 38.1 and 72.1 against 58.2, 41.5 and 63.7), with a standard deviation across paths of 19 points. The two regimes have nearly the same mean and different volatility, so a fit that is still close to a split by mean labels them almost at random. A model fitted to the price level instead of returns is also near chance.

What it does not show: which of these the paper did. An under-converged fit is one explanation that produces numbers of the right size. I have no evidence of the software or settings the authors used.

The paper reports its spread as "95% CI" without a formula (5.81 for WK-means total accuracy). The standard deviation across my 50 paths is 11.43, which would give a half-width of 3.2 as `1.96 sd / sqrt(50)`. I cannot tell which definition the paper used.

### Differences from Horvath, Issa and Muguruza in what was implemented

- Equation (3) as printed gives windows `(x_{1+h2(i-1)}, ..., x_{1+h1+h2(i-1)})`: `h1 + 1` points each, starting `h2` apart. The text says `(35, 28)` puts adjacent windows one day (seven hourly steps) apart and that one return can belong to five windows, which needs windows of 35 points starting `h1 - h2 = 7` apart. The code follows the text.
- The paper does not state which `p` its experiments use. The code uses `p = 1`, the case its barycenter result covers.
- The paper does not say what to do when a cluster becomes empty. The code restarts that centroid at the window farthest from its own centroid.
- The paper draws regime start points "randomly". The code draws them uniformly among placements in which no two changes touch.
- For the market data the paper uses hourly SPY returns from 2005 to 2020. This repository uses daily returns from 1926, with the same `(35, 28)` applied to days. Nothing from the paper's real-data section (its Figures 1 to 4, Tables 1 and 2, the MMD scores) is reproduced here.

## The experiment that is not in any source

The headline result is not a replication. It is one strategy run under three information sets on daily excess returns of the US market (French data library, 1926-07-01 to 2026-09-30).

- The model is the two-state Gaussian model above, fitted by EM. The calm state is the one with the smaller `sigma`.
- The rule is: hold the market on day `t + 1` if day `t` is labelled calm (probability above 0.5), otherwise hold bills.
- Smoothed labels use S1 with full-sample parameters. Filtered labels use F2 with full-sample parameters. Past-only labels refit the parameters on the last trading day of every month using data up to that day, and filter with them until the next refit.
- The first refit is on 1936-07-31, ten years into the sample. All versions are scored on the same 23,337 days from 1936-08-01.
- The drawdowns in `results/performance.csv` are on total returns (the strategy's excess return plus the bill rate), with a second column on excess returns.

Results: `results/performance.csv`, `results/sharpe_gaps.csv`, `results/subperiods.csv`.

| Version | Sharpe | 95% interval, model refitted | 95% interval, labels fixed | Switches | Sharpe at 10 bp per switch |
| --- | --- | --- | --- | --- | --- |
| Smoothed labels | 1.24 | 1.00 to 1.60 | 1.02 to 1.45 | 418 | 1.19 |
| Filtered labels, full-sample parameters | 0.56 | 0.31 to 0.80 | 0.33 to 0.80 | 1,312 | 0.42 |
| Filtered labels, past-only parameters | 0.53 | 0.31 to 0.82 | 0.30 to 0.76 | 1,372 | 0.38 |
| Buy and hold | 0.52 | 0.21 to 0.70 | 0.31 to 0.75 | 0 | 0.52 |
| Smoothed labels traded on the same day | 1.60 | not run | 1.37 to 1.82 | 418 | 1.55 |

| Gap in Sharpe ratio | Value | 95% interval, model refitted | 95% interval, labels fixed |
| --- | --- | --- | --- |
| Total look-ahead: smoothed minus past-only | 0.71 | 0.57 to 0.94 | 0.58 to 0.84 |
| From the smoother: smoothed minus filtered | 0.68 | 0.57 to 0.93 | 0.54 to 0.81 |
| From the parameters: full-sample minus past-only | 0.03 | -0.05 to 0.06 | -0.02 to 0.09 |

The same split for Wasserstein k-means labels gives 1.24 for labels built from every window containing the day (the way the paper colours a price path), 0.54 for the trailing window with full-sample centroids and 0.58 for the trailing window with past-only centroids. The centroid gap is -0.05 with a fixed-label interval of -0.11 to 0.02.

### The two kinds of interval

Labels fixed: 2,000 circular block bootstrap resamples of the daily strategy returns, 63-day blocks, the same days drawn for every strategy. The labels are computed once on the real data. These intervals describe noise in returns given the labels and do not include the uncertainty in the fitted parameters or in the labels.

Model refitted: 500 stationary bootstrap resamples of the raw daily excess returns, mean block 63 days (`experiments.refit_bootstrap`, `results/refit_bootstrap.csv`). On each resample the two-state model is fitted to the whole resample, the smoother and the filter are run, and the past-only parameters are refitted at the positions of the year-end refit dates, 91 fits per resample. The first refit is at the position of 1936-12-31, so these runs are scored from 1937, five months later than the monthly runs. The interval is the 2.5th to 97.5th percentile of the resampled statistic.

| Quantity | Monthly refits, real data | Yearly refits, real data | Mean over resamples | 95% interval |
| --- | --- | --- | --- | --- |
| Sharpe, smoothed | 1.24 | 1.24 | 1.30 | 1.00 to 1.60 |
| Sharpe, filtered, full-sample parameters | 0.56 | 0.56 | 0.55 | 0.31 to 0.80 |
| Sharpe, filtered, past-only parameters | 0.53 | 0.51 | 0.55 | 0.31 to 0.82 |
| Sharpe, buy and hold | 0.52 | 0.51 | 0.45 | 0.21 to 0.70 |
| Gap, total | 0.71 | 0.72 | 0.75 | 0.57 to 0.94 |
| Gap, from the smoother | 0.68 | 0.68 | 0.74 | 0.57 to 0.93 |
| Gap, from the parameters | 0.03 | 0.05 | 0.00 | -0.05 to 0.06 |

Three things to know when reading it.

- The share of resamples in which the parameter gap is at or below zero is 46%. With the labels fixed it was 8%. Neither is a p-value: the resamples are centred on the data and are not drawn under a hypothesis of no gap. The first version of this repository printed the 8% without that warning.
- The resampled parameter gap has a mean of 0.00, below the 0.05 on the real data with the same yearly schedule. A resample puts the blocks in random order, so the crash of 1929 to 1933 is no longer always inside the burn-in. The interval describes a history in which the early sample is typical of the rest.
- Joining blocks cuts regimes at the joins. The mean smoother gap over resamples (0.74) is a little above the real-data value (0.68), so this did not shrink the gap, but the intervals inherit whatever the cutting does.

### Trading one day later

`strategy.positions(calm, delay=1)` holds the market on day `t + 2` if day `t` is labelled calm. `results/execution_delay.csv`, with fixed-label intervals.

| Version | Same close | One day later | 95% interval, labels fixed |
| --- | --- | --- | --- |
| Smoothed labels | 1.24 | 1.02 | 0.80 to 1.24 |
| Filtered labels, full-sample parameters | 0.56 | 0.55 | 0.32 to 0.78 |
| Filtered labels, past-only parameters | 0.53 | 0.52 | 0.29 to 0.75 |
| WK-means, every window containing the day | 1.24 | 1.22 | 1.00 to 1.44 |
| WK-means, trailing window, full-sample centroids | 0.54 | 0.52 | 0.29 to 0.76 |
| WK-means, trailing window, past-only centroids | 0.58 | 0.59 | 0.33 to 0.83 |

One day later the total gap is 0.50 (fixed-label interval 0.38 to 0.62; refitted, yearly schedule, 0.36 to 0.61), the smoother gap 0.47 and the parameter gap 0.03. The Wasserstein labels barely move, because a 35-day window changes little in one day.

The past-only label does not exist on the day before the first refit, so the delayed past-only version holds bills on the first of the 23,337 evaluated days.

### Other series and three states

`experiments.cross_section`, `results/cross_section.csv`, `results/cross_section_summary.csv`. The 21 series are the market, the ten value-weighted industry portfolios and the ten value-weighted size deciles, all in excess of the bill rate, on the same 26,338 days and with the same burn-in, monthly refits and evaluation window as the market run.

With three states the states are ordered by standard deviation at every fit. Two rules are run:

- hold unless the probability of the most volatile state is above 0.5;
- hold only when the probability of the least volatile state is above 0.5.

| Model and rule | Execution | Smoother gap: median (min to max) | Parameter gap: median (min to max) | Smoother gap above zero | Past-only beats buy and hold |
| --- | --- | --- | --- | --- | --- |
| 2 states | same close | 0.64 (0.27 to 0.89) | 0.01 (-0.04 to 0.08) | 21 of 21 | 7 of 21 |
| 3 states, out of the most volatile | same close | 0.24 (0.04 to 0.35) | -0.00 (-0.04 to 0.09) | 21 of 21 | 18 of 21 |
| 3 states, in only the least volatile | same close | 1.24 (0.10 to 1.59) | -0.01 (-0.06 to 0.27) | 21 of 21 | 14 of 21 |
| 2 states | one day later | 0.41 (0.14 to 0.63) | 0.02 (-0.04 to 0.11) | 21 of 21 | 8 of 21 |
| 3 states, out of the most volatile | one day later | 0.20 (0.02 to 0.27) | 0.00 (-0.05 to 0.06) | 21 of 21 | 12 of 21 |
| 3 states, in only the least volatile | one day later | 0.98 (-0.07 to 1.30) | 0.01 (-0.16 to 0.26) | 20 of 21 | 8 of 21 |

The one negative smoother gap is the telecom industry under the last rule with the delay, -0.07.

The series are highly correlated, so the 21 values are not independent draws and the ranges are not intervals. No interval was computed for any cross-section number.

### Convergence and other starting points

`fit_em` returns a `converged` flag. The first version of this repository discarded it. It is now carried through `three_way_probabilities`, written to `results/em_convergence.csv` and to the `em_fits_converged` column of `results/cross_section.csv`, and `experiments.require_converged` stops the run if any fit reached the iteration limit of 20,000. No fit did. The slowest of the 1,083 fits behind the market result took 58 iterations. The cross-section made 45,486 fits (21 series, two models, 1,083 fits each); the slowest, a three-state fit on the second size decile, took 2,026 iterations. The refitted bootstrap made 45,500 more.

`experiments.em_multistart`, `results/em_multistart.csv`: for each of the 21 series and for two and three states, the full-sample model is refitted from 20 random starts and compared with the deterministic start. Two fits count as the same maximum when their log likelihoods are within 0.01. A fit is left out as degenerate if a state's standard deviation is below 1% of the sample's or a state carries fewer than ten days. In all 42 cases the deterministic start reached the best likelihood found. All 840 random starts converged, none was degenerate, and 839 of them reached that same likelihood. The one that stopped lower was a three-state fit on the telecom industry. The past-only refits were not restarted from random points.

## Variants run

Everything that was run on the market data, in order.

Before the first write-up:

1. The two-state specification above on the market index, once, during development.
2. `reproduce.py` in full, to write `results/`. Same specification, same numbers.
3. `reproduce.py --fast`: sample from 1990, five-year burn-in, yearly refits, 200 resamples, three k-means starts. These numbers are not in `results/`.

The threshold of 0.5, the 63-day block, monthly refits, 10 bp, `(35, 28)` and two states are the values in the first commit of the code. That commit and the first results were made on the same morning, so the git history does not prove the values were fixed before any run. They have not changed since. One code change was made between writing the EM routine and the first run on the strategy: the EM stopping tolerance went from `1e-8` to `1e-12` per observation after I saw that the looser setting stopped 0.0002 log-likelihood units short of the limit on the full sample.

After the first write-up, in response to a review. All of these are post hoc: they were chosen knowing the headline result.

4. The one-day execution delay, for all six label versions.
5. The 21 series with two and three states. Both three-state rules were written down before either was run, and both are reported. The whole cross-section was run once before the final run, with the same code and the same numbers.
6. Random EM starts, 20 per series and number of states, run once before the final run. After that run I changed one thing: when a random start ties the deterministic one within 0.01, the "best start" column now keeps the deterministic fit.
7. The refitted bootstrap: 500 resamples, mean block 63 days, yearly refits. These three settings were chosen for running time before the first run and not changed. It was run once before the final run.
8. The weak HMM fits on synthetic paths. An exploratory run on 10 paths with a different seed tried the five fits in the table above and one more, the inflated-variance start with a limit of 100 iterations, which was left out because the loose tolerance stopped it after 13 iterations on average, so it was nearly the same fit as the 10-iteration one. That run also scored each fit with the low-mean state taken as regime-on. I dropped that mapping because the two drifts differ by too little for it to identify a state.
9. The EM iteration limit went from 5,000 to 20,000 when the convergence check was added, as headroom for the three-state fits. No fit has reached either limit.

Nothing in 4 to 9 changed a number in the original tables, apart from the drawdown column, which is now on total returns.

For Hamilton's model one starting point was used for the optimiser and it converged to the published estimates on the first attempt.

For the synthetic test one seed was used for the 50 paths.
