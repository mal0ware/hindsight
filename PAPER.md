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

- The circular block bootstrap used for the intervals (`strategy.bootstrap_sharpes`) is written from the general description of the method: resample whole blocks of consecutive days with wrap-around. I did not read Politis and Romano (1992) or Künsch (1989). The block length of 63 days is my choice.
- The Sharpe ratio is the mean daily excess return over its standard deviation, times `sqrt(252)`.

## Comparison with the sources

### Hamilton (1989), Table I

US real GNP growth, 135 quarters from 1951Q2 to 1984Q4, 131 used after four lags. Full table: `results/hamilton1989_table1.csv`.

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

The largest difference is 0.0011, in `alpha0`. The log likelihood here is -181.2634 at my estimates and -181.2638 at the paper's printed estimates, so the printed values are 0.0004 log-likelihood units below the maximum I find. Likely reasons for the small differences: the data file I use stores the series in single precision, the paper prints four digits, and the likelihood is flat near the maximum. The paper does not print its log likelihood, so that number cannot be compared.

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

WK-means reproduces. The HMM does not, and the gap is large. The paper's HMM labels 58% of returns correctly; a two-state Gaussian HMM fitted here by EM to the same kind of path labels 99.5% correctly with smoothed probabilities and 98.0% with filtered ones. The paper does not say which HMM software, starting values or decoding rule it used. On a path where one state has 1.5 times the volatility of the other and each regime lasts 882 steps, a converged HMM should separate the states well, so my guess is that the paper's fit stopped at a poor local optimum or was decoded differently. I cannot confirm that from the paper.

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
- Intervals are from 2,000 circular block bootstrap resamples of the daily strategy returns, 63-day blocks, the same days drawn for every strategy. The labels are held fixed in the resampling, so the intervals describe noise in returns given the labels. They do not include the uncertainty in the fitted parameters.

Results: `results/performance.csv`, `results/sharpe_gaps.csv`, `results/subperiods.csv`.

| Version | Sharpe | 95% interval | Switches | Sharpe at 10 bp per switch |
| --- | --- | --- | --- | --- |
| Smoothed labels | 1.24 | 1.02 to 1.45 | 418 | 1.19 |
| Filtered labels, full-sample parameters | 0.56 | 0.33 to 0.80 | 1,312 | 0.42 |
| Filtered labels, past-only parameters | 0.53 | 0.30 to 0.76 | 1,372 | 0.38 |
| Buy and hold | 0.52 | 0.31 to 0.75 | 0 | 0.52 |
| Smoothed labels traded on the same day | 1.60 | 1.37 to 1.82 | 418 | 1.55 |

| Gap in Sharpe ratio | Value | 95% interval |
| --- | --- | --- |
| Total look-ahead: smoothed minus past-only | 0.71 | 0.58 to 0.84 |
| From the smoother: smoothed minus filtered | 0.68 | 0.54 to 0.81 |
| From the parameters: full-sample minus past-only | 0.03 | -0.02 to 0.09 |

The same split for Wasserstein k-means labels gives 1.24 for labels built from every window containing the day (the way the paper colours a price path), 0.54 for the trailing window with full-sample centroids and 0.58 for the trailing window with past-only centroids. The centroid gap is -0.05 with an interval of -0.11 to 0.02.

## Variants run

Everything that was run on the market data, in order:

1. The specification above, once, during development. Its numbers are the ones reported.
2. `reproduce.py` in full, to write `results/`. Same specification, same numbers.
3. `reproduce.py --fast`: sample from 1990, five-year burn-in, yearly refits, 200 resamples, three k-means starts. It gave Sharpe ratios of 1.51 (smoothed), 0.52 (filtered, full-sample parameters), 0.54 (filtered, past-only parameters) and 0.50 (buy and hold). These numbers are not in `results/`.

No threshold, block length, refit schedule, cost level, window length or number of states was changed after seeing a result. The choices of 0.5, 63 days, monthly refits, 10 bp, `(35, 28)` and two states were made before the first run. One code change was made between writing the EM routine and the first run on the strategy: the EM stopping tolerance went from `1e-8` to `1e-12` per observation after I saw that the looser setting stopped 0.0002 log-likelihood units short of the limit on the full sample.

For Hamilton's model one starting point was used for the optimiser and it converged to the published estimates on the first attempt.

For the synthetic test one seed was used for the 50 paths.
