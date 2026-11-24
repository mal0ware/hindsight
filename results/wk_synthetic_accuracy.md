| method | runs | total_mean_pct | total_sd_pct | total_paper_pct | on_mean_pct | on_sd_pct | on_paper_pct | off_mean_pct | off_sd_pct | off_paper_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WK-means | 50 | 90.39 | 11.43 | 90.60 | 86.68 | 8.69 | 87.24 | 91.63 | 12.48 | 91.72 |
| HMM, smoothed | 50 | 99.48 | 0.12 | 58.16 | 98.93 | 0.42 | 41.51 | 99.67 | 0.13 | 63.72 |
| HMM, filtered | 50 | 98.04 | 0.27 |  | 96.09 | 0.77 |  | 98.68 | 0.24 |  |
| HMM, usual start, stopped after 10 iterations | 50 | 33.25 | 2.66 |  | 99.82 | 0.13 |  | 11.06 | 3.57 |  |
| HMM, 2-means start with inflated variance, 10 iterations at most | 50 | 52.32 | 27.13 |  | 58.34 | 46.30 |  | 50.31 | 47.91 |  |
| HMM, 2-means start, 10 iterations at most | 50 | 63.61 | 18.80 |  | 38.05 | 30.16 |  | 72.13 | 30.94 |  |
| HMM, 2-means start with inflated variance, run to convergence | 50 | 99.49 | 0.12 |  | 98.93 | 0.42 |  | 99.67 | 0.13 |  |
| HMM fitted to the log price instead of returns | 50 | 47.77 | 7.41 |  | 62.58 | 14.46 |  | 42.83 | 11.12 |  |
