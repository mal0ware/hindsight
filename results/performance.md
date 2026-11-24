| strategy | ann_excess_return_pct | ann_vol_pct | sharpe | sharpe_ci_low | sharpe_ci_high | max_drawdown_total_return_pct | max_drawdown_excess_return_pct | time_in_market_pct | switches | sharpe_net_10bp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Buy and hold | 7.9848 | 15.4490 | 0.5169 | 0.3097 | 0.7544 | 54.5655 | 59.1708 | 100.0000 | 0 | 0.5169 |
| HMM, smoothed labels | 11.2306 | 9.0677 | 1.2385 | 1.0206 | 1.4500 | 24.1059 | 24.8505 | 79.7917 | 418 | 1.1890 |
| HMM, filtered labels, full-sample parameters | 5.7486 | 10.2273 | 0.5621 | 0.3327 | 0.7969 | 42.1723 | 42.1723 | 80.4216 | 1312 | 0.4232 |
| HMM, filtered labels, past-only parameters | 5.4563 | 10.3433 | 0.5275 | 0.3007 | 0.7624 | 60.3412 | 60.3412 | 80.7687 | 1372 | 0.3839 |
| HMM, smoothed labels traded on the same day | 13.8510 | 8.6691 | 1.5977 | 1.3695 | 1.8211 | 21.4455 | 23.1180 | 79.7917 | 418 | 1.5458 |
| WK-means, labels from all windows containing the day | 11.7903 | 9.5051 | 1.2404 | 1.0198 | 1.4562 | 21.1766 | 30.0661 | 79.3332 | 118 | 1.2271 |
| WK-means, trailing window, full-sample centroids | 5.4941 | 10.2263 | 0.5372 | 0.2933 | 0.7816 | 55.0635 | 55.0635 | 79.1833 | 286 | 0.5070 |
| WK-means, trailing window, past-only centroids | 6.2599 | 10.7445 | 0.5826 | 0.3396 | 0.8186 | 48.6527 | 48.6527 | 82.0157 | 234 | 0.5590 |
