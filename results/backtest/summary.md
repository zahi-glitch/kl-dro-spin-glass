# Results: KL-robust portfolios, 10 US stocks

Out-of-sample period 1991-12-31 to 2018-04-11. Window 504 days, monthly rebalancing, 10 bps per unit turnover, long-only. γ = 0.05, ρ = 0.05, τ = 0.1. Universe: AAPL, GE, AMD, WMT, BAC, T, XOM, BBY, PFE, JPM.

## Main comparison

| strategy | annual return | annual volatility | Sharpe (rf = 0) | max drawdown | worst day | CVaR 5% (daily) | turnover / rebalance | effective holdings |
|---|---|---|---|---|---|---|---|---|
| Equal weight | 18.9% | 22.3% | 0.89 | -57.8% | -10.7% | 3.2% | 0.00 | 10.00 |
| Markowitz | 20.0% | 21.6% | 0.95 | -48.9% | -9.5% | 3.1% | 0.28 | 3.07 |
| Minimum variance | 15.6% | 16.9% | 0.94 | -38.5% | -8.3% | 2.4% | 0.07 | 4.23 |
| Robust mean (EVaR) | 17.5% | 17.9% | 0.99 | -38.8% | -9.6% | 2.5% | 0.16 | 3.99 |
| Robust mean-variance (KSS23 dual) | 17.2% | 17.4% | 1.00 | -39.0% | -9.0% | 2.5% | 0.14 | 4.06 |

## Crises (total return over the period)

- Dot-com crash, 2000-03-24 to 2002-10-09: Equal weight: -39.9%; Markowitz: -44.0%; Minimum variance: -16.6%; Robust mean (EVaR): -30.1%; Robust mean-variance (KSS23 dual): -25.5%

- Financial crisis, 2007-10-09 to 2009-03-09: Equal weight: -56.9%; Markowitz: -38.6%; Minimum variance: -36.9%; Robust mean (EVaR): -38.5%; Robust mean-variance (KSS23 dual): -38.3%

![wealth and drawdown](equity.png)

## Sensitivity to the KL budget ρ (robust mean-variance)

How big is a budget? Within H(ν | μ) ≤ ρ the adversary can lower an asset's average daily return by about √(2ρ) daily standard deviations: 0.32σ at ρ = 0.05, 0.03σ at ρ = 0.0005. Daily means are only a few hundredths of σ, so a large budget wipes out the information in the averages and the robust portfolio becomes a minimum-variance portfolio; a small budget brings it back towards Markowitz.

| strategy | annual return | annual volatility | Sharpe (rf = 0) | max drawdown | worst day | CVaR 5% (daily) | turnover / rebalance | effective holdings | differs from min. variance | differs from Markowitz |
|---|---|---|---|---|---|---|---|---|---|---|
| rho=0.0005 | 19.5% | 20.4% | 0.98 | -45.7% | -9.6% | 2.9% | 0.26 | 3.35 | 47.7% | 8.4% |
| rho=0.001 | 19.3% | 20.0% | 0.98 | -44.5% | -9.6% | 2.8% | 0.25 | 3.45 | 45.2% | 11.3% |
| rho=0.002 | 19.0% | 19.5% | 0.99 | -43.0% | -9.7% | 2.8% | 0.24 | 3.56 | 41.8% | 15.1% |
| rho=0.005 | 18.5% | 18.9% | 0.99 | -40.5% | -9.7% | 2.7% | 0.21 | 3.71 | 36.4% | 21.1% |
| rho=0.01 | 18.0% | 18.4% | 1.00 | -39.1% | -9.8% | 2.6% | 0.19 | 3.83 | 31.4% | 26.3% |
| rho=0.02 | 17.7% | 17.9% | 1.00 | -39.0% | -9.5% | 2.5% | 0.17 | 3.96 | 25.9% | 31.7% |
| rho=0.05 | 17.2% | 17.4% | 1.00 | -39.0% | -9.0% | 2.5% | 0.14 | 4.06 | 19.4% | 38.1% |
| rho=0.1 | 16.9% | 17.2% | 0.99 | -39.3% | -8.7% | 2.4% | 0.12 | 4.08 | 15.7% | 42.2% |

## Sensitivity to the risk aversion γ

| strategy | annual return | annual volatility | Sharpe (rf = 0) | max drawdown | worst day | CVaR 5% (daily) | turnover / rebalance | effective holdings |
|---|---|---|---|---|---|---|---|---|
| markowitz gamma=0.02 | 21.8% | 29.2% | 0.82 | -67.3% | -14.6% | 4.1% | 0.31 | 2.03 |
| robust_mv gamma=0.02 | 17.4% | 17.7% | 0.99 | -38.8% | -9.4% | 2.5% | 0.15 | 4.02 |
| markowitz gamma=0.05 | 20.0% | 21.6% | 0.95 | -48.9% | -9.5% | 3.1% | 0.28 | 3.07 |
| robust_mv gamma=0.05 | 17.2% | 17.4% | 1.00 | -39.0% | -9.0% | 2.5% | 0.14 | 4.06 |
| markowitz gamma=0.1 | 18.7% | 18.7% | 1.01 | -39.1% | -9.7% | 2.7% | 0.20 | 3.78 |
| robust_mv gamma=0.1 | 17.0% | 17.2% | 1.00 | -39.3% | -8.7% | 2.4% | 0.13 | 4.08 |
| markowitz gamma=0.2 | 17.6% | 17.5% | 1.01 | -38.0% | -9.2% | 2.5% | 0.14 | 4.09 |
| robust_mv gamma=0.2 | 16.8% | 17.1% | 0.99 | -40.1% | -8.3% | 2.4% | 0.11 | 4.07 |

![sensitivity](sensitivity.png)

## Speed: the dual against a generic conic solver

Time per inner worst case (median of 5 random windows and portfolios), and the largest difference in value:

| days | dual | conic solver | speed-up | conic failures | max difference |
|---|---|---|---|---|---|
| 126 | 1.8 ms | 7.3 ms | 4x | 0 of 5 | 2e-08 |
| 252 | 2.0 ms | 16.1 ms | 8x | 0 of 5 | 4e-08 |
| 504 | 2.4 ms | 24.1 ms | 10x | 0 of 5 | 1e-08 |
| 1008 | 3.0 ms | 42.5 ms | 14x | 0 of 5 | 1e-08 |
| 2016 | 4.7 ms | 168.9 ms | 36x | 1 of 5 | 1e-08 |
| 4032 | 6.1 ms | 1202.8 ms | 196x | 1 of 5 | 1e-08 |

Whole robust mean-variance backtest over the last 24 monthly rebalances, run twice: **1.1 s with the dual, 69.0 s with the conic solver** (61x), same portfolios (largest weight difference 1e-04).

![speed](speed.png)

