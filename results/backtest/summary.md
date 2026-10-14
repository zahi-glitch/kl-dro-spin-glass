# Results: KL-robust portfolios, 10 US stocks

Out-of-sample period 1991-12-31 to 2018-04-11. Window 504 days, monthly rebalancing, 10 bps per unit turnover, long-only. γ = 0.05, ρ = 0.05, τ = 0.1. Universe: AAPL, GE, AMD, WMT, BAC, T, XOM, BBY, PFE, JPM.

## Main comparison

| strategy | annual return | annual volatility | Sharpe (rf = 0) | max drawdown | worst day | CVaR 5% (daily) | turnover / rebalance | effective holdings |
|---|---|---|---|---|---|---|---|---|
| Equal weight | 18.9% | 22.3% | 0.89 | -57.8% | -10.7% | 3.2% | 0.00 | 10.00 |
| Markowitz | 20.0% | 21.6% | 0.95 | -48.9% | -9.5% | 3.1% | 0.28 | 3.07 |
| Robust mean (EVaR) | 17.5% | 17.9% | 0.99 | -38.8% | -9.6% | 2.5% | 0.16 | 3.99 |
| Robust mean-variance (KSS23 dual) | 17.2% | 17.4% | 1.00 | -39.0% | -9.0% | 2.5% | 0.14 | 4.06 |

## Crises (total return over the period)

- Dot-com crash, 2000-03-24 to 2002-10-09: Equal weight: -39.9%; Markowitz: -44.0%; Robust mean (EVaR): -30.1%; Robust mean-variance (KSS23 dual): -25.5%

- Financial crisis, 2007-10-09 to 2009-03-09: Equal weight: -56.9%; Markowitz: -38.6%; Robust mean (EVaR): -38.5%; Robust mean-variance (KSS23 dual): -38.3%

![wealth and drawdown](equity.png)

## Sensitivity to the KL budget ρ (robust mean-variance)

| strategy | annual return | annual volatility | Sharpe (rf = 0) | max drawdown | worst day | CVaR 5% (daily) | turnover / rebalance | effective holdings |
|---|---|---|---|---|---|---|---|---|
| rho=0.01 | 18.0% | 18.4% | 1.00 | -39.1% | -9.8% | 2.6% | 0.19 | 3.83 |
| rho=0.02 | 17.7% | 17.9% | 1.00 | -39.0% | -9.5% | 2.5% | 0.17 | 3.96 |
| rho=0.05 | 17.2% | 17.4% | 1.00 | -39.0% | -9.0% | 2.5% | 0.14 | 4.06 |
| rho=0.1 | 16.9% | 17.2% | 0.99 | -39.3% | -8.7% | 2.4% | 0.12 | 4.08 |
| rho=0.2 | 16.7% | 17.1% | 0.99 | -39.9% | -8.3% | 2.4% | 0.11 | 4.04 |

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

## Speed of the inner worst case

Inner worst case on 20 random (window, portfolio) pairs, 504 days each:

| method | median time | max difference |
|---|---|---|
| Parisi-type dual (2 scalars) | 2.4 ms | reference |
| CVXPY + Clarabel, solver only | 24.0 ms | 1.6e-08 |
| CVXPY + Clarabel, incl. modelling | 35.9 ms | |

