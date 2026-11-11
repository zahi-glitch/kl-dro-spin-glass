# Risk report

Equal-weight portfolio of 10 US stocks, daily losses in % of value, 1990-12-27 to 2018-04-11 (6875 days). Every forecast uses only the previous 250 days. *Filtered*: past losses rescaled by today's EWMA volatility (RiskMetrics, λ = 0.94). *Robust*: worst case over a KL ball of radius ρ = 0.0077, a 95% confidence region for a tail probability estimated from 250 days.

## Value-at-risk 99%

Expected: 69 exceptions in total, about 2.5 per year.

| method | average VaR (%) | exceptions | in 2008 | Kupiec p | Christoffersen p | P(exception after exception) | 250-day windows green | red |
|---|---|---|---|---|---|---|---|---|
| historical | 3.23 | 94 | 16 | 0.004 | < 0.001 | 5% | 80% | 4% |
| historical + robust | 4.54 | 27 | 4 | < 0.001 | < 0.001 | 4% | 100% | 0% |
| filtered | 3.38 | 90 | 4 | 0.014 | 0.018 | 3% | 81% | 0% |
| filtered + robust | 4.76 | 25 | 1 | < 0.001 | < 0.001 | 0% | 100% | 0% |

Kupiec: is the exception rate right? Christoffersen: the right rate *and* no clustering. Small p-values reject the forecast. Basel zones: green 0-4 exceptions in 250 days, red 10 or more.

## Expected shortfall 97.5% and EVaR

| method | average ES (%) | Acerbi-Szekely Z2 |
|---|---|---|
| historical | 3.27 | 0.29 |
| historical + robust | 4.06 | -0.43 |
| filtered | 3.41 | 0.12 |
| filtered + robust | 4.28 | -0.46 |
| historical: EVaR 97.5% | 3.88 | (exceeded on 48 days) |
| filtered: EVaR 97.5% | 4.10 | (exceeded on 48 days) |

Z2 is about 0 when ES is right, positive when it is too low, negative when too high; roughly, |Z2| > 0.7 is significant at 5%. EVaR = worst-case mean over the KL ball of radius log(1/(1 − 0.975)): an upper bound on ES, not a quantile, so it has no target exception rate.

## How much distrust? VaR 99% exceptions as the KL radius grows

| ρ (× 95% radius) | 0 | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| historical | 94 (p 0.004) | 59 (p 0.226) | 59 (p 0.226) | 27 (p < 0.001) | 27 (p < 0.001) | 27 (p < 0.001) |
| filtered | 90 (p 0.014) | 57 (p 0.142) | 57 (p 0.142) | 25 (p < 0.001) | 25 (p < 0.001) | 25 (p < 0.001) |

Target: 69 exceptions. p is Kupiec's test of the exception rate.

