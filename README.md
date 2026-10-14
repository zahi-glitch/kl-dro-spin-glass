# kl-dro-spin-glass

From statistical mechanics to robust optimisation. A desktop app that tells the story order → disorder → memory → decisions (Curie–Weiss, spin glasses, Hopfield), and a Python package showing that for REM-type approximations of TAP free energies the Parisi formula is the Lagrangian dual of a KL-constrained sup, with a robust-portfolio backtest on 28 years of real stock data.

> Status: work in progress, built step by step.

## Outline (to be written)

1. The puzzle: free energies are naturally a **sup** over measures, yet for the
   SK model we only have an **inf** (the Parisi formula).
2. The ML/DRO view: KL-constrained worst-case expectations and their duals.
3. The REM/GREM approximation of TAP free energies: a non-linear model where
   both the sup and the inf are proved.
4. The dictionary: Parisi parameters as Lagrange multipliers.
5. Numerical primal-dual check and the phase transition.

## Case study: KL-robust portfolios on 28 years of US stocks

A walk-forward backtest on real daily prices of 10 US stocks (1990–2018, including the dot-com
crash and 2008). Every month each strategy is re-estimated on the last two years of returns,
long-only, with 10 bps transaction costs. All parameters were fixed before looking at results.

**Headline: the Parisi-type dual makes the robust portfolio 60× faster to compute.** The same
robust mean–variance backtest takes 1.1 s with the dual and 69 s with a generic conic solver
(CVXPY + Clarabel), and returns the same portfolios (weights differ by at most 10⁻⁴). One worst case
costs a few milliseconds whatever the window length; the conic solver slows down from 7 ms to 1.4 s
as the window grows to 16 years, and occasionally fails to converge.

![speed](results/backtest/speed.png)

*Left: time for one worst case (median of 5 random windows and portfolios; "failure" = Clarabel did
not converge and SCS was used instead). Right: the whole backtest over the last 24 monthly rebalances.*

The investment results:

| strategy | annual return | volatility | Sharpe | max drawdown | turnover |
|---|---|---|---|---|---|
| Equal weight | 18.9% | 22.3% | 0.89 | −57.8% | 0.00 |
| Markowitz (mean − γ·variance) | 20.0% | 21.6% | 0.95 | −48.9% | 0.28 |
| Robust mean (KL ball = EVaR) | 17.5% | 17.9% | 0.99 | −38.8% | 0.16 |
| **Robust mean–variance (KSS23 dual)** | 17.2% | 17.4% | 1.00 | −39.0% | 0.14 |

![wealth and drawdown](results/backtest/equity.png)

What the robust portfolios do: an adversary may reweight the past days within a KL budget ρ, and the
portfolio is chosen against the worst reweighting. For mean–variance risk this inner worst case is the
non-linear problem of [KSS23](https://doi.org/10.1088/1751-8121/acdf30), solved through its Parisi-type
dual in two scalars (q, m); the portfolio gradient comes for free from the worst-case weights
(Danskin's theorem).

Takeaways:

- **Less risk for a little return.** The robust portfolios cut volatility by about a fifth, the
  maximum drawdown from −49%/−58% to −39%, and turnover by half, for 1.7–2.8 points of annual return;
  risk-adjusted return is slightly higher. In the dot-com crash: −25% against −44% for Markowitz.
- **No tuning of risk aversion needed.** Markowitz swings from 29% to 17% volatility as γ goes from 0.02
  to 0.2; the robust portfolio stays at 17–18% for every γ and every budget ρ tried
  ([sensitivity](results/backtest/sensitivity.png)).
- **Exact and fast.** Why the dual is fast: the sup over 504 day-weights becomes an inf over two
  numbers (q, m), and the portfolio gradient is read off the worst-case weights. It agrees with the
  conic solver to 10⁻⁸ and, unlike it, has no convergence failures on long windows.
- **Honest limits.** Here the non-linear (mean–variance) robust portfolio performs like the linear one
  (EVaR): its edge in this experiment is computational (figure above), not higher returns. The universe has
  survivorship bias (well-known companies chosen today), returns are against a zero risk-free rate,
  and it is one dataset.

Full tables, crisis returns and sensitivities: [`results/backtest/summary.md`](results/backtest/summary.md).
Reproduce:

```bash
pip install -e ".[backtest]"       # pandas, matplotlib, cvxpy
python scripts/backtest.py         # about 8 minutes (most of it timing the conic solver); writes results/backtest/
```

Data: [`data/`](data/README.md), from PyPortfolioOpt (MIT licence).

## Desktop app

A local Mac/Linux/Windows app that tells the story from statistical mechanics to
robust optimisation, with live simulations:

```bash
pip install -e ".[app]"        # adds PySide6, pyqtgraph and matplotlib
python -m tapdual.app
```

Each section opens with what we saw so far, what changes now, and the question it answers:

1. **What is a statistical-mechanics system?** Many simple parts, an energy, noise; Gibbs
   probabilities and summary numbers; energy and entropy demos; free energy as an optimisation.
2. **Order: Curie–Weiss.** A crowd makes up its mind on its own: phase transition, the
   free-energy landscape, hysteresis of adoption with peer effects.
3. **Disorder: spin glasses.** Random signs and frustration; SK (two replicas, Max-Cut) and the
   REM (freezing, the entropy budget).
4. **Memory: Hopfield networks.** Storage and retrieval; the same network as a restricted
   Boltzmann machine; storage capacity and what is still open.
5. **Robust decisions.** KL-constrained DRO on 756 daily returns: the worst case is a Gibbs
   tilt of the data with temperature λ, and freezes onto the worst days as the budget grows.
   With mean–variance risk the non-linearity is concave and the sup becomes a Parisi-type inf
   over (q, m), checked live against a direct solution of the sup; plus a physics ↔ DRO dictionary.

Each section ends with two or three references for further reading. Blue words open a glossary entry with typeset formulas (rendered offline with matplotlib).

## Dashboard

Two interactive pages in `docs/`, computed live in the browser by `docs/solver.js`,
a JavaScript port of the Python package (checked against it by
`scripts/check_dashboard.py`, which needs Node.js):

- **`docs/index.html`: the KL budget as a robustness radius (spin-glass model).**
  Sliders for β, h and the budget ρ; the worst-case distribution against the
  Gaussian, the Parisi function along the KKT path, and complementary slackness.
- **`docs/mean-variance.html`: robust mean-variance on a dataset.** The same duality
  applied to KL-DRO on data: an adversary reweights daily returns within a KL ball
  to maximise E ℓ + γ Var ℓ − τ KL. Sliders for γ, τ and ρ, and you can load your
  own CSV of returns (it stays in the browser).

Open either file in a browser; double-clicking works, no server needed.

## Python package

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
python -m tapdual sk --beta 5 --h 0.5                  # spin-glass model
python -m tapdual mv --gamma 1 --tau 0.5 --rho 0.1     # robust mean-variance, built-in data
python -m tapdual mv --csv my_returns.csv              # ... on your own returns (in %)
```

`pyproject.toml` describes the package for `pip`; there is no need to open it.

### Checks

- primal (sup over measures) = dual (Parisi inf) to ~1e-12, in both phases;
- the KKT conditions, the jointly convex coordinates (p, λ), and the transition;
- the classical REM threshold β_c = √(2 log 2) is reproduced to 1e-8;
- on data, a direct optimisation over all sample weights gives the same value
  as the two-parameter dual.

## Papers

- N. Kistler, M. A. Schmidt, G. Sebastiani, *On the REM approximation of TAP free
  energies*, J. Phys. A: Math. Theor. **56** (2023) 294001.
  [arXiv:2212.11718](https://arxiv.org/abs/2212.11718)
- G. Sebastiani, M. A. Schmidt, *On the GREM approximation of TAP free energies*,
  J. Phys. A: Math. Theor. **57** (2024) 335206.
  [arXiv:2401.13507](https://arxiv.org/abs/2401.13507)
