"""Case study: KL-robust portfolios on 28 years of daily US stock returns.

    python scripts/backtest.py            # full run (a few minutes), writes results/backtest/
    python scripts/backtest.py --quick    # short run for a smoke test

Design fixed before looking at results: 10 stocks with full history (1990-2018), 2-year estimation
window (504 days), monthly rebalancing (21 days), 10 bps per unit of turnover, long-only.
Main parameters: gamma = 0.05 (risk aversion, daily % units: annual risk aversion of about 10),
rho = 0.05 (KL budget), tau = 0.1 (KL penalty of the KSS23 form). Sensitivities to rho and gamma
are reported separately.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tapdual.backtest import FULL_HISTORY, Strategy, backtest, load_prices, simple_returns_pct
from tapdual.dro import worst_case_mean_variance

OUT = Path(__file__).resolve().parents[1] / "results" / "backtest"
GAMMA, RHO, TAU = 0.05, 0.05, 0.1

MAIN = [
    Strategy("Equal weight", "equal"),
    Strategy("Markowitz", "markowitz", gamma=GAMMA),
    Strategy("Robust mean (EVaR)", "robust_mean", rho=RHO),
    Strategy("Robust mean-variance (KSS23 dual)", "robust_mv", gamma=GAMMA, rho=RHO, tau=TAU),
]
COLORS = ["#8a8983", "#eb6834", "#7f77dd", "#2a78d6"]


def fmt(k, v):
    pct = {"annual return", "annual volatility", "max drawdown", "worst day", "CVaR 5% (daily)"}
    return f"{100 * v:.1f}%" if k in pct else f"{v:.2f}"


def table(results) -> str:
    keys = list(results[0].metrics)
    lines = ["| strategy | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for r in results:
        lines.append(f"| {r.strategy.name} | " + " | ".join(fmt(k, r.metrics[k]) for k in keys) + " |")
    return "\n".join(lines)


def crisis(results, dates, start, end) -> str:
    sel = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    rows = []
    for r in results:
        s = (r.dates >= np.datetime64(start)) & (r.dates <= np.datetime64(end))
        rows.append(f"{r.strategy.name}: {100 * (np.prod(1 + r.returns[s] / 100) - 1):+.1f}%")
    return "; ".join(rows)


def speed_benchmark(R, n=20, seed=1) -> str:
    """Inner worst case: Parisi-type dual versus a generic conic solver (CVXPY + Clarabel)."""
    try:
        import cvxpy as cp
    except ImportError:
        return "CVXPY not installed: benchmark skipped (pip install cvxpy)."
    rng = np.random.default_rng(seed)
    t_dual, t_total, t_solve, diff = [], [], [], []
    for _ in range(n):
        t0 = rng.integers(0, R.shape[0] - 504)
        l = -R[t0:t0 + 504] @ rng.dirichlet(np.ones(R.shape[1]))
        t = time.perf_counter(); a = worst_case_mean_variance(l, GAMMA, TAU, RHO, check_primal=False).objective
        t_dual.append(time.perf_counter() - t)
        nu = cp.Variable(l.size, nonneg=True)
        kl = cp.sum(-cp.entr(nu)) + np.log(l.size)
        m = nu @ l
        prob = cp.Problem(cp.Maximize(m + GAMMA * (nu @ l ** 2) - GAMMA * cp.square(m) - TAU * kl),
                          [cp.sum(nu) == 1, kl <= RHO])
        t = time.perf_counter(); prob.solve(solver="CLARABEL"); t_total.append(time.perf_counter() - t)
        t_solve.append(prob.solver_stats.solve_time); diff.append(abs(prob.value - a))
    return (f"Inner worst case on {n} random (window, portfolio) pairs, 504 days each:\n\n"
            f"| method | median time | max difference |\n|---|---|---|\n"
            f"| Parisi-type dual (2 scalars) | {1e3 * np.median(t_dual):.1f} ms | reference |\n"
            f"| CVXPY + Clarabel, solver only | {1e3 * np.median(t_solve):.1f} ms | {max(diff):.1e} |\n"
            f"| CVXPY + Clarabel, incl. modelling | {1e3 * np.median(t_total):.1f} ms | |\n")


def plots(results, sens_rho, sens_gamma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for r, c in zip(results, COLORS):
        w = np.cumprod(1 + r.returns / 100)
        ax[0].plot(r.dates, w, color=c, lw=1.4, label=r.strategy.name)
        ax[1].plot(r.dates, w / np.maximum.accumulate(w) - 1, color=c, lw=1.0)
    ax[0].set_yscale("log"); ax[0].set_ylabel("wealth (log scale)"); ax[0].legend(frameon=False, fontsize=9)
    ax[1].set_ylabel("drawdown"); ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout(); fig.savefig(OUT / "equity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for axis, (name, sens, key) in zip(ax, (("KL budget ρ", sens_rho, "rho"), ("risk aversion γ", sens_gamma, "gamma"))):
        for kind, c, lab in (("robust_mv", COLORS[3], "robust mean-variance"), ("markowitz", COLORS[1], "Markowitz")):
            pts = [(getattr(r.strategy, key), r.metrics) for r in sens if r.strategy.kind == kind]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            axis.plot(xs, [-p[1]["max drawdown"] for p in pts], "o-", color=c, label=f"{lab}: max drawdown")
            axis.plot(xs, [p[1]["annual volatility"] for p in pts], "s--", color=c, alpha=0.7, label=f"{lab}: volatility")
        axis.set_xlabel(name); axis.set_xscale("log"); axis.minorticks_off()
        ticks = sorted({getattr(r.strategy, key) for r in sens})
        axis.set_xticks(ticks); axis.set_xticklabels([f"{t:g}" for t in ticks])
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "sensitivity.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="last 4 years only, no sensitivity sweeps")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dates, tickers, prices = load_prices(tickers=FULL_HISTORY, start="1990-01-02")
    R, dates = simple_returns_pct(prices), dates[1:]
    if a.quick:
        R, dates = R[-1100:], dates[-1100:]

    t = time.time()
    results = [backtest(s, dates, R) for s in MAIN]
    print(table(results)); print(f"({time.time() - t:.0f} s)")

    sens_rho, sens_gamma = [], []
    if not a.quick:
        sens_rho = [backtest(Strategy(f"rho={r}", "robust_mv", gamma=GAMMA, rho=r, tau=TAU), dates, R)
                    for r in (0.01, 0.02, 0.05, 0.1, 0.2)]
        sens_gamma = [backtest(Strategy(f"{k} gamma={g}", k, gamma=g, rho=RHO, tau=TAU), dates, R)
                      for g in (0.02, 0.05, 0.1, 0.2) for k in ("markowitz", "robust_mv")]
    plots(results, sens_rho, sens_gamma)

    first, last = str(results[0].dates[0])[:10], str(results[0].dates[-1])[:10]
    report = [
        "# Results: KL-robust portfolios, 10 US stocks",
        f"Out-of-sample period {first} to {last}. Window 504 days, monthly rebalancing, 10 bps per unit "
        f"turnover, long-only. γ = {GAMMA}, ρ = {RHO}, τ = {TAU}. Universe: {', '.join(tickers)}.",
        "## Main comparison", table(results),
        "## Crises (total return over the period)",
        "- Dot-com crash, 2000-03-24 to 2002-10-09: " + crisis(results, dates, "2000-03-24", "2002-10-09"),
        "- Financial crisis, 2007-10-09 to 2009-03-09: " + crisis(results, dates, "2007-10-09", "2009-03-09"),
        "![wealth and drawdown](equity.png)",
    ]
    if sens_rho:
        report += ["## Sensitivity to the KL budget ρ (robust mean-variance)", table(sens_rho),
                   "## Sensitivity to the risk aversion γ", table(sens_gamma), "![sensitivity](sensitivity.png)"]
    report += ["## Speed of the inner worst case", speed_benchmark(R)]
    (OUT / "summary.md").write_text("\n\n".join(report) + "\n")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
