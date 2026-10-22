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
SPEED: dict = {}   # filled by speed_study, saved to results.json for the app
GAMMA, RHO, TAU = 0.05, 0.05, 0.1

MAIN = [
    Strategy("Equal weight", "equal"),
    Strategy("Markowitz", "markowitz", gamma=GAMMA),
    Strategy("Minimum variance", "minvar"),
    Strategy("Robust mean (EVaR)", "robust_mean", rho=RHO),
    Strategy("Robust mean-variance (KSS23 dual)", "robust_mv", gamma=GAMMA, rho=RHO, tau=TAU),
]
COLORS = ["#8a8983", "#eb6834", "#1D9E75", "#7f77dd", "#2a78d6"]
RHOS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def fmt(k, v):
    pct = {"annual return", "annual volatility", "max drawdown", "worst day", "CVaR 5% (daily)",
           "differs from min. variance", "differs from Markowitz"}
    return f"{100 * v:.1f}%" if k in pct else f"{v:.2f}"


def table(results) -> str:
    keys = list(results[0].metrics)
    lines = ["| strategy | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for r in results:
        lines.append(f"| {r.strategy.name} | " + " | ".join(fmt(k, r.metrics[k]) for k in keys) + " |")
    return "\n".join(lines)


def crisis_return(r, start, end) -> float:
    s = (r.dates >= np.datetime64(start)) & (r.dates <= np.datetime64(end))
    return float(np.prod(1 + r.returns[s] / 100) - 1)


def crisis(results, dates, start, end) -> str:
    rows = []
    for r in results:
        s = (r.dates >= np.datetime64(start)) & (r.dates <= np.datetime64(end))
        rows.append(f"{r.strategy.name}: {100 * (np.prod(1 + r.returns[s] / 100) - 1):+.1f}%")
    return "; ".join(rows)


def speed_study(dates, R, reps=5, seed=0, rebalances=24) -> str:
    """Inner worst case: Parisi-type dual versus a generic conic solver (CVXPY + Clarabel, SCS fallback).

    1. Time per worst case as the window grows.  2. A whole robust mean-variance backtest run twice,
    once with each inner solver: wall-clock time and the largest difference in portfolio weights.
    """
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return "CVXPY not installed: speed study skipped (pip install cvxpy)."
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tapdual.backtest import ConicWorstCase

    rng = np.random.default_rng(seed)
    rows, ns, t_dual, t_conic, fails = [], (126, 252, 504, 1008, 2016, 4032), [], [], []
    for n in ns:
        conic, td, tc, diff = ConicWorstCase(n, GAMMA, RHO, TAU), [], [], []
        for _ in range(reps):
            t0 = rng.integers(0, R.shape[0] - n)
            l = -R[t0:t0 + n] @ rng.dirichlet(np.ones(R.shape[1]))
            t = time.perf_counter(); a = worst_case_mean_variance(l, GAMMA, TAU, RHO, check_primal=False).objective
            td.append(time.perf_counter() - t)
            t = time.perf_counter(); v, _ = conic(l); tc.append(time.perf_counter() - t); diff.append(abs(v - a))
        t_dual.append(np.median(td)); t_conic.append(np.median(tc)); fails.append(conic.failures)
        rows.append(f"| {n} | {1e3 * t_dual[-1]:.1f} ms | {1e3 * t_conic[-1]:.1f} ms | {t_conic[-1] / t_dual[-1]:.0f}x "
                    f"| {conic.failures} of {reps} | {max(diff):.0e} |")

    D, RR = dates[-504 - 21 * rebalances:], R[-504 - 21 * rebalances:]
    t = time.time(); a = backtest(Strategy("dual", "robust_mv", gamma=GAMMA, rho=RHO, tau=TAU), D, RR); e_dual = time.time() - t
    t = time.time(); b = backtest(Strategy("conic", "robust_mv_conic", gamma=GAMMA, rho=RHO, tau=TAU), D, RR)
    e_conic = time.time() - t
    wdiff = float(np.abs(a.weights - b.weights).max())
    SPEED.update(days=list(ns), dual_ms=[1e3 * x for x in t_dual], conic_ms=[1e3 * x for x in t_conic],
                 conic_failures=fails, reps=reps, backtest_dual_s=e_dual, backtest_conic_s=e_conic,
                 rebalances=rebalances, max_weight_difference=wdiff)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax[0].loglog(ns, 1e3 * np.array(t_conic), "o-", color="#8a8983", label="generic conic solver (CVXPY + Clarabel)")
    ax[0].loglog(ns, 1e3 * np.array(t_dual), "o-", color="#2a78d6", label="Parisi-type dual (2 scalars)")
    for n, tc, f in zip(ns, t_conic, fails):
        if f:
            ax[0].annotate(f"{f} failure{'s' if f > 1 else ''}", (n, 1e3 * tc), textcoords="offset points",
                           xytext=(-10, 8), fontsize=8, color="#eb6834", ha="right")
    ax[0].set_xticks(ns); ax[0].set_xticklabels([str(n) for n in ns]); ax[0].minorticks_off()
    ax[0].set_xlabel("days in the window"); ax[0].set_ylabel("time per worst case (ms)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("one worst case", fontsize=10)
    ax[1].barh([1, 0], [e_conic, e_dual], color=["#8a8983", "#2a78d6"])
    ax[1].set_yticks([1, 0]); ax[1].set_yticklabels(["conic solver", "dual"])
    for y, v in ((1, e_conic), (0, e_dual)):
        ax[1].text(v, y, f"  {v:.1f} s", va="center", fontsize=9)
    ax[1].set_xlim(0, e_conic * 1.3); ax[1].set_xlabel("seconds")
    ax[1].set_title(f"whole backtest, {rebalances} rebalances\nsame portfolios (weights differ ≤ {wdiff:.0e})", fontsize=10)
    for a_ in ax:
        a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "speed.png", dpi=160); plt.close(fig)

    return ("Time per inner worst case (median of %d random windows and portfolios), and the largest "
            "difference in value:\n\n| days | dual | conic solver | speed-up | conic failures | max difference |\n"
            "|---|---|---|---|---|---|\n" % reps + "\n".join(rows) +
            f"\n\nWhole robust mean-variance backtest over the last {rebalances} monthly rebalances, run twice: "
            f"**{e_dual:.1f} s with the dual, {e_conic:.1f} s with the conic solver** "
            f"({e_conic / e_dual:.0f}x), same portfolios (largest weight difference {wdiff:.0e}).\n\n"
            "![speed](speed.png)\n")


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

    if not sens_rho:
        return
    by = {r.strategy.name: r for r in results}
    rhos = [r.strategy.rho for r in sens_rho]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(rhos, [r.metrics["differs from min. variance"] for r in sens_rho], "o-", color=COLORS[2],
               label="from minimum variance")
    ax[0].plot(rhos, [r.metrics["differs from Markowitz"] for r in sens_rho], "o-", color=COLORS[1],
               label="from Markowitz")
    ax[0].set_ylabel("share of the portfolio held differently"); ax[0].set_ylim(0, None)
    ax[0].set_title("robust portfolio: who does it resemble?", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
    for key, style, lab in (("annual return", "o-", "return"), ("annual volatility", "s--", "volatility")):
        ax[1].plot(rhos, [r.metrics[key] for r in sens_rho], style, color=COLORS[4], label=f"robust: {lab}")
        for name, c in (("Markowitz", COLORS[1]), ("Minimum variance", COLORS[2])):
            ax[1].axhline(by[name].metrics[key], color=c, lw=1, ls=":" if key == "annual volatility" else "-")
    ax[1].set_title("return and volatility (lines: Markowitz, min. variance)", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.set_xscale("log"); a_.minorticks_off(); a_.set_xticks(rhos)
        a_.set_xticklabels([f"{r:g}" for r in rhos], rotation=45); a_.set_xlabel("KL budget ρ (per day)")
        a_.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout(); fig.savefig(OUT / "sensitivity.png", dpi=160); plt.close(fig)


def distance(a, b) -> float:
    """Average share of the portfolio held differently: half the L1 distance between weights, per rebalance."""
    return float(0.5 * np.abs(a.weights - b.weights).sum(1).mean())


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
        by = {r.strategy.name: r for r in results}
        sens_rho = [backtest(Strategy(f"rho={r}", "robust_mv", gamma=GAMMA, rho=r, tau=TAU), dates, R) for r in RHOS]
        for r in sens_rho:
            r.metrics["differs from min. variance"] = distance(r, by["Minimum variance"])
            r.metrics["differs from Markowitz"] = distance(r, by["Markowitz"])
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
        report += ["## Sensitivity to the KL budget ρ (robust mean-variance)",
                   "How big is a budget? Within H(ν | μ) ≤ ρ the adversary can lower an asset's average daily "
                   "return by about √(2ρ) daily standard deviations: 0.32σ at ρ = 0.05, 0.03σ at ρ = 0.0005. "
                   "Daily means are only a few hundredths of σ, so a large budget wipes out the information in the "
                   "averages and the robust portfolio becomes a minimum-variance portfolio; a small budget brings it "
                   "back towards Markowitz.",
                   table(sens_rho),
                   "## Sensitivity to the risk aversion γ", table(sens_gamma), "![sensitivity](sensitivity.png)"]
    report += ["## Speed: the dual against a generic conic solver", speed_study(dates, R)]
    (OUT / "summary.md").write_text("\n\n".join(report) + "\n")
    crises = {name: {r.strategy.name: crisis_return(r, a_, b_) for r in results}
              for name, a_, b_ in (("Dot-com crash 2000-02", "2000-03-24", "2002-10-09"),
                                   ("Financial crisis 2007-09", "2007-10-09", "2009-03-09"))}
    import json
    (OUT / "results.json").write_text(json.dumps({
        "period": [first, last], "universe": tickers, "gamma": GAMMA, "rho": RHO, "tau": TAU,
        "window": 504, "step": 21, "cost_bps": 10,
        "metrics": {r.strategy.name: r.metrics for r in results}, "crises": crises, "speed": SPEED,
        "rho_sweep": [{"rho": r.strategy.rho, **r.metrics} for r in sens_rho],
    }, indent=2))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
