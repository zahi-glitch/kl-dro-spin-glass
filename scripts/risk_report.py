"""Risk report: daily VaR / ES / EVaR forecasts with model risk, backtested on 28 years of US stocks.

Portfolio: equal weight in the 10 stocks of the bundled data (rebalanced daily), 1990-2018.
Four ways to forecast tomorrow's risk from the last 250 days:
  historical            the past losses as they were (historical simulation)
  historical + robust   the worst case over a KL ball around them (model risk)
  filtered              the past losses rescaled by today's EWMA volatility (filtered historical simulation)
  filtered + robust     both
Writes results/risk/{summary.md, results.json, crisis.png}.

    python scripts/risk_report.py          # about 2 minutes
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tapdual.backtest import DATA, FULL_HISTORY  # noqa: E402
from tapdual.risk import (ALPHA_VAR, WINDOW, backtest_report, ewma_volatility, kupiec, rho_confidence,  # noqa: E402
                          robust_var, rolling_risk)

OUT = Path(__file__).resolve().parents[1] / "results" / "risk"
SWEEP = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]          # multiples of the 95% radius


def p_fmt(v):
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def main():
    t0 = time.time()
    prices = pd.read_csv(DATA, parse_dates=[0], index_col=0)[FULL_HISTORY].dropna()
    losses = -100.0 * prices.pct_change().iloc[1:].mean(axis=1)
    rho = rho_confidence(WINDOW)
    frames = {"historical": rolling_risk(losses, WINDOW, rho),
              "filtered": rolling_risk(losses, WINDOW, rho, filtered=True)}
    reps = {k: backtest_report(f) for k, f in frames.items()}
    risk = frames["historical"]
    n = len(risk)
    y08 = risk.index.year == 2008

    methods = []                                   # (label, frame key, column prefix)
    for key in frames:
        methods += [(key, key, ""), (f"{key} + robust", key, "robust ")]

    lines = ["# Risk report", "",
             f"Equal-weight portfolio of {len(FULL_HISTORY)} US stocks, daily losses in % of value, "
             f"{risk.index[0].date()} to {risk.index[-1].date()} ({n} days). Every forecast uses only the "
             f"previous {WINDOW} days. *Filtered*: past losses rescaled by today's EWMA volatility "
             f"(RiskMetrics, λ = 0.94). *Robust*: worst case over a KL ball of radius ρ = {rho:.4f}, a 95% "
             f"confidence region for a tail probability estimated from {WINDOW} days.", "",
             "## Value-at-risk 99%", "",
             f"Expected: {0.01 * n:.0f} exceptions in total, about 2.5 per year.", "",
             "| method | average VaR (%) | exceptions | in 2008 | Kupiec p | Christoffersen p | "
             "P(exception after exception) | 250-day windows green | red |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, key, pre in methods:
        r, f = reps[key].loc[pre + "VaR 99%"], frames[key]
        e08 = int((f["loss"] > f[pre + "VaR 99%"])[y08].sum())
        lines.append(f"| {label} | {r['average level (%)']:.2f} | {int(r['exceptions'])} | {e08} | "
                     f"{p_fmt(r['Kupiec p'])} | {p_fmt(r['Christoffersen p (cc)'])} | "
                     f"{100 * r['P(exc | exc yesterday)']:.0f}% | {100 * r['windows green']:.0f}% | "
                     f"{100 * r['windows red']:.0f}% |")
    lines += ["", "Kupiec: is the exception rate right? Christoffersen: the right rate *and* no clustering. "
              "Small p-values reject the forecast. Basel zones: green 0-4 exceptions in 250 days, red 10 or more.",
              "", "## Expected shortfall 97.5% and EVaR", "",
              "| method | average ES (%) | Acerbi-Szekely Z2 |", "|---|---|---|"]
    for label, key, pre in methods:
        r = reps[key].loc[pre + "ES 97.5%"]
        lines.append(f"| {label} | {r['average level (%)']:.2f} | {r['Acerbi-Szekely Z2']:.2f} |")
    for key in frames:
        r = reps[key].loc["EVaR 97.5%"]
        lines.append(f"| {key}: EVaR 97.5% | {r['average level (%)']:.2f} | (exceeded on {int(r['exceptions'])} days) |")
    lines += ["", "Z2 is about 0 when ES is right, positive when it is too low, negative when too high; roughly, "
              "|Z2| > 0.7 is significant at 5%. EVaR = worst-case mean over the KL ball of radius "
              "log(1/(1 − 0.975)): an upper bound on ES, not a quantile, so it has no target exception rate.", ""]

    # rho sweep for VaR 99%
    L = losses.to_numpy(dtype=float)
    sig = ewma_volatility(L)
    sweep = {}
    lines += ["## How much distrust? VaR 99% exceptions as the KL radius grows", "",
              "| ρ (× 95% radius) | " + " | ".join(f"{s:g}" for s in SWEEP) + " |", "|---" * (len(SWEEP) + 1) + "|"]
    for key in frames:
        cells = []
        for s in SWEEP:
            exc = []
            for t in range(WINDOW, L.size):
                W = L[t - WINDOW:t] if key == "historical" else L[t - WINDOW:t] / sig[t - WINDOW:t] * sig[t]
                exc.append(L[t] > robust_var(W, ALPHA_VAR, s * rho))
            k = kupiec(np.array(exc), 0.01)
            sweep[f"{key} {s:g}"] = k
            cells.append(f"{k['exceptions']} (p {p_fmt(k['p_uc'])})")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", f"Target: {0.01 * n:.0f} exceptions. p is Kupiec's test of the exception rate.", ""]

    (OUT.mkdir(parents=True, exist_ok=True))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump({"rho": rho, "window": WINDOW,
               "report": {k: r.where(r.notna(), None).to_dict(orient="index") for k, r in reps.items()},
               "sweep": sweep}, open(OUT / "results.json", "w"), indent=2, default=float)

    # Crisis chart
    h, f = frames["historical"].loc["2007-06-01":"2009-12-31"], frames["filtered"].loc["2007-06-01":"2009-12-31"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(h.index, h["loss"].clip(lower=0), width=1.5, color="#b8bfc8", label="daily loss")
    ax.plot(h.index, h["VaR 99%"], color="#1f6fb4", lw=1.6, label="VaR 99%, historical")
    ax.plot(h.index, h["robust VaR 99%"], color="#1f6fb4", lw=1.2, ls="--", label="VaR 99%, historical + robust")
    ax.plot(f.index, f["VaR 99%"], color="#d1495b", lw=1.6, label="VaR 99%, filtered")
    ax.plot(f.index, f["robust VaR 99%"], color="#d1495b", lw=1.2, ls="--", label="VaR 99%, filtered + robust")
    exc = h[h["loss"] > h["VaR 99%"]]
    ax.scatter(exc.index, exc["loss"], s=16, color="black", zorder=3, label="exception of historical VaR 99%")
    ax.set_ylabel("loss, % of portfolio value")
    ax.set_title("2007-2009: daily losses and the VaR forecast made the day before")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "crisis.png", dpi=130); plt.close(fig)
    print(f"\nwritten to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
