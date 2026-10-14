"""Paper trading: today's robust portfolio from the latest market prices.

    pip install -e ".[live]"           # adds yfinance (free Yahoo Finance data) and pandas
    python scripts/today.py            # download prices, print today's weights, append to the log

Run it once a month (for example on the first working day). Each run appends one row per strategy to
paper_trading/log.csv, so after a few months you have a real out-of-sample record to compare.

This is an educational tool, not investment advice.

Defaults, chosen once and kept fixed:
  universe  ten liquid ETFs, one per major asset class (see UNIVERSE below)
  window    the last 504 trading days (two years) of daily returns
  settings  the same as the backtest: gamma = 0.05, rho = 0.05, tau = 0.1
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np

from tapdual.backtest import Strategy
from tapdual.dro import worst_case_mean_variance

UNIVERSE = {
    "SPY": "US stocks (S&P 500)",
    "EFA": "stocks, other developed countries",
    "EEM": "stocks, emerging markets",
    "VNQ": "US real estate",
    "TLT": "long-term US government bonds",
    "IEF": "medium-term US government bonds",
    "SHY": "short-term US government bonds (near cash)",
    "LQD": "US corporate bonds",
    "GLD": "gold",
    "DBC": "commodities",
}
WINDOW, GAMMA, RHO, TAU = 504, 0.05, 0.05, 0.1
STRATEGIES = [
    Strategy("Equal weight", "equal"),
    Strategy("Markowitz", "markowitz", gamma=GAMMA),
    Strategy("Robust mean-variance", "robust_mv", gamma=GAMMA, rho=RHO, tau=TAU),
]
LOG = Path(__file__).resolve().parents[1] / "paper_trading" / "log.csv"


def download_prices(tickers, years=3):
    """Daily closing prices adjusted for dividends and splits, from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit('yfinance is not installed. Run:  pip install -e ".[live]"')
    data = yf.download(list(tickers), period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    data = data[list(tickers)].dropna()
    if len(data) < WINDOW + 1:
        raise SystemExit(f"only {len(data)} days of prices downloaded; need at least {WINDOW + 1}")
    return data


def read_prices_csv(path, tickers):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    return df[list(tickers)].dropna()


def last_logged_weights(strategy_name, tickers):
    if not LOG.exists():
        return None
    rows = [r for r in csv.DictReader(LOG.open()) if r["strategy"] == strategy_name]
    if not rows or any(t not in rows[-1] for t in tickers):
        return None                                   # no earlier run with the same assets
    return np.array([float(rows[-1][t]) for t in tickers])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", help="use a CSV of prices (date column + one column per ticker) instead of downloading")
    ap.add_argument("--tickers", nargs="+", default=list(UNIVERSE), help="override the default universe")
    ap.add_argument("--no-log", action="store_true", help="print only, do not append to the log")
    a = ap.parse_args()
    tickers = a.tickers

    prices = read_prices_csv(a.prices, tickers) if a.prices else download_prices(tickers)
    as_of = prices.index[-1].date()
    P = prices.to_numpy(dtype=float)
    R = 100.0 * (P[1:] / P[:-1] - 1.0)[-WINDOW:]       # last two years of daily returns, in %

    print(f"\nPrices up to {as_of}; estimation window: the last {len(R)} trading days.")
    print("Educational tool, not investment advice.\n")
    weights = {s.name: s.weights(R) for s in STRATEGIES}

    width = max(len(t) for t in tickers)
    print(" " * (width + 2) + "".join(f"{s.name:>22}" for s in STRATEGIES) + "   asset")
    for i, t in enumerate(tickers):
        print(f"{t:<{width + 2}}" + "".join(f"{100 * weights[s.name][i]:>21.1f}%" for s in STRATEGIES)
              + f"   {UNIVERSE.get(t, '')}")

    x = weights["Robust mean-variance"]
    r = worst_case_mean_variance(-R @ x, GAMMA, TAU, RHO, check_primal=False)
    dates = prices.index[-len(R):]
    top = np.argsort(r.weights)[::-1][:5]
    print("\nThe robust portfolio is sized against these past days (the adversary's favourites):")
    for k in top:
        print(f"  {dates[k].date()}   portfolio return {R[k] @ x:+.2f}%   weight x{r.weights[k] * len(R):.1f}")

    prev = last_logged_weights("Robust mean-variance", tickers)
    if prev is not None:
        print("\nTrades for the robust portfolio since the last log entry (percentage points of the portfolio):")
        trades = [(t, d) for t, d in zip(tickers, 100 * (x - prev)) if abs(d) >= 0.5]
        for t, d in trades:
            print(f"  {t:<{width}}  {'buy ' if d > 0 else 'sell'} {abs(d):.1f}")
        if not trades:
            print("  none (all changes below 0.5 points)")

    if not a.no_log:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG.exists()
        with LOG.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["run_date", "prices_as_of", "strategy", *tickers])
            for s in STRATEGIES:
                w.writerow([date.today().isoformat(), as_of.isoformat(), s.name,
                            *[f"{v:.6f}" for v in weights[s.name]]])
        print(f"\nLogged to {LOG}")


if __name__ == "__main__":
    main()
