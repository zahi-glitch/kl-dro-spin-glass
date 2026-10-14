"""Paper trading: today's robust portfolio from the latest market prices.

    pip install -e ".[live]"           # adds yfinance (free Yahoo Finance data) and pandas
    python scripts/today.py            # download prices, print today's weights, append to the log

Run it once a month (for example on the first working day). Each run appends one row per strategy to
paper_trading/log.csv, so after a few months you have a real out-of-sample record to compare.
The same computation is in the app: section 5, tab "Today: live data".

This is an educational tool, not investment advice. Defaults and settings: see tapdual/live.py.
"""
from __future__ import annotations

import argparse

import numpy as np

from tapdual.live import (RHO, STRATEGIES, UNIVERSE, append_log, compute, get_prices, start_fresh_log_if_universe_changed,
                          trades_since_last_log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", help="use a CSV of prices (date column + one column per ticker) instead of downloading")
    ap.add_argument("--tickers", nargs="+", default=list(UNIVERSE), help="override the default universe")
    ap.add_argument("--no-log", action="store_true", help="print only, do not append to the log")
    a = ap.parse_args()

    try:
        today = compute(get_prices(a.tickers, a.prices))
    except RuntimeError as e:
        raise SystemExit(str(e))

    print(f"\nPrices up to {today.as_of}; estimation window: the last {today.window} trading days.")
    print(f"Budget ρ = {RHO}: the adversary may lower each fund's average daily return by about "
          f"{np.sqrt(2 * RHO):.2f} standard deviations, which is more than any fund's real daily average, "
          f"so the robust portfolio stays close to minimum variance.")
    print("Educational tool, not investment advice.\n")

    width = max(len(t) for t in today.tickers)
    print(" " * (width + 2) + "".join(f"{s.name:>21}" for s in STRATEGIES) + "   asset")
    for i, t in enumerate(today.tickers):
        print(f"{t:<{width + 2}}" + "".join(f"{100 * today.weights[s.name][i]:>20.1f}%" for s in STRATEGIES)
              + f"   {UNIVERSE.get(t, '')}")

    print("\nThe robust portfolio is sized against these past days (the adversary's favourites):")
    for d, ret, mult in today.worst_days:
        print(f"  {d}   portfolio return {ret:+.2f}%   weight x{mult:.1f}")

    if not a.no_log:
        moved = start_fresh_log_if_universe_changed(today.tickers)
        if moved:
            print("\n" + moved)
    trades = trades_since_last_log(today)
    if trades is not None:
        print("\nTrades for the robust portfolio since the last log entry (percentage points of the portfolio):")
        for t, d in trades:
            print(f"  {t:<{width}}  {'buy ' if d > 0 else 'sell'} {abs(d):.1f}")
        if not trades:
            print("  none (all changes below 0.5 points)")

    if not a.no_log:
        for m in append_log(today):
            print("\n" + m)


if __name__ == "__main__":
    main()
