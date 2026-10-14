"""Today's portfolios from the latest market prices (shared by scripts/today.py and the app).

Educational tool, not investment advice.

Defaults, chosen once and kept fixed:
  universe  nine liquid ETFs, one per major asset class. No cash-like fund: with one, the robust portfolio
            simply goes to cash, because at this budget it behaves like a minimum-variance portfolio.
  window    the last 504 trading days (two years) of daily returns
  settings  the same as the backtest: gamma = 0.05, rho = 0.05, tau = 0.1

Set the environment variable TAPDUAL_PRICES_CSV to a price file to work offline (no download).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .backtest import Strategy
from .dro import worst_case_mean_variance

UNIVERSE = {
    "SPY": "US stocks (S&P 500)",
    "EFA": "stocks, other developed countries",
    "EEM": "stocks, emerging markets",
    "VNQ": "US real estate",
    "TLT": "long-term US government bonds",
    "IEF": "medium-term US government bonds",
    "LQD": "US corporate bonds",
    "GLD": "gold",
    "DBC": "commodities",
}
WINDOW, GAMMA, RHO, TAU = 504, 0.05, 0.05, 0.1
STRATEGIES = [
    Strategy("Equal weight", "equal"),
    Strategy("Markowitz", "markowitz", gamma=GAMMA),
    Strategy("Minimum variance", "minvar"),
    Strategy("Robust mean-variance", "robust_mv", gamma=GAMMA, rho=RHO, tau=TAU),
]
ROBUST = "Robust mean-variance"
LOG = Path(__file__).resolve().parents[2] / "paper_trading" / "log.csv"
HEADER_FIXED = ["run_date", "prices_as_of", "strategy"]


# --------------------------------------------------------------------------- prices
def download_prices(tickers, years=3):
    """Daily closing prices adjusted for dividends and splits, from Yahoo Finance (needs internet)."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError('yfinance is not installed. In the project folder run:  pip install -e ".[live]"') from e
    data = yf.download(list(tickers), period=f"{years}y", auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise RuntimeError("no prices downloaded: check the internet connection and try again")
    data = data["Close"][list(tickers)].dropna()
    if len(data) < WINDOW + 1:
        raise RuntimeError(f"only {len(data)} days of prices downloaded; need at least {WINDOW + 1}")
    return data


def read_prices_csv(path, tickers):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    return df[list(tickers)].dropna()


def get_prices(tickers, csv_path=None):
    csv_path = csv_path or os.environ.get("TAPDUAL_PRICES_CSV")
    return read_prices_csv(csv_path, tickers) if csv_path else download_prices(tickers)


# --------------------------------------------------------------------------- portfolios
@dataclass
class Today:
    as_of: date
    tickers: list
    weights: dict                 # strategy name -> weights (array)
    worst_days: list              # (date, robust portfolio return in %, weight multiplier), largest first
    window: int


def compute(prices, n_worst=5) -> Today:
    tickers = list(prices.columns)
    P = prices.to_numpy(dtype=float)
    R = 100.0 * (P[1:] / P[:-1] - 1.0)[-WINDOW:]       # last two years of daily returns, in %
    weights = {s.name: s.weights(R) for s in STRATEGIES}
    x = weights[ROBUST]
    r = worst_case_mean_variance(-R @ x, GAMMA, TAU, RHO, check_primal=False)
    dates = prices.index[-len(R):]
    top = np.argsort(r.weights)[::-1][:n_worst]
    worst = [(dates[k].date(), float(R[k] @ x), float(r.weights[k] * len(R))) for k in top]
    return Today(prices.index[-1].date(), tickers, weights, worst, len(R))


# --------------------------------------------------------------------------- paper-trading log
def start_fresh_log_if_universe_changed(tickers) -> str | None:
    """One log per list of assets: if the assets changed, move the old log aside. Returns a message or None."""
    if not LOG.exists():
        return None
    with LOG.open() as f:
        header = next(csv.reader(f), [])
    if header != HEADER_FIXED + list(tickers):
        old = LOG.with_name(f"log-before-{date.today().isoformat()}.csv")
        LOG.rename(old)
        return f"The list of assets changed: the old log was moved to {old.name} and a new log starts today."
    return None


def last_logged_weights(strategy_name, tickers):
    if not LOG.exists():
        return None
    with LOG.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER_FIXED + list(tickers):
            return None
        rows = [r for r in reader if r["strategy"] == strategy_name]
    return np.array([float(rows[-1][t]) for t in tickers]) if rows else None


def trades_since_last_log(today: Today, threshold=0.5):
    """[(ticker, change in percentage points)] for the robust portfolio, or None if nothing is logged yet."""
    prev = last_logged_weights(ROBUST, today.tickers)
    if prev is None:
        return None
    return [(t, d) for t, d in zip(today.tickers, 100 * (today.weights[ROBUST] - prev)) if abs(d) >= threshold]


def append_log(today: Today) -> list:
    """Append today's weights to the log; returns messages for the user."""
    msgs = []
    moved = start_fresh_log_if_universe_changed(today.tickers)
    if moved:
        msgs.append(moved)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER_FIXED + list(today.tickers))
        for s in STRATEGIES:
            w.writerow([date.today().isoformat(), today.as_of.isoformat(), s.name,
                        *[f"{v:.6f}" for v in today.weights[s.name]]])
    msgs.append(f"Logged to {LOG}")
    return msgs
