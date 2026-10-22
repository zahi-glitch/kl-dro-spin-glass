"""Live data: today's portfolios and the virtual $10,000 (used by scripts/track.py and the app, section 5.2).

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
import json
import math
import os
import threading
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
# Standard benchmarks for the virtual investment: fixed mixes, rebalanced monthly like the strategies.
BENCHMARKS = {
    "Benchmark: US stocks (SPY)": {"SPY": 1.0},
    "Benchmark: 60/40 stocks/bonds": {"SPY": 0.6, "IEF": 0.4},
}
LOG = Path(__file__).resolve().parents[2] / "paper_trading" / "log.csv"
VIRTUAL = Path(__file__).resolve().parents[2] / "paper_trading" / "virtual_portfolio.json"
COST_BPS = 10.0
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
    # Yahoo sometimes lacks the latest close for one fund for a few hours: carry its previous close forward
    # (at most two days) instead of dropping the whole day for every fund.
    data = data["Close"][list(tickers)].dropna(how="all").ffill(limit=2).dropna()
    # During US trading hours Yahoo's last row is today's live price, not a close: leave it out.
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    if len(data) and data.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 30):
        data = data.iloc[:-1]
    if len(data) < WINDOW + 1:
        raise RuntimeError(f"only {len(data)} days of prices downloaded; need at least {WINDOW + 1}")
    return data


def read_prices_csv(path, tickers):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    return df[list(tickers)].dropna()


_CACHE: dict = {}
_LOCK = threading.Lock()


def get_prices(tickers, csv_path=None, years=3):
    """Prices from a CSV (argument or TAPDUAL_PRICES_CSV) or downloaded; downloads are cached for the session."""
    csv_path = csv_path or os.environ.get("TAPDUAL_PRICES_CSV")
    if csv_path:
        return read_prices_csv(csv_path, tickers)
    key = (tuple(tickers), years)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = download_prices(tickers, years)
        return _CACHE[key]


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


# --------------------------------------------------------------------------- virtual investment
@dataclass
class Virtual:
    start: date                  # the investment is made at this day's closing prices
    capital: float
    dates: list                  # trading days from start to the latest price
    values: dict                 # strategy name -> value on each of those days
    weights_now: dict            # strategy name -> current weights (after drift)
    rebalances: int              # number of monthly rebalances after the start
    tickers: list


def load_virtual():
    if not VIRTUAL.exists():
        return None
    d = json.loads(VIRTUAL.read_text())
    return date.fromisoformat(d["start"]), float(d["capital"]), d.get("tickers", list(UNIVERSE))


def start_virtual(prices, capital=10_000.0):
    """Invest `capital` at the latest closing prices available (yesterday's close when run during the day)."""
    start = prices.index[-1].date()
    VIRTUAL.parent.mkdir(parents=True, exist_ok=True)
    VIRTUAL.write_text(json.dumps({"start": start.isoformat(), "capital": capital,
                                   "tickers": list(prices.columns), "created": date.today().isoformat()}, indent=2))
    return start, capital


def years_needed(start: date) -> int:
    """Years of prices to download: the estimation window before the start, plus the time since."""
    return max(3, math.ceil(2.3 + (date.today() - start).days / 365.25))


def simulate(prices, start: date, capital: float) -> Virtual:
    """Value of `capital` invested at `start` in each strategy, rebalanced on the first trading day of each month.

    Each rebalance uses only the returns up to that day (no look-ahead) and pays COST_BPS per unit of turnover;
    between rebalances the number of shares of each fund is held fixed, so the weights drift with prices.
    """
    P = prices.to_numpy(dtype=float)
    days = [d.date() for d in prices.index]
    if start not in days:
        raise RuntimeError(f"no prices for the start day {start}")
    s = days.index(start)
    if s < WINDOW:
        raise RuntimeError("not enough price history before the start day")
    R = 100.0 * (P[1:] / P[:-1] - 1.0)               # R[k] = return from day k to day k+1, in %
    values, weights_now, n_reb = {}, {}, 0
    tickers = list(prices.columns)
    plans = [(st.name, st.weights) for st in STRATEGIES]
    for name, mix in BENCHMARKS.items():
        if all(t in tickers for t in mix):
            fixed = np.array([mix.get(t, 0.0) for t in tickers])
            plans.append((name, lambda R, w=fixed: w))
    for k, (name, weights_of) in enumerate(plans):
        shares, v, path = None, capital, []
        for i in range(s, len(days)):
            if shares is not None:
                v = float(shares @ P[i])
            if i == s or days[i].month != days[i - 1].month:      # start, or first trading day of a month
                w_old = (shares * P[i] / v) if shares is not None else np.zeros(P.shape[1])
                w = weights_of(R[i - WINDOW:i])                   # returns up to and including day i
                v -= COST_BPS / 1e4 * float(np.abs(w - w_old).sum()) * v
                shares = v * w / P[i]
                if i > s and k == 0:
                    n_reb += 1
            path.append(v)
        values[name] = np.array(path)
        weights_now[name] = shares * P[-1] / float(shares @ P[-1])
    return Virtual(start, capital, days[s:], values, weights_now, n_reb, list(prices.columns))
