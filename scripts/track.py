"""Automatic tracking (run every weekday by .github/workflows/paper-trading.yml, or by hand).

    python scripts/track.py

1. The virtual $10,000: starts it at the latest close if it does not exist yet, then writes its value day by
   day to paper_trading/virtual_values.csv and a short summary to paper_trading/virtual_report.md.
2. The monthly paper-trading log: on the first run of each calendar month, appends today's weights of the four
   strategies to paper_trading/log.csv.

Educational tool, not investment advice.
"""
from __future__ import annotations

import csv
from datetime import date

from tapdual import live

VALUES = live.VIRTUAL.with_name("virtual_values.csv")
REPORT = live.VIRTUAL.with_name("virtual_report.md")


def logged_this_month(today: date) -> bool:
    if not live.LOG.exists():
        return False
    with live.LOG.open() as f:
        return any(r["run_date"][:7] == today.isoformat()[:7] for r in csv.DictReader(f))


def main():
    saved = live.load_virtual()
    tickers = saved[2] if saved else list(live.UNIVERSE)
    prices = live.get_prices(tickers, years=live.years_needed(saved[0]) if saved else 3)
    if saved is None:
        start, capital = live.start_virtual(prices)
        print(f"Started a virtual ${capital:,.0f} at the close of {start}.")
    else:
        start, capital = saved[:2]

    v = live.simulate(prices, start, capital)
    names = list(v.values)
    VALUES.parent.mkdir(parents=True, exist_ok=True)
    with VALUES.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", *names])
        for i, d in enumerate(v.dates):
            w.writerow([d.isoformat(), *[f"{v.values[n][i]:.2f}" for n in names]])

    rows = "\n".join(f"| {n} | ${v.values[n][-1]:,.2f} | {v.values[n][-1] - capital:+,.2f} | "
                     f"{100 * (v.values[n][-1] / capital - 1):+.2f}% |" for n in names)
    REPORT.write_text(
        f"# Virtual ${capital:,.0f}\n\n"
        f"Invested at the close of {start}; latest prices {v.dates[-1]} ({len(v.dates) - 1} trading days, "
        f"{v.rebalances} monthly rebalances). Educational, no real money.\n\n"
        f"| strategy | worth now | gain ($) | gain (%) |\n|---|---|---|---|\n{rows}\n\n"
        f"Day-by-day values: `virtual_values.csv`. Updated automatically on weekdays.\n")
    print(REPORT.read_text())

    today = date.today()
    if not logged_this_month(today):
        for m in live.append_log(live.compute(live.get_prices(list(live.UNIVERSE)))):
            print(m)
    else:
        print("Monthly log: already saved this month.")


if __name__ == "__main__":
    main()
