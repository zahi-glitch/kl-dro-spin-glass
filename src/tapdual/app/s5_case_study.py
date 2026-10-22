"""Section 5.1: the case study. The robust portfolio on 28 years of real stock prices (results from scripts/backtest.py)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QHBoxLayout, QVBoxLayout, QWidget)

from .common import Picture, _SubSection, _big, _gap, _heading, _table
from .widgets import Card, label


RESULTS = Path(__file__).resolve().parents[3] / "results" / "backtest"



class CaseStudyPage(QWidget):
    ROBUST, MARKOWITZ, MINVAR, EQUAL = ("Robust mean-variance (KSS23 dual)", "Markowitz", "Minimum variance",
                                        "Equal weight")

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(16)
        path = RESULTS / "results.json"
        if not path.exists():
            lay.addWidget(label("No results yet. Run <b>python scripts/backtest.py</b> in the project folder "
                                "(about 10 minutes) and reopen the app.", "body"))
            return
        import json
        res = json.loads(path.read_text())
        M, S = res["metrics"], res["speed"]
        rob, mk, mv = M[self.ROBUST], M[self.MARKOWITZ], M[self.MINVAR]
        speedup = S["backtest_conic_s"] / S["backtest_dual_s"]
        pct = lambda v: f"{100 * v:.1f}%"

        lay.addWidget(label(
            "Everything so far used one sample of returns. Here the robust portfolio meets real market data.", "body"))

        names = {"AAPL": "Apple", "GE": "General Electric", "AMD": "AMD", "WMT": "Walmart",
                 "BAC": "Bank of America", "T": "AT&amp;T", "XOM": "ExxonMobil", "BBY": "Best Buy",
                 "PFE": "Pfizer", "JPM": "JPMorgan Chase"}
        first, last = res["period"][0], res["period"][1]
        data = Card("The data")
        data.add(_table(["", ""], [
            ["<b>What</b>", "daily closing prices of 10 large US companies, adjusted for dividends and splits"],
            ["<b>Companies</b>", ", ".join(f"{names.get(t, t)} ({t})" for t in res["universe"])],
            ["<b>Period</b>", f"January 1990 to April 2018; results are measured from {first[:7]} to {last[:7]} "
                              "(the first two years are only used for estimation)"],
            ["<b>Includes</b>", "the dot-com crash (2000–02) and the financial crisis (2007–09)"],
            ["<b>Source</b>", "the test data of PyPortfolioOpt, an open-source portfolio library (MIT licence); "
                              "file src/tapdual/data/stock_prices.csv in this project"],
            ["<b>Method</b>", "every month each strategy is re-estimated on the last two years of daily returns, "
                              "long-only, paying 0.1% on what it trades; settings fixed before looking at the "
                              f"results (γ = {res['gamma']}, ρ = {res['rho']} per day)"],
            ["<b>Caveat</b>", "these are well-known companies chosen today, so they survived: absolute returns look "
                              "better than a random choice made in 1990 would have (survivorship bias)"],
        ], first_width=120, text=True))
        lay.addWidget(data)

        # 1 ---------------------------------------------------------------- plain words
        lay.addWidget(_heading("1", "In plain words"))

        penalty = Card("A penalty for risk")
        penalty.add(label("To choose a portfolio, give each candidate a score and pick the highest:", "body"))
        penalty.add(label("<b>score = expected gain − penalty × risk</b>", "lead"))
        penalty.add(label("Risk is how much the portfolio swings up and down; the penalty is how much you dislike "
                          "those swings. Two candidates: <b>A</b> gains 10% a year and swings ±20%, <b>B</b> gains "
                          "7% and swings ±8%.", "body"))
        penalty.add(_table(["penalty", "score of A", "score of B", "winner"],
                           [["0 (swings don't matter)", "10", "7", "A"],
                            ["0.25", "10 − 5 = 5", "7 − 2 = 5", "tie"],
                            ["0.5", "10 − 10 = 0", "7 − 4 = 3", "B"]], first_width=190))
        penalty.add(label("A bigger penalty pushes you towards calmer portfolios.", "body"))

        knob = Card("The budget ρ is a knob")
        knob.add(label("Robustness adds exactly such a penalty. How big it is depends on one number, the budget ρ: "
                       "think of a volume knob.", "body"))
        knob.add(_table(["knob", "what you get", "what it believes"],
                        [["ρ = 0 (all the way down)", "Markowitz", "past average returns are right"],
                         ["ρ in between", "a mix of the two", "past averages are only roughly right"],
                         ["ρ large (all the way up)", "minimum variance", "past averages are noise: just avoid swings"]],
                        first_width=190, text=True))
        knob.add(label("Minimum variance is the portfolio with the smallest swings, chosen without looking at "
                       "average returns at all. It is widely used in practice (minimum-volatility funds and "
                       "indices), because average returns are very hard to estimate.", "body"))

        quality = Card("Is it a better portfolio?")
        quality.add(label("There is no single 'better': a risk penalty trades some return for a calmer ride. "
                          f"On the 10 US stocks above, {first[:4]}–{last[:4]}, from Markowitz to the robust "
                          "portfolio:", "body"))
        quality.add(_table(["", "Markowitz", "robust"],
                           [["return per year", pct(mk["annual return"]), pct(rob["annual return"])],
                            ["swings (volatility)", pct(mk["annual volatility"]), pct(rob["annual volatility"])],
                            ["worst fall from a peak", pct(mk["max drawdown"]), pct(rob["max drawdown"])],
                            ["return per unit of risk (Sharpe)", f"{mk['Sharpe (rf = 0)']:.2f}",
                             f"{rob['Sharpe (rf = 0)']:.2f}"]], first_width=230))
        quality.add(label("Some return is given up, but more risk is removed: the return per unit of risk is slightly "
                          "higher. Whether that suits you depends on how much a large fall would hurt you.", "body"))
        for c in (penalty, knob, quality):
            lay.addWidget(c)

        # 2 ---------------------------------------------------------------- results
        lay.addWidget(_heading("2", f"The results, {res['period'][0][:4]}–{res['period'][1][:4]}"))
        head = Card("In three numbers")
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 6, 0, 6); rl.setSpacing(40)
        rl.addWidget(_big(f"{speedup:.0f}×", "faster with the Parisi-type dual than with<br>a generic solver, "
                                             "same portfolios"))
        rl.addWidget(_big(pct(rob["max drawdown"]), f"worst fall, against {pct(mk['max drawdown'])} for Markowitz,"
                                                    f"<br>close to minimum variance ({pct(mv['max drawdown'])})"))
        rl.addWidget(_big(f"{rob['Sharpe (rf = 0)']:.2f}", f"Sharpe ratio, against {mv['Sharpe (rf = 0)']:.2f} "
                                                          f"(minimum variance)<br>and {mk['Sharpe (rf = 0)']:.2f} "
                                                          "(Markowitz)"))
        rl.addStretch(1)
        head.add(row)
        lay.addWidget(head)

        table = Card("All strategies")
        cols = (("annual return", "return / year", True), ("annual volatility", "volatility", True),
                ("Sharpe (rf = 0)", "Sharpe", False), ("max drawdown", "worst fall", True),
                ("turnover / rebalance", "trading", False))
        names = list(M)
        table.add(_table(["strategy"] + [c[1] for c in cols],
                         [[n.replace(" (KSS23 dual)", "")] + [pct(M[n][k]) if p else f"{M[n][k]:.2f}" for k, _, p in cols]
                          for n in names], bold_row=names.index(self.ROBUST), first_width=230))
        table.add(_gap())
        for crisis, vals in res["crises"].items():
            table.add(label(f"<b>{crisis}:</b> " + ", ".join(f"{k.split(' (')[0]} {100 * v:+.0f}%"
                                                             for k, v in vals.items()), "small"))
        lay.addWidget(table)

        wealth = Card("Wealth and drawdown over time", "Log scale. Below: how far each strategy was from its "
                                                       "previous peak.")
        wealth.add(Picture(RESULTS / "equity.png", 860))
        lay.addWidget(wealth)

        # 3 ---------------------------------------------------------------- the knob, measured
        lay.addWidget(_heading("3", "Turning the knob on real data"))
        dial = Card("Where the robust portfolio sits for each budget ρ")
        dial.add(label("Left: how much of the robust portfolio is held differently from Markowitz (orange) and from "
                       "minimum variance (green). Right: its return (solid) and volatility (dashed); the horizontal "
                       "lines are Markowitz (orange) and minimum variance (green).", "body"))
        dial.add(Picture(RESULTS / "sensitivity.png", 900))
        dial.add(label("Our default ρ = 0.05 sits near minimum variance. A budget matched to the statistical error of "
                       "the estimated averages (ρ ≈ 0.001–0.002) stays close to Markowitz; the two are equally far "
                       "around ρ ≈ 0.01.", "body"))
        lay.addWidget(dial)

        # 4 ---------------------------------------------------------------- speed
        lay.addWidget(_heading("4", "Where the mathematics pays off: speed"))
        speed = Card("The Parisi-type dual against a generic solver")
        speed.add(label("The worst case is an inf over two numbers (q, m) instead of a sup over 504 day-weights. "
                        f"Left: time for one worst case as the window grows. Right: the same robust backtest "
                        f"({S['rebalances']} monthly rebalances) computed twice: {S['backtest_dual_s']:.1f} s with the "
                        f"dual, {S['backtest_conic_s']:.0f} s with CVXPY + Clarabel, same portfolios (weights differ "
                        f"by at most {S['max_weight_difference']:.0e}).", "body"))
        speed.add(Picture(RESULTS / "speed.png", 900))
        lay.addWidget(speed)

        # 5 ---------------------------------------------------------------- honest limits
        lay.addWidget(_heading("5", "Honest limits"))
        limits = Card()
        limits.add(_table(["", ""], [
            ["<b>Performance</b>", "no clear edge over minimum variance; the risk reduction is mostly what minimum variance gives"],
            ["<b>Non-linearity</b>", "the mean–variance version performs like the simpler robust mean (EVaR)"],
            ["<b>Data</b>", "ten well-known US stocks chosen today (survivorship bias), one dataset"],
            ["<b>Simplifications</b>", "zero risk-free rate, simple trading costs, weights reset daily within the month"],
        ], first_width=160, text=True))
        lay.addWidget(limits)
        lay.addStretch(1)



class CaseStudySection(_SubSection):
    def __init__(self):
        super().__init__("5.1", "Case study: real data",
                         "The robust portfolio of section 5 on 28 years of real stock prices: what it does, how it "
                         "compares, and where the mathematics pays off.", CaseStudyPage())

