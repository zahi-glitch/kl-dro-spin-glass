"""Section 5.2: live data. A virtual $10,000 followed day by day against two benchmarks, and today's portfolios."""
from __future__ import annotations

import threading
from datetime import datetime, time as dtime

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from . import theme
from .common import _SubSection, _big, _table
from .widgets import Card, label


class LivePage(QWidget):
    """Live data: a virtual $10,000 followed since it was invested, and today's portfolios."""

    def __init__(self):
        super().__init__()
        from .. import live
        self.live = live
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(16)
        lay.addWidget(label(
            "The four strategies on live market data (Yahoo Finance, free, needs an internet connection), using nine "
            "funds (ETFs), one per major asset class. The first time you open this page, it invests a virtual "
            "<b>$10,000</b> (the funds trade in dollars) at the latest closing prices in each strategy and in two "
            "benchmarks; every time you come back, it downloads the new prices and shows what each is worth now. "
            "A robot on GitHub does the same every weekday (see the README). Like in the case study, each strategy rebalances on the "
            "first trading day of every month, using only the data available on that day, and pays 0.1% on what it "
            "trades (so each starts at $9,990). <b>No real money, educational only.</b>", "body"))
        bar = QWidget(); bl = QHBoxLayout(bar); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(10)
        self.refresh_btn = QPushButton("Update with the latest prices"); self.refresh_btn.setObjectName("primary")
        self.reset_btn = QPushButton("Start again from the latest close")
        bl.addWidget(self.refresh_btn); bl.addWidget(self.reset_btn); bl.addStretch(1)
        lay.addWidget(bar)
        self.status = label("Opening this page downloads the prices automatically.", "body")
        lay.addWidget(self.status)

        self.now_card = Card("What each $10,000 is worth now")
        self.now_row = QWidget(); self.now_lay = QHBoxLayout(self.now_row)
        self.now_lay.setContentsMargins(0, 6, 0, 6); self.now_lay.setSpacing(40)
        self.now_card.add(self.now_row)
        self.table = label("", "body"); self.now_card.add(self.table)

        self.chart_card = Card("Value day by day")
        self.plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
        theme.style_plot(self.plot.getPlotItem(), "", "value ($)")
        self.plot.setFixedHeight(300); self.plot.addLegend(offset=(10, 10))
        self.chart_card.add(self.plot)
        self.chart_note = label("", "small"); self.chart_card.add(self.chart_note)

        self.today_card = Card("Today's portfolios",
                               "What each strategy would hold if it chose today, from the last two years of prices.")
        self.today_box = label("", "body"); self.today_card.add(self.today_box)
        self.today_card.add(label("Minimum variance looks only at how much each fund swings; Markowitz also trusts the "
                                  "last two years' average returns; the robust portfolio sits between them, here close "
                                  "to minimum variance (see the case study).", "small"))
        self.days_card = Card("The past days the robust portfolio is protecting against")
        self.days_box = label("", "body"); self.days_card.add(self.days_box)
        self.cards = (self.now_card, self.chart_card, self.today_card, self.days_card)
        for c in self.cards:
            c.hide(); lay.addWidget(c)
        lay.addStretch(1)

        self.result, self.worker, self.started, self.reset = None, None, False, False
        self.poll = QTimer(self); self.poll.setInterval(200); self.poll.timeout.connect(self.collect)
        self.refresh_btn.clicked.connect(self.start)
        self.reset_btn.clicked.connect(self.start_again)

    def showEvent(self, e):
        super().showEvent(e)
        if not self.started:
            self.start()

    def start_again(self):
        from PySide6.QtWidgets import QMessageBox
        ok = QMessageBox.question(self, "Start again?",
                                  "This forgets the current virtual investment and invests $10,000 again at the "
                                  "latest closing prices. Continue?")
        if ok == QMessageBox.Yes:
            self.reset = True
            self.start()

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.started = True
        self.refresh_btn.setEnabled(False); self.reset_btn.setEnabled(False)
        self.status.setText("Downloading the latest prices and following your $10,000… (a few seconds)")
        live, reset = self.live, self.reset
        self.reset = False

        def run():
            try:
                saved = None if reset else live.load_virtual()
                tickers = saved[2] if saved else list(live.UNIVERSE)
                years = live.years_needed(saved[0]) if saved else 3
                prices = live.get_prices(tickers, years=years)
                new = saved is None
                start, capital = live.start_virtual(prices) if new else saved[:2]
                self.result = ("ok", (live.simulate(prices, start, capital), live.compute(prices), new))
            except Exception as err:          # shown to the user, never crashes the app
                self.result = ("error", str(err))
        self.result = None
        self.worker = threading.Thread(target=run, daemon=True); self.worker.start(); self.poll.start()

    def collect(self):
        if self.result is None:
            return
        self.poll.stop(); self.refresh_btn.setEnabled(True); self.reset_btn.setEnabled(True)
        kind, value = self.result
        if kind == "error":
            self.status.setText(f"<b>Could not update:</b> {value}")
            return
        v, today, new = value
        last = v.dates[-1]
        days = len(v.dates) - 1
        if new:
            self.status.setText(f"<b>Started.</b> $10,000 invested at the close of {v.start:%d %B %Y} in each "
                                "strategy. Come back on the next trading day to see how it moves.")
        else:
            self.status.setText(f"Invested at the close of <b>{v.start:%d %B %Y}</b>; latest prices "
                                f"<b>{last:%d %B %Y}</b> ({days} trading day{'s' if days != 1 else ''} later, "
                                f"{v.rebalances} monthly rebalance{'s' if v.rebalances != 1 else ''}).")
        while self.now_lay.count():
            w = self.now_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        names = list(v.values)
        rows = []
        for n in names:
            now = v.values[n][-1]
            if not n.startswith("Benchmark"):
                self.now_lay.addWidget(_big(f"${now:,.0f}", n))
            rows.append([n, f"${v.capital:,.0f}", f"${now:,.2f}", f"{now - v.capital:+,.2f}",
                         f"{100 * (now / v.capital - 1):+.2f}%"])
        self.now_lay.addStretch(1)
        self.table.setText(_table(["strategy", "invested", "worth now", "gain ($)", "gain (%)"], rows,
                                  bold_row=names.index(self.live.ROBUST), first_width=200).text())
        self.plot.clear()
        xs = [datetime.combine(d, dtime(16)).timestamp() for d in v.dates]
        colors = {"Equal weight": theme.INK_3, "Markowitz": theme.ACCENT_2, "Minimum variance": "#1D9E75",
                  self.live.ROBUST: theme.ACCENT}
        for n in names:
            bench = n.startswith("Benchmark")
            pen = pg.mkPen(colors.get(n, theme.INK), width=2.5 if n == self.live.ROBUST else 1.6,
                           style=Qt.DashLine if bench else Qt.SolidLine)
            if "60/40" in n:
                pen.setStyle(Qt.DotLine)
            self.plot.plot(xs, v.values[n], pen=pen,
                           symbol="o" if len(xs) < 40 and not bench else None, symbolSize=5,
                           symbolBrush=colors.get(n, theme.INK),
                           name=n)
        self.plot.addItem(pg.InfiniteLine(pos=v.capital, angle=0, pen=pg.mkPen(theme.BORDER, style=Qt.DashLine)))
        self.chart_note.setText("Dashed line: the $10,000 invested. The first point is the start: each strategy is "
                                "at $9,990 after paying 0.1% to buy." if days < 2 else
                                "Dashed line: the $10,000 invested.")
        strategies = [s.name for s in self.live.STRATEGIES]
        rows = [[f"{tk} &nbsp;<span style='color:{theme.INK_3};'>{self.live.UNIVERSE.get(tk, '')}</span>"]
                + [f"{100 * today.weights[n][i]:.1f}%" for n in strategies] for i, tk in enumerate(today.tickers)]
        self.today_box.setText(_table(["fund"] + strategies, rows, first_width=320).text())
        self.days_box.setText(_table(["day", "robust portfolio that day", "weight in the worst case"],
                                     [[f"{d:%d %b %Y}", f"{ret:+.2f}%", f"×{mult:.1f}"] for d, ret, mult in today.worst_days],
                                     first_width=140).text())
        for c in self.cards:
            c.show()



class LiveSection(_SubSection):
    def __init__(self):
        super().__init__("5.2", "Live: a virtual $10,000",
                         "Invest a virtual $10,000 once, then watch it day by day against two benchmarks; and see "
                         "what each strategy would hold today.", LivePage())
