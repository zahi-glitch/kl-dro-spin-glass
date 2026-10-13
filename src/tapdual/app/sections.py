"""The educational sections, ordered from order to disorder:
free energy -> Curie-Weiss -> Hopfield -> REM."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from .. import spinglass as sg
from . import theme
from .chapter1 import HopfieldPage, REMPage, _column, _two_columns
from .chapter3 import RBMPage
from .widgets import Card, Deeper, LabeledSlider, Practice, Segmented, SpinGrid, label


def section_header(lay, number: str, title: str, lead: str) -> None:
    lay.addWidget(label(f"SECTION {number}", "eyebrow"))
    lay.addWidget(label(title, "title"))
    lay.addWidget(label(lead, "lead"))


class Section(QWidget):
    def __init__(self, number, title, lead):
        super().__init__()
        self.setObjectName("page")
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(28, 24, 28, 24); self.lay.setSpacing(12)
        section_header(self.lay, number, title, lead)


# --------------------------------------------------------------------------- Curie-Weiss
class CurieWeissSection(Section):
    FPS, UPDATES, HISTORY = 30, 64, 300

    def __init__(self):
        super().__init__(
            "2", "Curie–Weiss: everyone agrees",
            "The simplest collective behaviour. Every [[spin]] feels the average of all the others, so the only "
            "thing that matters is the [[magnetisation]] m. Below a critical temperature the system picks a side "
            "on its own: a [[phase-transition|phase transition]]. Its [[free-energy|free energy]] is a supremum over "
            "one number, m.")
        self.rng = np.random.default_rng()
        self.s = self.rng.choice([-1.0, 1.0], sg.N_SPINS)
        self.hist_t, self.hist_m, self.t = [], [], 0.0
        self.sweep_dir, self.loop_h, self.loop_m = 0, [], []
        self.ms = np.linspace(-0.999, 0.999, 801)

        self.grid = SpinGrid(cell=14)
        self.m_label = label("", "small")
        sim = Card("256 spins, all coupled to all", "Black = +1, light = −1. Glauber dynamics at the chosen β and field h.")
        sim.add(self.grid, 1); sim.add(self.m_label)

        self.plot_l = pg.PlotWidget(); theme.style_plot(self.plot_l.getPlotItem(), "magnetisation m", "φ(m)")
        self.plot_l.setFixedHeight(240); self.plot_l.setXRange(-1, 1)
        self.curve_l = self.plot_l.plot(pen=pg.mkPen(theme.ACCENT, width=2.5))
        self.max_pts = pg.ScatterPlotItem(size=11, brush=theme.ACCENT, pen=pg.mkPen("white", width=2))
        self.plot_l.addItem(self.max_pts)
        self.now_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.ACCENT_2, width=1.5))
        self.plot_l.addItem(self.now_line)
        landscape = Card("The free-energy landscape",
                         f"φ(m) = βm²/2 + βhm − I(m): energy gain minus the cost of the rare event 'the average is m'. "
                         f"The free energy is log 2 + max φ. <span style='color:{theme.ACCENT_2}'>│</span> where the "
                         "simulation is now.")
        landscape.add(self.plot_l)
        landscape.add(Deeper(
            "Group the 2<sup>N</sup> configurations by their magnetisation: about 2<sup>N</sup>e<sup>−N I(m)</sup> of them have "
            "average m (Cramér), each with weight e<sup>Nβm²/2 + Nβhm</sup>. The sum is dominated by the best m "
            "(Laplace), so (1/N) log Z → log 2 + sup<sub>m</sub> φ(m). The maximiser solves m = tanh(β(m + h)). "
            "For h = 0 the maximum moves away from 0 when β &gt; 1. It is a <b>supremum</b> because the energy "
            "−Nm²/2 is a convex function of m: see [[legendre|Legendre duality]]. Open the [[curie-weiss]] entry "
            "for the formulas."))

        self.beta = LabeledSlider("Inverse temperature β", 0.0, 2.5, 1.4, 250, hint="β = 1 is the critical point (at h = 0).")
        self.h = LabeledSlider("External field h", -0.3, 0.3, 0.0, 120, fmt=lambda v: f"{v:+.3f}",
                               hint="A common push in one direction.")
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 4, 0, 0)
        self.run_btn = QPushButton("Run  ▶"); self.run_btn.setObjectName("primary")
        self.rand_btn = QPushButton("Random start")
        bl.addWidget(self.run_btn); bl.addWidget(self.rand_btn); bl.addStretch(1)
        controls = Card("Controls"); controls.add(self.beta); controls.add(self.h); controls.add(buttons)

        self.plot_h = pg.PlotWidget(); theme.style_plot(self.plot_h.getPlotItem(), "incentive h", "share adopting, m")
        self.plot_h.setFixedHeight(200); self.plot_h.setXRange(-0.3, 0.3); self.plot_h.setYRange(-1.05, 1.05)
        self.loop_curve = self.plot_h.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.sweep_btn = QPushButton("Sweep the incentive  ↔")
        self.loop_text = label("", "body")
        practice = Practice("Why do markets tip, and why don't they tip back?", rows=[
            ("Model", "N people choose σ<sub>i</sub> = +1 (adopt a product) or −1 (not). Each one's incentive is "
                      "h (price, private taste) plus the average choice m of the others (peer effect); β measures how "
                      "strongly people respond. This is the discrete-choice model with social interactions of "
                      "Brock and Durlauf (2001)."),
            ("Equilibrium", "m = tanh(β(m + h)): the same equation as the magnet."),
            ("Try it", "Press <b>Sweep</b>: h rises from −0.3 to +0.3 and falls back, starting with nobody adopting."),
            ("What you see", "β &lt; 1: adoption follows the price smoothly. β &gt; 1: it <b>jumps</b> up at one price "
                             "and jumps back only at a much lower one (hysteresis). Two markets with the same price can "
                             "sit in opposite states: history matters."),
        ])
        practice.add(self.sweep_btn); practice.add(self.plot_h); practice.add(self.loop_text)

        self.lay.addWidget(_two_columns(_column(sim, landscape), _column(controls, practice)), 1)
        self.timer = QTimer(self); self.timer.setInterval(1000 // self.FPS); self.timer.timeout.connect(self.tick)
        self.run_btn.clicked.connect(self.toggle)
        self.rand_btn.clicked.connect(self.random_start)
        self.sweep_btn.clicked.connect(self.start_sweep)
        self.beta.changed.connect(lambda _: self.refresh_landscape())
        self.h.changed.connect(lambda _: self.refresh_landscape())
        self.refresh_landscape(); self.refresh_sim()

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop(); self.run_btn.setText("Run  ▶")
        else:
            self.timer.start(); self.run_btn.setText("Pause  ❚❚")

    def random_start(self):
        self.s = self.rng.choice([-1.0, 1.0], sg.N_SPINS); self.refresh_sim()

    def start_sweep(self):
        self.s = -np.ones(sg.N_SPINS)                      # nobody adopts
        self.h.set_value(-0.3)
        self.loop_h, self.loop_m, self.sweep_dir = [], [], +1
        if not self.timer.isActive():
            self.toggle()

    def tick(self):
        if self.sweep_dir:
            h = self.h.value() + self.sweep_dir * 0.004
            if h >= 0.3:
                self.sweep_dir = -1
            elif h <= -0.3 and self.sweep_dir == -1:
                self.sweep_dir = 0
            self.h.set_value(float(np.clip(h, -0.3, 0.3)))
        sg.cw_sweep(self.s, self.beta.value(), self.h.value(), self.rng, self.UPDATES * 4 if self.sweep_dir else self.UPDATES)
        self.t += self.UPDATES / sg.N_SPINS
        if self.sweep_dir or self.loop_h:
            self.loop_h.append(self.h.value()); self.loop_m.append(float(self.s.mean()))
            self.loop_curve.setData(self.loop_h, self.loop_m)
            if not self.sweep_dir:
                self.describe_loop()
        self.refresh_sim()

    def describe_loop(self):
        h, m = np.array(self.loop_h), np.array(self.loop_m)
        up = h[: int(np.argmax(h)) + 1]; mu = m[: int(np.argmax(h)) + 1]
        down = h[int(np.argmax(h)):]; md = m[int(np.argmax(h)):]
        try:
            h_up = up[np.argmax(mu > 0)]; h_down = down[np.argmax(md < 0)]
            self.loop_text.setText(f"Adoption switched on at h ≈ {h_up:+.2f} and off only at h ≈ {h_down:+.2f}."
                                   if h_up - h_down > 0.03 else
                                   "No hysteresis at this β: adoption follows the price smoothly. Try β above 1.")
        except ValueError:
            pass
        self.loop_h, self.loop_m = [], []

    def refresh_landscape(self):
        b, h = self.beta.value(), self.h.value()
        phi = sg.cw_landscape(self.ms, b, h)
        self.curve_l.setData(self.ms, phi)
        roots = sg.cw_stationary(b, h)
        vals = [float(sg.cw_landscape(r, b, h)) for r in roots]
        best = max(vals)
        pts = [(r, v) for r, v in zip(roots, vals) if abs(v - best) < 1e-9 or
               (len(roots) == 3 and r != roots[1])]       # global and local maxima, not the middle minimum
        self.max_pts.setData([p[0] for p in pts], [p[1] for p in pts])
        self.plot_l.setYRange(float(phi.min()) - 0.02, best + 0.05)

    def refresh_sim(self):
        m = float(self.s.mean())
        self.grid.set_state(self.s)
        self.now_line.setPos(m)
        self.m_label.setText(f"magnetisation m = {m:+.2f}   ·   β = {self.beta.value():.2f}, h = {self.h.value():+.3f}")

    def hideEvent(self, e):
        self.timer.stop(); self.run_btn.setText("Run  ▶"); super().hideEvent(e)


# --------------------------------------------------------------------------- capacity
class CapacityPage(QWidget):
    N = 400
    ALPHAS = np.round(np.arange(0.02, 0.305, 0.01), 3)

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lay.addWidget(label(
            "Store more and more random memories, P = αN, and ask whether each one can still be recalled. As α grows, "
            "the cross-talk between memories acts like random couplings and the network turns into a "
            "[[spin-glass|spin glass]], close cousin of the [[sk|SK model]]: recall breaks down near the "
            "[[capacity|storage capacity]] α<sub>c</sub> ≈ 0.138.", "body"))

        self.plot = pg.PlotWidget(); theme.style_plot(self.plot.getPlotItem(), "load α = P / N", "overlap after recall")
        self.plot.setFixedHeight(300); self.plot.setXRange(0, 0.31); self.plot.setYRange(0, 1.05)
        self.plot.addItem(pg.InfiniteLine(pos=0.138, angle=90, pen=pg.mkPen(theme.INK_3),
                                          label="α_c ≈ 0.138 (N → ∞)", labelOpts={"color": theme.INK_3, "position": 0.25}))
        self.curve = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=2), symbol="o", symbolSize=7,
                                    symbolBrush=theme.ACCENT, symbolPen=pg.mkPen("white", width=1.5))
        self.status = label("", "small")
        self.run_btn = QPushButton("Run the experiment  ▶"); self.run_btn.setObjectName("primary")
        exp = Card(f"Experiment: N = {self.N} neurons, random memories",
                   "For each load, store P random patterns, start the network exactly at one of them, let it run at zero "
                   "temperature and measure how much of the memory survives (overlap 1 = perfect).")
        exp.add(self.run_btn); exp.add(self.plot); exp.add(self.status)

        known = Card("What is proven, and what is open")
        known.add(label(
            "<b>Few memories</b> (P fixed, or P/N → 0): solved rigorously. The free energy is a supremum over the "
            "[[overlap|overlaps]], a P-dimensional [[curie-weiss|Curie–Weiss]] formula.<br><br>"
            "<b>Many memories, high temperature or small α</b>: the replica-symmetric free energy is proved "
            "(Talagrand; Shcherbina and Tirozzi; and later work).<br><br>"
            "<b>Many memories, low temperature</b>, which is where recall lives: <b>open</b>. Physics has the "
            "replica-symmetric solution of Amit, Gutfreund and Sompolinsky (α<sub>c</sub> ≈ 0.138) and "
            "[[rsb|replica-symmetry-breaking]] corrections (Steffan and Kühn: α<sub>c</sub> ≈ 0.13819); Agliari, "
            "Albanese, Barra and Ottaviani derive broken-replica free energies by interpolation under an ansatz. "
            "A full proof, an analogue of the [[parisi|Parisi formula]] for the Hopfield model, is not available.",
            "body"))

        practice = Practice("How many memories fit in N neurons?", rows=[
            ("Classical", "Hebbian Hopfield network, pairwise couplings: about 0.138·N random memories. "
                          "This is the AGS value (Amit, Gutfreund, Sompolinsky, 1985), the curve on the left."),
            ("Dense", "Dense associative memories (Krotov and Hopfield, 2016) replace m² in the energy by mⁿ, "
                      "interactions among n neurons at a time: capacity grows like Nⁿ⁻¹."),
            ("Modern", "With an exponential interaction the capacity becomes exponential in N (Demircigil et al., "
                       "2017). The continuous version (Ramsauer et al., 2020) retrieves in one step, and that step is "
                       "a transformer's attention."),
        ])
        lay.addWidget(_two_columns(_column(exp), _column(known, practice)), 1)
        self.timer = QTimer(self); self.timer.setInterval(10); self.timer.timeout.connect(self.step)
        self.run_btn.clicked.connect(self.start)

    def start(self):
        self.k, self.xs, self.ys = 0, [], []
        self.run_btn.setEnabled(False)
        self.timer.start()

    def step(self):
        a = float(self.ALPHAS[self.k])
        P = max(1, int(round(a * self.N)))
        m = np.mean([sg.recall_overlap(self.N, P, self.rng, tests=6) for _ in range(2)])
        self.xs.append(P / self.N); self.ys.append(m)
        self.curve.setData(self.xs, self.ys)
        self.status.setText(f"α = {P / self.N:.3f}: P = {P} memories, mean overlap {m:.3f}")
        self.k += 1
        if self.k >= len(self.ALPHAS):
            self.timer.stop(); self.run_btn.setEnabled(True)
            self.status.setText("Done. With N = 400 the drop is smeared and sits a little above 0.138: finite-size "
                                "effects, which shrink as N grows.")


class HopfieldSection(Section):
    def __init__(self):
        super().__init__(
            "3", "Hopfield: between order and glass",
            "Store memories in the couplings and each memory becomes a valley. With one memory the "
            "[[hopfield|Hopfield model]] is the [[curie-weiss|Curie–Weiss]] magnet in disguise; with many it becomes a "
            "[[spin-glass|spin glass]]. In between, it is a working memory, and a machine-learning model.")
        self.seg = Segmented(["Memory", "= Boltzmann machine", "How much can it remember?"])
        self.lay.addWidget(self.seg)
        self.stack = QStackedWidget()
        for p in (HopfieldPage(), RBMPage(), CapacityPage()):
            self.stack.addWidget(p)
        self.lay.addWidget(self.stack, 1)
        self.seg.changed.connect(self.stack.setCurrentIndex)


class REMSection(Section):
    def __init__(self):
        super().__init__(
            "4", "REM: extreme disorder",
            "At the other end of the scale: forget the couplings and give every configuration its own independent "
            "random energy. Derrida's [[rem|Random Energy Model]] is the simplest [[spin-glass|spin glass]]. It "
            "<b>freezes</b>, and the reason is a budget: there are only 2<sup>N</sup> configurations, so the "
            "[[entropy]] cannot exceed log 2 per spin.")
        self.lay.addWidget(REMPage(), 1)
