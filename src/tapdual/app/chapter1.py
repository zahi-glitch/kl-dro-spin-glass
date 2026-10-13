"""Chapter 1: spins, energy, temperature. Three live models: Hopfield, SK, REM."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from .. import spinglass as sg
from . import theme
from .widgets import Card, Deeper, LabeledSlider, OverlapBars, Practice, Segmented, SpinGrid, label


def _two_columns(left: QWidget, right: QWidget) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(16)
    lay.addWidget(left, 5); lay.addWidget(right, 4)
    return w


def _column(*widgets) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(14)
    for x in widgets:
        lay.addWidget(x)
    lay.addStretch(1)
    return w


# --------------------------------------------------------------------------- Hopfield
class HopfieldPage(QWidget):
    FPS, UPDATES_PER_FRAME = 30, 24

    def __init__(self):
        super().__init__()
        self.net, self.rng = sg.Hopfield(), np.random.default_rng()
        self.target, self.sweeps = 0, 0.0
        self.s = self.net.corrupt(0, 0.3, self.rng)

        self.grid = SpinGrid(cell=22)
        self.status = label("", "small")
        state = Card("The network", "Each pixel is a neuron that is on (+1) or off (−1). "
                     "Press <b>Recall</b> and watch it roll downhill in energy.")
        state.add(self.grid, 1); state.add(self.status)

        thumbs = QWidget(); tl = QHBoxLayout(thumbs); tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(12)
        self.thumbs = []
        for mu, name in enumerate(sg.PATTERN_NAMES):
            t = SpinGrid(cell=5, selectable=True); t.set_state(sg.PATTERNS[mu]); t.setFixedSize(84, 84)
            t.clicked.connect(lambda k=mu: self.choose(k))
            tl.addWidget(t); self.thumbs.append(t)
        tl.addStretch(1)
        memories = Card("Stored memories", "Written into the couplings by Hebb's rule. Click one to corrupt it.")
        memories.add(thumbs)

        self.noise = LabeledSlider("Corruption", 0.0, 0.5, 0.3, 100, fmt=lambda v: f"{v:.0%} of pixels flipped")
        self.temp = LabeledSlider("Temperature T", 0.0, 2.0, 0.15, 200,
                                  hint="T = 0: always go downhill. Large T: random flips, memories melt.")
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 4, 0, 0)
        self.corrupt_btn = QPushButton("Corrupt")
        self.half_btn = QPushButton("Half unknown")
        self.run_btn = QPushButton("Recall  ▶"); self.run_btn.setObjectName("primary")
        bl.addWidget(self.corrupt_btn); bl.addWidget(self.half_btn); bl.addWidget(self.run_btn); bl.addStretch(1)
        controls = Card("Controls"); controls.add(self.noise); controls.add(self.temp); controls.add(buttons)

        self.bars = OverlapBars(sg.PATTERN_NAMES)
        overlaps = Card("Overlap with each memory", "m = +1: perfect recall. m ≈ 0: unrelated. "
                        "These overlaps are the <i>order parameters</i> of the network.")
        overlaps.add(self.bars)
        overlaps.add(Deeper(
            "Energy E(s) = −½ Σ<sub>ij</sub> J<sub>ij</sub> s<sub>i</sub> s<sub>j</sub>, with Hebbian couplings "
            "J<sub>ij</sub> = (1/N) Σ<sub>μ</sub> ξ<sub>i</sub><sup>μ</sup> ξ<sub>j</sub><sup>μ</sup>. In terms of the overlaps "
            "m<sub>μ</sub> = (1/N) Σ<sub>i</sub> ξ<sub>i</sub><sup>μ</sup> s<sub>i</sub>, the energy is −(N/2) Σ<sub>μ</sub> m<sub>μ</sub><sup>2</sup> "
            "+ const: each memory is a valley. The dynamics is Glauber's: s<sub>i</sub> = +1 with probability "
            "1/(1 + e<sup>−2h<sub>i</sub>/T</sup>), h<sub>i</sub> = Σ<sub>j</sub> J<sub>ij</sub> s<sub>j</sub>. "
            "Here N = 256 and 3 memories; recall survives up to about 0.138·N random memories "
            "(Amit, Gutfreund, Sompolinsky). Chapter 3: this network is a restricted Boltzmann machine in disguise."))

        practice = Practice(
            "Search by content, not by address",
            "Press <b>Half unknown</b>: the bottom half becomes random static, and the network still returns the "
            "whole symbol. That is a <b>content-addressable memory</b>: you retrieve a complete item from a fragment "
            "or a noisy copy, like completing a half-remembered phone number or cleaning a damaged scan of a code. "
            "The same idea lives on in machine learning: the update rule of modern Hopfield networks is the "
            "<b>attention</b> mechanism of transformers (Ramsauer et al., 2020), where the query is the noisy pattern "
            "and the keys are the stored memories.")
        self.layout_ = _two_columns(_column(state, practice), _column(memories, controls, overlaps))
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.layout_)

        self.timer = QTimer(self); self.timer.setInterval(1000 // self.FPS); self.timer.timeout.connect(self.tick)
        self.corrupt_btn.clicked.connect(self.corrupt)
        self.half_btn.clicked.connect(self.half_unknown)
        self.run_btn.clicked.connect(self.toggle)
        self.choose(0)

    def choose(self, mu: int):
        self.target = mu
        for k, t in enumerate(self.thumbs):
            t.set_selected(k == mu)
        self.corrupt()

    def corrupt(self):
        self.stop()
        self.s = self.net.corrupt(self.target, self.noise.value(), self.rng)
        self.sweeps = 0.0
        self.refresh()

    def half_unknown(self):
        """Keep the top half of the chosen memory; replace the bottom half by random static."""
        self.stop()
        self.s = sg.PATTERNS[self.target].copy()
        half = self.net.N // 2
        self.s[half:] = self.rng.choice([-1.0, 1.0], self.net.N - half)
        self.sweeps = 0.0
        self.refresh()

    def toggle(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.timer.start(); self.run_btn.setText("Pause  ❚❚")

    def stop(self):
        self.timer.stop(); self.run_btn.setText("Recall  ▶")

    def tick(self):
        sg.heat_bath(self.s, self.net.J, self.temp.value(), self.rng, self.UPDATES_PER_FRAME)
        self.sweeps += self.UPDATES_PER_FRAME / self.net.N
        self.refresh()
        if self.temp.value() == 0 and self.sweeps > 8:
            self.stop()

    def refresh(self):
        self.grid.set_state(self.s)
        self.bars.set_values(self.net.overlaps(self.s), self.target)
        self.status.setText(f"energy per neuron {self.net.energy(self.s):+.3f}   ·   "
                            f"{self.sweeps:.1f} sweeps (one sweep = every neuron updated once on average)")

    def hideEvent(self, e):
        self.stop(); super().hideEvent(e)


# --------------------------------------------------------------------------- SK
class SKPage(QWidget):
    FPS, UPDATES_PER_FRAME, HISTORY = 30, 48, 400

    def __init__(self):
        super().__init__()
        self.rng, self.seed = np.random.default_rng(), 0
        self.sk = sg.SK(seed=self.seed)

        grids = QWidget(); gl = QGridLayout(grids); gl.setContentsMargins(0, 0, 0, 0); gl.setSpacing(12)
        self.ga, self.gb = SpinGrid(cell=14), SpinGrid(cell=14)
        gl.addWidget(self.ga, 0, 0); gl.addWidget(self.gb, 0, 1)
        gl.addWidget(label("copy a", "small"), 1, 0, Qt.AlignHCenter); gl.addWidget(label("copy b", "small"), 1, 1, Qt.AlignHCenter)
        replicas = Card("Two copies, same random couplings",
                        "Started from different random states and cooled at the same temperature. "
                        "There is no image to find: just many valleys of similar depth.")
        replicas.add(grids, 1)

        self.plot_e = pg.PlotWidget(); theme.style_plot(self.plot_e.getPlotItem(), "time (sweeps)", "energy per spin")
        self.plot_e.setFixedHeight(190)
        self.plot_e.addItem(pg.InfiniteLine(pos=sg.SK.GROUND_STATE, angle=0,
                                            pen=pg.mkPen(theme.INK_3, width=1),
                                            label="Parisi ground state −0.763", labelOpts={"color": theme.INK_3, "position": 0.8}))
        self.ea = self.plot_e.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.eb = self.plot_e.plot(pen=pg.mkPen(theme.ACCENT_2, width=2))
        self.plot_e.setYRange(-0.85, 0.1)
        self.plot_q = pg.PlotWidget(); theme.style_plot(self.plot_q.getPlotItem(), "time (sweeps)", "overlap q(a, b)")
        self.plot_q.setFixedHeight(150); self.plot_q.setYRange(-1, 1)
        self.qline = self.plot_q.plot(pen=pg.mkPen(theme.INK, width=2))
        traces = Card("Energy and overlap over time")
        legend = label(f"<span style='color:{theme.ACCENT}'>━</span> copy a &nbsp;&nbsp; "
                       f"<span style='color:{theme.ACCENT_2}'>━</span> copy b", "small")
        traces.add(legend); traces.add(self.plot_e); traces.add(self.plot_q)

        self.temp = LabeledSlider("Temperature T", 0.1, 2.0, 0.4, 190,
                                  hint="Above T = 1 the system is a paramagnet; below, it is a spin glass.")
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 4, 0, 0)
        self.run_btn = QPushButton("Cool from random  ▶"); self.run_btn.setObjectName("primary")
        self.new_btn = QPushButton("New couplings")
        bl.addWidget(self.run_btn); bl.addWidget(self.new_btn); bl.addStretch(1)
        controls = Card("Controls"); controls.add(self.temp); controls.add(buttons)
        self.readout = label("", "body")
        explain = Card("What to look for")
        explain.add(label("Random couplings J<sub>ij</sub> ask some pairs to agree and others to disagree, and no state "
                          "satisfies them all: <b>frustration</b>. Both copies reach nearly the same low energy, "
                          "yet they usually end in <b>different valleys</b>: their overlap q stays well below 1 and "
                          "changes from run to run. Parisi's theory is a theory of the distribution of q.", "body"))
        explain.add(self.readout)
        explain.add(Deeper(
            "H(s) = −Σ<sub>i&lt;j</sub> J<sub>ij</sub> s<sub>i</sub> s<sub>j</sub>, J<sub>ij</sub> ~ N(0, 1/N) independent. "
            "Overlap q<sub>ab</sub> = (1/N) Σ<sub>i</sub> s<sub>i</sub><sup>a</sup> s<sub>i</sub><sup>b</sup>. "
            "For T &gt; 1 the free energy is log 2 + β²/4 (annealed); below T = 1 replica symmetry breaks and the "
            "free energy is given by the Parisi formula, an infimum over a functional order parameter "
            "(Guerra; Talagrand). The minimum energy per spin tends to −0.7633. With N = 256 the run here is a small, "
            "honest finite-size picture, not the limit."))

        self.practice_readout = label("", "body")
        practice = Practice(
            "Many good answers that look nothing alike",
            "Minimising this energy is a <b>Max-Cut</b> problem: split 256 nodes into two groups so that the "
            "total weight of the links you cut (with weights −J<sub>ij</sub>) is as large as possible. Problems of "
            "this kind (scheduling, chip layout, portfolio selection with pairwise interactions) are routinely "
            "written as Ising energies, and Ising machines and quantum annealers are built to minimise them. "
            "The glass teaches the key practical lesson: there are many near-optimal solutions that are "
            "<b>very different</b> from each other.")
        practice.add(self.practice_readout)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_two_columns(_column(replicas, traces), _column(controls, explain, practice)))

        self.timer = QTimer(self); self.timer.setInterval(1000 // self.FPS); self.timer.timeout.connect(self.tick)
        self.run_btn.clicked.connect(self.restart)
        self.new_btn.clicked.connect(self.new_couplings)
        self.reset_states()

    def reset_states(self):
        self.a = self.rng.choice([-1.0, 1.0], self.sk.N)
        self.b = self.rng.choice([-1.0, 1.0], self.sk.N)
        self.t, self.hist = 0.0, {"t": [], "ea": [], "eb": [], "q": []}
        self.refresh()

    def restart(self):
        self.reset_states(); self.timer.start()

    def new_couplings(self):
        self.timer.stop(); self.seed += 1; self.sk = sg.SK(seed=self.seed); self.reset_states()

    def tick(self):
        T = self.temp.value()
        sg.heat_bath(self.a, self.sk.J, T, self.rng, self.UPDATES_PER_FRAME)
        sg.heat_bath(self.b, self.sk.J, T, self.rng, self.UPDATES_PER_FRAME)
        self.t += self.UPDATES_PER_FRAME / self.sk.N
        if self.t > 60:
            self.timer.stop()
        self.refresh()

    def refresh(self):
        ea, eb, q = self.sk.energy(self.a), self.sk.energy(self.b), self.sk.overlap(self.a, self.b)
        h = self.hist
        h["t"].append(self.t); h["ea"].append(ea); h["eb"].append(eb); h["q"].append(q)
        for k in h:
            h[k] = h[k][-self.HISTORY:]
        self.ga.set_state(self.a); self.gb.set_state(self.b)
        self.ea.setData(h["t"], h["ea"]); self.eb.setData(h["t"], h["eb"]); self.qline.setData(h["t"], h["q"])
        self.readout.setText(f"Now: energy per spin {ea:+.3f} and {eb:+.3f} · overlap q = {q:+.2f}")
        if hasattr(self, "practice_readout"):
            agree = (1 + abs(q)) / 2
            gap = abs(ea - eb) / max(abs(ea), abs(eb), 1e-9)
            self.practice_readout.setText(
                f"Right now the two partitions differ in quality by <b>{gap:.1%}</b>, yet they put only "
                f"<b>{agree:.0%}</b> of the nodes on the same side (50% would be unrelated).")

    def hideEvent(self, e):
        self.timer.stop(); super().hideEvent(e)


# --------------------------------------------------------------------------- REM
class REMPage(QWidget):
    TOP = 40

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.rem = sg.REM(n=16, seed=self.seed)
        self.betas = np.linspace(0, 3, 121)

        self.plot_w = pg.PlotWidget(); theme.style_plot(self.plot_w.getPlotItem(), "the 40 most probable states (ranked)", "Gibbs weight")
        self.plot_w.setFixedHeight(260); self.plot_w.setYRange(0, 1); self.plot_w.setXRange(0, self.TOP + 1)
        self.bars = pg.BarGraphItem(x=np.arange(1, self.TOP + 1), height=np.zeros(self.TOP), width=0.7,
                                    brush=theme.ACCENT, pen=pg.mkPen(None))
        self.plot_w.addItem(self.bars)
        weights = Card(f"Where the probability sits, among 2¹⁶ = 65,536 states",
                       "Each configuration has its own independent random energy. Lower the temperature and watch "
                       "the probability pile up on a handful of states.")
        weights.add(self.plot_w)

        self.plot_s = pg.PlotWidget(); theme.style_plot(self.plot_s.getPlotItem(), "inverse temperature β", "entropy per spin")
        self.plot_s.setFixedHeight(230); self.plot_s.setYRange(0, 0.75); self.plot_s.setXRange(0, 3)
        self.plot_s.plot(self.betas, [sg.REM.entropy_theory(b) for b in self.betas], pen=pg.mkPen(theme.INK_3, width=2))
        self.curve_s = self.plot_s.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.plot_s.addItem(pg.InfiniteLine(pos=np.log(2), angle=0, pen=pg.mkPen(theme.BORDER),
                                            label="log 2: all 2ᴺ states equally likely", labelOpts={"color": theme.INK_3, "position": 0.78}))
        self.plot_s.addItem(pg.InfiniteLine(pos=sg.REM.beta_c, angle=90, pen=pg.mkPen(theme.INK_3),
                                            label="β_c = √(2 log 2)", labelOpts={"color": theme.INK_3, "position": 0.6}))
        self.marker = pg.ScatterPlotItem(size=11, brush=theme.ACCENT, pen=pg.mkPen("white", width=2))
        self.plot_s.addItem(self.marker)
        entropy = Card("Entropy runs out", f"<span style='color:{theme.ACCENT}'>━</span> this sample (N = 16) &nbsp; "
                       f"<span style='color:{theme.INK_3}'>━</span> N → ∞: log 2 − β²/2, until it hits 0 at β<sub>c</sub>.")
        entropy.add(self.plot_s)

        self.beta = LabeledSlider("Inverse temperature β", 0.0, 3.0, 0.6, 300, hint="β = 1/T. Freezing at β_c ≈ 1.177.")
        self.new_btn = QPushButton("New random energies")
        controls = Card("Controls"); controls.add(self.beta); controls.add(self.new_btn)

        self.k_states, self.k_entropy = label("", "value"), label("", "value")
        stats = Card("Right now")
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(24)
        for title, val in (("effective number of states", self.k_states), ("entropy / log 2", self.k_entropy)):
            box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(0)
            bl.addWidget(label(title, "small")); bl.addWidget(val); rl.addWidget(box)
        rl.addStretch(1)
        stats.add(row)
        self.phase = label("", "body"); stats.add(self.phase)

        explain = Card("Why this matters later")
        explain.add(label("There are only 2<sup>N</sup> configurations, so the entropy can never exceed log 2 per spin. "
                          "That is a <b>budget</b>. Above β<sub>c</sub> the Gibbs measure has spent it all and "
                          "<b>freezes</b> onto a few states. In chapter 4 the same budget reappears as a "
                          "KL-divergence constraint, and the freezing as the moment the constraint binds.", "body"))
        explain.add(Deeper(
            "Derrida's REM: energies E<sub>α</sub> ~ N(0, N) independent for α = 1…2<sup>N</sup>. "
            "Free energy (1/N) log Σ<sub>α</sub> e<sup>−βE<sub>α</sub></sup> → log 2 + β²/2 for β ≤ β<sub>c</sub> = √(2 log 2), "
            "and β·√(2 log 2) above. In the frozen phase the normalised weights of the lowest states converge to a "
            "Poisson–Dirichlet point process with parameter m = β<sub>c</sub>/β: the 1-step replica-symmetry-breaking picture, "
            "and the same m as in the Parisi formula."))

        self.tokens = label("", "body")
        practice = Practice(
            "Temperature when a language model picks the next word",
            "A language model chooses the next token with a softmax over a vocabulary of tens of thousands of "
            "tokens: probability ∝ e<sup>logit/T</sup>. Read the 65,536 states here as a vocabulary and the "
            "random energies as (minus) logits, a deliberately crude toy. The <b>effective number of states</b> "
            "is then the number of tokens really in play. Lowering the temperature makes the choice collapse "
            "onto a handful of tokens: the sampler <b>freezes</b> and becomes almost deterministic.")
        practice.add(self.tokens)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_two_columns(_column(weights, entropy), _column(controls, stats, practice, explain)))
        self.beta.changed.connect(lambda _: self.refresh())
        self.new_btn.clicked.connect(self.new_sample)
        self.new_sample(first=True)

    def new_sample(self, first=False):
        if not first:
            self.seed += 1
        self.rem = sg.REM(n=16, seed=self.seed)
        self.curve_s.setData(self.betas, [self.rem.entropy_per_spin(b) for b in self.betas])
        self.refresh()

    def refresh(self):
        b = self.beta.value()
        w = np.sort(self.rem.weights(b))[::-1][: self.TOP]
        self.bars.setOpts(height=w)
        s = self.rem.entropy_per_spin(b)
        self.marker.setData([b], [s])
        self.k_states.setText(f"{self.rem.effective_states(b):,.0f}")
        self.k_entropy.setText(f"{s / np.log(2):.0%}")
        self.tokens.setText(f"At β = {b:.2f} (temperature {1 / b:.2f}): about <b>{self.rem.effective_states(b):,.0f}</b> "
                            f"of 65,536 tokens in play." if b > 0 else "At infinite temperature every token is equally likely.")
        if b < sg.REM.beta_c:
            self.phase.setText(f"β below β<sub>c</sub>: the probability is spread over exponentially many states (high temperature).")
        else:
            self.phase.setText(f"β above β<sub>c</sub>: frozen. A few states carry almost all the probability "
                               f"(Poisson–Dirichlet with m = β<sub>c</sub>/β = {sg.REM.beta_c / b:.2f}).")


# --------------------------------------------------------------------------- chapter
class Chapter1(QWidget):
    TITLE = "Spins, energy, temperature"

    def __init__(self):
        super().__init__()
        self.setObjectName("page")
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 24); lay.setSpacing(12)
        lay.addWidget(label("CHAPTER 1", "eyebrow"))
        lay.addWidget(label(self.TITLE, "title"))
        lay.addWidget(label("Three classic models, one idea: many ±1 variables, an energy, and a temperature that "
                            "decides how strictly the system follows the energy. <b>Hopfield</b> stores memories, "
                            "<b>SK</b> is a glass, the <b>REM</b> is the simplest model of a glass.", "lead"))
        self.seg = Segmented(["Hopfield · memory", "SK · glass", "REM · freezing"])
        lay.addWidget(self.seg)
        self.stack = QStackedWidget()
        self.pages = [HopfieldPage(), SKPage(), REMPage()]
        for p in self.pages:
            self.stack.addWidget(p)
        lay.addWidget(self.stack, 1)
        self.seg.changed.connect(self.stack.setCurrentIndex)
