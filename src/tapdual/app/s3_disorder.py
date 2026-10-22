"""Section 3: disorder. Spin glasses: SK (frustration, two replicas) and the REM (freezing)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget)

from .. import spinglass as sg
from . import theme
from .common import Section, _column, _two_columns
from .references import reading_card
from .widgets import Card, Deeper, FitStack, FrustrationTriangle, LabeledSlider, Practice, Segmented, SpinGrid, label


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
                        "There is no hidden picture to find: just many valleys of similar depth.")
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

        self.temp = LabeledSlider("Temperature T = 1/β", 0.1, 2.0, 0.4, 190,
                                  hint="Above T = 1 the spins fluctuate freely (paramagnet); below, they freeze into a glass.")
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 4, 0, 0)
        self.run_btn = QPushButton("Cool from random  ▶"); self.run_btn.setObjectName("primary")
        self.new_btn = QPushButton("New couplings")
        bl.addWidget(self.run_btn); bl.addWidget(self.new_btn); bl.addStretch(1)
        controls = Card("Controls"); controls.add(self.temp); controls.add(buttons)

        puzzle = Card("Frustration in three spins",
                      f"<span style='color:{theme.ACCENT}'><b>=</b></span> the pair wants to agree, "
                      f"<span style='color:{theme.ACCENT_2}'><b>≠</b></span> it wants to disagree; the signs are random. "
                      "Click the spins to flip them and try to satisfy every link.")
        self.triangle = FrustrationTriangle()
        puzzle.add(self.triangle)
        tri_btn = QPushButton("New random signs"); tri_btn.clicked.connect(self.triangle.randomize)
        puzzle.add(tri_btn)
        puzzle.add(label("Spin-glass models combine two ingredients: <b>disorder</b> (random signs) and <b>frustration</b> "
                         "(loops whose signs multiply to −1, about half of them here). All '≠' on a triangle would "
                         "also be frustrated, but without randomness: that is an antiferromagnet on a triangular "
                         "lattice, not a glass.", "small"))

        self.readout = label("", "body")
        explain = Card("What to look for")
        explain.add(label("In the Sherrington–Kirkpatrick ([[sk|SK]]) model every pair of spins gets its own random "
                          "coupling J<sub>ij</sub>: some pairs want to agree, others to disagree. Like the triangle, no "
                          "configuration satisfies them all. The landscape becomes rugged: two copies cooled the same way "
                          "reach almost the same low energy, yet usually land in <b>different valleys</b>. Their "
                          "[[overlap]] q stays well below 1 and changes from run to run.", "body"))
        explain.add(self.readout)
        explain.add(Deeper(
            "Energy H(σ) = −Σ<sub>i&lt;j</sub> J<sub>ij</sub> σ<sub>i</sub> σ<sub>j</sub> with J<sub>ij</sub> ~ N(0, 1/N) independent. "
            "Overlap of two copies q = (1/N) Σ<sub>i</sub> σ<sub>i</sub><sup>a</sup> σ<sub>i</sub><sup>b</sup>. "
            "For T &gt; 1 the free energy is log 2 + β²/4; below T = 1 [[rsb|replica symmetry breaks]] and the free energy "
            "is the [[parisi|Parisi formula]], an <b>infimum</b> over a functional order parameter (proved by Guerra and "
            "Talagrand). The lowest energy per spin tends to −0.7633. With N = 256 this run is a small, honest "
            "finite-size picture, not the limit."))

        self.practice_readout = label("", "body")
        practice = Practice("Is there one best way to split a network in two?", rows=[
            ("The problem", "<b>Max-Cut</b>: split the nodes of a weighted network into two groups so that the total "
                            "weight of the links between the groups is as large as possible. With weights −J<sub>ij</sub>, "
                            "this is exactly finding the lowest energy of the SK model."),
            ("Where it appears", "Chip layout, clustering, scheduling, portfolio selection with pairwise effects. "
                                 "Ising machines and quantum annealers are built to minimise such energies."),
            ("Live", self.practice_readout),
            ("Lesson", "No: there are <b>many</b> near-optimal splits that look nothing alike. An algorithm that returns "
                       "'the' answer is picking one valley among many."),
        ])
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_two_columns(_column(replicas, traces), _column(puzzle, controls, explain, practice)))

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
        weights = Card("Where the probability sits, among 2¹⁶ = 65,536 states",
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
                          "<b>freezes</b> onto a few states. In the robust-optimisation section the same budget reappears as a "
                          "constraint on the [[kl|relative entropy]], H(ν | μ) ≤ log 2, and the freezing as the moment the "
                          "constraint binds.", "body"))
        explain.add(Deeper(
            "Derrida's REM: energies H(σ) ~ N(0, N) independent for the 2<sup>N</sup> configurations σ. "
            "Free energy (1/N) log Σ<sub>σ</sub> e<sup>−βH(σ)</sup> → log 2 + β²/2 for β ≤ β<sub>c</sub> = √(2 log 2), "
            "and β·√(2 log 2) above. In the frozen phase the normalised weights of the lowest states converge to a "
            "[[poisson-dirichlet|Poisson–Dirichlet]] distribution with parameter m = β<sub>c</sub>/β ∈ (0, 1): the "
            "one-step [[rsb|replica-symmetry-breaking]] (1RSB) picture, and the same m as in the [[parisi|Parisi formula]]."))

        self.tokens = label("", "body")
        practice = Practice("What does 'temperature' do when a chatbot writes?", rows=[
            ("Setting", "A large language model (LLM) picks the next token (a word or word piece) from a vocabulary "
                        "of tens of thousands, with probability p<sub>i</sub> ∝ e<sup>ℓ<sub>i</sub>/T</sup>: ℓ<sub>i</sub> are its "
                        "scores (logits), T is the sampling temperature."),
            ("Toy model", "Read the 65,536 states here as tokens and the random energies H(σ) as minus the logits "
                          "(so β = 1/T). "
                          "This is deliberately crude: real logits are not independent random numbers."),
            ("Measure", "Effective number of tokens in play: N<sub>eff</sub> = 1 / Σ<sub>i</sub> p<sub>i</sub>² "
                        "(all equal: N<sub>eff</sub> = vocabulary size; one certain token: N<sub>eff</sub> = 1)."),
            ("Live", self.tokens),
            ("Lesson", "Below a critical temperature the choice <b>freezes</b> onto a handful of tokens: the REM "
                       "transition at β<sub>c</sub> = √(2 log 2), in this model's scaling."),
        ])
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lay.addWidget(label(
            "Push disorder to the limit: forget couplings and give every configuration its own independent random "
            "energy. Derrida's Random Energy Model ([[rem|REM]]) is the simplest spin-glass model. It <b>freezes</b>, and the "
            "reason is a budget: there are only 2<sup>N</sup> configurations, so the [[entropy]] cannot exceed "
            "log 2 per spin. When the budget runs out, the probability piles up on a handful of states.", "body"))
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
            self.phase.setText("β below β<sub>c</sub>: the probability is spread over exponentially many states (high temperature).")
        else:
            self.phase.setText(f"β above β<sub>c</sub>: frozen. A few states carry almost all the probability "
                               f"(their sizes follow a Poisson–Dirichlet law with parameter m = β<sub>c</sub>/β = "
                               f"{sg.REM.beta_c / b:.2f}; details in the box below).")



class DisorderSection(Section):
    def __init__(self):
        super().__init__(
            "3", "Disorder: spin glasses",
            "Replace 'everyone agrees' by random rules: some pairs want to agree, others to disagree. Random rules "
            "can contradict each other, the landscape becomes rugged, and at low temperature the system can freeze "
            "into one of many disordered states: the behaviour that [[spin-glass|spin-glass]] models are built to "
            "describe.",
            story=("When everyone wants to agree (Curie–Weiss), one number m tells the whole story and order appears "
                   "below a critical temperature.",
                   "Each pair of spins gets its own random coupling. We look at two models: SK, where pairs interact "
                   "randomly, and the REM, the most extreme case, where every configuration gets a random energy.",
                   "What happens when the rules contradict each other?"))
        self.seg = Segmented(["SK: frustration", "REM: extreme disorder"])
        self.lay.addWidget(self.seg)
        self.stack = FitStack()
        self.stack.addWidget(SKPage())
        self.stack.addWidget(REMPage())
        self.lay.addWidget(self.stack)
        self.lay.addWidget(reading_card("disorder"))
        self.lay.addStretch(1)
        self.stack.setCurrentIndex(0)
        self.seg.changed.connect(self.stack.setCurrentIndex)

