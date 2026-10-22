"""Section 4: memory. Hopfield networks: recall, the same network as a restricted Boltzmann machine, capacity."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QHBoxLayout, QPushButton, QVBoxLayout, QWidget)

from .. import spinglass as sg
from . import theme
from .common import Section, _column, _two_columns
from .references import reading_card
from .widgets import Card, Deeper, FitStack, LabeledSlider, OverlapBars, Practice, Segmented, SpinGrid, label


class HopfieldPage(QWidget):
    FPS, UPDATES_PER_FRAME = 30, 24

    def __init__(self):
        super().__init__()
        self.net, self.rng = sg.Hopfield(), np.random.default_rng()
        self.target, self.sweeps = 0, 0.0
        self.s = self.net.corrupt(0, 0.3, self.rng)

        self.grid = SpinGrid(cell=22)
        self.status = label("", "small")
        state = Card("The network", "Each pixel is a neuron σ<sub>i</sub>, on (+1) or off (−1). "
                     "Press <b>Recall</b> and watch it roll downhill in energy.")
        state.add(self.grid, 1); state.add(self.status)

        thumbs = QWidget(); tl = QHBoxLayout(thumbs); tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(12)
        self.thumbs = []
        for mu, name in enumerate(sg.PATTERN_NAMES):
            t = SpinGrid(cell=5, selectable=True); t.set_state(sg.PATTERNS[mu]); t.setFixedSize(84, 84)
            t.clicked.connect(lambda k=mu: self.choose(k))
            tl.addWidget(t); self.thumbs.append(t)
        tl.addStretch(1)
        memories = Card("Stored memories", "Written into the couplings by [[hebb|Hebb's rule]]. Click one to corrupt it.")
        memories.add(thumbs)

        self.noise = LabeledSlider("Corruption", 0.0, 0.5, 0.3, 100, fmt=lambda v: f"{v:.0%} of pixels flipped")
        self.temp = LabeledSlider("Temperature T = 1/β", 0.0, 2.0, 0.15, 200,
                                  hint="T = 0: always go downhill. Large T: random flips, memories melt.")
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 4, 0, 0)
        self.corrupt_btn = QPushButton("Corrupt")
        self.half_btn = QPushButton("Half unknown")
        self.run_btn = QPushButton("Recall  ▶"); self.run_btn.setObjectName("primary")
        bl.addWidget(self.corrupt_btn); bl.addWidget(self.half_btn); bl.addWidget(self.run_btn); bl.addStretch(1)
        controls = Card("Controls"); controls.add(self.noise); controls.add(self.temp); controls.add(buttons)

        self.bars = OverlapBars(sg.PATTERN_NAMES)
        overlaps = Card("Overlap with each memory", "m = +1: perfect recall. m ≈ 0: unrelated. "
                        "These [[overlap|overlaps]] are the [[order-parameter|order parameters]] of the network.")
        overlaps.add(self.bars)
        overlaps.add(Deeper(
            "Energy H(σ) = −½ Σ<sub>i≠j</sub> J<sub>ij</sub> σ<sub>i</sub> σ<sub>j</sub>, with Hebbian couplings "
            "J<sub>ij</sub> = (1/N) Σ<sub>μ</sub> ξ<sub>i</sub><sup>μ</sup> ξ<sub>j</sub><sup>μ</sup>. In terms of the overlaps "
            "m<sub>μ</sub>(σ) = (1/N) Σ<sub>i</sub> ξ<sub>i</sub><sup>μ</sup> σ<sub>i</sub>, the energy is −(N/2) Σ<sub>μ</sub> m<sub>μ</sub><sup>2</sup> "
            "+ const: each memory is a valley. The dynamics is [[glauber|Glauber's]]: σ<sub>i</sub> = +1 with probability "
            "1/(1 + e<sup>−2h<sub>i</sub>/T</sup>), local field h<sub>i</sub> = Σ<sub>j</sub> J<sub>ij</sub> σ<sub>j</sub>. "
            "Here N = 256 and only 3 memories, so recall is easy. <b>With many memories</b> each one disturbs the "
            "recall of the others (cross-talk). If the number P of memories grows with N, P = αN, this disturbance "
            "does not average out: the memories act as random couplings and the network turns into a disordered, "
            "glass-like system. Recall survives up to α<sub>c</sub> ≈ 0.138, i.e. about 0.138·N random memories "
            "(Amit, Gutfreund, Sompolinsky); above it, only spurious states are left. Its rigorous solution at low "
            "temperature is still open (third tab). Open the [[hopfield|Hopfield]] entry for the formulas. "
            "Next tab: this network is a restricted Boltzmann machine in disguise."))

        practice = Practice("Can you find something when you only remember half of it?", rows=[
            ("The idea", "A <b>content-addressable memory</b> retrieves a stored item from a piece of its content, "
                         "not from an address. A phone book works by address (the name); a Hopfield network works by "
                         "content: give it a fragment or a noisy copy, get the whole item back."),
            ("Try it", "Press <b>Half unknown</b>: the bottom half of the chosen symbol is replaced by random ±1 "
                       "static. Press <b>Recall</b>."),
            ("Why it works", "Each step lowers the [[energy|energy]] H(σ) = −(N/2) Σ<sub>μ</sub> m<sub>μ</sub>(σ)². The known half "
                             "already has a large [[overlap]] with one memory, so the nearest valley is that memory."),
            ("Where it is used", "Modern Hopfield networks (Ramsauer et al., 2020): their update rule is the "
                                 "<b>attention</b> step of a transformer. The query is the noisy pattern, the keys are "
                                 "the stored memories, and a softmax picks the closest one."),
        ])
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



class NetworkDiagram(QWidget):
    """Left: fully connected Hopfield network. Right: bipartite RBM with Gaussian hidden units."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(250)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        panel = W / 2
        f = QFont(self.font()); f.setPixelSize(13)
        p.setFont(f)
        visible, hidden = QColor(theme.SPIN_UP), QColor(theme.ACCENT)
        edge = QPen(QColor(theme.INK_3)); edge.setWidthF(0.8)

        # Hopfield: 8 nodes on a circle, all pairs connected
        cx, cy, r = panel * 0.5, H * 0.46, min(panel, H) * 0.30
        pts = [QPointF(cx + r * np.cos(2 * np.pi * k / 8 - np.pi / 2), cy + r * np.sin(2 * np.pi * k / 8 - np.pi / 2)) for k in range(8)]
        p.setPen(edge)
        for i in range(8):
            for j in range(i + 1, 8):
                p.drawLine(pts[i], pts[j])
        p.setPen(Qt.NoPen); p.setBrush(visible)
        for q in pts:
            p.drawEllipse(q, 7, 7)
        p.setPen(QColor(theme.INK))
        p.drawText(QRectF(0, H - 44, panel, 20), Qt.AlignHCenter, "Hopfield: all pairs coupled")
        p.setPen(QColor(theme.INK_3))
        p.drawText(QRectF(0, H - 24, panel, 20), Qt.AlignHCenter, "couplings J = (1/N) Σ ξ ξᵀ")

        # RBM: 8 visible bottom, 3 hidden top
        x0 = panel + panel * 0.12
        xs_v = [x0 + k * (panel * 0.76) / 7 for k in range(8)]
        xs_h = [x0 + panel * 0.76 * f for f in (0.18, 0.5, 0.82)]
        yv, yh = H * 0.70, H * 0.20
        p.setPen(edge)
        for xv in xs_v:
            for xh in xs_h:
                p.drawLine(QPointF(xv, yv), QPointF(xh, yh))
        p.setPen(Qt.NoPen); p.setBrush(visible)
        for xv in xs_v:
            p.drawEllipse(QPointF(xv, yv), 7, 7)
        p.setBrush(hidden)
        for xh, name in zip(xs_h, sg.PATTERN_NAMES):
            p.drawRoundedRect(QRectF(xh - 11, yh - 11, 22, 22), 5, 5)
            p.setPen(QColor("white")); p.drawText(QRectF(xh - 11, yh - 11, 22, 22), Qt.AlignCenter, name); p.setPen(Qt.NoPen)
        p.setPen(QColor(theme.INK_3))
        p.drawText(QRectF(panel, yh - 36, panel, 18), Qt.AlignHCenter, "hidden: one Gaussian unit per memory")
        p.drawText(QRectF(panel, yv + 12, panel, 18), Qt.AlignHCenter, "visible: the pixels, ±1")
        p.setPen(QColor(theme.INK))
        p.drawText(QRectF(panel, H - 44, panel, 20), Qt.AlignHCenter, "RBM: links only between layers")
        p.setPen(QColor(theme.INK_3))
        p.drawText(QRectF(panel, H - 24, panel, 20), Qt.AlignHCenter, "weights = the memories ξ")

        # arrow between panels
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        ay = H * 0.46
        p.drawLine(QPointF(panel - 22, ay), QPointF(panel + 22, ay))
        p.drawLine(QPointF(panel - 22, ay), QPointF(panel - 14, ay - 6)); p.drawLine(QPointF(panel - 22, ay), QPointF(panel - 14, ay + 6))
        p.drawLine(QPointF(panel + 22, ay), QPointF(panel + 14, ay - 6)); p.drawLine(QPointF(panel + 22, ay), QPointF(panel + 14, ay + 6))
        p.end()



class RBMPage(QWidget):
    FPS = 5

    def __init__(self):
        super().__init__()
        self.rbm, self.rng = sg.RBM(beta=3.0), np.random.default_rng()
        self.target, self.steps = 0, 0
        self.s = sg.PATTERNS[0].copy()

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lay.addWidget(label("The memory network can be redrawn as a two-layer machine: the pixels below, one hidden unit "
                            "per memory above, and links only between the layers. Same probabilities, far fewer "
                            "connections: a [[hopfield|Hopfield network]] is a [[rbm|restricted Boltzmann machine]] "
                            "(Barra, Bernacchia, Santucci, Contucci). The hidden units are the dual variables of a "
                            "[[legendre|Legendre transform]].", "body"))

        diagram = Card("Two drawings of the same machine")
        diagram.add(NetworkDiagram())
        diagram.add(label("For the 256-pixel network with 3 memories: the Hopfield network needs <b>32,640</b> couplings, "
                          "the RBM only <b>768</b> weights (256 × 3), and the weights <i>are</i> the memories.", "body"))

        self.grid = SpinGrid(cell=16)
        self.hidden = OverlapBars(sg.PATTERN_NAMES)
        self.status = label("", "small")
        demo_row = QWidget(); dl = QHBoxLayout(demo_row); dl.setContentsMargins(0, 0, 0, 0); dl.setSpacing(18)
        dl.addWidget(self.grid, 3)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(label("<b>Hidden units</b> (scaled z<sub>μ</sub> / √(βN))", "body"))
        rl.addWidget(self.hidden); rl.addWidget(label("Each hidden unit lights up for its own memory.", "small")); rl.addStretch(1)
        dl.addWidget(right, 2)
        demo = Card("The RBM recalls, layer by layer",
                    "One step = sample all hidden units given the pixels, then all pixels given the hidden units.")
        demo.add(demo_row); demo.add(self.status)

        self.choice = Segmented([f"memory {n}" for n in sg.PATTERN_NAMES])
        self.beta = LabeledSlider("Inverse temperature β", 0.3, 5.0, 3.0, 470,
                                  hint="Small β: the hidden units barely listen, nothing is recalled.")
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 4, 0, 0)
        self.noise_btn = QPushButton("Corrupt 30%")
        self.half_btn = QPushButton("Half unknown")
        self.run_btn = QPushButton("Run  ▶"); self.run_btn.setObjectName("primary")
        for b in (self.noise_btn, self.half_btn, self.run_btn):
            bl.addWidget(b)
        bl.addStretch(1)
        controls = Card("Controls"); controls.add(self.choice); controls.add(self.beta); controls.add(buttons)

        why = Card("Why they are the same")
        why.add(label("A Gaussian hidden unit can 'absorb' a square: averaging e<sup>az</sup> over a standard Gaussian z "
                      "gives e<sup>a²/2</sup>. The Hopfield weight is e<sup>(βN/2) Σ m<sub>μ</sub>²</sup>, a sum of squares of "
                      "overlaps, so each square becomes one hidden unit. Integrate the hidden units out and you are "
                      "back to Hopfield; keep them and you have an RBM.", "body"))
        why.add(Deeper(
            "Joint weight of the RBM: exp( −½ Σ<sub>μ</sub> z<sub>μ</sub>² + √(β/N) Σ<sub>i,μ</sub> ξ<sub>i</sub><sup>μ</sup> σ<sub>i</sub> z<sub>μ</sub> ). "
            "Conditionals: z<sub>μ</sub> | σ ~ N(√(βN) m<sub>μ</sub>, 1) and P(σ<sub>i</sub> = +1 | z) = 1/(1 + e<sup>−2h<sub>i</sub></sup>), "
            "h<sub>i</sub> = √(β/N) Σ<sub>μ</sub> ξ<sub>i</sub><sup>μ</sup> z<sub>μ</sub>. Integrating z gives exp((βN/2) Σ<sub>μ</sub> m<sub>μ</sub>²), the Hopfield "
            "Gibbs weight. The package's test checks this on all 2⁶ states of a small machine. Integrating instead over σ "
            "gives the hidden layer an energy Σ<sub>i</sub> log cosh(h<sub>i</sub>): the same free energy seen from the other side "
            "(Barra et al. 2012). With more layers, a three-layer RBM reduces to a two-species Hopfield network "
            "(Agliari, Migliozzi, Tantari); its learning and retrieval regimes are studied in "
            "<i>Learning and retrieval operational modes for three-layer restricted Boltzmann machines</i> "
            "(Agliari, Sebastiani, J. Stat. Phys.)."))

        practice = Practice("How does a recommender guess the films you have not rated?", rows=[
            ("The model", "A restricted Boltzmann machine (RBM): visible units = a user's ratings, hidden units = "
                          "latent 'taste' features. 'Restricted' means no links inside a layer, only between layers."),
            ("History", "RBMs were among the models used in the Netflix Prize (Salakhutdinov, Mnih, Hinton, 2007)."),
            ("Try it", "Press <b>Half unknown</b>, then <b>Run</b>: the bottom half plays the films the user has not "
                       "seen. The machine alternates 'hidden given visible' and 'visible given hidden' and fills them in."),
            ("Difference", "Here the weights are the stored memories ξ (Hebb's rule). In a recommender they are "
                           "learned from data, but the filling-in step is the same."),
        ])
        lay.addWidget(_two_columns(_column(diagram, demo), _column(controls, why, practice)), 1)

        self.timer = QTimer(self); self.timer.setInterval(1000 // self.FPS); self.timer.timeout.connect(self.tick)
        self.choice.changed.connect(self.choose)
        self.noise_btn.clicked.connect(self.corrupt)
        self.half_btn.clicked.connect(self.half_unknown)
        self.run_btn.clicked.connect(self.toggle)
        self.beta.changed.connect(lambda v: setattr(self.rbm, "beta", v))
        self.choose(0)

    def choose(self, mu: int):
        self.target = mu
        self.half_unknown()

    def corrupt(self):
        self.stop()
        self.s = sg.PATTERNS[self.target].copy()
        self.s[self.rng.random(self.s.size) < 0.3] *= -1
        self.steps = 0; self.refresh(np.zeros(3))

    def half_unknown(self):
        self.stop()
        self.s = sg.PATTERNS[self.target].copy()
        self.s[128:] = self.rng.choice([-1.0, 1.0], 128)
        self.steps = 0; self.refresh(np.zeros(3))

    def toggle(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.timer.start(); self.run_btn.setText("Pause  ❚❚")

    def stop(self):
        self.timer.stop(); self.run_btn.setText("Run  ▶")

    def tick(self):
        z = self.rbm.sample_hidden(self.s, self.rng)
        self.s = self.rbm.sample_visible(z, self.rng)
        self.steps += 1
        self.refresh(z / np.sqrt(self.rbm.beta * self.rbm.N))
        if self.steps >= 30:
            self.stop()

    def refresh(self, hidden):
        self.grid.set_state(self.s)
        self.hidden.set_values(np.clip(hidden, -1, 1), self.target)
        m = self.rbm.overlaps(self.s)
        self.status.setText(f"step {self.steps} · overlap of the pixels with memory {sg.PATTERN_NAMES[self.target]}: "
                            f"{m[self.target]:+.2f}")

    def hideEvent(self, e):
        self.stop(); super().hideEvent(e)



class CapacityPage(QWidget):
    N = 400
    ALPHAS = np.round(np.arange(0.02, 0.305, 0.01), 3)

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lay.addWidget(label(
            "Store more and more random memories, P = αN, and ask whether each one can still be recalled. As α grows, "
            "the cross-talk between memories acts like random couplings and the network starts to behave like a "
            "[[spin-glass|spin glass]], close cousin of the Sherrington–Kirkpatrick ([[sk|SK]]) model: recall breaks down near the "
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
            "<b>Many memories, high temperature or small α</b>: the [[replica-symmetry|replica-symmetric]] free energy "
            "is proved (Pastur, Shcherbina and Tirozzi; Talagrand; and later work).<br><br>"
            "<b>Many memories, low temperature</b>, which is where recall lives: <b>open</b>. Physics has the "
            "replica-symmetric solution of Amit, Gutfreund and Sompolinsky (AGS, α<sub>c</sub> ≈ 0.138) and "
            "[[rsb|replica-symmetry-breaking]] corrections (Steffan and Kühn: α<sub>c</sub> ≈ 0.13819). Agliari, "
            "Albanese, Barra and Ottaviani derive broken-replica free energies by [[interpolation|Guerra's interpolation]], "
            "under an <i>ansatz</i>: an assumed form of the solution that is not itself proved. "
            "A full proof, an analogue of the [[parisi|Parisi formula]] for the Hopfield model, is not available.",
            "body"))

        practice = Practice("How many memories fit in N neurons?", rows=[
            ("Classical", "Hebbian Hopfield network, pairwise couplings: about 0.138·N random memories. "
                          "This is the AGS value (Amit, Gutfreund, Sompolinsky, 1985), the curve on the left."),
            ("Dense", "Dense associative memories (Krotov and Hopfield, 2016) replace m<sub>μ</sub>² in the energy by "
                      "m<sub>μ</sub><sup>p</sup>, interactions among p neurons at a time: capacity grows like N<sup>p−1</sup>."),
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
            "4", "Memory: Hopfield networks",
            story=("Random couplings (spin-glass models) create a rugged landscape with many valleys, but nobody chose them.",
                   "Choose the couplings on purpose, with Hebb's rule, so that the valleys are memories you picked.",
                   "Can a physical system store information and retrieve it from a fragment?"),
            lead="Store memories in the couplings and each memory becomes a valley. With one memory the "
            "[[hopfield|Hopfield model]] is the [[curie-weiss|Curie–Weiss]] magnet in disguise; with many, the memories interfere and it behaves like a "
            "[[spin-glass|spin glass]]. In between, it is a working memory, and a machine-learning model.")
        self.seg = Segmented(["Memory", "= Boltzmann machine", "How much can it remember?"])
        self.lay.addWidget(self.seg)
        self.stack = FitStack()
        for p in (HopfieldPage(), RBMPage(), CapacityPage()):
            self.stack.addWidget(p)
        self.lay.addWidget(self.stack)
        self.lay.addWidget(reading_card("hopfield"))
        self.lay.addStretch(1)
        self.stack.setCurrentIndex(0)
        self.seg.changed.connect(self.stack.setCurrentIndex)

