"""Hopfield section, second tab: a Hopfield network is a restricted Boltzmann machine in disguise."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .. import spinglass as sg
from . import theme
from .chapter1 import _column, _two_columns
from .widgets import Card, Deeper, LabeledSlider, OverlapBars, Practice, Segmented, SpinGrid, label


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
            "Joint weight of the RBM: exp( −½ Σ<sub>μ</sub> z<sub>μ</sub>² + √(β/N) Σ<sub>i,μ</sub> ξ<sub>i</sub><sup>μ</sup> s<sub>i</sub> z<sub>μ</sub> ). "
            "Conditionals: z<sub>μ</sub> | s ~ N(√(βN) m<sub>μ</sub>, 1) and P(s<sub>i</sub> = +1 | z) = 1/(1 + e<sup>−2h<sub>i</sub></sup>), "
            "h<sub>i</sub> = √(β/N) Σ<sub>μ</sub> ξ<sub>i</sub><sup>μ</sup> z<sub>μ</sub>. Integrating z gives exp((βN/2) Σ<sub>μ</sub> m<sub>μ</sub>²), the Hopfield "
            "Gibbs weight. The package's test checks this on all 2⁶ states of a small machine. Integrating instead over s "
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
