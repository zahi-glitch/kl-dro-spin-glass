"""Section 1: free energy is an optimisation (the Gibbs variational principle)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from . import theme
from .chapter1 import _column, _two_columns
from .widgets import Card, Deeper, LabeledSlider, Practice, label

from ..spinglass import ENERGIES, boltzmann, log_Z, terms


class Chapter2(QWidget):
    TITLE = "Free energy is an optimisation"

    def __init__(self, header: bool = True):
        super().__init__()
        self.ts = np.linspace(0, 4, 161)

        lay = QVBoxLayout(self); lay.setSpacing(12)
        if header:
            self.setObjectName("page")
            lay.setContentsMargins(28, 24, 28, 24)
            lay.addWidget(label("SECTION 1", "eyebrow"))
            lay.addWidget(label(self.TITLE, "title"))
        else:
            lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label("Nature does not compute partition functions: it <b>optimises</b>. Among all ways of spreading "
                            "probability over the states, the [[gibbs|Gibbs distribution]] is the one that best balances "
                            "low [[energy|energy]] against high [[entropy]], and the best value it reaches is the "
                            "[[free-energy|free energy]] log Z. Every model in the next sections is an instance of this.",
                            "lead" if header else "body"))
        lay.addWidget(label("<b>The setting.</b> A toy system with n = 10 states σ = 1, …, 10 and [[energy|energies]] H(σ). "
                            "β = 1/T is the [[temperature|inverse temperature]]. A <i>trial distribution</i> ν is any "
                            "probability vector on the states; its [[entropy]] is Ent(ν) = −Σ ν(σ) log ν(σ). The "
                            "[[gibbs|Gibbs distribution]] is μ<sub>β</sub>(σ) = e<sup>−βH(σ)</sup>/Z, with Z = Σ<sub>σ</sub> e<sup>−βH(σ)</sup>.", "body"))

        # left: the distributions and the objective
        self.plot_p = pg.PlotWidget(); theme.style_plot(self.plot_p.getPlotItem(), "states, from lowest to highest energy", "probability")
        self.plot_p.setFixedHeight(240); self.plot_p.setYRange(0, 1); self.plot_p.setXRange(0.3, 10.7)
        x = np.arange(1, 11)
        self.bar_gibbs = pg.BarGraphItem(x=x + 0.18, height=np.zeros(10), width=0.34, brush=theme.GRID, pen=pg.mkPen(theme.INK_3))
        self.bar_trial = pg.BarGraphItem(x=x - 0.18, height=np.zeros(10), width=0.34, brush=theme.ACCENT, pen=pg.mkPen(None))
        self.plot_p.addItem(self.bar_gibbs); self.plot_p.addItem(self.bar_trial)
        dists = Card("Your guess against the Gibbs distribution",
                     f"<span style='color:{theme.ACCENT}'>■</span> your trial distribution ν &nbsp; "
                     f"<span style='color:{theme.INK_3}'>□</span> Gibbs distribution μ<sub>β</sub>")
        dists.add(self.plot_p)

        self.plot_o = pg.PlotWidget(); theme.style_plot(self.plot_o.getPlotItem(), "trial inverse temperature t", "value")
        self.plot_o.setFixedHeight(260); self.plot_o.setXRange(0, 4)
        self.c_e = self.plot_o.plot(pen=pg.mkPen(theme.ACCENT_2, width=2))
        self.c_h = self.plot_o.plot(pen=pg.mkPen(theme.INK_3, width=2))
        self.c_sum = self.plot_o.plot(pen=pg.mkPen(theme.ACCENT, width=2.5))
        self.logz_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.INK, width=1),
                                         label="log Z", labelOpts={"color": theme.INK, "position": 0.06})
        self.plot_o.addItem(self.logz_line)
        self.dot = pg.ScatterPlotItem(size=11, brush=theme.ACCENT, pen=pg.mkPen("white", width=2))
        self.plot_o.addItem(self.dot)
        objective = Card("Energy against entropy",
                         f"<span style='color:{theme.ACCENT_2}'>━</span> −β E<sub>ν</sub>[H] (energy term) &nbsp; "
                         f"<span style='color:{theme.INK_3}'>━</span> entropy Ent(ν) &nbsp; "
                         f"<span style='color:{theme.ACCENT}'>━</span> their sum. "
                         "The sum is highest exactly when ν is the Gibbs distribution (t = β), "
                         "and its maximum is log Z.")
        objective.add(self.plot_o)

        # right: controls and readouts
        self.beta = LabeledSlider("Inverse temperature β", 0.0, 3.0, 1.2, 300,
                                  hint="β = 1/T: how strongly the system prefers low energy.")
        self.t = LabeledSlider("Trial distribution ν: parameter t", 0.0, 4.0, 0.4, 400,
                               hint="Here ν = μ<sub>t</sub>, the Gibbs distribution at a trial value t. t = 0: uniform. Large t: all on the lowest state.")
        self.snap = label("", "small")
        controls = Card("Controls"); controls.add(self.beta); controls.add(self.t)

        self.v_obj, self.v_logz, self.v_kl = label("", "value"), label("", "value"), label("", "value")
        nums = Card("Right now")
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(22)
        for title, val in (("your value", self.v_obj), ("log Z (best)", self.v_logz), ("gap = H(ν | μ<sub>β</sub>)", self.v_kl)):
            box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(0)
            bl.addWidget(label(title, "small")); bl.addWidget(val); rl.addWidget(box)
        rl.addStretch(1)
        nums.add(row)
        nums.add(label("The gap between your value and log Z is exactly the [[kl|relative entropy]] H(ν | μ<sub>β</sub>), "
                       "also called the Kullback–Leibler (KL) divergence. It is zero only at ν = μ<sub>β</sub>, so the optimum is unique.", "small"))

        ml = Card("The same formula in machine learning")
        ml.add(label("<b>Softmax.</b> With logits −H(σ) and temperature T = 1/β, the softmax <i>is</i> the Gibbs distribution: "
                     "the probabilities that best trade a high score against spread.<br>"
                     "<b>Variational inference.</b> log p(x) = max<sub>ν</sub> { E<sub>ν</sub>[log p(x, z)] + Ent(ν) }, ν a distribution of the latent z: "
                     "the ELBO (evidence lower bound) is this principle with −βH = log p(x, z), and the gap is again a KL divergence.<br>"
                     "<b>Entropy regularisation.</b> Adding +Ent to an objective (soft-max policies, label smoothing) "
                     "is choosing a Gibbs distribution.", "body"))
        ml.add(Deeper(
            "<b>Gibbs variational principle.</b> For every trial distribution ν, "
            "−β E<sub>ν</sub>[H] + Ent(ν) = log Z − H(ν | μ<sub>β</sub>). Since H(ν | μ<sub>β</sub>) ≥ 0, "
            "log Z = max<sub>ν</sub> { −β E<sub>ν</sub>[H] + Ent(ν) }, attained only at ν = μ<sub>β</sub>.<br>"
            "<b>Robust reading.</b> Let μ be the uniform distribution. Then Ent(ν) = log n − H(ν | μ), so "
            "log Z = log n + max<sub>ν</sub> { E<sub>ν</sub>[−βH] − H(ν | μ) }. Read ν as an <i>adversary</i>: it may move "
            "away from the reference μ to make E<sub>ν</sub>[−βH] large, but pays H(ν | μ) for the move. "
            "This is a KL-penalised worst case; in the robust-optimisation section the penalty becomes a budget H(ν | μ) ≤ ρ."))

        self.reco = label("", "body")
        practice = Practice("How much screen time should each product get?", rows=[
            ("Setting", "A recommender shows one of n = 10 products. Product i has a relevance score "
                        "s<sub>i</sub> = −H(i). A policy is a probability vector ν = (ν<sub>1</sub>, …, ν<sub>n</sub>): "
                        "how often each product is shown."),
            ("The rule", "Maximise relevance plus variety: max<sub>ν</sub> { E<sub>ν</sub>[s] + β<sup>−1</sup>·Ent(ν) }. "
                         "The answer is the softmax ν<sub>i</sub> = e<sup>βs<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>βs<sub>j</sub></sup>: "
                         "exactly the [[gibbs|Gibbs distribution]] of this section."),
            ("Read the dial", "β = 0: every product equally often (pure exploration). β → ∞: only the best one "
                              "(pure exploitation, and the system never learns about the others)."),
            ("Live", self.reco),
        ])
        lay.addWidget(_two_columns(_column(dists, objective), _column(controls, nums, practice, ml)), 1)
        self.beta.changed.connect(lambda _: self.refresh())
        self.t.changed.connect(lambda _: self.refresh())
        self.refresh()

    def refresh(self):
        beta, t = self.beta.value(), self.t.value()
        q, p = boltzmann(t), boltzmann(beta)
        self.bar_trial.setOpts(height=q)
        self.bar_gibbs.setOpts(height=p)
        curves = np.array([terms(boltzmann(s), beta) for s in self.ts])
        self.c_e.setData(self.ts, curves[:, 0]); self.c_h.setData(self.ts, curves[:, 1]); self.c_sum.setData(self.ts, curves[:, 2])
        lz = log_Z(beta)
        self.logz_line.setPos(lz)
        e, h, val = terms(q, beta)
        self.dot.setData([t], [val])
        lo = min(curves.min(), 0) - 0.2
        self.plot_o.setYRange(lo, max(curves.max(), lz) + 0.3)
        self.v_obj.setText(f"{val:.3f}"); self.v_logz.setText(f"{lz:.3f}"); self.v_kl.setText(f"{lz - val:.3f}")
        top = np.sort(p)[::-1]
        smooth = f" The free energy divided by β, (1/β) log Z = <b>{lz / beta:.2f}</b>, is a <i>smooth maximum</i> of the scores (the hard maximum is {-ENERGIES.min():.2f}); machine learning uses this log-sum-exp wherever it needs a differentiable max." if beta > 0.05 else ""
        self.reco.setText(f"At β = {beta:.2f}: the top product gets <b>{top[0]:.0%}</b> of the exposure, the top three "
                          f"<b>{top[:3].sum():.0%}</b>.{smooth}")
