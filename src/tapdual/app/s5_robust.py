"""Section 5: robust decisions. The free energy read as a worst case over reweighted data (KL-DRO); with a
non-linear risk the sup becomes a Parisi-type inf over (q, m)."""
from __future__ import annotations

import threading

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget)

from .. import Quadrature, mean_variance_model, solve_primal
from ..dro import load_returns, worst_case_mean, worst_case_mean_variance
from ..dual import parisi_function
from . import theme
from .common import Section, _column, _hist, _metrics, _two_columns
from .references import reading_card
from .widgets import Card, Deeper, FitStack, LabeledSlider, Practice, Segmented, formula, label


RETURNS = load_returns()   # real data: daily returns in % of an equal-weight portfolio of 10 US stocks, 2015-2018
LOSSES = -RETURNS
N_DAYS = LOSSES.size
TAU = 0.5                  # weight of the KL penalty in the mean–variance example



def _dict_rows(pairs) -> QWidget:
    w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(12); g.setVerticalSpacing(6)
    for r, (a, b) in enumerate(pairs):
        x = label(a, "body"); x.setFixedWidth(330)
        g.addWidget(x, r, 0, Qt.AlignTop); g.addWidget(label(b, "body"), r, 1)
    g.setColumnStretch(1, 1)
    return w



class LinearPage(QWidget):
    EDGES = np.linspace(-6.5, 8.5, 31)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lay.addWidget(label(
            "You hold an equal-weight portfolio of 10 large US stocks (the data of the case study, section 5.1) and "
            "record its daily loss l (in %, a gain is a negative loss) over the 756 trading days from April 2015 to "
            "April 2018. "
            "The data define the <i>empirical distribution</i> μ: weight 1/756 on each day. An adversary may "
            "reweight the days into any ν with [[kl|relative entropy]] H(ν | μ) ≤ ρ, and picks the ν with the "
            "largest average loss. That worst average is your robust estimate.", "body"))

        self.plot_h = pg.PlotWidget(); theme.style_plot(self.plot_h.getPlotItem(), "daily loss l (%)", "probability")
        self.plot_h.setFixedHeight(250)
        mids, width = (self.EDGES[1:] + self.EDGES[:-1]) / 2, self.EDGES[1] - self.EDGES[0]
        nominal = _hist(LOSSES, np.full(N_DAYS, 1 / N_DAYS), self.EDGES)
        self.plot_h.addItem(pg.BarGraphItem(x=mids - width * 0.2, height=nominal, width=width * 0.4,
                                            brush=theme.GRID, pen=pg.mkPen(theme.INK_3)))
        self.bars = pg.BarGraphItem(x=mids + width * 0.2, height=nominal, width=width * 0.4,
                                    brush=theme.ACCENT, pen=pg.mkPen(None))
        self.plot_h.addItem(self.bars)
        hist = Card("The data and the adversary's reweighting",
                    f"<span style='color:{theme.INK_3}'>□</span> past data μ &nbsp; "
                    f"<span style='color:{theme.ACCENT}'>■</span> worst case ν*: probability moves to the bad days.")
        hist.add(self.plot_h)

        self.plot_w = pg.PlotWidget(); theme.style_plot(self.plot_w.getPlotItem(), "daily loss l (%)",
                                                        "weight multiplier n·ν*(day)")
        self.plot_w.setFixedHeight(220); self.plot_w.setLogMode(y=True)
        self.plot_w.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.BORDER)))  # log10(1) = 0
        self.curve_w = self.plot_w.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=theme.ACCENT,
                                        symbolPen=pg.mkPen(None))
        tilt = Card("Each day's weight is multiplied by e<sup>l/λ</sup>",
                    "Log scale. A straight line: the worst case is a [[gibbs|Gibbs distribution]] with energy −l "
                    "and temperature λ.")
        tilt.add(self.plot_w)

        self.rho = LabeledSlider("Budget ρ (how much you distrust the data)", 0.0, 7.0, 0.1, 700,
                                 hint="ρ = 0: trust the data. The largest possible budget is log 756 ≈ 6.63: then "
                                      "the adversary puts everything on the single worst day.")
        controls = Card("Controls"); controls.add(self.rho)

        nums = Card("Right now")
        row, (self.v_nom, self.v_worst, self.v_lam, self.v_eff) = _metrics(
            ["average loss, data", "worst-case average", "temperature λ*", f"days in play (of {N_DAYS})"])
        nums.add(row)
        self.phase = label("", "body"); nums.add(self.phase)

        formula_card = Card("The formula")
        formula_card.add(formula(r"$\sup_{\nu:\,H(\nu|\mu)\leq\rho}\ \mathbb{E}_\nu[l]\ =\ "
                                 r"\inf_{\lambda>0}\left\{\lambda\log\mathbb{E}_\mu e^{l/\lambda}+\lambda\rho\right\}$"))
        formula_card.add(formula(r"$\nu^*(i)\ \propto\ e^{l_i/\lambda^*},\qquad H(\nu^*|\mu)=\rho$"))
        formula_card.add(label(
            "Left: a sup over distributions (the adversary). Right: an inf over one number λ, the price of the budget. "
            "The right side is the [[free-energy|free energy]] of section 1 with energy −l at temperature λ, plus "
            "the cost λρ of the budget. As ρ grows, λ* drops: the adversary 'cools down' and "
            "<b>freezes</b> onto the worst days, exactly like the [[rem|REM]] below its critical temperature.", "body"))

        self.top = label("", "body")
        practice = Practice("How bad can an average day get if the past is only roughly right?", rows=[
            ("Setting", "756 real daily returns (10 US stocks, equal weight, 2015–2018). Loss l = −return. "
                        "On the average day the portfolio gained 0.06%."),
            ("The rule", "Distrust the data by a budget ρ and let an adversary reweight the days. The worst case "
                         "multiplies each day's weight by e<sup>l/λ</sup>: bad days count more, good days less."),
            ("Live", self.top),
            ("Lesson", "ρ is a dial between 'trust the past' (ρ = 0) and 'plan for the worst day' (ρ = log n). "
                       "Stress tests that reweight historical scenarios work the same way."),
        ])
        ml = Practice("Why would a machine-learning model train on a worst case?", rows=[
            ("Problem", "A model trained on average loss can fail on rare groups or on tomorrow's data "
                        "(distribution shift)."),
            ("Fix", "Minimise the worst-case loss over a KL ball around the training data. At a fixed temperature "
                    "this is <i>tilted empirical risk minimisation</i>: minimise λ log (1/n) Σ e<sup>ℓ<sub>i</sub>/λ</sup> "
                    "instead of (1/n) Σ ℓ<sub>i</sub> (Li, Beirami, Sanjabi, Smith, ICLR 2021)."),
            ("Same maths", "The log-sum-exp is a free energy; the hard examples are up-weighted by a Gibbs factor."),
        ])
        lay.addWidget(_two_columns(_column(hist, tilt, formula_card), _column(controls, nums, practice, ml)))
        self.rho.changed.connect(lambda _: self.refresh())
        self.refresh()

    def refresh(self):
        r = worst_case_mean(LOSSES, self.rho.value())
        self.bars.setOpts(height=_hist(LOSSES, r.weights, self.EDGES))
        order = np.argsort(LOSSES)
        mult = np.maximum(r.weights[order] * N_DAYS, 1e-6)
        self.curve_w.setData(LOSSES[order], mult)
        ent = -float(np.sum(r.weights[r.weights > 0] * np.log(r.weights[r.weights > 0])))
        self.v_nom.setText(f"{r.nominal:+.3f}%"); self.v_worst.setText(f"{r.value:+.3f}%")
        self.v_lam.setText("0 (frozen)" if r.frozen else f"{r.lam:.3f}")
        self.v_eff.setText(f"{np.exp(ent):,.0f}")
        self.phase.setText("Frozen: all the weight sits on the single worst day." if r.frozen else
                           "The budget is fully used: H(ν* | μ) = ρ. 'Days in play' = e<sup>Ent(ν*)</sup>, the "
                           "effective number of days the worst case relies on.")
        k = int(np.argmax(r.weights))
        self.top.setText(f"At ρ = {self.rho.value():.2f} the worst day (loss {LOSSES[k]:+.2f}%) counts "
                         f"<b>{r.weights[k] * N_DAYS:,.1f}×</b> its share in the data, and the robust average loss is "
                         f"<b>{r.value:+.2f}%</b> instead of {r.nominal:+.2f}%.")



class TangentsPicture(QWidget):
    """Convex: max of tangents (sup). Concave: min of tangents (inf)."""

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        x = np.linspace(-1, 1, 201)
        for title, sign, color in (("convex: max of tangents → sup", 1, theme.ACCENT_2),
                                   ("concave: min of tangents → inf", -1, theme.ACCENT)):
            p = pg.PlotWidget(); theme.style_plot(p.getPlotItem(), "", "")
            p.setFixedHeight(150); p.hideAxis("left"); p.hideAxis("bottom"); p.setMouseEnabled(False, False)
            p.setTitle(title, color=theme.INK_2, size="10pt")
            for a in np.linspace(-0.8, 0.8, 5):
                p.plot(x, sign * (a * a / 2 + a * (x - a)), pen=pg.mkPen(theme.INK_3, width=1))
            p.plot(x, sign * x * x / 2, pen=pg.mkPen(color, width=3))
            p.setYRange(-0.75, 0.75)
            lay.addWidget(p)



class NonlinearPage(QWidget):
    EDGES = np.linspace(-8.5, 6.5, 31)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        lay.addWidget(label(
            "Now measure risk by <b>mean + γ × variance</b> (Markowitz). Variance is not an average of something "
            "per day: Var<sub>ν</sub>(l) = E<sub>ν</sub>[l²] − (E<sub>ν</sub>[l])². The square of an average makes "
            "the payoff a <b>concave</b> function of q = E<sub>ν</sub>[l]. This is exactly the structure of the "
            "REM approximation of TAP free energies (Kistler, Schmidt, Sebastiani 2023), and the sup over ν turns "
            "into a <b>Parisi-type inf</b> over two numbers (q, m).", "body"))

        self.plot_w = pg.PlotWidget(); theme.style_plot(self.plot_w.getPlotItem(), "daily return (%)",
                                                        "weight multiplier n·ν*(day)")
        self.plot_w.setFixedHeight(240); self.plot_w.setLogMode(y=True)
        self.plot_w.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.BORDER)))
        self.c_lin = self.plot_w.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=theme.INK_3, symbolPen=pg.mkPen(None))
        self.c_mv = self.plot_w.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=theme.ACCENT, symbolPen=pg.mkPen(None))
        weights = Card("Which days does the worst case boost?",
                       f"<span style='color:{theme.INK_3}'>●</span> average loss only (tab 1, same budget) &nbsp; "
                       f"<span style='color:{theme.ACCENT}'>●</span> mean + γ × variance. Log scale.")
        weights.add(self.plot_w)

        self.plot_p = pg.PlotWidget(); theme.style_plot(self.plot_p.getPlotItem(), "q (the worst-case mean loss, %)",
                                                        "P(q, m*)")
        self.plot_p.setFixedHeight(220)
        self.c_p = self.plot_p.plot(pen=pg.mkPen(theme.ACCENT, width=2.5))
        self.dot = pg.ScatterPlotItem(size=11, brush=theme.ACCENT, pen=pg.mkPen("white", width=2))
        self.plot_p.addItem(self.dot)
        self.primal_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.ACCENT_2, width=1.5, style=Qt.DashLine))
        self.primal_line.hide(); self.plot_p.addItem(self.primal_line)
        profile = Card("The dual side: an inf, as in the Parisi formula",
                       "The Parisi-type function P(q, m) at the optimal m*, as a function of q. Its minimum is the "
                       f"robust value. <span style='color:{theme.ACCENT_2}'>- -</span> the sup computed directly "
                       "(press Check).")
        profile.add(self.plot_p)

        self.gamma = LabeledSlider("Risk aversion γ (weight of the variance)", 0.1, 3.0, 1.0, 290)
        self.rho = LabeledSlider("Budget ρ", 0.01, 1.0, 0.1, 99)
        self.check_btn = QPushButton("Check: solve the sup directly (about 15 s)")
        controls = Card("Controls"); controls.add(self.gamma); controls.add(self.rho); controls.add(self.check_btn)

        nums = Card("Right now")
        row1, (self.v_nom, self.v_worst, self.v_val) = _metrics(["risk, data", "risk, worst case", "robust value (inf)"])
        row2, (self.v_q, self.v_m, self.v_sup) = _metrics(["q* (mean loss)", "m* (Parisi)", "sup, solved directly"])
        nums.add(row1); nums.add(row2)
        self.status = label("", "small"); nums.add(self.status)

        why = Card("Why an inf instead of a sup")
        why.add(TangentsPicture())
        why.add(formula(r"$\sup_{\nu:\,H(\nu|\mu)\leq\rho}\left\{\Phi(\mathbb{E}_\nu l)+\mathbb{E}_\nu f-H(\nu|\mu)\right\}"
                        r"\ =\ \inf_{q,\ m\in(0,1]}\ P(q,m)$"))
        why.add(label(
            "Write the non-linearity through its tangent lines. A <b>convex</b> one (the m²/2 of Curie–Weiss and "
            "Hopfield) is the <i>maximum</i> of its tangents: the free energy stays a sup, now over the tangent point "
            "too. A <b>concave</b> one (−γq² here, −β²q²/4 in REM-TAP) is the <i>minimum</i> of its tangents: the "
            "tangent point q comes out as an inf. Swapping it with the sup over ν (a minimax theorem) leaves an inf "
            "over q and over m, the price of the budget: the shape of the [[parisi|Parisi formula]].", "body"))
        why.add(Deeper(
            "Objective (KSS23 form, divided by the penalty weight τ): Ω(ν) = Φ(E<sub>ν</sub>g) + E<sub>ν</sub>f − H(ν | μ) with "
            "g = l, f = (l + γl²)/τ, Φ(q) = −γq²/τ, maximised under H(ν | μ) ≤ ρ. Dual: "
            "P(q, m) = Φ(q) − qΦ′(q) + (1/m)(log E<sub>μ</sub> e<sup>m(Φ′(q)g + f)</sup> + ρ) − ρ, and "
            "sup Ω = inf<sub>q, m ∈ (0, 1]</sub> P(q, m). Optimality: q = E<sub>ν</sub>[g] for the tilted ν, and "
            "m = 1 (budget slack) or H(ν | μ) = ρ (budget binding, m &lt; 1). 1/m is the total price of relative "
            f"entropy. Here τ = {TAU}; values are shown multiplied back by τ."))

        self.tails = label("", "body")
        practice = Practice("Why does a cautious investor's worst case boost crashes <i>and</i> rallies?", rows=[
            ("Setting", "Risk = average loss + γ × variance, the classic Markowitz trade-off; γ is how much you "
                        "dislike fluctuations."),
            ("What you see", "With average loss only, the adversary boosts the bad days. With variance in the risk, "
                             "it boosts <b>both tails</b>: big rallies increase the variance too."),
            ("Live", self.tails),
            ("The maths", "The worst case is no longer a single Gibbs tilt found by a sup: it comes from an inf over "
                          "(q, m), the same Parisi-type duality as in the spin-glass papers."),
            ("In ML", "Variance-regularised training (Namkoong and Duchi, NeurIPS 2017) penalises the variance of "
                      "the loss across examples for the same reason: robustness."),
        ])

        lay.addWidget(_two_columns(_column(weights, profile, why), _column(controls, nums, practice)))
        self.gamma.changed.connect(lambda _: self.refresh())
        self.rho.changed.connect(lambda _: self.refresh())
        self.check_btn.clicked.connect(self.start_check)
        self.worker, self.result = None, None
        self.poll = QTimer(self); self.poll.setInterval(200); self.poll.timeout.connect(self.collect)
        self.refresh()

    def refresh(self):
        g, rho = self.gamma.value(), self.rho.value()
        r = worst_case_mean_variance(LOSSES, g, TAU, rho, check_primal=False)
        lin = worst_case_mean(LOSSES, rho)
        self.c_lin.setData(RETURNS, np.maximum(lin.weights * N_DAYS, 1e-6))
        self.c_mv.setData(RETURNS, np.maximum(r.weights * N_DAYS, 1e-6))
        model = mean_variance_model(LOSSES, g, TAU)
        quad = Quadrature(model)
        qs = r.worst_mean + np.linspace(-2.5, 2.5, 121)
        vals = np.array([TAU * parisi_function(quad, q, r.m, rho) for q in qs])
        self.c_p.setData(qs, vals); self.dot.setData([r.worst_mean], [r.objective])
        top = float(np.percentile(vals, 70)); span = max(top - vals.min(), 1e-3)
        self.plot_p.setYRange(vals.min() - 0.08 * span, top)
        self.primal_line.hide(); self.v_sup.setText("–")
        self.v_nom.setText(f"{r.nominal_risk:.3f}"); self.v_worst.setText(f"{r.worst_risk:.3f}")
        self.v_val.setText(f"{r.objective:.4f}")
        self.v_q.setText(f"{r.worst_mean:+.3f}"); self.v_m.setText(f"{r.m:.3f}")
        self.status.setText(f"Budget {'binding (m* &lt; 1): H(ν* | μ) = ρ' if r.phase == 'binding' else 'slack (m* = 1)'}. "
                            "Robust value = worst-case risk − τ·H(ν* | μ).")
        lo, hi = np.percentile(RETURNS, [5, 95])
        w = r.weights
        self.tails.setText(f"The worst 5% of days carry <b>{w[RETURNS <= lo].sum():.0%}</b> of the worst-case weight, "
                           f"the best 5% carry <b>{w[RETURNS >= hi].sum():.0%}</b> (5% each in the data). "
                           f"Average loss only: {lin.weights[RETURNS <= lo].sum():.0%} and "
                           f"{lin.weights[RETURNS >= hi].sum():.0%}.")
        self.current, self.current_value = (g, rho), r.objective

    def start_check(self):
        if self.worker and self.worker.is_alive():
            return
        g, rho = self.current
        self.check_btn.setText("Solving the sup over ν…"); self.check_btn.setEnabled(False)

        def run():
            self.result = (g, rho, TAU * solve_primal(mean_variance_model(LOSSES, g, TAU), rho).value)
        self.result = None
        self.worker = threading.Thread(target=run, daemon=True); self.worker.start(); self.poll.start()

    def collect(self):
        if self.result is None:
            return
        self.poll.stop(); self.check_btn.setEnabled(True)
        self.check_btn.setText("Check: solve the sup directly (about 15 s)")
        g, rho, value = self.result
        if (g, rho) != self.current:
            return
        self.v_sup.setText(f"{value:.4f}")
        self.primal_line.setPos(value); self.primal_line.show()
        gap = abs(value - self.current_value)
        self.status.setText(f"sup over ν (solved directly) = inf over (q, m), up to {gap:.0e}.")



class RobustSection(Section):
    def __init__(self):
        super().__init__(
            "5", "Robust decisions",
            "Every decision is tested on past data, but the future is not the past. Distributionally robust "
            "optimisation (DRO) asks for the worst average loss over all distributions close to the data, and "
            "protects against it. With closeness measured by relative entropy, this is the free energy of "
            "section 1 again, and the budget behaves like the entropy budget of the REM.",
            story=("Free energy = max over ν of payoff − relative entropy (section 1). In the REM the entropy budget "
                   "log 2 runs out and the measure freezes (section 3).",
                   "Read ν as an adversary who may reweight your data within a budget H(ν | μ) ≤ ρ.",
                   "How bad can things get if the data are slightly wrong, and what plays the role of temperature?"))
        self.seg = Segmented(["Worst case in a KL ball", "Nonlinear risk: the sup becomes an inf", "Dictionary"])
        self.lay.addWidget(self.seg)
        self.stack = FitStack()
        self.stack.addWidget(LinearPage()); self.stack.addWidget(NonlinearPage()); self.stack.addWidget(self._dictionary())
        self.stack.setCurrentIndex(0)
        self.lay.addWidget(self.stack)
        self.seg.changed.connect(self.stack.setCurrentIndex)
        self.lay.addWidget(reading_card("robust"))
        self.lay.addStretch(1)

    def _dictionary(self) -> QWidget:
        card = Card("Statistical mechanics ↔ robust optimisation")
        card.add(_dict_rows([
            ("<b>Statistical mechanics</b>", "<b>Robust optimisation</b>"),
            ("reference measure μ (all configurations equally likely)", "the data: weight 1/n on each observation"),
            ("Gibbs measure ν", "the adversary's reweighting of the data"),
            ("entropy budget log 2 per spin (REM)", "ambiguity budget ρ: how far from the data ν may move"),
            ("temperature 1/β", "λ, the price of the budget (the multiplier of the KL constraint)"),
            ("freezing below the critical temperature", "the worst case piles onto a few extreme observations"),
            ("no non-linearity (the plain REM): a single Gibbs tilt", "risk that is an average loss (tab 1)"),
            ("convex non-linearity (Curie–Weiss, Hopfield): still a sup", "not used here"),
            ("concave non-linearity (REM-TAP): Parisi-type inf over (q, m)",
             "risk with a variance: inf over (q, m), q = worst-case mean, 1/m = price of the budget"),
            ("m = 1 (high temperature) / m &lt; 1 (frozen)", "budget slack / budget binding"),
        ]))
        lead = label("The same objects under two names. Left: the spin-glass papers. Right: the worked examples of this "
                     "section.", "body")
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(12)
        l.addWidget(lead); l.addWidget(card); l.addStretch(1)
        return w

