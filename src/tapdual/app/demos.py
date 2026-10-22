"""Small interactive demos shown inside glossary entries (block kind "demo").

Both use n = 6 states and bars you drag with the mouse:
- EntropyDemo: drag a distribution ν; see Ent(ν) + H(ν | uniform) = log n.
- EnergyDemo: drag the energies H(σ); see the Gibbs distribution they produce at inverse temperature β.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from . import theme
from .widgets import LabeledSlider, label

N = 6


class Bars(QWidget):
    """n bars on [lo, hi]. Click or drag inside a column to set its height (if editable)."""
    edited = Signal()

    def __init__(self, values, lo=0.0, hi=1.0, color=theme.ACCENT, editable=True, ref=None, fmt="{:.2f}",
                 height=150):
        super().__init__()
        self.values, self.lo, self.hi, self.color = list(values), lo, hi, QColor(color)
        self.editable, self.ref, self.fmt = editable, ref, fmt
        self.setFixedHeight(height)
        if editable:
            self.setCursor(Qt.PointingHandCursor)

    def set_values(self, values):
        self.values = list(values); self.update()

    def _geom(self):
        W, H = self.width(), self.height()
        top, bottom = 6, 22
        return W / len(self.values), top, H - bottom

    def _y(self, v, top, base):
        return base - (v - self.lo) / (self.hi - self.lo) * (base - top)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        col, top, base = self._geom()
        f = QFont(self.font()); f.setPixelSize(12); p.setFont(f)
        zero = self._y(max(self.lo, min(0.0, self.hi)), top, base)
        p.setPen(QPen(QColor(theme.BORDER), 1)); p.drawLine(0, int(zero), self.width(), int(zero))
        for i, v in enumerate(self.values):
            x = i * col
            y = self._y(v, top, base)
            p.setPen(Qt.NoPen); p.setBrush(self.color)
            p.drawRoundedRect(QRectF(x + col * 0.2, min(y, zero), col * 0.6, max(abs(zero - y), 1.5)), 3, 3)
            p.setPen(QColor(theme.INK_3))
            p.drawText(QRectF(x, base + 3, col, 18), Qt.AlignHCenter, self.fmt.format(v))
        if self.ref is not None:
            pen = QPen(QColor(theme.INK_3), 1.5, Qt.DashLine); p.setPen(pen)
            y = self._y(self.ref, top, base); p.drawLine(0, int(y), self.width(), int(y))
        p.end()

    def _set_from(self, pos):
        if not self.editable:
            return
        col, top, base = self._geom()
        i = int(pos.x() // col)
        if 0 <= i < len(self.values):
            v = self.lo + (base - pos.y()) / (base - top) * (self.hi - self.lo)
            self.values[i] = min(self.hi, max(self.lo, v))
            self.update(); self.edited.emit()

    def mousePressEvent(self, e):
        self._set_from(e.position())

    def mouseMoveEvent(self, e):
        self._set_from(e.position())


class Split(QWidget):
    """One horizontal bar of fixed total, split into a blue and an orange part."""

    def __init__(self):
        super().__init__(); self.setFixedHeight(26); self.frac = 1.0

    def set_fraction(self, f):
        self.frac = min(1.0, max(0.0, f)); self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setPen(Qt.NoPen)
        W, H = self.width(), self.height()
        p.setBrush(QColor(theme.ACCENT_2)); p.drawRoundedRect(QRectF(0, 0, W, H), 6, 6)
        p.setBrush(QColor(theme.ACCENT)); p.drawRoundedRect(QRectF(0, 0, W * self.frac, H), 6, 6)
        p.end()


def _metric(title):
    box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(0)
    val = label("", "value"); bl.addWidget(label(title, "small", wrap=False)); bl.addWidget(val)
    return box, val


def _presets(pairs, fn):
    row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
    for name, vals in pairs:
        b = QPushButton(name); b.clicked.connect(lambda _=False, v=vals: fn(v)); rl.addWidget(b)
    rl.addStretch(1)
    return row


class EntropyDemo(QWidget):
    """Ent(ν) + H(ν | μ) = log n, with μ uniform."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(8)
        lay.addWidget(label("<b>Try it.</b> Drag the blue bars to choose a distribution ν on 6 states. "
                            "Dashed line: the uniform distribution μ = 1/6.", "body"))
        lay.addWidget(_presets([("Uniform", [1] * 6), ("Tilted", [3, 2, 1, 1, .5, .5]),
                                ("Half the states", [1, 1, 1, 0, 0, 0]), ("One state", [1, 0, 0, 0, 0, 0])],
                               self.set_weights))
        self.bars = Bars([1] * N, 0.0, 1.0, ref=1 / N)
        self.bars.edited.connect(self.normalise)
        lay.addWidget(self.bars)
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(22)
        (b1, self.v_ent), (b2, self.v_kl), (b3, v_sum) = _metric("entropy Ent(ν)"), _metric("relative entropy H(ν | μ)"), _metric("sum = log 6")
        v_sum.setText(f"{math.log(N):.3f}")
        for b in (b1, b2, b3):
            rl.addWidget(b)
        rl.addStretch(1); lay.addWidget(row)
        self.split = Split(); lay.addWidget(self.split)
        lay.addWidget(label(f"<span style='color:{theme.ACCENT}'>■</span> spread (entropy) &nbsp; "
                            f"<span style='color:{theme.ACCENT_2}'>■</span> distance from uniform. "
                            "The total never changes: whatever spread you lose, you gain as distance.", "small"))
        self.eff = label("", "body"); lay.addWidget(self.eff)
        self.set_weights([3, 2, 1, 1, .5, .5])

    def set_weights(self, w):
        s = sum(w); self.bars.set_values([x / s for x in w]); self.refresh()

    def normalise(self):
        w = self.bars.values
        if sum(w) <= 0:
            w = [1.0] * N
        self.set_weights(w)

    def refresh(self):
        p = self.bars.values
        ent = -sum(x * math.log(x) for x in p if x > 0)
        kl = max(0.0, math.log(N) - ent)
        self.v_ent.setText(f"{ent:.3f}"); self.v_kl.setText(f"{kl:.3f}")
        self.split.set_fraction(ent / math.log(N))
        self.eff.setText(f"Effective number of states e<sup>Ent(ν)</sup> = <b>{math.exp(ent):.2f}</b> of 6.")


class EnergyDemo(QWidget):
    """Energies H(σ) and inverse temperature β produce the Gibbs distribution μ_β ∝ e^{−βH}."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(8)
        lay.addWidget(label("<b>Try it.</b> Six configurations σ. Drag the orange bars to set their energies "
                            "H(σ); the blue bars show the Gibbs probabilities μ<sub>β</sub>(σ) = "
                            "e<sup>−βH(σ)</sup>/Z they produce.", "body"))
        lay.addWidget(_presets([("All equal", [0] * 6), ("One ground state", [-2, 0, 0, 0, 0, 0]),
                                ("Two ground states", [-2, -2, 0, 0, 1, 1]), ("Staircase", [-2.5, -1.5, -.5, .5, 1.5, 2.5])],
                               self.set_energies))
        lay.addWidget(label("energy H(σ): low = preferred", "small"))
        self.energy = Bars([0] * N, -3.0, 3.0, color=theme.ACCENT_2, fmt="{:+.1f}", height=140)
        self.energy.edited.connect(self.refresh)
        lay.addWidget(self.energy)
        self.beta = LabeledSlider("Inverse temperature β = 1/T", 0.0, 4.0, 1.0, 400,
                                  hint="β = 0: energy is ignored, all equally likely. Large β: only the lowest energy survives.")
        self.beta.changed.connect(lambda _: self.refresh())
        lay.addWidget(self.beta)
        lay.addWidget(label("Gibbs probability μ<sub>β</sub>(σ)", "small"))
        self.prob = Bars([1 / N] * N, 0.0, 1.0, editable=False, height=120)
        lay.addWidget(self.prob)
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(22)
        (b1, self.v_e), (b2, self.v_ent), (b3, self.v_lz) = (_metric("average energy E[H]"), _metric("entropy Ent(μ<sub>β</sub>)"),
                                                              _metric("log Z = −β E[H] + Ent"))
        for b in (b1, b2, b3):
            rl.addWidget(b)
        rl.addStretch(1); lay.addWidget(row)
        self.note = label("", "body"); lay.addWidget(self.note)
        self.set_energies([-2.5, -1.5, -.5, .5, 1.5, 2.5])

    def set_energies(self, h):
        self.energy.set_values(h); self.refresh()

    def refresh(self):
        b, h = self.beta.value(), self.energy.values
        lo = min(h)
        w = [math.exp(-b * (x - lo)) for x in h]
        z = sum(w); p = [x / z for x in w]
        mean = sum(pi * x for pi, x in zip(p, h))
        ent = -sum(pi * math.log(pi) for pi in p if pi > 0)
        log_z = -b * lo + math.log(z)
        self.prob.set_values(p)
        self.v_e.setText(f"{mean:+.2f}"); self.v_ent.setText(f"{ent:.2f}"); self.v_lz.setText(f"{log_z:.2f}")
        ground = sum(pi for pi, x in zip(p, h) if x - lo < 1e-9)
        self.note.setText(f"The lowest-energy configurations hold <b>{ground:.0%}</b> of the probability. "
                          "Only <i>differences</i> of energy matter: shifting all bars up by the same amount "
                          "changes nothing but log Z.")


DEMOS = {"entropy": EntropyDemo, "energy": EnergyDemo}
