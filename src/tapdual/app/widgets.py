"""Small reusable widgets: cards, spin grids, overlap bars, sliders, 'go deeper' boxes."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QStackedWidget, QVBoxLayout, QWidget)

from . import theme


_GLOSSARY_OPENER = None


def set_glossary_opener(fn) -> None:
    """Register the function that opens a glossary entry (called with its key)."""
    global _GLOSSARY_OPENER
    _GLOSSARY_OPENER = fn


def _on_link(href: str) -> None:
    if href.startswith("gloss:") and _GLOSSARY_OPENER is not None:
        _GLOSSARY_OPENER(href[6:])


def label(text: str, kind: str = "body", wrap: bool = True) -> QLabel:
    """Rich-text label. [[term|shown]] becomes a link to the glossary entry `term`."""
    from .glossary import linkify
    lab = QLabel(linkify(text, theme.ACCENT))
    lab.setObjectName(kind)
    lab.setWordWrap(wrap)
    lab.setTextFormat(Qt.RichText)
    if "gloss:" in lab.text():
        lab.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        lab.linkActivated.connect(_on_link)
        lab.setCursor(Qt.ArrowCursor)
    return lab


def set_label_text(lab: QLabel, text: str) -> None:
    from .glossary import linkify
    lab.setText(linkify(text, theme.ACCENT))


class Card(QFrame):
    def __init__(self, title: str = "", subtitle: str = ""):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(18, 16, 18, 16)
        self.lay.setSpacing(8)
        if title:
            self.lay.addWidget(label(title, "h2"))
        if subtitle:
            self.lay.addWidget(label(subtitle, "body"))

    def add(self, w, stretch: int = 0):
        self.lay.addWidget(w, stretch)
        return w


class Practice(Card):
    """'In practice' card as a question and a structured answer.

    rows: list of (label, html or widget) pairs, e.g. ("Setting", ...), ("Try it", ...), ("Live", a QLabel).
    """

    def __init__(self, question: str, rows=None, body: str = ""):
        super().__init__()
        self.setObjectName("practice")
        self.lay.addWidget(label("IN PRACTICE", "tag"))
        self.lay.addWidget(label(question, "h2"))
        if body:
            self.lay.addWidget(label(body, "body"))
        if rows:
            from PySide6.QtWidgets import QGridLayout
            grid = QWidget(); g = QGridLayout(grid); g.setContentsMargins(0, 2, 0, 0)
            g.setHorizontalSpacing(12); g.setVerticalSpacing(8)
            for r, (name, html) in enumerate(rows):
                key = label(f"<b>{name}</b>", "small"); key.setAlignment(Qt.AlignTop | Qt.AlignLeft)
                key.setFixedWidth(96)
                g.addWidget(key, r, 0)
                g.addWidget(html if isinstance(html, QWidget) else label(html, "body"), r, 1)
            g.setColumnStretch(1, 1)
            self.lay.addWidget(grid)


class SpinGrid(QWidget):
    """A square grid of +-1 spins, painted as crisp cells."""
    clicked = Signal()

    def __init__(self, side: int = 16, cell: int = 18, selectable: bool = False):
        super().__init__()
        self.side, self.state, self.selected, self.selectable = side, np.ones(side * side), False, selectable
        self.setMinimumSize(side * 6, side * 6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pref = side * cell
        if selectable:
            self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(self._pref, self._pref)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return w

    def set_state(self, s):
        self.state = np.asarray(s)
        self.update()

    def set_selected(self, on: bool):
        self.selected = on
        self.update()

    def mousePressEvent(self, e):
        if self.selectable:
            self.clicked.emit()

    def paintEvent(self, e):
        p = QPainter(self)
        size = min(self.width(), self.height())
        x0, y0 = (self.width() - size) / 2, (self.height() - size) / 2
        c = size / self.side
        up, down = QColor(theme.SPIN_UP), QColor(theme.SPIN_DOWN)
        p.setPen(Qt.NoPen)
        for k, v in enumerate(self.state):
            i, j = divmod(k, self.side)
            p.setBrush(up if v > 0 else down)
            p.drawRect(QRectF(x0 + j * c + 0.5, y0 + i * c + 0.5, c - 1, c - 1))
        if self.selected:
            pen = p.pen()
            p.setPen(QColor(theme.ACCENT))
            p.setBrush(Qt.NoBrush)
            from PySide6.QtGui import QPen
            p.setPen(QPen(QColor(theme.ACCENT), 3))
            p.drawRoundedRect(QRectF(x0 + 1.5, y0 + 1.5, size - 3, size - 3), 4, 4)
        p.end()


class OverlapBars(QWidget):
    """Horizontal bars for the overlaps m_mu in [-1, 1], one row per pattern."""

    def __init__(self, names):
        super().__init__()
        self.names, self.values, self.target = list(names), np.zeros(len(names)), 0
        self.setMinimumHeight(34 * len(names) + 8)

    def set_values(self, values, target: int):
        self.values, self.target = np.asarray(values), target
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W = self.width()
        lab_w, val_w = 34, 56
        x0, x1 = lab_w, W - val_w
        mid = (x0 + x1) / 2
        f = QFont(self.font()); f.setPixelSize(15)
        for r, (name, v) in enumerate(zip(self.names, self.values)):
            y = 6 + r * 34
            p.setPen(QColor(theme.INK if r == self.target else theme.INK_2))
            p.setFont(f)
            p.drawText(QRectF(0, y, lab_w - 8, 22), Qt.AlignVCenter | Qt.AlignRight, name)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.GRID))
            p.drawRoundedRect(QRectF(x0, y + 7, x1 - x0, 8), 4, 4)
            p.setBrush(QColor(theme.ACCENT if r == self.target else theme.INK_3))
            a, b = sorted((mid, mid + v * (x1 - x0) / 2))
            p.drawRoundedRect(QRectF(a, y + 5, max(b - a, 2), 12), 4, 4)
            p.setPen(QColor(theme.INK_3))
            p.drawLine(int(mid), y + 2, int(mid), y + 20)
            p.setPen(QColor(theme.INK))
            p.drawText(QRectF(x1 + 6, y, val_w - 6, 22), Qt.AlignVCenter | Qt.AlignLeft, f"{v:+.2f}")
        p.end()


class LabeledSlider(QWidget):
    """Slider with a name, a live value and an optional hint; maps an int range to floats."""
    changed = Signal(float)

    def __init__(self, name: str, lo: float, hi: float, value: float, steps: int = 200,
                 fmt=lambda v: f"{v:.2f}", hint: str = ""):
        super().__init__()
        self.lo, self.hi, self.steps, self.fmt = lo, hi, steps, fmt
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
        top = QHBoxLayout()
        self.name = label(f"<b>{name}</b>", "body", wrap=False)
        self.val = label("", "body", wrap=False)
        top.addWidget(self.name); top.addStretch(1); top.addWidget(self.val)
        lay.addLayout(top)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, steps)
        lay.addWidget(self.slider)
        if hint:
            lay.addWidget(label(hint, "small"))
        self.slider.valueChanged.connect(self._emit)
        self.set_value(value)

    def value(self) -> float:
        return self.lo + (self.hi - self.lo) * self.slider.value() / self.steps

    def set_value(self, v: float):
        self.slider.setValue(round((v - self.lo) / (self.hi - self.lo) * self.steps))
        self._emit()

    def _emit(self, *_):
        self.val.setText(self.fmt(self.value()))
        self.changed.emit(self.value())


class Deeper(QWidget):
    """A collapsible 'Go deeper' box for the mathematics."""

    def __init__(self, html: str, title: str = "Go deeper"):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
        self.btn = QPushButton(f"▸  {title}"); self.btn.setObjectName("deeper")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.body = label(html, "body"); self.body.setVisible(False)
        self.title = title
        lay.addWidget(self.btn); lay.addWidget(self.body)
        self.btn.clicked.connect(self.toggle)

    def toggle(self):
        on = not self.body.isVisible()
        self.body.setVisible(on)
        self.btn.setText(f"{'▾' if on else '▸'}  {self.title}")


class Segmented(QWidget):
    """Row of pill buttons; exactly one is checked."""
    changed = Signal(int)

    def __init__(self, names):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        self.buttons = []
        for i, n in enumerate(names):
            b = QPushButton(n); b.setObjectName("seg"); b.setCheckable(True); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=i: self.select(k))
            lay.addWidget(b); self.buttons.append(b)
        lay.addStretch(1)
        self.select(0)

    def select(self, k: int):
        for i, b in enumerate(self.buttons):
            b.setChecked(i == k)
        self.changed.emit(k)


class FrustrationTriangle(QWidget):
    """Three spins with couplings of random sign: '=' (want to agree) or '≠' (want to disagree).

    A link is satisfied when J_ij σ_i σ_j > 0. The loop is frustrated when the product of the three signs is
    negative (Toulouse): then no configuration satisfies all three links. Click a spin to flip it.
    """
    EDGES = ((0, 1), (1, 2), (0, 2))

    def __init__(self):
        super().__init__()
        self.s, self.J = [1, 1, 1], [1, 1, -1]
        self.rng = np.random.default_rng()
        self.setFixedHeight(214); self.setCursor(Qt.PointingHandCursor)

    def randomize(self):
        self.J = [int(x) for x in self.rng.choice([-1, 1], 3)]
        self.update()

    def frustrated(self) -> bool:
        return self.J[0] * self.J[1] * self.J[2] < 0

    def _nodes(self):
        W, H = self.width(), self.height()
        cx, r = W / 2, min(W, H) * 0.34
        return [QPointF(cx, 22), QPointF(cx - r * 1.15, H - 70), QPointF(cx + r * 1.15, H - 70)]

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        pts = self._nodes()
        ok = 0
        for (i, j), J in zip(self.EDGES, self.J):
            good = J * self.s[i] * self.s[j] > 0
            ok += good
            col = QColor(theme.ACCENT if J > 0 else theme.ACCENT_2)
            p.setPen(QPen(col, 3 if good else 2, Qt.SolidLine if good else Qt.DashLine)); p.drawLine(pts[i], pts[j])
            mid = (pts[i] + pts[j]) / 2
            p.setPen(QPen(col, 1.5)); p.setBrush(QColor(theme.CARD)); p.drawEllipse(mid, 12, 12)
            f = QFont(self.font()); f.setPixelSize(15); f.setBold(True); p.setFont(f)
            p.drawText(QRectF(mid.x() - 12, mid.y() - 12, 24, 24), Qt.AlignCenter, "=" if J > 0 else "≠")
        f = QFont(self.font()); f.setPixelSize(15); f.setBold(True); p.setFont(f)
        for q, v in zip(pts, self.s):
            p.setPen(QPen(QColor(theme.INK_3), 1)); p.setBrush(QColor(theme.SPIN_UP if v > 0 else theme.SPIN_DOWN))
            p.drawEllipse(q, 17, 17)
            p.setPen(QColor("white" if v > 0 else theme.INK))
            p.drawText(QRectF(q.x() - 17, q.y() - 17, 34, 34), Qt.AlignCenter, "+" if v > 0 else "−")
        f.setPixelSize(13); f.setBold(False); p.setFont(f); p.setPen(QColor(theme.INK_2))
        best = 2 if self.frustrated() else 3
        p.drawText(QRectF(0, self.height() - 44, self.width(), 20), Qt.AlignHCenter,
                   f"satisfied links (solid): {ok} of 3" + ("  ·  the best possible" if ok == best else ""))
        p.setPen(QColor(theme.ACCENT_2 if self.frustrated() else theme.ACCENT))
        sign = "−" if self.frustrated() else "+"
        p.drawText(QRectF(0, self.height() - 22, self.width(), 20), Qt.AlignHCenter,
                   f"product of the signs around the loop: {sign}1  →  "
                   + ("frustrated: at most 2 of 3" if self.frustrated() else "not frustrated: all 3 can be satisfied"))
        p.end()

    def mousePressEvent(self, e):
        for k, q in enumerate(self._nodes()):
            if (e.position() - q).manhattanLength() < 30:
                self.s[k] *= -1; self.update(); return


class FitStack(QStackedWidget):
    """A stacked widget whose height is that of the current page, not of the tallest one."""

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w else super().minimumSizeHint()

    def setCurrentIndex(self, i):
        super().setCurrentIndex(i)
        for k in range(self.count()):
            self.widget(k).setSizePolicy(QSizePolicy.Preferred,
                                         QSizePolicy.Preferred if k == i else QSizePolicy.Ignored)
        self.updateGeometry()


def hline() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setStyleSheet(f"color: {theme.BORDER};")
    return f


class GlossaryDialog:
    """A small window explaining one term, with typeset formulas."""

    def __init__(self, parent=None):
        from PySide6.QtWidgets import QDialog, QScrollArea
        self.dlg = QDialog(parent)
        self.dlg.setWindowTitle("Glossary")
        self.dlg.resize(640, 720)
        self.dlg.setStyleSheet(f"QDialog {{ background: {theme.CARD}; }}")
        outer = QVBoxLayout(self.dlg); outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(f"QScrollArea {{ background: {theme.CARD}; border: none; }}")
        outer.addWidget(self.scroll)
        self.history = []

    def open(self, key: str):
        from PySide6.QtGui import QPixmap
        from .glossary import ENTRIES, formula_png
        if key not in ENTRIES:
            return
        title, blocks = ENTRIES[key]
        body = QWidget(); body.setStyleSheet(f"background: {theme.CARD};")
        lay = QVBoxLayout(body); lay.setContentsMargins(24, 20, 24, 20); lay.setSpacing(10)
        lay.addWidget(label("GLOSSARY", "eyebrow")); lay.addWidget(label(title, "title"))
        for kind, content in blocks:
            if kind == "p":
                lay.addWidget(label(content, "body"))
            elif kind == "demo":
                from .demos import DEMOS
                lay.addWidget(DEMOS[content]())
            else:
                pm = QPixmap(); pm.loadFromData(formula_png(content)); pm.setDevicePixelRatio(220 / 96)
                img = QLabel(); img.setPixmap(pm); img.setAlignment(Qt.AlignHCenter)
                img.setStyleSheet("padding: 6px 0;")
                lay.addWidget(img)
        lay.addStretch(1)
        self.scroll.setWidget(body)
        self.dlg.setWindowTitle(f"Glossary · {title}")
        self.dlg.show(); self.dlg.raise_(); self.dlg.activateWindow()
