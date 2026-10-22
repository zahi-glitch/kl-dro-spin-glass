"""Layout helpers shared by the sections: columns, the section header with its story bridge, small tables."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget)

from . import theme
from .widgets import label


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


def bridge(previously: str, now: str, question: str) -> QWidget:
    """The thread of the story: what we saw, what changes now, and the question this section answers."""
    box = QFrame(); box.setObjectName("bridge")
    lay = QGridLayout(box); lay.setContentsMargins(16, 12, 16, 12); lay.setHorizontalSpacing(14); lay.setVerticalSpacing(4)
    for r, (tag, text) in enumerate((("SO FAR", previously), ("NOW", now), ("QUESTION", question))):
        t = label(tag, "tag"); t.setFixedWidth(78)
        lay.addWidget(t, r, 0, Qt.AlignTop)
        lay.addWidget(label(f"<b>{text}</b>" if tag == "QUESTION" else text, "body"), r, 1)
    lay.setColumnStretch(1, 1)
    return box


def section_header(lay, number: str, title: str, lead: str, story=None) -> None:
    lay.addWidget(label(f"SECTION {number}", "eyebrow"))
    lay.addWidget(label(title, "title"))
    if story:
        lay.addWidget(bridge(*story))
    lay.addWidget(label(lead, "lead"))


class Section(QWidget):
    def __init__(self, number, title, lead, story=None):
        super().__init__()
        self.setObjectName("page")
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(28, 24, 28, 24); self.lay.setSpacing(12)
        section_header(self.lay, number, title, lead, story)


def _rows(pairs, tag_width=150) -> QWidget:
    w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(14); g.setVerticalSpacing(12)
    for r, (head, text) in enumerate(pairs):
        h = label(head, "h2"); h.setFixedWidth(tag_width)
        g.addWidget(h, r, 0, Qt.AlignTop); g.addWidget(label(text, "body"), r, 1)
    g.setColumnStretch(1, 1)
    return w


def _metrics(pairs) -> tuple[QWidget, list]:
    row = QWidget(); g = QGridLayout(row); g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(22); g.setVerticalSpacing(2)
    vals = []
    for c, title in enumerate(pairs):
        v = label("", "value")
        g.addWidget(label(title, "small", wrap=False), 0, c); g.addWidget(v, 1, c); vals.append(v)
    g.setColumnStretch(len(pairs), 1)
    return row, vals


def _hist(values, weights, edges):
    idx = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    return np.bincount(idx, weights=weights, minlength=len(edges) - 1)


class Picture(QWidget):
    """An image that scales to the available width (up to max_width), keeps its aspect ratio, centred."""

    def __init__(self, path, max_width=900):
        super().__init__()
        self.pm, self.max_width = QPixmap(str(path)), max_width
        sp = self.sizePolicy(); sp.setHeightForWidth(True); self.setSizePolicy(sp)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        w = min(w, self.max_width)
        return int(w * self.pm.height() / max(self.pm.width(), 1))

    def sizeHint(self):
        return QSize(600, self.heightForWidth(600))

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.SmoothPixmapTransform)
        w = min(self.width(), self.max_width); h = self.heightForWidth(w)
        p.drawPixmap(QRect((self.width() - w) // 2, 0, w, h), self.pm); p.end()


def _big(value, caption) -> QWidget:
    w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(0)
    v = label(value, "value"); v.setStyleSheet(f"font-size: 30px; color: {theme.ACCENT};")
    l.addWidget(v); l.addWidget(label(caption, "small"))
    return w


def _heading(number: str, text: str) -> QWidget:
    h = label(f"<span style='color:{theme.ACCENT};'>{number}</span>&nbsp;&nbsp;{text}", "title")
    h.setStyleSheet("font-size: 20px; padding-top: 20px; padding-bottom: 2px;")
    return h


def _table(headers, rows, bold_row=None, first_width=None, text=False) -> QWidget:
    """A small table as one rich-text label (wraps correctly); numbers right-aligned unless text=True."""
    pad = "padding: 5px 24px 5px 0px;"
    first = f"width: {first_width}px;" if first_width else ""
    html = ["<table cellspacing='0' cellpadding='0' style='margin: 2px 0px;'>"]
    if any(headers):
        html.append("<tr>" + "".join(
            f"<td style='{pad}{first if c == 0 else ''} color: {theme.INK_3}; font-size: 12px;' "
            f"align='{'left' if c == 0 or text else 'right'}'><b>{h}</b></td>" for c, h in enumerate(headers)) + "</tr>")
    for r, row in enumerate(rows):
        cells = []
        for c, cell in enumerate(row):
            txt = f"<b>{cell}</b>" if r == bold_row else str(cell)
            cells.append(f"<td style='{pad}{first if c == 0 else ''}' valign='top' "
                         f"align='{'left' if c == 0 or text else 'right'}'>{txt}</td>")
        html.append("<tr>" + "".join(cells) + "</tr>")
    html.append("</table>")
    return label("".join(html), "body")


def _gap(h=6) -> QWidget:
    w = QWidget(); w.setFixedHeight(h)
    return w


class _SubSection(Section):
    def __init__(self, number, title, lead, page):
        super().__init__(number, title, lead)
        self.lay.addWidget(page)
        self.lay.addStretch(1)

