"""Main window: the sections in the sidebar on the left, the current section on the right."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QListWidget, QListWidgetItem, QMainWindow, QScrollArea, QStackedWidget,
                               QVBoxLayout, QWidget)

from .s1_statmech import StatMechSection
from .s2_curie_weiss import CurieWeissSection
from .s3_disorder import DisorderSection
from .s4_hopfield import HopfieldSection
from .s5_case_study import CaseStudySection
from .s5_live import LiveSection
from .s5_robust import RobustSection
from .widgets import GlossaryDialog, label, set_glossary_opener

SECTIONS = [
    ("1", "What is a statistical-mechanics system?", StatMechSection),
    ("2", "Order: Curie–Weiss", CurieWeissSection),
    ("3", "Disorder: spin glasses", DisorderSection),
    ("4", "Memory: Hopfield networks", HopfieldSection),
    ("5", "Robust decisions", RobustSection),
    ("5.1", "Case study: real data", CaseStudySection),
    ("5.2", "Live: a virtual $10,000", LiveSection),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Free energy and robust optimisation")
        self.resize(1360, 880)

        side = QWidget(); side.setFixedWidth(290)
        sl = QVBoxLayout(side); sl.setContentsMargins(16, 24, 8, 16); sl.setSpacing(6)
        sl.addWidget(label("STATISTICAL MECHANICS ↔ ROBUST OPTIMISATION", "eyebrow"))
        sl.addWidget(label("From crowds and memories to worst-case decisions", "h2"))
        self.nav = QListWidget(); self.nav.setObjectName("nav")
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.nav.setWordWrap(True)
        for number, title, _ in SECTIONS:
            sub = "." in number
            item = QListWidgetItem(f"      {number}   {title}" if sub else f"{number}   {title}")
            if sub:
                f = item.font(); f.setPixelSize(13); item.setFont(f)
            self.nav.addItem(item)
        sl.addWidget(self.nav, 1)
        sl.addWidget(label("Blue words open a short explanation with the formulas. "
                           "All symbols: [[notation|notation table]].", "small"))
        sl.addWidget(label("Further reading: at the end of each section.", "small"))

        self.stack = QStackedWidget()
        for _, _, cls in SECTIONS:
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(cls())
            self.stack.addWidget(scroll)

        central = QWidget(); cl = QHBoxLayout(central); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        cl.addWidget(side); cl.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.glossary = GlossaryDialog(self)
        set_glossary_opener(self.glossary.open)
        self.nav.setCurrentRow(0)
