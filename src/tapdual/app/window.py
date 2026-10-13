"""Main window: chapter list on the left, the current chapter on the right."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from . import theme
from .chapter2 import Chapter2
from .sections import CurieWeissSection, HopfieldSection, REMSection
from .widgets import GlossaryDialog, label, set_glossary_opener

CHAPTERS = [
    ("1", "Free energy is an optimisation", Chapter2),
    ("2", "Curie–Weiss: everyone agrees", CurieWeissSection),
    ("3", "Hopfield: memories", HopfieldSection),
    ("4", "REM: extreme disorder", REMSection),
    ("5", "Robust optimisation", None),
]


class Placeholder(QWidget):
    def __init__(self, number, title):
        super().__init__()
        self.setObjectName("page")
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 24)
        lay.addWidget(label(f"CHAPTER {number}", "eyebrow")); lay.addWidget(label(title, "title"))
        lay.addWidget(label("Coming next.", "lead")); lay.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Free energy and robust optimisation")
        self.resize(1360, 880)

        side = QWidget(); side.setFixedWidth(290)
        sl = QVBoxLayout(side); sl.setContentsMargins(16, 24, 8, 16); sl.setSpacing(6)
        sl.addWidget(label("FREE ENERGY ↔ ROBUST OPTIMISATION", "eyebrow"))
        sl.addWidget(label("From spin glasses to worst-case decisions", "h2"))
        self.nav = QListWidget(); self.nav.setObjectName("nav")
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.nav.setWordWrap(True)
        for number, title, cls in CHAPTERS:
            item = QListWidgetItem(f"{number}   {title}")
            if cls is None:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.nav.addItem(item)
        sl.addWidget(self.nav, 1)
        sl.addWidget(label("Blue words open a short explanation with the formulas.", "small"))
        sl.addWidget(label("Based on Kistler, Schmidt, Sebastiani (2023) and Sebastiani, Schmidt (2024).", "small"))

        self.stack = QStackedWidget()
        for number, title, cls in CHAPTERS:
            page = cls() if cls else Placeholder(number, title)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
            self.stack.addWidget(scroll)

        central = QWidget(); cl = QHBoxLayout(central); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        cl.addWidget(side); cl.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.glossary = GlossaryDialog(self)
        set_glossary_opener(self.glossary.open)
        self.nav.setCurrentRow(0)
