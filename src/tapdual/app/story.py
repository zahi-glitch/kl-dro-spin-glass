"""The story path:

1 What is a statistical-mechanics system?   (ingredients, energy and entropy, free energy)
2 Order: Curie–Weiss                        (sections.CurieWeissSection)
3 Disorder: spin glasses                    (SK frustration, REM freezing)
4 Memory: Hopfield networks                 (sections.HopfieldSection)
5 Robust decisions                          (next)
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QStackedWidget, QWidget

from .chapter1 import REMPage, SKPage, _column, _two_columns
from .chapter2 import Chapter2
from .demos import EnergyDemo, EntropyDemo
from .references import reading_card
from .sections import Section
from .widgets import Card, FitStack, Segmented, label


def _rows(pairs, tag_width=150) -> QWidget:
    w = QWidget(); g = QGridLayout(w); g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(14); g.setVerticalSpacing(12)
    for r, (head, text) in enumerate(pairs):
        h = label(head, "h2"); h.setFixedWidth(tag_width)
        g.addWidget(h, r, 0, Qt.AlignTop); g.addWidget(label(text, "body"), r, 1)
    g.setColumnStretch(1, 1)
    return w


class StatMechSection(Section):
    def __init__(self):
        super().__init__(
            "1", "What is a statistical-mechanics system?",
            "You cannot predict what one voter will do, but you can predict an election. Statistical mechanics is "
            "the art of predicting the crowd without tracking the individuals.")
        self.seg = Segmented(["The ingredients", "Energy against entropy"])
        self.lay.addWidget(self.seg)
        self.stack = FitStack()
        self.stack.addWidget(self._ingredients())
        self.stack.addWidget(Chapter2(header=False))
        self.lay.addWidget(self.stack)
        self.lay.addWidget(reading_card("statmech"))
        self.lay.addStretch(1)
        self.stack.setCurrentIndex(0)
        self.seg.changed.connect(self.stack.setCurrentIndex)

    def _ingredients(self) -> QWidget:
        recipe = Card("Three ingredients")
        recipe.add(_rows([
            ("1 · Many simple parts",
             "N variables σ<sub>i</sub> = +1 or −1, called [[spin|spins]]. A spin can be a tiny magnet (up or down), a "
             "neuron (firing or silent), a pixel (black or white), a person (yes or no). A <i>configuration</i> "
             "σ = (σ<sub>1</sub>, …, σ<sub>N</sub>) lists them all: 2<sup>N</sup> possibilities. With N = 300 that is "
             "more than the number of atoms in the visible universe, so nobody can list them."),
            ("2 · A score",
             "The [[energy|energy]] H(σ) scores each configuration: <b>low energy = preferred</b>. It encodes who "
             "wants to agree with whom. Think of it as a landscape over all configurations, with valleys where the "
             "system likes to be."),
            ("3 · Noise",
             "The [[temperature|temperature]] T, or its inverse β = 1/T. At T = 0 the system only rolls downhill. "
             "At high T it is shaken so hard that it ignores the landscape."),
        ]))
        result = Card("What comes out")
        result.add(_rows([
            ("Probabilities, not paths",
             "We give up on predicting the exact configuration and ask how likely each one is: the "
             "[[gibbs|Gibbs distribution]] μ<sub>β</sub>(σ) = e<sup>−βH(σ)</sup>/Z. Low energy is more likely, "
             "and β decides how much more."),
            ("A few summary numbers",
             "For large N, a handful of averages become predictable even though single spins are not, like the "
             "share of 'yes' in an election. These are the [[order-parameter|order parameters]]; the "
             "[[magnetisation]] m = (1/N) Σ σ<sub>i</sub> is the first one."),
            ("Phase transitions",
             "When β crosses a critical value, the summary numbers can jump: a magnet magnetises, a memory "
             "becomes retrievable, a glass freezes. Finding these points is the main game."),
        ], tag_width=170))

        energy = Card("Play: from energy to probability")
        energy.add(EnergyDemo())
        entropy = Card("Play: entropy, the cost of being certain")
        entropy.add(EntropyDemo())

        story = Card("The path through this app")
        story.add(_rows([
            ("2 · Order", "Curie–Weiss: everyone wants to agree. A crowd can make up its mind on its own."),
            ("3 · Disorder", "Spin glasses (SK, REM): random rules that contradict each other. Many valleys, no "
                             "single answer."),
            ("4 · Memory", "Hopfield networks: choose the rules on purpose so that the valleys are memories. "
                           "Storage and retrieval of information."),
            ("5 · Decisions", "The same mathematics as robust optimisation: decisions that survive when the data "
                              "are wrong."),
        ], tag_width=110))
        return _two_columns(_column(recipe, result, story), _column(energy, entropy))


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
