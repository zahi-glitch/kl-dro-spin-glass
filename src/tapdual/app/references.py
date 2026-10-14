"""Further reading: at most three references per section, each with one line on why to read it.

Entries are taken from the bibliography of the introduction to Sebastiani's thesis (2024), except
Bolthausen–Sznitman (cited in KSS23) and Brock–Durlauf (the source of the Curie–Weiss market example).
"""
from __future__ import annotations

from .widgets import Card, label

READING = {
    "statmech": [
        ("A. Bovier", "Statistical Mechanics of Disordered Systems: A Mathematical Perspective",
         "Cambridge Series in Statistical and Probabilistic Mathematics, Cambridge University Press, 2006",
         "the whole path of this app in one monograph: Gibbs measures, Curie–Weiss, the REM and GREM, SK and the "
         "Hopfield model."),
        ("S. Friedli, Y. Velenik", "Statistical Mechanics of Lattice Systems: A Concrete Mathematical Introduction",
         "Cambridge University Press, 2017, Chapter 1",
         "an intuitive first contact with the subject."),
    ],
    "curie-weiss": [
        ("S. Friedli, Y. Velenik", "Statistical Mechanics of Lattice Systems: A Concrete Mathematical Introduction",
         "Cambridge University Press, 2017, Chapter 2",
         "the Curie–Weiss model done carefully: free energy, phase transition, large deviations."),
        ("W. A. Brock, S. N. Durlauf", "Discrete choice with social interactions",
         "Rev. Econ. Stud. 68 (2001) 235–260",
         "the same equations as a model of peer effects and social tipping."),
    ],
    "disorder": [
        ("D. Sherrington, S. Kirkpatrick", "Solvable model of a spin-glass",
         "Phys. Rev. Lett. 35 (1975) 1792",
         "where the SK model starts."),
        ("B. Derrida", "Random-energy model: an exactly solvable model of disordered systems",
         "Phys. Rev. B 24 (1981) 2613",
         "the REM, freezing at β<sub>c</sub> = √(2 log 2), in a few pages."),
        ("E. Bolthausen, A.-S. Sznitman", "Ten Lectures on Random Media",
         "DMV Seminar 32, Birkhäuser, 2002",
         "a mathematical tour of disordered systems, including REM-type models and Ruelle cascades."),
    ],
    "hopfield": [
        ("J. J. Hopfield", "Neural networks and physical systems with emergent collective computational abilities",
         "PNAS 79 (1982) 2554–2558",
         "the original associative memory."),
        ("D. J. Amit, H. Gutfreund, H. Sompolinsky", "Spin-glass models of neural networks",
         "Phys. Rev. A 32 (1985) 1007",
         "the storage capacity α<sub>c</sub> ≈ 0.138 and the phase diagram."),
        ("E. Agliari, L. Albanese, A. Barra, G. Ottaviani",
         "Replica symmetry breaking in neural networks: a few steps toward rigorous results",
         "J. Phys. A 53 (2020) 415005",
         "where the mathematics of the loaded Hopfield model stands."),
    ],
    "robust": [
        ("N. Kistler, M. A. Schmidt, G. Sebastiani", "On the REM approximation of TAP free energies",
         "J. Phys. A 56 (2023) 294001",
         "the Boltzmann–Gibbs sup with an entropy budget and its Parisi-type dual."),
        ("G. Sebastiani, M. A. Schmidt", "On the GREM approximation of TAP free energies",
         "J. Phys. A 57 (2024) 335206",
         "the same duality for hierarchical (GREM) models."),
    ],
}


def reading_card(key: str) -> Card:
    card = Card("Further reading")
    items = "".join(
        f"<p style='margin:0 0 8px 0;'>{authors}. <i>{title}</i>. {venue}.<br>"
        f"<span style='color:#8a8983;'>Read it for {why}</span></p>"
        for authors, title, venue, why in READING[key])
    card.add(label(items, "body"))
    return card
