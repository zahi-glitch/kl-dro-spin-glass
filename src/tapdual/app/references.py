"""Further reading: at most three references per section, each with one line on why to read it.

Most entries come from the bibliographies of Kistler–Schmidt–Sebastiani (2023), Sebastiani–Schmidt (2024)
and Sebastiani's thesis; the Curie–Weiss and Hopfield entries add the standard sources those lists lack.
"""
from __future__ import annotations

from .widgets import Card, label

READING = {
    "statmech": [
        ("Y. Polyanskiy, Y. Wu", "Lecture Notes on Information Theory", "MIT, 2017/2019",
         "entropy, relative entropy and their variational formulas, from scratch."),
        ("M. D. Donsker, S. R. S. Varadhan",
         "Asymptotic evaluation of certain Markov process expectations for large time, I",
         "Comm. Pure Appl. Math. 28 (1975) 1–47",
         "the variational formula behind 'free energy = max of payoff minus relative entropy'."),
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
        ("H. Nishimori", "Statistical Physics of Spin Glasses and Information Processing: An Introduction",
         "Oxford University Press, 2001",
         "spin glasses and their use in memory, error correction and learning, at an introductory level."),
        ("A. Barra, A. Bernacchia, E. Santucci, P. Contucci",
         "On the equivalence of Hopfield networks and Boltzmann machines",
         "Neural Networks 34 (2012) 1–9",
         "the Hopfield = restricted Boltzmann machine duality shown in the second tab."),
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
