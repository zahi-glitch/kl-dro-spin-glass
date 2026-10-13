"""Small simulations of three classical spin systems, used by the desktop app.

* Hopfield network: Hebbian couplings J = (1/N) sum_mu xi^mu xi^mu^T storing
  three 16x16 images; recall by heat-bath (Glauber) dynamics.
* Sherrington-Kirkpatrick (SK): Gaussian couplings J_ij ~ N(0, 1/N); two
  replicas with the same couplings, to watch frustration and overlaps.
* Random Energy Model (REM): 2^N independent energies E ~ N(0, N); exact Gibbs
  weights, entropy and free energy, freezing at beta_c = sqrt(2 log 2).
"""
from __future__ import annotations

import numpy as np

SIDE = 16
N_SPINS = SIDE * SIDE

# Three nearly orthogonal symbols (pairwise overlap ~0.25), so plain Hebbian learning recalls them.
_BITMAPS = {
    "Σ": [
        "................", "....#########...", "....#########...", "....####........",
        ".....####.......", "......####......", ".......####.....", "........####....",
        ".......####.....", "......####......", "......###.......", ".....###........",
        "....####........", "....##########..", "....##########..", "................",
    ],
    "β": [
        "....########....", "...####..###....", "...###...###....", "...###...###....",
        "..####..####....", "..####.####.....", "..####.######...", "..####...####...",
        "..####....####..", "..####....####..", "..#####..#####..", "..###########...",
        "..####.#####....", "..####..........", "..####..........", "..####..........",
    ],
    "∞": [
        "................", "................", "................", "....##....##....",
        "...####..####...", "..##.#########..", "..#...###...##..", "..#....##....#..",
        "..#...####..##..", "..############..", "...####..####...", "................",
        "................", "................", "................", "................",
    ],
}
PATTERN_NAMES = list(_BITMAPS)
PATTERNS = np.array([[1 if c == "#" else -1 for row in _BITMAPS[k] for c in row]
                     for k in PATTERN_NAMES], dtype=float)


def heat_bath(s: np.ndarray, J: np.ndarray, T: float, rng, n_updates: int) -> None:
    """In-place random-site Glauber dynamics: P(s_i = +1) = 1 / (1 + exp(-2 h_i / T))."""
    n = s.size
    for i in rng.integers(0, n, n_updates):
        h = J[i] @ s
        if T <= 1e-9:
            s[i] = 1.0 if h >= 0 else -1.0
        else:
            s[i] = 1.0 if rng.random() < 1.0 / (1.0 + np.exp(-2.0 * h / T)) else -1.0


class Hopfield:
    """Hebbian network storing the three symbols."""

    def __init__(self, patterns: np.ndarray = PATTERNS):
        self.xi = np.asarray(patterns, dtype=float)
        self.P, self.N = self.xi.shape
        self.J = self.xi.T @ self.xi / self.N
        np.fill_diagonal(self.J, 0.0)

    def corrupt(self, mu: int, noise: float, rng) -> np.ndarray:
        s = self.xi[mu].copy()
        s[rng.random(self.N) < noise] *= -1
        return s

    def overlaps(self, s: np.ndarray) -> np.ndarray:
        """Mattis magnetisations m_mu = (1/N) xi^mu . s, the order parameters of retrieval."""
        return self.xi @ s / self.N

    def energy(self, s: np.ndarray) -> float:
        return float(-0.5 * s @ self.J @ s / self.N)


class SK:
    """Sherrington-Kirkpatrick couplings with two replicas."""

    GROUND_STATE = -0.7633  # Parisi: lim E_min / N

    def __init__(self, n: int = N_SPINS, seed: int = 0):
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((n, n))
        self.J = np.triu(g, 1)
        self.J = (self.J + self.J.T) / np.sqrt(n)
        self.N = n

    def energy(self, s: np.ndarray) -> float:
        """E / N with E = -sum_{i<j} J_ij s_i s_j."""
        return float(-0.5 * s @ self.J @ s / self.N)

    @staticmethod
    def overlap(s1: np.ndarray, s2: np.ndarray) -> float:
        return float(s1 @ s2 / s1.size)


class REM:
    """Random Energy Model with 2^n independent energies E ~ N(0, n)."""

    def __init__(self, n: int = 16, seed: int = 0):
        self.n = n
        self.E = np.sqrt(n) * np.random.default_rng(seed).standard_normal(2 ** n)

    beta_c = float(np.sqrt(2 * np.log(2)))

    def weights(self, beta: float) -> np.ndarray:
        a = -beta * self.E
        w = np.exp(a - a.max())
        return w / w.sum()

    def entropy_per_spin(self, beta: float) -> float:
        w = self.weights(beta)
        w = w[w > 0]
        return float(-(w * np.log(w)).sum() / self.n)

    def free_energy(self, beta: float) -> float:
        """(1/n) log Z for this sample."""
        a = -beta * self.E
        return float((a.max() + np.log(np.exp(a - a.max()).sum())) / self.n)

    def effective_states(self, beta: float) -> float:
        """Participation number 1 / sum_alpha w_alpha^2."""
        w = self.weights(beta)
        return float(1.0 / (w * w).sum())

    @classmethod
    def entropy_theory(cls, beta: float) -> float:
        return max(0.0, np.log(2) - beta * beta / 2)

    @classmethod
    def free_energy_theory(cls, beta: float) -> float:
        return np.log(2) + beta * beta / 2 if beta <= cls.beta_c else beta * cls.beta_c


# --------------------------------------------------------------------------- Gibbs variational principle
# Ten states with fixed energies: think of ten configurations, or ten classes with logits -E.
ENERGIES = np.array([-2.1, -1.6, -1.2, -0.9, -0.4, 0.0, 0.3, 0.8, 1.3, 1.9])
LOGN = np.log(ENERGIES.size)


def boltzmann(t: float) -> np.ndarray:
    a = -t * ENERGIES
    w = np.exp(a - a.max())
    return w / w.sum()


def terms(q: np.ndarray, beta: float):
    """Energy term -beta E_q[E], entropy H(q), and their sum."""
    e = -beta * float(q @ ENERGIES)
    h = float(-(q[q > 0] * np.log(q[q > 0])).sum())
    return e, h, e + h


def log_Z(beta: float) -> float:
    a = -beta * ENERGIES
    return float(a.max() + np.log(np.exp(a - a.max()).sum()))


# --------------------------------------------------------------------------- Hopfield = RBM
class RBM:
    """Restricted Boltzmann machine with +-1 visible units and Gaussian hidden units.

    Joint weight  exp( -|z|^2 / 2 + sqrt(beta / N) * sum_{i,mu} xi_i^mu s_i z_mu ).
    Integrating out z (a Gaussian integral) leaves exp( (beta N / 2) sum_mu m_mu(s)^2 ),
    the Hopfield Gibbs weight at inverse temperature beta with the patterns xi as
    weights (Barra, Bernacchia, Santucci, Contucci 2012).
    """

    def __init__(self, patterns: np.ndarray = PATTERNS, beta: float = 3.0):
        self.xi = np.asarray(patterns, dtype=float)
        self.P, self.N = self.xi.shape
        self.beta = beta

    @property
    def scale(self) -> float:
        return np.sqrt(self.beta / self.N)

    def sample_hidden(self, s: np.ndarray, rng) -> np.ndarray:
        """z_mu | s  ~  N( sqrt(beta N) m_mu(s), 1 )."""
        return self.scale * (self.xi @ s) + rng.standard_normal(self.P)

    def sample_visible(self, z: np.ndarray, rng) -> np.ndarray:
        """s_i | z independent:  P(s_i = +1) = 1 / (1 + exp(-2 h_i)),  h_i = sqrt(beta/N) sum_mu xi_i^mu z_mu."""
        h = self.scale * (self.xi.T @ z)
        return np.where(rng.random(self.N) < 1.0 / (1.0 + np.exp(-2.0 * h)), 1.0, -1.0)

    def overlaps(self, s: np.ndarray) -> np.ndarray:
        return self.xi @ s / self.N

    def hopfield_log_weight(self, s: np.ndarray) -> float:
        """log of the Hopfield Gibbs weight exp((beta N / 2) sum_mu m_mu^2), up to a constant."""
        m = self.overlaps(s)
        return 0.5 * self.beta * self.N * float(m @ m)
