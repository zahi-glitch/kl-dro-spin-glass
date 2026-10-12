"""Expectations under mu = N(0,1) and under exponentially tilted measures.

Integrals against mu use probabilists' Gauss-Hermite quadrature, which is
exact for polynomials and converges fast for the smooth, at most linearly
growing exponents used here. All exponentials are evaluated with a
log-sum-exp shift, so no overflow occurs for large tilts.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .model import Model

DEFAULT_NODES = 300


@lru_cache(maxsize=8)
def gauss_hermite(n: int = DEFAULT_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalised weights with sum_i w_i g(x_i) ~ E_mu[g]."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


@dataclass(frozen=True)
class Tilt:
    """Summary of nu ~ exp(a phi^2 + b f) mu."""

    log_Z: float     # log E_mu exp(a phi^2 + b f)
    E_phi2: float    # E_nu[phi^2]
    E_f: float       # E_nu[f]
    H: float         # H(nu | mu)


class Quadrature:
    """Caches phi^2 and f on the quadrature nodes for one model."""

    def __init__(self, model: Model, n_nodes: int = DEFAULT_NODES):
        self.model = model
        self.x, self.w = gauss_hermite(n_nodes)
        self.phi2 = model.phi2(self.x)
        self.f = model.f(self.x)

    def tilt(self, a: float, b: float) -> Tilt:
        """Statistics of the tilted measure nu ~ exp(a phi^2 + b f) mu."""
        expo = a * self.phi2 + b * self.f
        shift = expo.max()
        weights = self.w * np.exp(expo - shift)
        Z = weights.sum()
        weights /= Z
        log_Z = shift + np.log(Z)
        E_phi2 = float(weights @ self.phi2)
        E_f = float(weights @ self.f)
        H = a * E_phi2 + b * E_f - log_Z  # Gibbs identity
        return Tilt(float(log_Z), E_phi2, E_f, float(max(H, 0.0)))

    def gibbs(self, q: float, m: float) -> Tilt:
        """Generalised Gibbs measure nu^{q,m} ~ exp(m [Phi'(q) phi^2 + f]) mu."""
        return self.tilt(m * self.model.dPhi(q), m)
