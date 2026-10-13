"""Expectations under mu = N(0,1) and under exponentially tilted measures.

Integrals against mu use the trapezoid rule on a uniform grid of [-L, L]
with the Gaussian density folded into the weights. For integrands that are
analytic in a strip, as here (tanh(h + beta x) has its nearest poles at
distance pi/(2 beta) from the real axis), the trapezoid rule converges
geometrically in the number of points, and it does not suffer from the
loss of accuracy of Gauss-Hermite quadrature when tanh(beta x) varies on
scales finer than the Hermite node spacing (large beta).

All exponentials are evaluated with a log-sum-exp shift, so no overflow
occurs for large tilts. The same grid is used by the dashboard (docs/solver.js).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .model import Model

DEFAULT_HALF_WIDTH = 14.0
DEFAULT_POINTS = 2801


@lru_cache(maxsize=8)
def gaussian_grid(half_width: float = DEFAULT_HALF_WIDTH,
                  n_points: int = DEFAULT_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Points x_i and weights w_i with sum_i w_i g(x_i) ~ E_mu[g], mu = N(0,1)."""
    x = np.linspace(-half_width, half_width, n_points)
    dx = x[1] - x[0]
    w = np.full(n_points, dx)
    w[[0, -1]] *= 0.5
    w *= np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return x, w


@dataclass(frozen=True)
class Tilt:
    """Summary of nu ~ exp(a phi^2 + b f) mu."""

    log_Z: float     # log E_mu exp(a phi^2 + b f)
    E_phi2: float    # E_nu[phi^2]
    E_f: float       # E_nu[f]
    H: float         # H(nu | mu)


class Quadrature:
    """Caches phi^2 and f on the integration grid for one model."""

    def __init__(self, model: Model, n_points: int = DEFAULT_POINTS,
                 half_width: float = DEFAULT_HALF_WIDTH):
        self.model = model
        self.x, self.w = gaussian_grid(half_width, n_points)
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
