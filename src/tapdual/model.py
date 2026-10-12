"""Model specification.

The single-site model is described by three functions and a reference measure:

    phi : R -> [-1, 1]   enters the non-linearity through phi(x)^2
    f   : R -> R         linear part (continuous, bounded below, all
                         exponential moments finite under mu)
    Phi : R -> R         concave non-linearity, Phi'' < 0 on [0, 1]

and mu = N(0, 1), the reference ("nominal") measure.

The objective over probability measures nu is

    Omega(nu) = Phi( E_nu[phi^2] ) + E_nu[f] - H(nu | mu),

maximised under the KL budget H(nu | mu) <= rho (rho = log 2 in the REM).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Model:
    """A single-site model: phi, f, Phi (and Phi') with mu = N(0, 1)."""

    phi: Callable[[Array], Array]
    f: Callable[[Array], Array]
    Phi: Callable[[float], float]
    dPhi: Callable[[float], float]
    name: str = "custom"
    params: dict = field(default_factory=dict)

    def phi2(self, x: Array) -> Array:
        return self.phi(x) ** 2


def sk_model(beta: float, h: float = 0.0) -> Model:
    """REM approximation of the SK TAP free energy (KSS23, Section 3.1).

    Phi(x) = beta^2/4 (1 - x^2)
    phi(x) = tanh(h + beta x)
    f(x)   = -1/2 log(1 - tanh^2(h + beta x)) - beta/2 x tanh(h + beta x)
           = log cosh(h + beta x) - beta/2 x tanh(h + beta x)

    f is bounded below (by -|h|/2) and grows like beta/2 |x|; under the
    Gaussian mu it has exponential moments of all orders.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")

    def phi(x):
        return np.tanh(h + beta * x)

    def f(x):
        y = h + beta * x
        log_cosh = np.logaddexp(y, -y) - np.log(2.0)  # stable for large |y|
        return log_cosh - 0.5 * beta * x * np.tanh(y)

    def Phi(q):
        return 0.25 * beta**2 * (1.0 - q * q)

    def dPhi(q):
        return -0.5 * beta**2 * q

    return Model(phi=phi, f=f, Phi=Phi, dPhi=dPhi, name="sk", params={"beta": beta, "h": h})
