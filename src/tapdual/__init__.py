"""Primal (KL-constrained sup) and dual (Parisi inf) for REM-type
approximations of TAP free energies."""
from .model import Model, sk_model
from .gibbs import Quadrature, Tilt
from .dual import (DualSolution, fixed_point_q, parisi_function,
                   parisi_function_convex, solve_dual)
from .primal import PrimalSolution, omega, solve_primal

__all__ = [
    "Model", "sk_model", "Quadrature", "Tilt",
    "DualSolution", "fixed_point_q", "parisi_function", "parisi_function_convex", "solve_dual",
    "PrimalSolution", "omega", "solve_primal",
]
