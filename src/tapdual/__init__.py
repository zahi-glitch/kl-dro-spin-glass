"""Primal (KL-constrained sup) and dual (Parisi inf) for REM-type
approximations of TAP free energies, and for KL-DRO problems on data."""
from .model import Model, mean_variance_model, sk_model
from .gibbs import Quadrature, Tilt
from .dual import (DualSolution, fixed_point_q, parisi_function,
                   parisi_function_convex, solve_dual)
from .primal import PrimalSolution, omega, solve_primal
from .dro import RobustMeanVariance, load_returns, worst_case_mean_variance

__all__ = [
    "Model", "sk_model", "mean_variance_model", "Quadrature", "Tilt",
    "DualSolution", "fixed_point_q", "parisi_function", "parisi_function_convex", "solve_dual",
    "PrimalSolution", "omega", "solve_primal",
    "RobustMeanVariance", "load_returns", "worst_case_mean_variance",
]
