import numpy as np
import pytest
from scipy.optimize import minimize

from tapdual import (Quadrature, fixed_point_q, parisi_function, parisi_function_convex,
                     sk_model, solve_dual, solve_primal)

LOG2 = np.log(2.0)
GRID = [(b, h) for b in (1.0, 2.5, 3.3, 4.0, 5.5) for h in (0.0, 0.5)]


@pytest.mark.parametrize("beta,h", GRID)
def test_primal_equals_dual(beta, h):
    model = sk_model(beta, h)
    d, p = solve_dual(model), solve_primal(model)
    assert p.nu.H <= LOG2 + 1e-10            # primal point is feasible
    assert abs(p.value - d.value) < 1e-8


@pytest.mark.parametrize("beta,h", GRID)
def test_kkt_conditions(beta, h):
    d = solve_dual(sk_model(beta, h))
    assert abs(d.q - d.gibbs.E_phi2) < 1e-10           # stationarity in q
    if d.phase == "high":
        assert d.m == 1.0 and d.gibbs.H <= LOG2         # budget slack
    else:
        assert d.m < 1.0 and abs(d.gibbs.H - LOG2) < 1e-10  # budget binding


def test_phases_and_critical_beta():
    assert solve_dual(sk_model(3.6, 0.0)).phase == "high"
    assert solve_dual(sk_model(3.7, 0.0)).phase == "low"   # beta_c(h=0) ~ 3.6312


@pytest.mark.parametrize("beta,h", [(2.0, 0.3), (4.5, 0.0)])
def test_dual_matches_brute_force(beta, h):
    """Independent check: direct minimisation of P(q, m) over [0,1] x (0,1]."""
    model = sk_model(beta, h)
    quad = Quadrature(model)
    obj = lambda z: parisi_function(quad, np.clip(z[0], 0, 1), np.clip(z[1], 1e-4, 1.0))
    best = min((minimize(obj, z0, method="Nelder-Mead",
                         options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000))
                for z0 in ([0.2, 0.9], [0.6, 0.5], [0.9, 0.99])), key=lambda r: r.fun)
    assert abs(best.fun - solve_dual(model).value) < 1e-8


@pytest.mark.parametrize("q,m", [(0.1, 1.0), (0.5, 0.7), (0.9, 0.3)])
def test_convex_coordinates_agree(q, m):
    model = sk_model(4.0, 0.2)
    quad = Quadrature(model)
    assert abs(parisi_function(quad, q, m)
               - parisi_function_convex(quad, model.dPhi(q), 1.0 / m)) < 1e-12


def test_convexity_in_p_lambda():
    model = sk_model(4.0, 0.2)
    quad = Quadrature(model)
    rng = np.random.default_rng(0)
    lo, hi = model.dPhi(1.0), model.dPhi(0.0)
    for _ in range(200):
        a = np.array([rng.uniform(lo, hi), rng.uniform(1, 10)])
        b = np.array([rng.uniform(lo, hi), rng.uniform(1, 10)])
        F = lambda z: parisi_function_convex(quad, *z)
        assert F((a + b) / 2) <= (F(a) + F(b)) / 2 + 1e-10


def test_larger_budget_raises_critical_beta():
    # with rho = 1 instead of log 2 the budget is slack at beta = 3.7
    assert solve_dual(sk_model(3.7, 0.0), rho=1.0).phase == "high"


def test_fixed_point_unique_bracket():
    quad = Quadrature(sk_model(4.0, 0.0))
    q = fixed_point_q(quad, 0.8)
    assert 0.0 < q < 1.0 and abs(q - quad.gibbs(q, 0.8).E_phi2) < 1e-12


@pytest.mark.parametrize("beta,h,a,b", [(7.5, 0.0, 0.0, 0.0), (7.5, 0.3, -3.0, 0.6), (12.0, 0.5, -10.0, 0.4)])
def test_quadrature_accurate_at_large_beta(beta, h, a, b):
    """Grid expectations against adaptive quadrature, where tanh(beta x) is steep."""
    from scipy.integrate import quad
    model = sk_model(beta, h)
    nu = Quadrature(model).tilt(a, b)
    dens = lambda x: np.exp(a * model.phi2(x) + b * model.f(x) - 0.5 * x * x) / np.sqrt(2 * np.pi)
    opts = dict(points=[-h / beta], limit=1000, epsabs=1e-15, epsrel=1e-13)
    Z = quad(dens, -14, 14, **opts)[0]
    E2 = quad(lambda x: model.phi2(x) * dens(x), -14, 14, **opts)[0] / Z
    assert abs(nu.log_Z - np.log(Z)) < 1e-10
    assert abs(nu.E_phi2 - E2) < 1e-10
