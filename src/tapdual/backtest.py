"""Walk-forward portfolio backtest: classical versus KL-robust allocations.

Each month the portfolio is re-estimated on the last `window` trading days of returns R (T x d, in %),
long-only and fully invested (x >= 0, sum x = 1). The daily loss of portfolio x is l_t = -R_t . x.

Strategies
----------
equal weight      x = 1/d.
markowitz         min_x  E_mu[l] + gamma Var_mu(l)                       (mu = empirical, 1/T per day)
minimum variance  min_x  Var_mu(l)                                       (ignores average returns)
robust mean       min_x  sup { E_nu[l] : H(nu | mu) <= rho }            (= entropic value-at-risk, EVaR)
robust mean-var   min_x  sup { E_nu[l] + gamma Var_nu(l) - tau H(nu | mu) : H(nu | mu) <= rho }

The inner sup of the last strategy is the non-linear problem of Kistler-Schmidt-Sebastiani (2023):
it is evaluated through its Parisi-type dual (two scalars q, m) by `worst_case_mean_variance`.
Gradients in x come from Danskin's theorem: the gradient of a sup is the gradient at the maximiser,
so with nu* the worst-case reweighting,

    d/dx E_nu*[l] = -E_nu*[R],      d/dx Var_nu*(l) = 2 Cov_nu*(R) x.

Each robust objective is a sup of functions convex in x, hence convex in x.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .dro import PRICES as DATA, STOCKS as FULL_HISTORY, worst_case_mean, worst_case_mean_variance  # noqa: F401


# --------------------------------------------------------------------------- data
def load_prices(path=None, tickers=None, start=None):
    """Wide CSV with a date column and one price column per asset -> (dates, tickers, prices)."""
    import pandas as pd
    df = pd.read_csv(path or DATA, parse_dates=[0], index_col=0).sort_index()
    if tickers:
        df = df[list(tickers)]
    if start:
        df = df.loc[start:]
    df = df.dropna()
    return df.index.to_numpy(), list(df.columns), df.to_numpy(dtype=float)


def simple_returns_pct(prices: np.ndarray) -> np.ndarray:
    return 100.0 * (prices[1:] / prices[:-1] - 1.0)


# --------------------------------------------------------------------------- objectives
def _simplex_minimise(fun_grad, d, x0=None):
    x0 = np.full(d, 1.0 / d) if x0 is None else x0
    res = minimize(fun_grad, x0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * d,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(d)}],
                   options={"ftol": 1e-10, "maxiter": 200})
    x = np.clip(res.x, 0.0, None)
    return x / x.sum()


def markowitz_objective(R, x, gamma):
    mu, S = R.mean(0), np.cov(R, rowvar=False, bias=True)
    return -mu @ x + gamma * x @ S @ x, -mu + 2 * gamma * S @ x


def robust_mean_objective(R, x, rho):
    r = worst_case_mean(-R @ x, rho)
    return r.value, -r.weights @ R


def robust_mean_variance_objective(R, x, gamma, rho, tau):
    r = worst_case_mean_variance(-R @ x, gamma, tau, rho, check_primal=False)
    w = r.weights
    m = w @ R
    C = (R - m).T @ (w[:, None] * (R - m))
    return r.objective, -m + 2 * gamma * C @ x


class ConicWorstCase:
    """The same inner worst case solved by a generic conic solver (CVXPY + Clarabel), for comparison.

    Parameterised once per window length (CVXPY's DPP), so repeated solves skip the re-modelling.
    If Clarabel fails, SCS is tried; `failures` counts Clarabel failures.
    """

    def __init__(self, n, gamma, rho, tau):
        import cvxpy as cp
        self.cp = cp
        self.l, self.l2 = cp.Parameter(n), cp.Parameter(n)
        self.nu = cp.Variable(n, nonneg=True)
        kl = cp.sum(-cp.entr(self.nu)) + np.log(n)
        m = self.nu @ self.l
        self.problem = cp.Problem(cp.Maximize(m + gamma * (self.nu @ self.l2) - gamma * cp.square(m) - tau * kl),
                                  [cp.sum(self.nu) == 1, kl <= rho])
        self.failures = 0

    def __call__(self, losses):
        self.l.value, self.l2.value = losses, losses ** 2
        try:
            self.problem.solve(solver="CLARABEL")
            if self.problem.status != "optimal":
                raise self.cp.error.SolverError(self.problem.status)
        except self.cp.error.SolverError:
            self.failures += 1
            self.problem.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)
        w = np.clip(self.nu.value, 0.0, None)
        return float(self.problem.value), w / w.sum()


def robust_mean_variance_objective_conic(R, x, gamma, rho, tau, solver: ConicWorstCase):
    value, w = solver(-R @ x)
    m = w @ R
    C = (R - m).T @ (w[:, None] * (R - m))
    return value, -m + 2 * gamma * C @ x


# --------------------------------------------------------------------------- strategies
@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str                     # "equal", "markowitz", "minvar", "robust_mean", "robust_mv", "robust_mv_conic"
    gamma: float = 0.05
    rho: float = 0.05
    tau: float = 0.1

    def weights(self, R, x0=None):
        d = R.shape[1]
        if self.kind == "equal":
            return np.full(d, 1.0 / d)
        if self.kind == "markowitz":
            return _simplex_minimise(lambda x: markowitz_objective(R, x, self.gamma), d, x0)
        if self.kind == "minvar":
            S = np.cov(R, rowvar=False, bias=True)
            return _simplex_minimise(lambda x: (x @ S @ x, 2 * S @ x), d, x0)
        if self.kind == "robust_mean":
            return _simplex_minimise(lambda x: robust_mean_objective(R, x, self.rho), d, x0)
        if self.kind == "robust_mv":
            return _simplex_minimise(lambda x: robust_mean_variance_objective(R, x, self.gamma, self.rho, self.tau),
                                     d, x0)
        if self.kind == "robust_mv_conic":
            key = (R.shape[0], self.gamma, self.rho, self.tau)
            if key not in _CONIC:
                _CONIC[key] = ConicWorstCase(*key)
            return _simplex_minimise(lambda x: robust_mean_variance_objective_conic(
                R, x, self.gamma, self.rho, self.tau, _CONIC[key]), d, x0)
        raise ValueError(self.kind)


_CONIC: dict = {}


# --------------------------------------------------------------------------- backtest
@dataclass
class BacktestResult:
    strategy: Strategy
    dates: np.ndarray             # out-of-sample days
    returns: np.ndarray           # daily portfolio returns in %, after costs
    weights: np.ndarray           # weights at each rebalance (k x d)
    rebalance_dates: np.ndarray
    turnover: np.ndarray          # sum |x_new - x_old| at each rebalance
    metrics: dict = field(default_factory=dict)


def backtest(strategy: Strategy, dates, R, window=504, step=21, cost_bps=10.0) -> BacktestResult:
    """Walk-forward: estimate on R[t-window:t], hold the weights (rebalanced daily to target) on R[t:t+step].

    Transaction costs: cost_bps basis points per unit of turnover, charged on the first day of each period.
    """
    T, d = R.shape
    out, w_hist, reb, turn = [], [], [], []
    x_prev = np.full(d, 1.0 / d)
    for t in range(window, T, step):
        x = strategy.weights(R[t - window:t], x0=x_prev)
        tv = float(np.abs(x - x_prev).sum())
        period = R[t:t + step] @ x
        period[0] -= cost_bps / 100.0 * tv          # bps of the portfolio value, in % units
        out.append(period); w_hist.append(x); reb.append(dates[t]); turn.append(tv)
        x_prev = x
    rets = np.concatenate(out)
    res = BacktestResult(strategy, dates[window:window + rets.size], rets, np.array(w_hist), np.array(reb),
                         np.array(turn))
    res.metrics = performance(rets, res.turnover, res.weights)
    return res


def performance(rets_pct, turnover=None, weights=None, days_per_year=252) -> dict:
    r = rets_pct / 100.0
    wealth = np.cumprod(1.0 + r)
    years = r.size / days_per_year
    peak = np.maximum.accumulate(wealth)
    q = np.quantile(r, 0.05)
    m = {
        "annual return": wealth[-1] ** (1.0 / years) - 1.0,
        "annual volatility": r.std() * np.sqrt(days_per_year),
        "Sharpe (rf = 0)": r.mean() / r.std() * np.sqrt(days_per_year),
        "max drawdown": float((wealth / peak - 1.0).min()),
        "worst day": float(r.min()),
        "CVaR 5% (daily)": float(-r[r <= q].mean()),
    }
    if turnover is not None:
        m["turnover / rebalance"] = float(turnover.mean())
    if weights is not None:
        m["effective holdings"] = float(np.mean(1.0 / (weights ** 2).sum(1)))
    return m
