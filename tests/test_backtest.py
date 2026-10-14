import numpy as np
import pytest

pytest.importorskip("pandas")

from tapdual.backtest import (FULL_HISTORY, Strategy, backtest, load_prices, markowitz_objective,  # noqa: E402
                              robust_mean_objective, robust_mean_variance_objective, simple_returns_pct)


@pytest.fixture(scope="module")
def returns():
    dates, _, prices = load_prices(tickers=FULL_HISTORY, start="2014-01-01")
    return dates[1:], simple_returns_pct(prices)


def test_data_loads(returns):
    dates, R = returns
    assert R.shape[1] == 10 and np.isfinite(R).all() and len(dates) == len(R)


@pytest.mark.parametrize("objective", [
    lambda R, x: markowitz_objective(R, x, 0.05),
    lambda R, x: robust_mean_objective(R, x, 0.05),
    lambda R, x: robust_mean_variance_objective(R, x, 0.05, 0.05, 0.1),
])
def test_danskin_gradients(returns, objective):
    """Gradients from the worst-case reweighting nu* (Danskin) match finite differences."""
    _, R = returns
    W, x, e = R[:504], np.random.default_rng(0).dirichlet(np.ones(10)), 1e-6
    _, g = objective(W, x)
    fd = [(objective(W, x + e * u)[0] - objective(W, x - e * u)[0]) / (2 * e) for u in np.eye(10)]
    assert np.abs(g - fd).max() < 1e-6


def test_robust_objective_dominates_nominal(returns):
    """The worst case over a KL ball is at least the nominal value (nu = mu is feasible)."""
    _, R = returns
    W, x = R[:504], np.full(10, 0.1)
    nominal = markowitz_objective(W, x, 0.05)[0]
    assert robust_mean_variance_objective(W, x, 0.05, 0.05, 0.1)[0] >= nominal - 1e-9


def test_backtest_runs_and_weights_are_valid(returns):
    dates, R = returns
    res = backtest(Strategy("rmv", "robust_mv"), dates[-800:], R[-800:], window=504, step=63)
    assert np.allclose(res.weights.sum(1), 1) and (res.weights >= -1e-12).all()
    assert res.returns.size == 800 - 504 and np.isfinite(list(res.metrics.values())).all()
