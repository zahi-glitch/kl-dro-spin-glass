import math

import numpy as np
import pandas as pd
import pytest

from tapdual.dro import load_returns, worst_case_mean
from tapdual.risk import (acerbi_szekely_z2, christoffersen, es, evar, kupiec, rho_confidence, robust_es,
                          robust_var, rolling_risk, traffic_light, var, worst_case_probability)

LOSSES = -load_returns()[-250:]


def _kl(w):
    n = w.size
    nz = w > 0
    return float(np.sum(w[nz] * np.log(w[nz] * n)))


def _es_weighted(l, w, alpha):
    """ES under weights w: min over data points v of v + E_w (l - v)_+ / (1 - alpha)."""
    return min(v + w @ np.maximum(l - v, 0.0) / (1 - alpha) for v in l)


def test_historical_measures():
    l = np.arange(1, 101, dtype=float)                 # losses 1..100
    assert var(l, 0.99) == 99.0                        # one day strictly above
    assert var(l, 0.95) == 95.0
    assert es(l, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert var(LOSSES, 0.99) <= es(LOSSES, 0.99) <= evar(LOSSES, 0.99) <= LOSSES.max()


def test_evar_is_es_bound_and_kl_formula():
    for a in (0.9, 0.975):
        assert es(LOSSES, a) <= evar(LOSSES, a) + 1e-12
        assert evar(LOSSES, a) == pytest.approx(worst_case_mean(LOSSES, math.log(1 / (1 - a))).value)


def test_worst_case_probability():
    assert worst_case_probability(0.0, 0.1) == 0.0
    assert worst_case_probability(0.3, 0.0) == pytest.approx(0.3)
    q = worst_case_probability(0.01, 0.01)
    assert q > 0.01 and q * math.log(q / 0.01) + (1 - q) * math.log((1 - q) / 0.99) == pytest.approx(0.01)


def test_robust_measures_reduce_to_historical_at_rho_zero():
    for a in (0.99, 0.975):
        assert robust_var(LOSSES, a, 0.0) == var(LOSSES, a)
        assert robust_es(LOSSES, a, 0.0) == pytest.approx(es(LOSSES, a))


def test_robust_measures_grow_with_rho():
    r = [rho_confidence(250) * s for s in (0, 1, 3)]
    v = [robust_var(LOSSES, 0.99, x) for x in r]
    e = [robust_es(LOSSES, 0.975, x) for x in r]
    assert v == sorted(v) and e == sorted(e) and e[1] > e[0]


def test_robust_var_is_attained():
    """Some distribution in the ball has VaR equal to robust VaR: tilt the mass of the days above it."""
    rho, a = rho_confidence(250), 0.99
    v = robust_var(LOSSES, a, rho)
    above = LOSSES >= v
    q = worst_case_probability(np.mean(above), rho)
    w = np.where(above, q / above.sum(), (1 - q) / (~above).sum())
    assert _kl(w) <= rho + 1e-9 and w[LOSSES >= v].sum() > 1 - a - 1e-9


def test_robust_es_saddle_point():
    """Exactness of the minimax formula: the worst-case measure at the optimal v has ES = robust ES,
    and no random measure in the ball does better."""
    l, a, rho = LOSSES, 0.975, 3 * rho_confidence(250)
    value = robust_es(l, a, rho)
    vs = np.sort(l)
    obj = [v + worst_case_mean(np.maximum(l - v, 0.0), rho).value / (1 - a) for v in vs]
    v_star = vs[int(np.argmin(obj))]
    w_star = worst_case_mean(np.maximum(l - v_star, 0.0), rho).weights
    assert _kl(w_star) <= rho + 1e-9
    assert _es_weighted(l, w_star, a) == pytest.approx(value, rel=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.dirichlet(np.full(l.size, 50.0))
        if _kl(w) <= rho:
            assert _es_weighted(l, w, a) <= value + 1e-9


def test_kupiec_and_christoffersen():
    e = np.zeros(1000, dtype=bool); e[::100] = True     # 10 exceptions, exactly the 1% rate, spread out
    k, c = kupiec(e, 0.01), christoffersen(e, 0.01)
    assert k["exceptions"] == 10 and k["LR_uc"] == pytest.approx(0.0, abs=1e-9) and c["p_ind"] > 0.5
    bad = np.zeros(1000, dtype=bool); bad[500:510] = True   # same count, all clustered
    assert christoffersen(bad, 0.01)["p_ind"] < 1e-4
    assert kupiec(np.r_[np.ones(50), np.zeros(950)].astype(bool), 0.01)["p_uc"] < 1e-6


def test_traffic_light_and_z2():
    e = np.zeros(500, dtype=int); e[100:110] = 1
    tl = traffic_light(e)
    assert tl["worst window"] == 10 and tl["red"] > 0
    rng = np.random.default_rng(1)
    l = rng.standard_normal(200_000)
    from scipy.stats import norm
    v, s = norm.ppf(0.975), norm.pdf(norm.ppf(0.975)) / 0.025
    assert abs(acerbi_szekely_z2(l, np.full_like(l, v), np.full_like(l, s), 0.975)) < 0.03
    assert acerbi_szekely_z2(l, np.full_like(l, v), np.full_like(l, 0.7 * s), 0.975) > 0.3


def test_rolling_uses_only_the_past():
    s = pd.Series(-load_returns()[-300:])
    r = rolling_risk(s, window=250)
    s2 = s.copy(); s2.iloc[-1] = 99.0                      # change the last day
    r2 = rolling_risk(s2, window=250)
    assert len(r) == 50
    pd.testing.assert_frame_equal(r.drop(columns="loss"), r2.drop(columns="loss"))
