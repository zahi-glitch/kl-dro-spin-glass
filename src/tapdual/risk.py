"""Risk measurement with model risk: VaR, Expected Shortfall, EVaR and their KL-robust versions, with backtests.

Losses l are in % of the portfolio value (l = -return). A forecast for day t uses only the losses of the
`window` days before t (historical simulation, the method most banks use).

Measures
--------
VaR_a              value-at-risk: the loss exceeded with probability 1 - a
ES_a               expected shortfall: the average loss beyond VaR_a (Basel's market-risk measure, a = 97.5%)
EVaR_a             entropic value-at-risk = sup { E_nu[l] : KL(nu | mu) <= log(1 / (1 - a)) },
                   the worst-case mean over a KL ball; always >= ES_a >= VaR_a
robust VaR, ES     model risk: the worst VaR / ES over all distributions within KL(nu | mu) <= rho of the
                   empirical one. With rho = chi2_{1, 95%} / (2n) the ball is (approximately) a 95% confidence
                   region for a tail probability estimated from n days, so the robust measures add the
                   estimation error of a short window on top of the historical ones.

Exact formulas
--------------
robust VaR   For an event A, the largest nu(A) in the ball solves the Bernoulli equation kl(q | mu(A)) = rho
             (reweight uniformly inside and outside A). So robust VaR is the smallest data value v with
             q_max(mu(l > v)) <= 1 - a: a historical VaR at a stricter level.
robust ES    ES_a(nu) = min_v { v + E_nu[(l - v)_+] / (1 - a) } (Rockafellar-Uryasev). The bracket is linear in
             nu and convex in v, so sup and min can be swapped (Sion), and
                 robust ES = min_v { v + WC_rho((l - v)_+) / (1 - a) },
             with WC_rho the worst-case mean over the KL ball (dro.worst_case_mean, a 1-D dual).

Filtered historical simulation (filtered=True in rolling_risk) rescales the window by today's EWMA
volatility (RiskMetrics, lambda = 0.94): the robust adjustment fixes the level, the filtering the timing.

Limit: nu must be absolutely continuous w.r.t. the data, so no measure here can exceed the largest loss in
the window. Historical simulation cannot imagine a crash worse than the ones it has seen.

Backtests
---------
Kupiec (unconditional coverage)   is the exception rate the promised 1 - a?
Christoffersen (independence)     do exceptions cluster (one bad day predicting the next)?
Basel traffic light               exceptions of VaR 99% in 250 days: green 0-4, yellow 5-9, red 10 or more
Acerbi-Szekely Z2                 for ES: about 0 if right, positive if ES is too low. Rough 5% warning
                                  level: Z2 > 0.7 (Acerbi and Szekely 2014, for 250 days at 97.5%).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2

from .dro import worst_case_mean

WINDOW = 250
ALPHA_VAR, ALPHA_ES = 0.99, 0.975


def rho_confidence(n: int, level: float = 0.95) -> float:
    """KL radius whose ball is (approximately) a `level` confidence region for a probability from n samples."""
    return float(chi2.ppf(level, 1) / (2 * n))


# --------------------------------------------------------------------------- measures on one window
def var(losses, alpha: float) -> float:
    """Historical VaR: the smallest observed loss v with (share of days with loss > v) <= 1 - alpha."""
    l = np.sort(np.asarray(losses, dtype=float))[::-1]
    k = int(np.floor(l.size * (1 - alpha) + 1e-12))          # days allowed strictly above VaR
    return float(l[min(k, l.size - 1)])


def es(losses, alpha: float) -> float:
    """Historical expected shortfall, Rockafellar-Uryasev form at v = VaR."""
    l = np.asarray(losses, dtype=float)
    v = var(l, alpha)
    return float(v + np.mean(np.maximum(l - v, 0.0)) / (1 - alpha))


def evar(losses, alpha: float) -> float:
    """Entropic value-at-risk: the worst-case mean over the KL ball of radius log(1 / (1 - alpha))."""
    return worst_case_mean(losses, math.log(1.0 / (1.0 - alpha))).value


def _bernoulli_kl(q, p):
    out = 0.0
    if q > 0:
        out += q * math.log(q / p)
    if q < 1:
        out += (1 - q) * math.log((1 - q) / (1 - p))
    return out


def worst_case_probability(p: float, rho: float) -> float:
    """Largest nu(A) over KL(nu | mu) <= rho, when mu(A) = p."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0 or rho >= -math.log(p):
        return 1.0
    return brentq(lambda q: _bernoulli_kl(q, p) - rho, p, 1.0 - 1e-15, xtol=1e-14)


def robust_var(losses, alpha: float, rho: float) -> float:
    """Worst VaR_alpha over the KL ball of radius rho around the data."""
    l = np.sort(np.asarray(losses, dtype=float))[::-1]
    n = l.size
    best = l[0]
    for k in range(n):                                       # candidate VaR = l[k]; k days strictly above it
        if worst_case_probability(k / n, rho) <= 1 - alpha:
            best = l[k]
        else:
            break
    return float(best)


def robust_es(losses, alpha: float, rho: float) -> float:
    """Worst ES_alpha over the KL ball: min over v of v + WC_rho((l - v)_+) / (1 - alpha).

    The function of v is convex and piecewise linear with kinks at the data, so the minimum sits at a data
    point; a ternary search over the sorted losses finds it.
    """
    l = np.sort(np.asarray(losses, dtype=float))
    f = lambda i: l[i] + worst_case_mean(np.maximum(l - l[i], 0.0), rho).value / (1 - alpha)   # noqa: E731
    lo, hi = 0, l.size - 1
    while hi - lo > 2:
        m1, m2 = lo + (hi - lo) // 3, hi - (hi - lo) // 3
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    return float(min(f(i) for i in range(lo, hi + 1)))


# --------------------------------------------------------------------------- rolling forecasts
def ewma_volatility(losses, lam: float = 0.94, init: int = WINDOW) -> np.ndarray:
    """RiskMetrics volatility: sigma_t^2 = lam sigma_{t-1}^2 + (1 - lam) l_{t-1}^2, so sigma_t uses days < t only."""
    L = np.asarray(losses, dtype=float)
    s2 = np.empty(L.size)
    s2[0] = np.mean(L[:init] ** 2)
    for t in range(1, L.size):
        s2[t] = lam * s2[t - 1] + (1 - lam) * L[t - 1] ** 2
    return np.sqrt(s2)


COLUMNS = ["VaR 99%", "robust VaR 99%", "VaR 97.5%", "robust VaR 97.5%", "ES 97.5%", "robust ES 97.5%",
           "EVaR 97.5%"]


def rolling_risk(losses: pd.Series, window: int = WINDOW, rho: float | None = None,
                 filtered: bool = False, lam: float = 0.94) -> pd.DataFrame:
    """One-day-ahead risk forecasts: row t uses only the losses of days t-window .. t-1.

    filtered=False   historical simulation: the past losses as they were
    filtered=True    filtered historical simulation: each past loss is divided by the EWMA volatility of its
                     day and multiplied by today's, so a calm window still gives a high forecast once
                     volatility rises (and the window's tail shape is kept)
    """
    rho = rho_confidence(window) if rho is None else rho
    L = losses.to_numpy(dtype=float)
    sigma = ewma_volatility(L, lam) if filtered else None
    rows = []
    for t in range(window, L.size):
        W = L[t - window:t] if not filtered else L[t - window:t] / sigma[t - window:t] * sigma[t]
        rows.append([L[t],
                     var(W, ALPHA_VAR), robust_var(W, ALPHA_VAR, rho),
                     var(W, ALPHA_ES), robust_var(W, ALPHA_ES, rho),
                     es(W, ALPHA_ES), robust_es(W, ALPHA_ES, rho),
                     evar(W, ALPHA_ES)])
    return pd.DataFrame(rows, index=losses.index[window:], columns=["loss", *COLUMNS])


# --------------------------------------------------------------------------- backtests
def kupiec(exceptions, p: float) -> dict:
    """Unconditional coverage: LR = -2 log [ p^x (1-p)^(T-x) / phat^x (1-phat)^(T-x) ] ~ chi2(1)."""
    e = np.asarray(exceptions, dtype=bool)
    T, x = e.size, int(e.sum())
    ph = x / T

    def ll(q):
        return (x * math.log(q) if x else 0.0) + ((T - x) * math.log(1 - q) if T - x else 0.0)
    lr = -2 * (ll(p) - ll(ph)) if 0 < ph < 1 else -2 * (ll(p) - 0.0)
    return {"exceptions": x, "expected": p * T, "rate": ph, "LR_uc": lr, "p_uc": float(chi2.sf(lr, 1))}


def christoffersen(exceptions, p: float) -> dict:
    """Independence (do exceptions follow exceptions?) and conditional coverage = Kupiec + independence."""
    e = np.asarray(exceptions, dtype=int)
    a, b = e[:-1], e[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))

    def term(k, q):
        return k * math.log(q) if k else 0.0
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l_ind = term(n00, 1 - p01) + term(n01, p01) + term(n10, 1 - p11) + term(n11, p11)
    l_one = term(n00 + n10, 1 - p1) + term(n01 + n11, p1)
    lr_ind = max(-2 * (l_one - l_ind), 0.0)
    lr_cc = kupiec(e, p)["LR_uc"] + lr_ind
    return {"LR_ind": lr_ind, "p_ind": float(chi2.sf(lr_ind, 1)),
            "LR_cc": lr_cc, "p_cc": float(chi2.sf(lr_cc, 2)),
            "P(exception | exception yesterday)": p11}


def traffic_light(exceptions, window: int = WINDOW) -> dict:
    """Share of 250-day windows in each Basel zone, for VaR 99% exceptions."""
    e = np.asarray(exceptions, dtype=int)
    counts = np.convolve(e, np.ones(window, dtype=int), "valid")
    return {"green": float(np.mean(counts <= 4)), "yellow": float(np.mean((counts >= 5) & (counts <= 9))),
            "red": float(np.mean(counts >= 10)), "worst window": int(counts.max())}


def acerbi_szekely_z2(losses, var_forecast, es_forecast, alpha: float) -> float:
    """Z2 = sum_t l_t 1{l_t > VaR_t} / (T (1 - alpha) ES_t) - 1: about 0 if ES is right, > 0 if too low."""
    l, v, s = (np.asarray(x, dtype=float) for x in (losses, var_forecast, es_forecast))
    return float(np.sum(l * (l > v) / s) / (l.size * (1 - alpha)) - 1.0)


def backtest_report(risk: pd.DataFrame) -> pd.DataFrame:
    """One row per VaR / ES forecast: levels, exceptions and test results."""
    rows = {}
    for col, alpha in [("VaR 99%", ALPHA_VAR), ("robust VaR 99%", ALPHA_VAR),
                       ("VaR 97.5%", ALPHA_ES), ("robust VaR 97.5%", ALPHA_ES)]:
        exc = (risk["loss"] > risk[col]).to_numpy()
        k, c = kupiec(exc, 1 - alpha), christoffersen(exc, 1 - alpha)
        row = {"average level (%)": risk[col].mean(), "exceptions": k["exceptions"],
               "expected": k["expected"], "Kupiec p": k["p_uc"], "Christoffersen p (cc)": c["p_cc"],
               "P(exc | exc yesterday)": c["P(exception | exception yesterday)"]}
        if alpha == ALPHA_VAR:
            tl = traffic_light(exc)
            row.update({"windows green": tl["green"], "windows red": tl["red"]})
        rows[col] = row
    for col, vcol in [("ES 97.5%", "VaR 97.5%"), ("robust ES 97.5%", "robust VaR 97.5%")]:
        rows[col] = {"average level (%)": risk[col].mean(),
                     "Acerbi-Szekely Z2": acerbi_szekely_z2(risk["loss"], risk[vcol], risk[col], ALPHA_ES)}
    rows["EVaR 97.5%"] = {"average level (%)": risk["EVaR 97.5%"].mean(),
                          "exceptions": int((risk["loss"] > risk["EVaR 97.5%"]).sum())}
    return pd.DataFrame(rows).T
