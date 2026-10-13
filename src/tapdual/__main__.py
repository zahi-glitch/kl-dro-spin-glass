"""Command line:

    python -m tapdual sk --beta 5 --h 0.5          # spin-glass model (REM approximation of SK TAP)
    python -m tapdual mv --gamma 1 --tau 0.5 --rho 0.1 [--csv returns.csv]   # robust mean-variance
"""
import argparse

import numpy as np

from . import sk_model, solve_dual, solve_primal
from .dro import load_returns, worst_case_mean_variance


def run_sk(a) -> None:
    model = sk_model(a.beta, a.h)
    d, p = solve_dual(model, a.rho), solve_primal(model, a.rho)
    print(f"Spin-glass model  beta={a.beta}  h={a.h}  KL budget rho={a.rho:.6f}")
    print(f"  Parisi side   inf P(q,m)   = {d.value:.12f}   q*={d.q:.6f}  m*={d.m:.6f}  (lambda*={d.lam:.6f})")
    print(f"  DRO side      sup Omega(nu) = {p.value:.12f}   KL(nu*|mu)={p.nu.H:.6f}")
    print(f"  |difference|               = {abs(p.value - d.value):.2e}")
    print(f"  phase: {d.phase} temperature "
          f"({'budget slack, m* = 1' if d.phase == 'high' else 'budget binding, KL = rho'})")


def run_mv(a) -> None:
    returns = load_returns(a.csv)
    r = worst_case_mean_variance(-returns, a.gamma, a.tau, a.rho)
    src = a.csv or "built-in synthetic sample"
    print(f"Robust mean-variance  data: {src} ({returns.size} returns, loss = -return)")
    print(f"  gamma={a.gamma}  tau={a.tau}  KL radius rho={a.rho}")
    print(f"  nominal risk  E[l] + gamma Var(l)            = {r.nominal_risk:.6f}")
    print(f"  worst-case risk under nu*                     = {r.worst_risk:.6f}")
    print(f"  robust objective  sup risk - tau KL           = {r.objective:.6f}")
    print(f"  worst-case mean loss q* = {r.worst_mean:.6f}   KL used = {r.kl:.6f} of {a.rho}")
    print(f"  Parisi parameter m* = {r.m:.6f}   KL multiplier tau/m* = {r.kl_multiplier:.6f}")
    print(f"  budget {r.phase};  primal-dual gap {r.primal_dual_gap:.1e}")
    top = np.argsort(r.weights)[::-1][:5]
    print("  most up-weighted days (return %, weight x n):",
          ", ".join(f"{returns[i]:+.2f}% (x{r.weights[i] * returns.size:.1f})" for i in top))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m tapdual", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sk", help="REM approximation of the SK TAP free energy")
    s.add_argument("--beta", type=float, default=4.0)
    s.add_argument("--h", type=float, default=0.0)
    s.add_argument("--rho", type=float, default=float(np.log(2.0)), help="KL budget (default log 2)")
    s.set_defaults(run=run_sk)
    m = sub.add_parser("mv", help="worst-case mean-variance risk over a KL ball (data)")
    m.add_argument("--gamma", type=float, default=1.0, help="risk aversion (weight of the variance)")
    m.add_argument("--tau", type=float, default=0.5, help="weight of the KL penalty")
    m.add_argument("--rho", type=float, default=0.1, help="KL radius")
    m.add_argument("--csv", default=None, help="CSV file, returns in %% in the first column")
    m.set_defaults(run=run_mv)
    a = ap.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
