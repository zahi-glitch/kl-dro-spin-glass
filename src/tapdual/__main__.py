"""Command line report:  python -m tapdual --beta 4 --h 0"""
import argparse

import numpy as np

from . import sk_model, solve_dual, solve_primal


def main() -> None:
    ap = argparse.ArgumentParser(description="Primal-dual check for the REM-TAP model (SK functions).")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=float(np.log(2.0)), help="KL budget (default log 2)")
    a = ap.parse_args()

    model = sk_model(a.beta, a.h)
    d = solve_dual(model, a.rho)
    p = solve_primal(model, a.rho)
    print(f"beta={a.beta}  h={a.h}  rho={a.rho:.6f}")
    print(f"  dual   inf P(q,m)       = {d.value:.12f}   q*={d.q:.6f}  m*={d.m:.6f}  (lambda*={d.lam:.6f})")
    print(f"  primal sup Omega(nu)    = {p.value:.12f}   H(nu*|mu)={p.nu.H:.6f}")
    print(f"  |primal - dual|         = {abs(p.value - d.value):.2e}")
    print(f"  phase: {d.phase} temperature "
          f"({'KL budget slack, m* = 1' if d.phase == 'high' else 'KL budget binding, H = rho'})")


if __name__ == "__main__":
    main()
