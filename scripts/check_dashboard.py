"""Check that the dashboard's JavaScript solver (docs/solver.js) agrees with
the Python package.  Requires Node.js.  Usage:  python scripts/check_dashboard.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

from tapdual import load_returns, sk_model, solve_dual, solve_primal, worst_case_mean_variance

ROOT = Path(__file__).resolve().parents[1]
CASES = [(b, h, rho) for b in (1.0, 3.0, 3.7, 5.0, 7.5) for h in (0.0, 0.8)
         for rho in (math.log(2), 1.0)]
MV_CASES = [(1.0, 0.5, 0.1), (1.0, 0.05, 0.05), (0.5, 10.0, 0.1), (2.0, 1.0, 0.5), (0.3, 0.02, 0.3)]

JS = """
const { skModel, meanVarianceModel, solve } = require(%r);
const [cases, mvCases, losses] = JSON.parse(process.argv.at(-1));
const out = (s) => ({ q: s.q, m: s.m, dual: s.dual, primal: s.primal, H: s.nu.H, phase: s.phase });
console.log(JSON.stringify([
  cases.map(([b, h, rho]) => out(solve(skModel(b, h), rho))),
  mvCases.map(([g, t, rho]) => out(solve(meanVarianceModel(losses, g, t), rho))),
]));
""" % str(ROOT / "docs" / "solver.js")


def main() -> int:
    losses = (-load_returns()).tolist()
    out = subprocess.run(["node", "-e", JS, json.dumps([CASES, MV_CASES, losses])],
                         capture_output=True, text=True, check=True).stdout
    js_sk, js_mv = json.loads(out)
    worst = 0.0
    for (b, h, rho), r in zip(CASES, js_sk):
        d = solve_dual(sk_model(b, h), rho)
        p = solve_primal(sk_model(b, h), rho)
        errs = [abs(r["q"] - d.q), abs(r["m"] - d.m), abs(r["dual"] - d.value),
                abs(r["primal"] - p.value), abs(r["H"] - d.gibbs.H)]
        worst = max(worst, *errs)
        flag = "ok" if max(errs) < 1e-6 and r["phase"] == d.phase else "MISMATCH"
        print(f"SK  beta={b:4.1f} h={h:3.1f} rho={rho:.3f}  phase={r['phase']:4s}  max err={max(errs):.1e}  {flag}")
    for (g, t, rho), r in zip(MV_CASES, js_mv):
        res = worst_case_mean_variance(losses, g, t, rho)
        errs = [abs(r["q"] - res.worst_mean), abs(r["m"] - res.m) / max(res.m, 1e-12),
                abs(r["dual"] * t - res.objective) / max(1.0, abs(res.objective)), abs(r["H"] - res.kl)]
        worst = max(worst, *errs)
        phase = "high" if res.phase == "slack" else "low"
        flag = "ok" if max(errs) < 1e-6 and r["phase"] == phase else "MISMATCH"
        print(f"MV  gamma={g} tau={t} rho={rho}  phase={r['phase']:4s}  max err={max(errs):.1e}  {flag}")
    print(f"worst error {worst:.1e}")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
