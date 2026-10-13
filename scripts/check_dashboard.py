"""Check that the dashboard's JavaScript solver (docs/solver.js) agrees with
the Python package.  Requires Node.js.  Usage:  python scripts/check_dashboard.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

from tapdual import sk_model, solve_dual, solve_primal

ROOT = Path(__file__).resolve().parents[1]
CASES = [(b, h, rho) for b in (1.0, 3.0, 3.7, 5.0, 7.5) for h in (0.0, 0.8)
         for rho in (math.log(2), 1.0)]

JS = """
const { skModel, solve } = require(%r);
const cases = JSON.parse(process.argv.at(-1));
console.log(JSON.stringify(cases.map(([b, h, rho]) => {
  const s = solve(skModel(b, h), rho);
  return { q: s.q, m: s.m, dual: s.dual, primal: s.primal, H: s.nu.H, phase: s.phase };
})));
""" % str(ROOT / "docs" / "solver.js")


def main() -> int:
    out = subprocess.run(["node", "-e", JS, json.dumps(CASES)],
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    worst = 0.0
    for (b, h, rho), r in zip(CASES, js):
        d = solve_dual(sk_model(b, h), rho)
        p = solve_primal(sk_model(b, h), rho)
        errs = [abs(r["q"] - d.q), abs(r["m"] - d.m), abs(r["dual"] - d.value),
                abs(r["primal"] - p.value), abs(r["H"] - d.gibbs.H)]
        worst = max(worst, *errs)
        flag = "ok" if max(errs) < 1e-6 and r["phase"] == d.phase else "MISMATCH"
        print(f"beta={b:4.1f} h={h:3.1f} rho={rho:.3f}  phase={r['phase']:4s}  "
              f"max err={max(errs):.1e}  {flag}")
    print(f"worst error {worst:.1e}")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
