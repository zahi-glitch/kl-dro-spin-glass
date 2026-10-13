# kl-dro-spin-glass

KL-constrained DRO meets spin glasses: in REM-type approximations of TAP free energies, the Parisi formula is the Lagrangian dual of an entropy-constrained sup. Interactive dashboard on the entropy budget, convex reparametrisation and the phase transition, with a primal–dual numerical check.

> Status: work in progress, built step by step.

## Outline (to be written)

1. The puzzle: free energies are naturally a **sup** over measures, yet for the
   SK model we only have an **inf** (the Parisi formula).
2. The ML/DRO view: KL-constrained worst-case expectations and their duals.
3. The REM/GREM approximation of TAP free energies: a non-linear model where
   both the sup and the inf are proved.
4. The dictionary: Parisi parameters as Lagrange multipliers.
5. Numerical primal-dual check and the phase transition.

## Dashboard

`docs/index.html` is an interactive dashboard, computed live in the browser by
`docs/solver.js`, a JavaScript port of the Python package.

- **Panel 1: the KL budget as a robustness radius.** Sliders for β, h and the
  budget ρ; the worst-case distribution against the Gaussian, the Parisi function
  along the KKT path, and complementary slackness (m* = 1 or H = ρ).
- Panels 2 and 3 (convex reparametrisation, phase diagram): coming next.

Open `docs/index.html` in a browser (double-click works, no server needed).
`python scripts/check_dashboard.py` checks the JavaScript solver against the
Python package (requires Node.js).

## Python package

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
python -m tapdual --beta 5 --h 0.5
```

## Papers

- N. Kistler, M. A. Schmidt, G. Sebastiani, *On the REM approximation of TAP free
  energies*, J. Phys. A: Math. Theor. **56** (2023) 294001.
  [arXiv:2212.11718](https://arxiv.org/abs/2212.11718)
- G. Sebastiani, M. A. Schmidt, *On the GREM approximation of TAP free energies*,
  J. Phys. A: Math. Theor. **57** (2024) 335206.
  [arXiv:2401.13507](https://arxiv.org/abs/2401.13507)
