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

## Desktop app (work in progress)

A local Mac/Linux/Windows app that tells the story from spin glasses to robust
optimisation, chapter by chapter, with live simulations:

```bash
pip install -e ".[app]"        # adds PySide6 and pyqtgraph
python -m tapdual.app
```

Chapter 1 (ready): Hopfield memory recall, two SK replicas, REM freezing.

## Dashboard

Two interactive pages in `docs/`, computed live in the browser by `docs/solver.js`,
a JavaScript port of the Python package (checked against it by
`scripts/check_dashboard.py`, which needs Node.js):

- **`docs/index.html`: the KL budget as a robustness radius (spin-glass model).**
  Sliders for β, h and the budget ρ; the worst-case distribution against the
  Gaussian, the Parisi function along the KKT path, and complementary slackness.
- **`docs/mean-variance.html`: robust mean-variance on a dataset.** The same duality
  applied to KL-DRO on data: an adversary reweights daily returns within a KL ball
  to maximise E ℓ + γ Var ℓ − τ KL. Sliders for γ, τ and ρ, and you can load your
  own CSV of returns (it stays in the browser).

Open either file in a browser; double-clicking works, no server needed.

## Python package

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
python -m tapdual sk --beta 5 --h 0.5                  # spin-glass model
python -m tapdual mv --gamma 1 --tau 0.5 --rho 0.1     # robust mean-variance, built-in data
python -m tapdual mv --csv my_returns.csv              # ... on your own returns (in %)
```

`pyproject.toml` describes the package for `pip`; there is no need to open it.

### Checks

- primal (sup over measures) = dual (Parisi inf) to ~1e-12, in both phases;
- the KKT conditions, the jointly convex coordinates (p, λ), and the transition;
- the classical REM threshold β_c = √(2 log 2) is reproduced to 1e-8;
- on data, a direct optimisation over all sample weights gives the same value
  as the two-parameter dual.

## Papers

- N. Kistler, M. A. Schmidt, G. Sebastiani, *On the REM approximation of TAP free
  energies*, J. Phys. A: Math. Theor. **56** (2023) 294001.
  [arXiv:2212.11718](https://arxiv.org/abs/2212.11718)
- G. Sebastiani, M. A. Schmidt, *On the GREM approximation of TAP free energies*,
  J. Phys. A: Math. Theor. **57** (2024) 335206.
  [arXiv:2401.13507](https://arxiv.org/abs/2401.13507)
