// Browser port of tapdual (src/tapdual): KL-constrained sup vs Parisi inf
// for the REM approximation of the SK TAP free energy, mu = N(0, 1).
// Checked against the Python package by scripts/check_dashboard.py.
// Plain script (no ES module) so the dashboard also works when opened from disk.
(function (globalObj) {
"use strict";

const L = 14, N = 1401;                       // uniform grid on [-L, L] (trapezoid rule)
const DX = (2 * L) / (N - 1);
const X = Float64Array.from({ length: N }, (_, i) => -L + i * DX);
const LOGW = (() => {                          // log of trapezoid weight x Gaussian density
  const lw = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const trap = (i === 0 || i === N - 1) ? 0.5 : 1.0;
    lw[i] = Math.log(trap * DX) - 0.5 * X[i] * X[i] - 0.5 * Math.log(2 * Math.PI);
  }
  return lw;
})();
const LOG2 = Math.log(2);

const logCosh = (y) => { const a = Math.abs(y); return a + Math.log1p(Math.exp(-2 * a)) - LOG2; };

/** SK specialisation: Phi(x) = b^2/4 (1-x^2), phi = tanh(h+bx),
 *  f = log cosh(h+bx) - b/2 x tanh(h+bx). */
function skModel(beta, h) {
  const phi2 = new Float64Array(N), f = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const y = h + beta * X[i], t = Math.tanh(y);
    phi2[i] = t * t;
    f[i] = logCosh(y) - 0.5 * beta * X[i] * t;
  }
  return {
    beta, h, phi2, f,
    Phi: (q) => 0.25 * beta * beta * (1 - q * q),
    dPhi: (q) => -0.5 * beta * beta * q,
  };
}

/** nu ~ exp(a phi^2 + b f) mu  ->  { logZ, Ephi2, Ef, H } */
function tilt(model, a, b) {
  const { phi2, f } = model;
  let shift = -Infinity;
  const e = new Float64Array(N);
  for (let i = 0; i < N; i++) { e[i] = a * phi2[i] + b * f[i] + LOGW[i]; if (e[i] > shift) shift = e[i]; }
  let Z = 0, S2 = 0, Sf = 0;
  for (let i = 0; i < N; i++) {
    const w = Math.exp(e[i] - shift);
    Z += w; S2 += w * phi2[i]; Sf += w * f[i];
  }
  const logZ = shift + Math.log(Z), Ephi2 = S2 / Z, Ef = Sf / Z;
  return { logZ, Ephi2, Ef, H: Math.max(0, a * Ephi2 + b * Ef - logZ) };
}

const gibbs = (model, q, m) => tilt(model, m * model.dPhi(q), m);

// Illinois (modified regula falsi): bracketing like bisection, superlinear in practice.
function root(g, lo, hi, tol = 1e-12, maxIter = 200) {
  let a = lo, b = hi, fa = g(a), fb = g(b), side = 0;
  for (let k = 0; k < maxIter; k++) {
    const c = (a * fb - b * fa) / (fb - fa), fc = g(c);
    if (Math.abs(b - a) < tol || fc === 0) return c;
    if ((fc > 0) === (fb > 0)) { b = c; fb = fc; if (side === -1) fa /= 2; side = -1; }
    else { a = c; fa = fc; if (side === 1) fb /= 2; side = 1; }
    if (Math.abs(b - a) < tol) return 0.5 * (a + b);
  }
  return 0.5 * (a + b);
}

/** Unique q in [0,1] with q = E_{nu^{q,m}}[phi^2]. */
function fixedPointQ(model, m) {
  const g = (q) => q - gibbs(model, q, m).Ephi2;
  if (g(0) >= 0) return 0;
  if (g(1) <= 0) return 1;
  return root(g, 0, 1, 1e-13);
}

/** Parisi function P(q, m), eq. (34) of KSS23 with the -rho convention. */
function parisi(model, q, m, rho) {
  return model.Phi(q) - q * model.dPhi(q) + (gibbs(model, q, m).logZ + rho) / m - rho;
}

/** Objective of the DRO side: Omega(nu) = Phi(E phi^2) + E f - H. */
const omega = (model, nu) => model.Phi(nu.Ephi2) + nu.Ef - nu.H;

/** Solve inf P via KKT: q = E phi^2, and m = 1 or H = rho. */
function solve(model, rho, mMin = 1e-4) {
  const gap = (m) => gibbs(model, fixedPointQ(model, m), m).H - rho;
  let m, phase;
  if (gap(1) <= 0) { m = 1; phase = "high"; }
  else { m = root(gap, mMin, 1, 1e-12); phase = "low"; }
  const q = fixedPointQ(model, m);
  const nu = gibbs(model, q, m);
  return { q, m, lam: 1 / m, p: model.dPhi(q), nu, phase,
           dual: parisi(model, q, m, rho), primal: omega(model, nu) };
}

/** Critical beta at which the KL budget becomes binding (H at m=1 equals rho). */
function criticalBeta(h, rho, lo = 0.05, hi = 40) {
  const g = (b) => { const M = skModel(b, h); return gibbs(M, fixedPointQ(M, 1), 1).H - rho; };
  if (g(lo) > 0) return lo;
  if (g(hi) < 0) return Infinity;
  return root(g, lo, hi, 1e-6, 60);
}

/** The dual along the KKT path: m -> P(q*(m), m) and H(nu^{q*(m), m}). */
function dualPath(model, rho, ms) {
  return ms.map((m) => {
    const q = fixedPointQ(model, m), nu = gibbs(model, q, m);
    return { m, q, H: nu.H, P: model.Phi(q) - q * model.dPhi(q) + (nu.logZ + rho) / m - rho };
  });
}

/** Density of nu^{q,m} with respect to Lebesgue, on the grid points xs. */
function density(model, q, m, xs, logZ) {
  const a = m * model.dPhi(q), b = m;
  const beta = model.beta, h = model.h;
  return xs.map((x) => {
    const y = h + beta * x, t = Math.tanh(y);
    const fx = logCosh(y) - 0.5 * beta * x * t;
    return Math.exp(a * t * t + b * fx - logZ - 0.5 * x * x) / Math.sqrt(2 * Math.PI);
  });
}

const api = { X, LOG2, skModel, tilt, gibbs, fixedPointQ, parisi, omega, solve, criticalBeta, dualPath, density };
if (typeof module !== "undefined" && module.exports) module.exports = api;
globalObj.TapDual = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
