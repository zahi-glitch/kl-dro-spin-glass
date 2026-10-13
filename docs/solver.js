// Browser port of the tapdual Python package (src/tapdual): KL-constrained
// sup vs Parisi inf, for a reference measure mu that is either N(0, 1)
// (spin-glass model) or an empirical measure on data (DRO model).
// Checked against the Python package by scripts/check_dashboard.py.
// Plain script (no ES module) so the dashboard also works when opened from disk.
(function (globalObj) {
"use strict";

const LOG2 = Math.log(2);

// N(0,1): trapezoid rule on a uniform grid of [-L, L], Gaussian density in the weights.
const L = 14, N = 1401, DX = (2 * L) / (N - 1);
const X = Float64Array.from({ length: N }, (_, i) => -L + i * DX);
const LOGW_GAUSS = Float64Array.from(X, (x, i) =>
  Math.log(((i === 0 || i === N - 1) ? 0.5 : 1.0) * DX) - 0.5 * x * x - 0.5 * Math.log(2 * Math.PI));

const logCosh = (y) => { const a = Math.abs(y); return a + Math.log1p(Math.exp(-2 * a)) - LOG2; };

/** A model: points with log-weights (the reference measure mu), g and f at
 *  the points, the concave Phi with derivative dPhi, and the range of q = E_nu[g]. */
function makeModel({ points, logw, g, f, Phi, dPhi, qLo, qHi, ...rest }) {
  if (qLo === undefined) { qLo = Infinity; qHi = -Infinity; for (const v of g) { qLo = Math.min(qLo, v); qHi = Math.max(qHi, v); } }
  return { points, logw, g, f, Phi, dPhi, qLo, qHi, ...rest };
}

/** REM approximation of the SK TAP free energy: Phi(x) = b^2/4 (1-x^2),
 *  g = tanh^2(h+bx), f = log cosh(h+bx) - b/2 x tanh(h+bx), mu = N(0,1). */
function skModel(beta, h) {
  const g = new Float64Array(N), f = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const y = h + beta * X[i], t = Math.tanh(y);
    g[i] = t * t;
    f[i] = logCosh(y) - 0.5 * beta * X[i] * t;
  }
  return makeModel({
    kind: "sk", beta, h, points: X, logw: LOGW_GAUSS, g, f, qLo: 0, qHi: 1,
    Phi: (q) => 0.25 * beta * beta * (1 - q * q),
    dPhi: (q) => -0.5 * beta * beta * q,
  });
}

/** Robust mean-variance on data: sup E[l] + gamma Var(l) - tau KL, divided by tau:
 *  g = l, f = (l + gamma l^2)/tau, Phi(x) = -gamma x^2 / tau, mu = empirical. */
function meanVarianceModel(losses, gamma, tau) {
  const n = losses.length, lw = -Math.log(n);
  const points = Float64Array.from(losses);
  return makeModel({
    kind: "mv", gamma, tau, points,
    logw: new Float64Array(n).fill(lw),
    g: Float64Array.from(points),
    f: Float64Array.from(points, (l) => (l + gamma * l * l) / tau),
    Phi: (q) => -gamma * q * q / tau,
    dPhi: (q) => -2 * gamma * q / tau,
  });
}

/** nu ~ exp(a g + b f) mu  ->  { logZ, Eg, Ef, H } */
function tilt(model, a, b) {
  const { g, f, logw } = model, n = g.length;
  let shift = -Infinity;
  const e = new Float64Array(n);
  for (let i = 0; i < n; i++) { e[i] = a * g[i] + b * f[i] + logw[i]; if (e[i] > shift) shift = e[i]; }
  let Z = 0, Sg = 0, Sf = 0;
  for (let i = 0; i < n; i++) {
    const w = Math.exp(e[i] - shift);
    Z += w; Sg += w * g[i]; Sf += w * f[i];
  }
  const logZ = shift + Math.log(Z), Eg = Sg / Z, Ef = Sf / Z;
  return { logZ, Eg, Ef, H: Math.max(0, a * Eg + b * Ef - logZ) };
}

const gibbs = (model, q, m) => tilt(model, m * model.dPhi(q), m);

/** Probabilities of nu^{q,m} at the model's points. */
function weights(model, q, m) {
  const a = m * model.dPhi(q), { g, f, logw } = model, n = g.length;
  const e = new Float64Array(n);
  let shift = -Infinity;
  for (let i = 0; i < n; i++) { e[i] = a * g[i] + m * f[i] + logw[i]; if (e[i] > shift) shift = e[i]; }
  let Z = 0;
  for (let i = 0; i < n; i++) { e[i] = Math.exp(e[i] - shift); Z += e[i]; }
  for (let i = 0; i < n; i++) e[i] /= Z;
  return e;
}

// Illinois (modified regula falsi): bracketing like bisection, superlinear in practice.
function root(fn, lo, hi, tol = 1e-12, maxIter = 200) {
  let a = lo, b = hi, fa = fn(a), fb = fn(b), side = 0;
  for (let k = 0; k < maxIter; k++) {
    const c = (a * fb - b * fa) / (fb - fa), fc = fn(c);
    if (Math.abs(b - a) < tol || fc === 0) return c;
    if ((fc > 0) === (fb > 0)) { b = c; fb = fc; if (side === -1) fa /= 2; side = -1; }
    else { a = c; fa = fc; if (side === 1) fb /= 2; side = 1; }
    if (Math.abs(b - a) < tol) return 0.5 * (a + b);
  }
  return 0.5 * (a + b);
}

/** Unique q in [qLo, qHi] with q = E_{nu^{q,m}}[g]. */
function fixedPointQ(model, m) {
  const { qLo: a, qHi: b } = model;
  const gap = (q) => q - gibbs(model, q, m).Eg;
  if (gap(a) >= 0) return a;
  if (gap(b) <= 0) return b;
  return root(gap, a, b, 1e-13 * Math.max(1, Math.abs(a), Math.abs(b)));
}

/** Parisi function P(q, m), eq. (34) of KSS23 with the -rho convention. */
function parisi(model, q, m, rho) {
  return model.Phi(q) - q * model.dPhi(q) + (gibbs(model, q, m).logZ + rho) / m - rho;
}

/** Objective of the DRO side: Omega(nu) = Phi(E g) + E f - H. */
const omega = (model, nu) => model.Phi(nu.Eg) + nu.Ef - nu.H;

/** KL used by the unconstrained optimum (m = 1): the budget binds iff rho is below it. */
function criticalRho(model) { return gibbs(model, fixedPointQ(model, 1), 1).H; }

/** Solve inf P via KKT: q = E g, and m = 1 or H = rho. */
function solve(model, rho, mMin = 1e-6) {
  const gap = (m) => gibbs(model, fixedPointQ(model, m), m).H - rho;
  let m, phase;
  if (gap(1) <= 0) { m = 1; phase = "high"; }
  else { m = root(gap, mMin, 1, 1e-13); phase = "low"; }
  const q = fixedPointQ(model, m);
  const nu = gibbs(model, q, m);
  return { q, m, lam: 1 / m, p: model.dPhi(q), nu, phase,
           dual: parisi(model, q, m, rho), primal: omega(model, nu) };
}

/** SK model: beta at which the KL budget becomes binding (H at m = 1 equals rho). */
function criticalBeta(h, rho, lo = 0.05, hi = 40) {
  const fn = (b) => criticalRho(skModel(b, h)) - rho;
  if (fn(lo) > 0) return lo;
  if (fn(hi) < 0) return Infinity;
  return root(fn, lo, hi, 1e-6, 60);
}

/** The dual along the KKT path: m -> P(q*(m), m) and H(nu^{q*(m), m}). */
function dualPath(model, rho, ms) {
  return ms.map((m) => {
    const q = fixedPointQ(model, m), nu = gibbs(model, q, m);
    return { m, q, H: nu.H, P: model.Phi(q) - q * model.dPhi(q) + (nu.logZ + rho) / m - rho };
  });
}

/** SK model: density of nu^{q,m} with respect to Lebesgue at the points xs. */
function density(model, q, m, xs, logZ) {
  const a = m * model.dPhi(q), b = m, beta = model.beta, h = model.h;
  return xs.map((x) => {
    const y = h + beta * x, t = Math.tanh(y);
    const fx = logCosh(y) - 0.5 * beta * x * t;
    return Math.exp(a * t * t + b * fx - logZ - 0.5 * x * x) / Math.sqrt(2 * Math.PI);
  });
}

const api = { X, LOG2, skModel, meanVarianceModel, tilt, gibbs, weights, fixedPointQ, parisi, omega,
              solve, criticalBeta, criticalRho, dualPath, density };
if (typeof module !== "undefined" && module.exports) module.exports = api;
globalObj.TapDual = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
