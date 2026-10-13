// Small dependency-free SVG line charts with a hover crosshair, shared by the dashboard pages.
(function (globalObj) {
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 4) => Number.isFinite(v) ? v.toFixed(d) : "–";
// tick labels: integers without decimals, otherwise as few decimals as needed (never round -7.5 to -8)
const tickFmt = (v) => { for (let d = 0; d < 4; d++) { if (Math.abs(v - +v.toFixed(d)) < 1e-9) return v.toFixed(d); } return v.toFixed(4); };
const gapText = (g) => (Math.abs(g) < 1e-12 ? "< 1e-12" : Math.abs(g).toExponential(0));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SVGNS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= n) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9 * span; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

function logTicks(lo, hi) {
  const ticks = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) ticks.push(10 ** e);
  return ticks;
}

/** Line chart. All series share the x values `xs`; one y axis; hover crosshair
 *  with one tooltip listing every series. Options:
 *  xLog (log-scale x), yDomain, hlines [{y, label, short}], points [{x, y, color, label}]. */
function lineChart(container, { xs, series, xLabel, yLabel, yDomain, hlines = [], points = [], xLog = false,
                                xFmt = (v) => v.toFixed(2), yFmt = (v) => v.toFixed(3), tipX }) {
  container.textContent = "";
  const W = Math.max(320, container.clientWidth || 600), H = Math.round(Math.min(300, Math.max(200, W * 0.5)));
  const m = { l: 60, r: 14, t: 14, b: 38 };
  const tx = xLog ? Math.log10 : (v) => v;
  const x0 = tx(xs[0]), x1 = tx(xs[xs.length - 1]);
  const [y0, y1] = yDomain || (() => {
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.ys) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const h of hlines) { lo = Math.min(lo, h.y); hi = Math.max(hi, h.y); }
    const pad = (hi - lo) * 0.08 || 0.1;
    return [lo - pad, hi + pad];
  })();
  const sx = (x) => m.l + ((tx(x) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${yLabel} against ${xLabel}` }, container);
  const grid = el("g", {}, svg);
  for (const t of niceTicks(y0, y1, 4)) {
    el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), stroke: css("--grid"), "stroke-width": 1 }, grid);
    const lab = el("text", { x: m.l - 6, y: sy(t) + 3.5, "text-anchor": "end" }, grid); lab.textContent = yFmt(t);
  }
  const xticks = xLog ? logTicks(xs[0], xs[xs.length - 1]) : niceTicks(x0, x1, 6);
  for (const t of xticks) {
    const lab = el("text", { x: sx(t), y: H - m.b + 15, "text-anchor": "middle" }, grid); lab.textContent = xFmt(t);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css("--border"), "stroke-width": 1 }, grid);
  const xt = el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, svg); xt.textContent = xLabel;
  const yc = (m.t + H - m.b) / 2;
  const yt = el("text", { x: 12, y: yc, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 12 ${yc})` }, svg); yt.textContent = yLabel;

  for (const h of hlines) {
    if (h.y < y0 || h.y > y1) continue;
    el("line", { x1: m.l, x2: W - m.r, y1: sy(h.y), y2: sy(h.y), stroke: css("--text-muted"), "stroke-width": 1 }, svg);
    // put the label on the side away from the highlighted point; shorten on narrow charts
    const px = points.length ? sx(points[0].x) : W;
    const onLeft = px > (m.l + W - m.r) / 2;
    const t = el("text", { x: onLeft ? m.l + 6 : W - m.r - 4, y: sy(h.y) - 5, "text-anchor": onLeft ? "start" : "end", class: "annot" }, svg);
    t.textContent = W < 520 && h.short ? h.short : h.label;
  }
  for (const s of series) {
    // points outside the y-domain are not drawn (no flat clipped segments)
    let d = "", pen = false;
    xs.forEach((x, i) => {
      const v = s.ys[i];
      if (!Number.isFinite(v) || v > y1 || v < y0) { pen = false; return; }
      d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(v).toFixed(1)}`; pen = true;
    });
    if (s.area && d) el("path", { d: `${d}L${sx(xs[xs.length - 1])},${sy(Math.max(y0, 0))}L${sx(xs[0])},${sy(Math.max(y0, 0))}Z`, fill: css(s.wash) }, svg);
    el("path", { d, fill: "none", stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  }
  for (const p of points) {
    el("circle", { cx: sx(p.x), cy: sy(p.y), r: 5, fill: css(p.color), stroke: css("--surface-1"), "stroke-width": 2 }, svg);
    const right = sx(p.x) > W - 110, left = sx(p.x) < m.l + 60;
    const t = el("text", { x: sx(p.x) + (right ? -2 : left ? 2 : 0), y: sy(p.y) - 12,
                           "text-anchor": right ? "end" : left ? "start" : "middle", class: "annot" }, svg);
    t.textContent = p.label;
  }

  // hover layer: crosshair + one tooltip listing every series
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--text-muted"), "stroke-width": 1, visibility: "hidden" }, svg);
  const tip = document.createElement("div"); tip.className = "tooltip"; container.appendChild(tip);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", tabindex: 0 }, svg);
  const hide = () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); };
  const show = (clientX) => {
    const r = svg.getBoundingClientRect(), scale = W / r.width;
    const px = (clientX - r.left) * scale;
    let i = 0, best = Infinity;
    xs.forEach((x, k) => { const dd = Math.abs(sx(x) - px); if (dd < best) { best = dd; i = k; } });
    cross.setAttribute("x1", sx(xs[i])); cross.setAttribute("x2", sx(xs[i])); cross.setAttribute("visibility", "visible");
    tip.textContent = "";
    const head = document.createElement("div"); head.className = "head";
    head.textContent = tipX ? tipX(xs[i], i) : `${xLabel.split(" ")[0]} = ${xFmt(xs[i])}`;
    tip.appendChild(head);
    for (const s of series) {
      const row = document.createElement("div"); row.className = "row";
      const key = document.createElement("i"); key.className = "key"; key.style.borderColor = css(s.color);
      const b = document.createElement("b"); b.textContent = yFmt(s.ys[i]);
      const sp = document.createElement("span"); sp.textContent = s.name;
      row.append(key, b, sp); tip.appendChild(row);
    }
    tip.style.display = "block";
    const left = sx(xs[i]) / scale;
    tip.style.left = `${Math.max(0, Math.min(left + 12, r.width - tip.offsetWidth - 4))}px`;
    tip.style.top = `${m.t / scale}px`;
  };
  hit.addEventListener("pointermove", (e) => show(e.clientX));
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => { const r = svg.getBoundingClientRect(); show(r.left + r.width / 2); });
  hit.addEventListener("blur", hide);
}

/** Re-render on width changes and on light/dark switches. */
function onRelayout(rootEl, rerender) {
  let lastW = 0;
  new ResizeObserver((es) => {
    const w = Math.round(es[0].contentRect.width);
    if (w !== lastW) { lastW = w; rerender(); }
  }).observe(rootEl);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
}

function fillTable(tbody, rows) {
  tbody.replaceChildren(...rows.map(([k, v]) => {
    const tr = document.createElement("tr");
    const a = document.createElement("td"); a.textContent = k;
    const b = document.createElement("td"); b.className = "num"; b.textContent = v;
    tr.append(a, b); return tr;
  }));
}

globalObj.Charts = { $, fmt, css, lineChart, onRelayout, fillTable, tickFmt, gapText };
})(window);
