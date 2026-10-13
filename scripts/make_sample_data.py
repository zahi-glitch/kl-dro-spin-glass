"""Generate the synthetic daily-return sample used by the examples and the dashboard.

Three years (756 trading days) of fat-tailed daily returns, in percent:
Student-t with 4 degrees of freedom, daily volatility 1.2%, drift 0.04%.
Seeded, so the file is reproducible. Writes
  src/tapdual/data/synthetic_returns.csv   (used by Python)
  docs/sample_returns.js                   (used by the dashboard)
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261013)
df, vol, drift, n = 4, 1.2, 0.04, 756
r = drift + vol * rng.standard_t(df, n) / np.sqrt(df / (df - 2))
r = np.round(r, 4)

csv = ROOT / "src" / "tapdual" / "data" / "synthetic_returns.csv"
csv.write_text("return_pct\n" + "\n".join(f"{x:.4f}" for x in r) + "\n")
js = ROOT / "docs" / "sample_returns.js"
js.write_text("// Synthetic daily returns in % (scripts/make_sample_data.py)\n"
              "window.SAMPLE_RETURNS = [" + ",".join(f"{x:.4f}" for x in r) + "];\n")
print(f"{n} returns: mean {r.mean():.4f}%, std {r.std():.4f}%, min {r.min():.2f}%, max {r.max():.2f}%")
