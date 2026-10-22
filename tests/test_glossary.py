"""Every glossary formula must typeset, and every [[link]] in the app must point to an entry."""
import re
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from tapdual.app.glossary import ENTRIES, formula_png  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "tapdual" / "app"


@pytest.mark.parametrize("key", sorted(ENTRIES))
def test_formulas_render(key):
    for kind, body in ENTRIES[key][1]:
        if kind == "eq":
            assert formula_png(body)[:4] == b"\x89PNG"


def test_links_resolve():
    used = set()
    for f in APP.glob("*.py"):
        used |= set(re.findall(r"\[\[([a-z0-9\-]+)(?:\|[^\]]+)?\]\]", f.read_text()))  # complete [[term|text]] links
    used.discard("term")  # the placeholder in docstrings
    assert used <= set(ENTRIES), used - set(ENTRIES)


def test_demos_compute_the_identities():
    import math
    import os
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])  # keep a Qt application alive
    from tapdual.app.demos import EnergyDemo, EntropyDemo
    d = EntropyDemo()
    d.set_weights([1, 1, 1, 0, 0, 0])                      # half the states
    assert d.v_ent.text() == f"{math.log(3):.3f}" and d.v_kl.text() == f"{math.log(2):.3f}"
    e = EnergyDemo()
    e.set_energies([-2, 0, 0, 0, 0, 0]); e.beta.set_value(0.0)
    assert all(abs(p - 1 / 6) < 1e-12 for p in e.prob.values)     # β = 0: uniform
    e.beta.set_value(4.0)
    assert e.prob.values[0] > 0.99                                # large β: ground state
