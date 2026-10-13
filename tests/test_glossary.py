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
        used |= set(re.findall(r"\[\[([a-z0-9\-]+)", f.read_text()))
    used.discard("term")  # the placeholder in docstrings
    assert used <= set(ENTRIES), used - set(ENTRIES)
