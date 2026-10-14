"""Smoke test: the desktop app builds every section and every tab without errors."""
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_all_sections_and_tabs_build(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from tapdual.live import UNIVERSE
    R = np.random.default_rng(0).normal(0.0002, 0.008, (700, len(UNIVERSE)))
    path = tmp_path / "prices.csv"
    pd.DataFrame(100 * np.cumprod(1 + R, 0), index=pd.bdate_range("2023-01-02", periods=700),
                 columns=list(UNIVERSE)).to_csv(path)
    monkeypatch.setenv("TAPDUAL_PRICES_CSV", str(path))   # the live tab must not go online in tests
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from tapdual.app.window import CHAPTERS, MainWindow
    w = MainWindow(); w.show()
    for row in range(len(CHAPTERS)):
        w.nav.setCurrentRow(row); app.processEvents()
        section = w.stack.widget(row).widget()
        if hasattr(section, "seg"):
            for i in range(section.stack.count()):
                section.seg.changed.emit(i); app.processEvents()
    w.close()
