"""Smoke test: the desktop app builds every section and every tab without errors."""
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_all_sections_and_tabs_build():
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
