"""Start the desktop app:  python -m tapdual.app"""
import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The app needs PySide6 and pyqtgraph. Install them with:\n"
              '    pip install -e ".[app]"')
        return 1
    from . import theme
    from .window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Free energy and robust optimisation")
    theme.configure_pyqtgraph()
    app.setStyleSheet(theme.STYLE)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
