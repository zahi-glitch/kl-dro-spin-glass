"""Colours, fonts and the Qt style sheet: one quiet light theme, one accent colour."""
from __future__ import annotations

import pyqtgraph as pg

BG = "#f6f5f2"          # window
CARD = "#ffffff"        # panels
BORDER = "#e4e2dc"
GRID = "#ebe9e4"
INK = "#14161a"         # primary text
INK_2 = "#55544f"       # secondary text
INK_3 = "#8a8983"       # muted text, axes
ACCENT = "#2a78d6"      # the one accent (blue)
ACCENT_SOFT = "#e6effb"
ACCENT_2 = "#eb6834"    # second series only (orange)
SPIN_UP = "#1d2733"     # +1 cell
SPIN_DOWN = "#e9edf2"   # -1 cell

STYLE = f"""
QMainWindow, QWidget#page {{ background: {BG}; }}
QWidget {{ color: {INK}; font-size: 14px; }}
QLabel#title {{ font-size: 26px; font-weight: 600; }}
QLabel#eyebrow {{ font-size: 11px; letter-spacing: 1px; color: {INK_3}; }}
QLabel#lead {{ font-size: 15px; color: {INK_2}; }}
QLabel#h2 {{ font-size: 16px; font-weight: 600; }}
QLabel#body {{ color: {INK_2}; }}
QLabel#small {{ font-size: 12px; color: {INK_3}; }}
QLabel#value {{ font-size: 22px; font-weight: 600; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#practice {{ background: {ACCENT_SOFT}; border: 1px solid #cddff5; border-radius: 12px; }}
QLabel#tag {{ font-size: 11px; letter-spacing: 1px; color: {ACCENT}; font-weight: 600; }}
QListWidget#nav {{
  background: {BG}; border: none; outline: none; padding: 8px 4px; font-size: 14px;
}}
QListWidget#nav::item {{ padding: 9px 12px; border-radius: 8px; color: {INK_2}; margin: 1px 0; }}
QListWidget#nav::item:selected {{ background: {CARD}; color: {INK}; border: 1px solid {BORDER}; }}
QListWidget#nav::item:disabled {{ color: #b8b6b0; }}
QPushButton {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 14px; color: {INK};
}}
QPushButton:hover {{ border-color: {INK_3}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border-color: {ACCENT}; font-weight: 600; }}
QPushButton#seg {{ border-radius: 16px; padding: 6px 16px; color: {INK_2}; }}
QPushButton#seg:checked {{ background: {INK}; color: white; border-color: {INK}; }}
QPushButton#deeper {{ border: none; background: transparent; color: {ACCENT}; padding: 4px 0; text-align: left; }}
QSlider::groove:horizontal {{ height: 4px; background: {GRID}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
  background: white; border: 2px solid {ACCENT}; width: 14px; height: 14px; margin: -7px 0; border-radius: 9px;
}}
QScrollArea {{ border: none; background: {BG}; }}
"""


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True, background=CARD, foreground=INK_3)


def style_plot(plot: pg.PlotItem, x_label: str = "", y_label: str = "") -> None:
    """Hairline axes, light grid, muted labels."""
    plot.showGrid(x=False, y=True, alpha=0.18)
    for name in ("left", "bottom"):
        ax = plot.getAxis(name)
        ax.setPen(pg.mkPen(BORDER))
        ax.setTextPen(pg.mkPen(INK_3))
        ax.enableAutoSIPrefix(False)
    if x_label:
        plot.setLabel("bottom", x_label, color=INK_2)
    if y_label:
        plot.setLabel("left", y_label, color=INK_2)
    plot.getViewBox().setMouseEnabled(x=False, y=False)
    plot.hideButtons()
