"""
Central look of the Qt demo: one dark palette for all panels and one color per track (bb, label, selection).
"""
from typing import Tuple

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication

BACKGROUND = "#14161a"
PANEL = "#1c1f24"
PANEL_BORDER = "#2b2f36"
TEXT = "#e6e8eb"
TEXT_MUTED = "#8e96a1"
ACCENT = "#ffb020"  # active actions, "sees the camera"
INACTIVE = "#4a515c"  # inactive bars / objects

# distinguishable on dark backgrounds (based on the Okabe-Ito / Tableau palettes)
TRACK_COLORS = ["#4cc2ff", "#ff7a59", "#7bd88f", "#c792ea", "#ffd166", "#f78fb3", "#5eead4", "#a3be8c"]

STYLESHEET = f"""
QMainWindow, QWidget {{ background: {BACKGROUND}; color: {TEXT}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {PANEL_BORDER}; border-radius: 6px; margin: 4px;
             padding-top: 22px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left; left: 10px; top: 4px;
                    color: {TEXT_MUTED}; }}
QToolBar {{ background: {BACKGROUND}; border: none; spacing: 4px; padding: 2px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 3px 6px;
               color: {TEXT_MUTED}; }}
QToolButton:checked {{ background: {PANEL}; border-color: {PANEL_BORDER}; color: {TEXT}; }}
QToolButton:hover {{ border-color: {TEXT_MUTED}; }}
QMenuBar {{ background: {BACKGROUND}; color: {TEXT}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {PANEL_BORDER}; }}
QMenu {{ background: {PANEL}; color: {TEXT}; border: 1px solid {PANEL_BORDER}; }}
QStatusBar {{ background: {BACKGROUND}; color: {TEXT_MUTED}; border-top: 1px solid {PANEL_BORDER}; }}
QStatusBar QLabel {{ color: {TEXT_MUTED}; padding: 0 8px; }}
QLabel#img_view, QLabel#img_ehpi {{ background: {PANEL}; border: none; }}
"""


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, BACKGROUND), (QPalette.ColorRole.Base, PANEL),
                        (QPalette.ColorRole.AlternateBase, BACKGROUND), (QPalette.ColorRole.Button, PANEL),
                        (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.WindowText, TEXT),
                        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, PANEL_BORDER),
                        (QPalette.ColorRole.ToolTipBase, PANEL), (QPalette.ColorRole.ToolTipText, TEXT)):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


def track_color(uid: int) -> QColor:
    if uid is None or uid < 0:
        return QColor(TEXT_MUTED)
    return QColor(TRACK_COLORS[uid % len(TRACK_COLORS)])


def rgb_float(color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    c = QColor(color)
    return c.redF(), c.greenF(), c.blueF(), alpha
