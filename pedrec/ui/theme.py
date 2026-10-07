"""
Central look of the Qt demo: one light palette for all panels and one color per track (bb, label, selection).
"""
from typing import Tuple

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication

BACKGROUND = "#eef0f3"
PANEL = "#ffffff"
PANEL_BORDER = "#d5d9df"
TEXT = "#1f2328"
TEXT_MUTED = "#5f6773"
ACCENT = "#d9480f"  # recognized actions, "sees the camera"
INACTIVE = "#c5cbd3"  # inactive bars
LABEL_TEXT = "#ffffff"  # text on track colored labels

# distinguishable on light backgrounds and on video frames (Tableau 10)
TRACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b"]

STYLESHEET = f"""
QMainWindow {{ background: {BACKGROUND}; }}
QWidget {{ color: {TEXT}; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {PANEL_BORDER}; border-radius: 6px; margin: 4px;
             padding-top: 22px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left; left: 10px; top: 4px;
                    color: {TEXT_MUTED}; }}
QToolBar {{ background: {BACKGROUND}; border: none; spacing: 4px; padding: 2px 6px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 2px 4px;
                        color: {TEXT_MUTED}; }}
QToolBar QToolButton:checked {{ background: {PANEL}; border-color: {PANEL_BORDER}; color: {TEXT}; }}
QToolBar QToolButton:hover {{ border-color: {TEXT_MUTED}; }}
QMenuBar {{ background: {BACKGROUND}; color: {TEXT}; }}
QStatusBar {{ background: {BACKGROUND}; color: {TEXT_MUTED}; border-top: 1px solid {PANEL_BORDER}; }}
QStatusBar QLabel {{ color: {TEXT_MUTED}; padding: 0 8px; }}
QLabel#img_view, QLabel#img_ehpi {{ background: {PANEL}; border: none; }}
QSlider::groove:horizontal {{ height: 4px; background: {PANEL_BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {TRACK_COLORS[0]}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {PANEL}; border: 2px solid {TRACK_COLORS[0]}; width: 10px; height: 10px;
                              margin: -5px 0; border-radius: 7px; }}
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
