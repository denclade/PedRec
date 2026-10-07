"""Overlay of the detections in the image view (QPainter, coordinates already in widget pixels)."""
import math
import os
from typing import List, Optional

import numpy as np
from qtpy import QtSvg
from qtpy.QtCore import Qt, QRectF
from qtpy.QtGui import QPen, QPainter, QColor, QFont, QFontMetrics, QBrush

from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.constants.class_mappings import COCO_CLASSES
from pedrec.ui import theme
from pedrec.utils.bb_helper import get_img_coordinates_from_bb, get_bb_class_idx

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EYE_RENDERER = None
LABEL_FONT = QFont("Sans Serif", 9, QFont.Weight.DemiBold)
ACTION_FONT = QFont("Sans Serif", 8)
MAX_ACTION_LABELS = 4


def sees_camera(head_orientation: np.ndarray) -> bool:
    """Head turned towards the camera (theta / phi in radians, as the original "sees car" flag)."""
    theta, phi = math.degrees(head_orientation[0]), math.degrees(head_orientation[1])
    return 208 <= phi <= 332 and 40 <= theta <= 160


def _eye_renderer():
    global _EYE_RENDERER
    if _EYE_RENDERER is None:
        _EYE_RENDERER = QtSvg.QSvgRenderer(os.path.join(UI_DIR, "eye.svg"))
    return _EYE_RENDERER


def _scaled_rect(bb: np.ndarray, scale_factor: float) -> QRectF:
    tl_x, tl_y, br_x, br_y = get_img_coordinates_from_bb(bb)
    return QRectF(tl_x / scale_factor, tl_y / scale_factor, (br_x - tl_x) / scale_factor,
                  (br_y - tl_y) / scale_factor)


def _chip(painter: QPainter, x: float, y: float, text: str, font: QFont, background: QColor, foreground: QColor,
          icon: bool = False) -> QRectF:
    """Rounded text label with its bottom left corner at (x, y). Returns its rectangle."""
    metrics = QFontMetrics(font)
    height = metrics.height() + 4
    icon_size = height - 4 if icon else 0
    width = metrics.horizontalAdvance(text) + 10 + (icon_size + 4 if icon else 0)
    rect = QRectF(x, y - height, width, height)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(background))
    painter.drawRoundedRect(rect, 3, 3)
    painter.setFont(font)
    painter.setPen(foreground)
    text_rect = rect.adjusted(5, 0, -5 - (icon_size + 4 if icon else 0), 0)
    painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
    if icon:
        _eye_renderer().render(painter, QRectF(rect.right() - icon_size - 4, rect.top() + 2, icon_size, icon_size))
    return rect


def draw_human_box(painter: QPainter, bb: np.ndarray, uid: int, selected: bool, scale_factor: float):
    """Track colored bb, filled when selected (drawn below the skeleton)."""
    color = theme.track_color(uid)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    fill = QColor(color)
    fill.setAlpha(40 if selected else 0)
    painter.setBrush(QBrush(fill))
    painter.setPen(QPen(color, 3 if selected else 1.5))
    painter.drawRoundedRect(_scaled_rect(bb, scale_factor), 3, 3)


def draw_human_labels(painter: QPainter, bb: np.ndarray, uid: int, score: float, scale_factor: float,
                      actions: Optional[List[ACTION]] = None, action_probabilities: Optional[np.ndarray] = None,
                      action_list: Optional[List[ACTION]] = None, sees_camera_icon: bool = False):
    """Label above the bb (track id, score, "sees the camera" icon) and the recognized actions below it."""
    color = theme.track_color(uid)
    rect = _scaled_rect(bb, scale_factor)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    label = f"#{uid}  {int(round(score * 100))}%" if uid is not None and uid >= 0 else f"{int(round(score * 100))}%"
    background = QColor(color)
    background.setAlpha(230)
    _chip(painter, rect.left(), rect.top() - 2, label, LABEL_FONT, background, QColor(theme.LABEL_TEXT),
          sees_camera_icon)
    if actions:
        y = rect.bottom() + 2
        line_height = QFontMetrics(ACTION_FONT).height() + 4
        shown = actions[:MAX_ACTION_LABELS]
        for action in shown:
            text = action.name.lower().replace("_", " ")
            if action_probabilities is not None and action_list is not None and action in action_list:
                text += f"  {action_probabilities[action_list.index(action)]:.2f}"
            chip_bg = QColor(theme.PANEL)
            chip_bg.setAlpha(215)
            y = _chip(painter, rect.left(), y + line_height, text, ACTION_FONT, chip_bg, QColor(theme.ACCENT)).bottom() + 2
        if len(actions) > len(shown):  # all of them are listed in the action chart
            chip_bg = QColor(theme.PANEL)
            chip_bg.setAlpha(215)
            _chip(painter, rect.left(), y + line_height, f"+{len(actions) - len(shown)} more", ACTION_FONT, chip_bg,
                  QColor(theme.TEXT_MUTED))


def draw_object(painter: QPainter, bb: np.ndarray, scale_factor: float):
    """Other detected objects: thin neutral box with the class name."""
    rect = _scaled_rect(bb, scale_factor)
    color = QColor(theme.TEXT_MUTED)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
    painter.drawRect(rect)
    class_idx = get_bb_class_idx(bb)
    name = COCO_CLASSES[class_idx] if 0 <= class_idx < len(COCO_CLASSES) else str(class_idx)
    background = QColor(theme.PANEL)
    background.setAlpha(200)
    _chip(painter, rect.left(), rect.top() - 1, name, ACTION_FONT, background, color)
