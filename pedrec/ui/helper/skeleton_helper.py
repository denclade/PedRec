import math

import numpy as np
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QPen, QPainter, QColor, QPolygonF, QBrush

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_LIMB_COLORS, SKELETON_PEDREC, SKELETON_PEDREC_JOINT_COLORS
from pedrec.utils.skeleton_helper import get_joint_score


def draw_skeleton(painter: QPainter, skeleton_orig: np.ndarray, min_joint_score: float = 0.3,
                  scale_factor: float = 1.0):
    """Limbs in the left / right colors of the skeleton, joints as small dots with a dark outline."""
    skeleton = skeleton_orig[:, :3].copy()
    skeleton[:, :2] /= scale_factor
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for idx, (a, b) in enumerate(SKELETON_PEDREC):
        joint_a, joint_b = skeleton[a], skeleton[b]
        if (get_joint_score(joint_a) + get_joint_score(joint_b)) / 2 > min_joint_score:
            color = SKELETON_PEDREC_LIMB_COLORS[idx]
            painter.setPen(QPen(QColor(color.r, color.g, color.b, 230), 2.5, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(joint_a[0], joint_a[1]), QPointF(joint_b[0], joint_b[1]))
    painter.setPen(QPen(QColor(20, 22, 26, 200), 1))
    for idx, joint in enumerate(skeleton):
        if get_joint_score(joint) > min_joint_score:
            color = SKELETON_PEDREC_JOINT_COLORS[idx]
            painter.setBrush(QBrush(QColor(color.r, color.g, color.b, 255)))
            painter.drawEllipse(QPointF(joint[0], joint[1]), 3, 3)


def orientation_direction(phi: float):
    """
    Image direction of a ground plane orientation phi (radians): 0 = right, 90 deg = away from the camera (up),
    270 deg = towards the camera (down); the ellipse is flattened as seen by a camera looking slightly down.
    """
    return math.cos(phi), -math.sin(phi) * 0.45


def draw_orientation(painter: QPainter, orientation: np.ndarray, center: np.ndarray, color: QColor,
                     radius: float = 18):
    """Orientation (phi) as an arrow on a flattened ground ellipse around ``center`` (widget pixels)."""
    phi = float(orientation[1])
    cx, cy = float(center[0]), float(center[1])
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    ellipse_color = QColor(color)
    ellipse_color.setAlpha(110)
    painter.setPen(QPen(ellipse_color, 1.2))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPointF(cx, cy), radius, radius * 0.45)
    dx, dy = orientation_direction(phi)
    tip = QPointF(cx + dx * radius * 1.35, cy + dy * radius * 1.35)
    painter.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(QPointF(cx, cy), tip)
    angle = math.atan2(tip.y() - cy, tip.x() - cx)
    head = QPolygonF([tip,
                      QPointF(tip.x() - 7 * math.cos(angle - 0.45), tip.y() - 7 * math.sin(angle - 0.45)),
                      QPointF(tip.x() - 7 * math.cos(angle + 0.45), tip.y() - 7 * math.sin(angle + 0.45))])
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(color))
    painter.drawPolygon(head)
