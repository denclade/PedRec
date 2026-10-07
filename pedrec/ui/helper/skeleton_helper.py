import math

import numpy as np
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QPen, QPainter, QColor, QPolygonF, QBrush

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_LIMB_COLORS, SKELETON_PEDREC, SKELETON_PEDREC_JOINT_COLORS
from pedrec.utils.skeleton_helper import get_joint_score


def draw_skeleton(painter: QPainter, skeleton_orig: np.ndarray, min_joint_score: float = 0.3,
                  scale_factor: float = 1.0):
    skeleton = skeleton_orig.copy()
    skeleton[:, :2] /= scale_factor
    for idx, joint in enumerate(skeleton):
        if get_joint_score(joint) > min_joint_score:  # check score
            x_coord, y_coord = int(joint[0]), int(joint[1])
            color = SKELETON_PEDREC_JOINT_COLORS[idx]
            pen = QPen(QColor(color.r, color.g, color.b, 127), 10, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawEllipse(x_coord, y_coord, 2, 2)
    for idx, pair in enumerate(SKELETON_PEDREC):
        joint_a = skeleton[pair[0]]
        joint_b = skeleton[pair[1]]
        if (get_joint_score(joint_a) + get_joint_score(joint_b)) / 2 > min_joint_score:
            color = SKELETON_PEDREC_LIMB_COLORS[idx]
            pen = QPen(QColor(color.r, color.g, color.b, 255), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(int(joint_a[0]), int(joint_a[1]), int(joint_b[0]), int(joint_b[1]))


def draw_arrow_head(painter, line_start, line_end, color, line_width: int = 0):
    polygon = QPolygonF()
    rotation = math.degrees(math.atan2(line_start[1] - line_end[1], line_end[0] - line_start[0])) + 90
    points = ((line_end[0] + 5 * math.sin(math.radians(rotation)), line_end[1] + 5 * math.cos(math.radians(rotation))),
              (line_end[0] + 5 * math.sin(math.radians(rotation - 120)), line_end[1] + 5 * math.cos(math.radians(rotation - 120))),
              (line_end[0] + 5 * math.sin(math.radians(rotation + 120)), line_end[1] + 5 * math.cos(math.radians(rotation + 120))))
    pen = QPen(QColor(*color), line_width, Qt.SolidLine)
    brush = QBrush(QColor(*color))
    painter.setPen(pen)
    painter.setBrush(brush)
    for point in points:
        polygon.append(QPointF(point[0], point[1]))

    painter.drawPolygon(polygon)

def draw_arrow(painter, start, yaw, pitch, roll):
    x1 = 25 * (math.cos(yaw) * math.cos(roll))
    y1 = 25 * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw))

    x2 = 25 * (-math.cos(yaw) * math.sin(roll))
    y2 = 25 * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll))

    x3 = 25 * (math.sin(yaw))
    y3 = 25 * (-math.cos(yaw) * math.sin(pitch))

    # start = (100, 100)
    roll_end = (start[0] + int(x1), start[1] + int(y1))
    pitch_end = (start[0] + int(x2), start[1] + int(y2))
    yaw_end = (start[0] + int(x3), start[1] + int(y3))                     # set lineColor

    draw_line(painter, start, yaw_end, (255, 0, 0, 125))
    draw_arrow_head(painter, start, yaw_end, (255, 0, 0, 125))

    draw_line(painter, start, pitch_end, (0, 255, 0, 125))
    draw_arrow_head(painter, start, pitch_end, (255, 0, 0, 125))

    draw_line(painter, start, roll_end, (0, 0, 255, 255), 4)
    draw_arrow_head(painter, start, roll_end, (255, 0, 0, 125))


def draw_line(painter, start, end, color, line_width: int = 2):
    pen = QPen(QColor(*color), line_width, Qt.SolidLine)
    painter.setPen(pen)
    painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))


def draw_orientation(painter: QPainter, orientations: np.ndarray, nose_joint_2d: np.ndarray):
    theta = orientations[0] - (0.5 * math.pi)  # -90 - 90°
    phi = orientations[1]

    roll = 0
    pitch = 1 * math.pi - theta
    yaw = phi
    draw_arrow(painter, (nose_joint_2d[0], nose_joint_2d[1]), yaw, pitch, roll)

