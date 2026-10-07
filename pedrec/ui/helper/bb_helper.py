import math
import os
from typing import List

import numpy as np
from qtpy import QtSvg
from qtpy.QtCore import Qt, QPoint, QPointF, QRect, QRectF
from qtpy.QtGui import QPen, QPainter, QColor, QFont, QBrush

from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.constants.class_mappings import COCO_CLASSES
from pedrec.models.constants.color_palettes import DETECTION_COLOR_PALETTE
from pedrec.models.data_structures import ImageSize
from pedrec.ui.models.pedrec_ui_config import PedRecUIConfig
from pedrec.utils.bb_helper import get_img_coordinates_from_bb, get_bb_class_idx

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def draw_bb(painter: QPainter,
            bb: np.ndarray,
            img_size: ImageSize,
            cfg: PedRecUIConfig,
            orientations: np.ndarray = None,
            title: str = None,
            selected: bool = False,
            scale_factor: float = 1.0,
            alpha: int = 255,
            action_list: List[ACTION] = None):
    bb_tl_x, bb_tl_y, bb_br_x, bb_br_y = get_img_coordinates_from_bb(bb)
    bb_tl_y = max(bb_tl_y - 40, 0)
    # Qt only accepts integer pixel coordinates for QPoint / QRect
    bb_tl_x = int(bb_tl_x / scale_factor)
    bb_tl_y = int(bb_tl_y / scale_factor)
    bb_br_x = int(bb_br_x / scale_factor)
    bb_br_y = int(bb_br_y / scale_factor)
    # cls_conf = get_bb_score(bb)

    if cfg.show_human_bb:
        cls_id = get_bb_class_idx(bb)
        color = DETECTION_COLOR_PALETTE[cls_id]
        pen = QPen(QColor(color.r, color.g, color.b, alpha), 2, Qt.SolidLine)

        painter.setPen(pen)
        point_tl = QPoint(bb_tl_x, bb_tl_y)
        rect = QRect(point_tl, QPoint(bb_br_x, bb_br_y))

        transparent = QBrush(QColor(color.r, color.g, color.b, 0))
        painter.setBrush(transparent)
        if selected:
            painter.setBrush(QBrush(QColor(color.r, color.g, color.b, 100)))
        # painter.setBrush(brush)
        painter.drawRect(rect)
        painter.setBrush(transparent)

        painter.setPen(QColor(255, 255, 255, alpha))
        painter.setFont(QFont('Decorative', 10))
        text_rect = QRect(QPoint(bb_tl_x, bb_tl_y + 20), QPoint(bb_br_x, bb_tl_y))
        painter.fillRect(text_rect, QColor(color.r, color.g, color.b, alpha))
        box_title = COCO_CLASSES[cls_id]
        if title is not None:
            box_title = title

        painter.drawText(text_rect, Qt.AlignCenter, box_title)

    if cfg.show_sees_car_flag and orientations is not None:
        theta = math.degrees(orientations[0])
        phi = math.degrees(orientations[1])

        if 208 <= phi <= 332 and 40 <= theta <= 160:
            draw_eye_symbol(painter, bb_tl_x, bb_tl_y, img_size)

    if cfg.show_actions and action_list is not None:
        draw_actions(painter, bb_tl_x, bb_tl_y, bb_br_x, bb_br_y, img_size, action_list)


def draw_eye_symbol(painter: QPainter, bb_tl_x: float, bb_tl_y: float, img_size: ImageSize):
    point_tl = QPointF(max(bb_tl_x - 10, 0), max(bb_tl_y - 10, 0))
    point_br = QPointF(min(bb_tl_x + 10, img_size.width), min(bb_tl_y + 10, img_size.height))
    rect = QRectF(point_tl, point_br)
    renderer = QtSvg.QSvgRenderer(os.path.join(UI_DIR, 'eye.svg'))
    renderer.render(painter, rect)

icon_actions = [ACTION.WALK, ACTION.JOG, ACTION.STAND, ACTION.SIT]


def draw_actions(painter: QPainter,
                 bb_tl_x: float,
                 bb_tl_y: float,
                 bb_br_x: float,
                 bb_br_y: float,
                 img_size: ImageSize,
                 action_list: List[ACTION]):
    point_tl = QPointF(max(bb_br_x, 0), max(bb_br_y - 25, 0))
    point_br = QPointF(min(bb_br_x + 15, img_size.width), min(bb_br_y, img_size.height))
    rect = QRectF(point_tl, point_br)
    if ACTION.WALK in action_list:
        renderer = QtSvg.QSvgRenderer(os.path.join(UI_DIR, 'action_walk.svg'))
        renderer.render(painter, rect)
    elif ACTION.JOG in action_list:
        renderer = QtSvg.QSvgRenderer(os.path.join(UI_DIR, 'action_jog.svg'))
        renderer.render(painter, rect)
    elif ACTION.STAND in action_list:
        renderer = QtSvg.QSvgRenderer(os.path.join(UI_DIR, 'action_stand.svg'))
        renderer.render(painter, rect)
    elif ACTION.SIT in action_list:
        renderer = QtSvg.QSvgRenderer(os.path.join(UI_DIR, 'action_sit.svg'))
        renderer.render(painter, rect)

    point_a = QPoint(int(bb_br_x), int(bb_tl_y))
    point_b = QPoint(int(bb_br_x) + 150, int(bb_tl_y) + 10)
    for action in action_list:
        if action in icon_actions:
            continue
        # print("JOOO")
        painter.setPen(QColor(0, 0, 0, 255))
        painter.setFont(QFont('Decorative', 10))
        text_rect = QRect(point_a, point_b)
        painter.fillRect(text_rect, QColor(255, 255, 255, 255))
        painter.drawText(text_rect, Qt.AlignCenter, action.name)
        point_a.setY(point_a.y() + 12)
        point_b.setY(point_b.y() + 12)

