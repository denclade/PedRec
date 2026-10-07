from typing import List, Optional

import numpy as np
from qtpy import QtGui
from qtpy.QtCore import Qt, Signal, QPointF
from qtpy.QtWidgets import QLabel, QWidget

from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.ui import theme
from pedrec.ui.helper.bb_helper import draw_human_box, draw_human_labels, draw_object, sees_camera
from pedrec.ui.helper.skeleton_helper import draw_skeleton, draw_orientation
from pedrec.ui.models.pedrec_ui_config import PedRecUIConfig
from pedrec.utils.bb_helper import get_img_coordinates_from_bb, get_human_bb_from_joints


class ImageWidget(QLabel):
    """Video frame (centered, aspect ratio kept) with the detections drawn on top."""
    human_selected = Signal(int)

    def __init__(self, parent: QWidget = None, img_size: ImageSize = None, text: str = None):
        super().__init__(parent)
        self.img_size = img_size  # set by PedRecApp.init_img_view
        if text is not None:
            self.setText(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.selected_human_uid: Optional[int] = None
        self.humans: List[Human] = []
        self.object_bbs: np.ndarray = None
        self.action_list: Optional[List[ACTION]] = None  # set by PedRecApp
        self.scale_factor = 1
        self.offset = QPointF(0, 0)
        self.cfg: PedRecUIConfig = None
        self._scaled_key = None
        self._scaled_pixmap = None

    def set_humans(self, humans: List[Human]):
        self.humans = humans
        # initial selection: the first recognized human (only signalled when the selection changes)
        if self.selected_human_uid is None and len(humans) > 0:
            self.selected_human_uid = self.humans[0].uid
            self.human_selected.emit(self.selected_human_uid)

    def set_object_bbs(self, object_bbs: np.ndarray):
        self.object_bbs = object_bbs.copy()

    def _scaled(self, pixmap: QtGui.QPixmap) -> QtGui.QPixmap:
        key = (pixmap.cacheKey(), self.width(), self.height())
        if self._scaled_key != key:  # scale every frame only once
            self._scaled_key = key
            self._scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
        return self._scaled_pixmap

    def paintEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return super().paintEvent(event)
        with QtGui.QPainter(self) as painter:
            painter.fillRect(self.rect(), QtGui.QColor(theme.PANEL))
            scaled = self._scaled(pixmap)
            self.offset = QPointF((self.width() - scaled.width()) / 2, (self.height() - scaled.height()) / 2)
            self.scale_factor = self.img_size.width / scaled.width()
            painter.translate(self.offset)
            painter.drawPixmap(0, 0, scaled)
            if self.cfg.show_object_bb and self.object_bbs is not None and self.object_bbs.shape[0] > 0:
                for object_bb in self.object_bbs:
                    draw_object(painter, object_bb, self.scale_factor)
            for human in self.humans:
                self._draw_human(painter, human)

    def _draw_human(self, painter: QtGui.QPainter, human: Human):
        selected = self.selected_human_uid is not None and self.selected_human_uid == human.uid
        bb = get_human_bb_from_joints(human.skeleton_2d, max_x_val=self.img_size.width,
                                      max_y_val=self.img_size.height, confidence=human.score, class_idx=0)
        color = theme.track_color(human.uid)
        if self.cfg.show_human_bb:
            draw_human_box(painter, bb, human.uid, selected, self.scale_factor)
        if self.cfg.show_pose_2d:
            draw_skeleton(painter, human.skeleton_2d, scale_factor=self.scale_factor)
        if human.orientation is not None:
            if self.cfg.show_body_orientation_2d:
                draw_orientation(painter, human.orientation[0],
                                 human.skeleton_2d[SKELETON_PEDREC_JOINT.hip_center.value] / self.scale_factor,
                                 color, radius=22)
            if self.cfg.show_head_orientation_2d:
                draw_orientation(painter, human.orientation[1],
                                 human.skeleton_2d[SKELETON_PEDREC_JOINT.nose.value] / self.scale_factor,
                                 QtGui.QColor(theme.TEXT), radius=12)
        if self.cfg.show_human_bb or self.cfg.show_actions:
            draw_human_labels(painter, bb, human.uid, human.score, self.scale_factor,
                              actions=human.actions if self.cfg.show_actions else None,
                              action_probabilities=human.action_probabilities, action_list=self.action_list,
                              sees_camera_icon=self.cfg.show_sees_car_flag and human.orientation is not None
                              and sees_camera(human.orientation[1]))

    def mousePressEvent(self, event):
        click_pos = event.position() if hasattr(event, 'position') else QPointF(event.pos())
        click_x = (click_pos.x() - self.offset.x()) * self.scale_factor
        click_y = (click_pos.y() - self.offset.y()) * self.scale_factor
        selected_human_uid = -1
        for human in self.humans:
            tl_x, tl_y, br_x, br_y = get_img_coordinates_from_bb(human.bb)
            if tl_x <= click_x <= br_x and tl_y <= click_y <= br_y:
                selected_human_uid = human.uid
                break
        self.selected_human_uid = selected_human_uid
        self.human_selected.emit(selected_human_uid)
