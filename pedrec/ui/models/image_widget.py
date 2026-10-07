from typing import List

import numpy as np
from qtpy import QtGui, QtSvg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QWidget
from qtpy import QtCore

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS, SKELETON_PEDREC_JOINT
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.ui.helper.bb_helper import draw_bb
from pedrec.ui.helper.skeleton_helper import draw_skeleton, draw_orientation
from pedrec.ui.models.pedrec_ui_config import PedRecUIConfig
from pedrec.utils.bb_helper import get_img_coordinates_from_bb, get_human_bb_from_joints


class ImageWidget(QLabel):
    human_selected = Signal(int)

    def __init__(self, parent: QWidget = None, img_size: ImageSize = None, text: str = None):
        super().__init__(parent)
        self.img_size = img_size  # set by PedRecApp.init_img_view
        if text is not None:
            self.setText(text)
        self.setScaledContents(True)
        self.setMinimumSize(320, 180)
        self.setMaximumSize(1920, 1080)
        self.selected_human_uid: int = None
        self.humans: List[Human] = []
        self.object_bbs: np.ndarray = None
        self.scale_factor = 1
        self.cfg: PedRecUIConfig = None

    def set_humans(self, humans: List[Human]):
        self.humans = humans

        # Initialize selection
        if self.selected_human_uid is None and len(humans) > 0:
            # First recognition to have one initial selection
            self.selected_human_uid = self.humans[0].uid
        self.human_selected.emit(self.selected_human_uid)

    def set_object_bbs(self, object_bbs: np.ndarray):
        self.object_bbs = object_bbs.copy()
        # self.object_bbs[:, :2] /= self.scale_factor

    def paintEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        with QtGui.QPainter(self) as painter:
            label_size = self.size()
            start_point = QtCore.QPoint(0, 0)
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            self.scale_factor = self.img_size.width / scaled_pixmap.size().width()
            painter.drawPixmap(start_point, scaled_pixmap)

            if len(self.humans) > 0:
                for human in self.humans:
                    selected = False
                    if self.selected_human_uid is not None and self.selected_human_uid == human.uid:
                        selected = True
                    if self.cfg.show_pose_2d:
                        draw_skeleton(painter, human.skeleton_2d, scale_factor=self.scale_factor)
                    if self.cfg.show_head_orientation_2d:
                        draw_orientation(painter, human.orientation[1],
                                         human.skeleton_2d[SKELETON_PEDREC_JOINT.nose.value] / self.scale_factor)
                    if self.cfg.show_body_orientation_2d:
                        draw_orientation(painter, human.orientation[0],
                                         human.skeleton_2d[SKELETON_PEDREC_JOINT.hip_center.value] / self.scale_factor)
                    # draw_eye(painter, human.orientation[1])
                    if self.cfg.show_human_bb:
                        bb = get_human_bb_from_joints(human.skeleton_2d,
                                                      max_x_val=self.img_size.width,
                                                      max_y_val=self.img_size.height,
                                                      confidence=human.score,
                                                      class_idx=0)
                        draw_bb(painter, bb, img_size=self.img_size, cfg=self.cfg, orientations=human.orientation[1], title=str(human.uid), selected=selected, scale_factor=self.scale_factor, alpha=125,
                                action_list=human.actions)

            if self.cfg.show_object_bb and self.object_bbs is not None and self.object_bbs.shape[0] > 0:
                for i in range(self.object_bbs.shape[0]):
                    object_bb = self.object_bbs[i, :]
                    draw_bb(painter, object_bb, img_size=self.img_size, cfg=self.cfg, scale_factor=self.scale_factor)

    def mousePressEvent(self, event):
        click_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        click_x = click_pos.x()
        click_y = click_pos.y()
        selected_human_uid = -1
        for human in self.humans:
            tl_x, tl_y, br_x, br_y = get_img_coordinates_from_bb(human.bb)
            tl_x /= self.scale_factor
            tl_y /= self.scale_factor
            br_x /= self.scale_factor
            br_y /= self.scale_factor
            if tl_x <= click_x <= br_x and tl_y <= click_y <= br_y:
                selected_human_uid = human.uid
                break
        self.selected_human_uid = selected_human_uid
        self.human_selected.emit(selected_human_uid)
