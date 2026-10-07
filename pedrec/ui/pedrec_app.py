import os
import sys
from typing import List

import cv2
import numpy as np
from qtpy.QtCore import Slot
from qtpy.QtGui import QImage, QPixmap, QFontDatabase
from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QStyle, QGroupBox, QDialog, QHBoxLayout
from qtpy import uic, QtCore

from pedrec.configs.app_config import AppConfig
from pedrec.models.human import Human
from pedrec.ui.models.pedrec_ui_config import PedRecUIConfig
from pedrec.ui.pedrec_worker import PedRecWorker
from pedrec.utils.skeleton_helper_3d import get_human_size_from_skeleton_3d

# Load icons
import pedrec.ui.qt_icon_resources  # noqa: F401

UI_DIR = os.path.dirname(os.path.abspath(__file__))


class PedRecApp(QMainWindow):
    def __init__(self, app: QApplication, worker: PedRecWorker, app_cfg: AppConfig):
        super().__init__()
        app.setStyle('Breeze')
        uic.loadUi(os.path.join(UI_DIR, "mainwindow.ui"), self)
        self.cfg_path = os.path.join(UI_DIR, ".ui_cfg.pkl")
        self.cfg = PedRecUIConfig()
        self.cfg.load(self.cfg_path)
        self.app_cfg = app_cfg
        self.play = True
        self.playback_action = None
        self.play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        self.selected_human_uid: int = None
        self.humans: List[Human] = None
        self.worker = worker
        self.frame_nr_label = QLabel("Frame: No data")
        self.fps_label = QLabel("FPS: No data")
        self.actions_label = QLabel("Actions: No data")
        self.human_score_label = QLabel("-")
        self.human_size_label = QLabel("-")
        self.actions_bar_chart_view.initialize_actions_chart(self.app_cfg.inference.action_list)

        self.init_menu(app)
        self.init_status_bar()
        self.init_img_view(app_cfg.inference.img_size)
        self.init_buttons()
        self.__clear_ehpi()
        # self.init_metadata_view()
        self.show()
        self.start_worker(app)

    def init_menu(self, app):
        menu_bar = self.menuBar()
        # File menu
        file_menu = menu_bar.addMenu('&File')
        exit_action = QAction(self.style().standardIcon(QStyle.SP_DialogCancelButton), '&Exit', self)
        exit_action.setShortcut('Esc')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(app.quit)
        file_menu.addAction(exit_action)
        # Settings menu
        # settings_menu = menu_bar.addMenu('&Settings')
        # show_object_bbs_toggle = QAction('&Show objects', self, checkable=True, checked=True)
        # show_object_bbs_toggle.chec
        # show_object_bbs_toggle.triggered(true)
        # show_object_bbs_toggle.setShortcut('o')
        # show_object_bbs_toggle.setStatusTip('Show object bounding boxes')
        # show_object_bbs_toggle.triggered.connect(self.toggle_show_objects)
        # settings_menu.addAction(show_object_bbs_toggle)

        # Control actions
        self.playback_action = QAction(self.pause_icon, '', self)
        self.playback_action.setShortcut('Space')
        self.playback_action.setStatusTip('Pause')
        self.playback_action.triggered.connect(self.toggle_worker_playback)
        menu_bar.addAction(self.playback_action)

    #######################################################################
    ############################ Buttons ##################################
    #######################################################################

    def init_buttons(self):
        # pose_2d
        self.action_toggle_pose_2d.setChecked(self.cfg.show_pose_2d)
        self.action_toggle_pose_2d.triggered.connect(
            lambda status: self.toggle_button(f'{self.cfg.show_pose_2d=}'.split('=')[0].split('.')[-1],
                                              status))
        
        # object_bb
        self.action_toggle_object_bb.setChecked(self.cfg.show_object_bb)
        self.action_toggle_object_bb.triggered.connect(
            lambda status: self.toggle_button(f'{self.cfg.show_object_bb=}'.split('=')[0].split('.')[-1],
                                              status))
        
        # human_bb
        self.action_toggle_human_bb.setChecked(self.cfg.show_human_bb)
        self.action_toggle_human_bb.triggered.connect(
            lambda status: self.toggle_button(f'{self.cfg.show_human_bb=}'.split('=')[0].split('.')[-1],
                                              status))
        
        # head_orientation_2d
        self.action_toggle_head_orientation_2d.setChecked(self.cfg.show_head_orientation_2d)
        self.action_toggle_head_orientation_2d.triggered.connect(
            lambda status: self.toggle_button(f'{self.cfg.show_head_orientation_2d=}'.split('=')[0].split('.')[-1],
                                              status))

        # body_orientation_2d
        self.action_toggle_body_orientation_2d.setChecked(self.cfg.show_body_orientation_2d)
        self.action_toggle_body_orientation_2d.triggered.connect(
            lambda status: self.toggle_button(f'{self.cfg.show_body_orientation_2d=}'.split('=')[0].split('.')[-1],
                                              status))
        
        # actions
        self.action_toggle_actions.setChecked(self.cfg.show_actions)
        self.action_toggle_actions.triggered.connect(
            lambda status: self.toggle_button(f'{self.cfg.show_actions=}'.split('=')[0].split('.')[-1],
                                              status))
        
        # sees_car
        self.action_toggle_sees_car.setChecked(self.cfg.show_sees_car_flag)
        self.action_toggle_sees_car.triggered.connect(
            lambda status: self.toggle_button(f'{self.cfg.show_sees_car_flag=}'.split('=')[0].split('.')[-1],
                                              status))


    # Button Actions

    def toggle_button(self, property_name: str, active: bool):
        setattr(self.cfg, property_name, active)
        self.cfg.save(self.cfg_path)

    def toggle_worker_playback(self):
        if self.play:
            self.playback_action.setStatusTip('Play')
            self.playback_action.setIcon(self.play_icon)
            self.worker.pause()
        else:
            self.playback_action.setStatusTip('Pause')
            self.playback_action.setIcon(self.pause_icon)
            self.worker.resume()
        self.play = not self.play

    def init_status_bar(self):
        self.statusBar().addWidget(self.frame_nr_label)
        self.statusBar().addWidget(self.fps_label)
        self.statusBar().addWidget(self.human_score_label)
        self.statusBar().addWidget(self.human_size_label)
        self.statusBar().addWidget(self.actions_label)

    def init_img_view(self, img_size):
        self.dialog = QDialog()
        self.hlayout = QHBoxLayout()
        self.img_view.img_size = img_size
        self.img_view.cfg = self.cfg
        self.img_view.human_selected.connect(self.set_selected_human_uid)

        # show img in seperat window
        # self.grid.removeWidget(self.groupBox_2)
        # self.hlayout.addWidget(self.img_view)
        # self.hlayout.setContentsMargins(0,0,0,0)
        # self.dialog.setLayout(self.hlayout)
        # # self.dialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        # self.dialog.show()

    # def init_img_ephi(self):
    #     self.img_ehpi.img_size = ImageSize(200, 100)

    def __update_ehpi(self, ehpi: np.ndarray):
        if ehpi is None:
            return
        # ehpi = np.transpose(np.copy(ehpi_normalized), (1, 0, 2))
        # ehpi *= 255
        # ehpi = ehpi.astype(np.uint8)
        ehpi = cv2.resize(ehpi, (640, 320), interpolation=cv2.INTER_NEAREST)
        # ehpi = cv2.cvtColor(ehpi, cv2.COLOR_BGR2RGB)
        h, w, ch = ehpi.shape
        bytes_per_line = ch * w
        qt_img = QImage(ehpi.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.img_ehpi.setPixmap(QPixmap.fromImage(qt_img))

    def __clear_ehpi(self):
        ehpi = np.zeros((320, 640, 3), dtype=np.uint8)
        h, w, ch = ehpi.shape
        bytes_per_line = ch * w
        qt_img = QImage(ehpi.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.img_ehpi.setPixmap(QPixmap.fromImage(qt_img))

    # Event Handler
    @Slot(int, np.ndarray, list, np.ndarray, int)
    def update_worker_data(self, frame_nr: int, img: np.ndarray, humans: List[Human], object_bbs: np.ndarray, fps: int):
        # frame nr label:
        self.frame_nr_label.setText(f"Frame: {frame_nr:05}")
        self.humans = humans

        # img_view
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.img_view.setPixmap(QPixmap.fromImage(qt_img))
        self.img_view.set_humans(humans)
        self.img_view.set_object_bbs(object_bbs)

        # Skeleton 2.5d
        self.update_selected_human()

        # fps_label
        self.fps_label.setText(f"FPS: {fps:03}")
        self.worker.frame_consumed()

    @Slot(int)
    def set_selected_human_uid(self, selected_human_uid: int):
        self.selected_human_uid = selected_human_uid
        self.update_selected_human()
        self.img_view.update()

    def __clear_selected_human(self):
        self.__clear_ehpi()
        self.body_orientation_view.clear()
        self.human_score_label.setText("Score: -%")
        self.human_size_label.setText("Size: -mm")
        self.actions_label.setText("Actions: -")
        self.actions_bar_chart_view.clear_data()
        # self.human_score_label.setText("-")

    def update_selected_human(self):
        self.skeleton_view.clear()
        if self.selected_human_uid == -1:
            self.__clear_selected_human()
            return

        selected_human = next((human for human in self.humans if human.uid == self.selected_human_uid), None)
        if selected_human is None:
            self.__clear_selected_human()
            return
        self.__update_ehpi(selected_human.ehpi)
        self.__update_metadata(selected_human)
        self.__update_skeleton_3d(selected_human)
        self.__update_orientation(selected_human)
        self.actions_bar_chart_view.set_actions(selected_human.action_probabilities)

    def __update_metadata(self, selected_human: Human):
        self.human_score_label.setText(f"Score: {int(selected_human.score * 100):03}%")
        self.actions_label.setText(f"Actions: {', '.join([a.name for a in selected_human.actions])}")
        size = get_human_size_from_skeleton_3d(selected_human.skeleton_3d)
        self.human_size_label.setText(f"Size: {int(size)}mm" if size > 0 else "Size: -mm")

    def __update_skeleton_3d(self, selected_human: Human):
        skeleton = selected_human.skeleton_3d.copy()
        self.skeleton_view.add_skeleton(skeleton)

    def __update_orientation(self, selected_human: Human):
        orientation = selected_human.orientation.copy()
        self.body_orientation_view.set_orientation(orientation[0, 0], orientation[0, 1], orientation[1, 0],
                                                   orientation[1, 1])

    def start_worker(self, app):
        self.worker.data_updated.connect(self.update_worker_data)
        self.worker.start()
        app.aboutToQuit.connect(self.worker.stop)
