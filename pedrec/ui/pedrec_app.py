import os
from typing import List

import numpy as np
from qtpy.QtCore import Slot, Qt
from qtpy.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QStyle, QDialog, QHBoxLayout
from qtpy import uic

from pedrec.configs.app_config import AppConfig
from pedrec.models.human import Human
from pedrec.ui import theme
from pedrec.ui.models.pedrec_ui_config import PedRecUIConfig
from pedrec.ui.models.player_bar import PlayerBar
from pedrec.ui.pedrec_worker import PedRecWorker
from pedrec.utils.skeleton_helper_3d import get_human_size_from_skeleton_3d

# Load icons
import pedrec.ui.qt_icon_resources  # noqa: F401

UI_DIR = os.path.dirname(os.path.abspath(__file__))


class PedRecApp(QMainWindow):
    def __init__(self, app: QApplication, worker: PedRecWorker, app_cfg: AppConfig):
        super().__init__()
        theme.apply_theme(app)
        uic.loadUi(os.path.join(UI_DIR, "mainwindow.ui"), self)
        self.cfg_path = os.path.join(UI_DIR, ".ui_cfg.pkl")
        self.cfg = PedRecUIConfig()
        self.cfg.load(self.cfg_path)
        self.app_cfg = app_cfg
        self.selected_human_uid: int = None
        self.humans: List[Human] = None
        self.worker = worker
        self.frame_nr_label = QLabel("Frame -")
        self.fps_label = QLabel("- FPS")
        self.persons_label = QLabel("0 persons")
        self.selection_label = QLabel("no selection")
        pipeline = getattr(worker, "pipeline", None)
        action_thresh = pipeline.cfg.action_thresh if pipeline is not None else None
        self.actions_bar_chart_view.initialize_actions_chart(self.app_cfg.inference.action_list, action_thresh)
        self.init_layout()

        self.init_menu(app)
        self.init_player()
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

    def init_player(self):
        # player controls below the video; keyboard: Space play / pause, Left / Right frame by frame while paused
        # (skip 5 s while playing), Home replay
        self.player_bar = PlayerBar(self.worker, self)
        self.verticalLayout_2.addWidget(self.player_bar)
        for key, slot in ((Qt.Key.Key_Space, self.worker.toggle_play), (Qt.Key.Key_Left, self.player_bar.left),
                          (Qt.Key.Key_Right, self.player_bar.right), (Qt.Key.Key_Home, self.worker.replay)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(slot)

    def init_layout(self):
        # image and 3D pose on top (larger), EHPI / actions / orientation below
        self.grid.setRowStretch(1, 3)
        self.grid.setRowStretch(3, 2)
        for column in range(3):
            self.grid.setColumnStretch(column, 1)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.img_ehpi.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_ehpi.setMinimumSize(1, 1)

    def init_status_bar(self):
        self.statusBar().addWidget(self.frame_nr_label)
        self.statusBar().addWidget(self.fps_label)
        self.statusBar().addWidget(self.persons_label)
        self.statusBar().addWidget(self.selection_label)

    def init_img_view(self, img_size):
        self.dialog = QDialog()
        self.hlayout = QHBoxLayout()
        self.img_view.img_size = img_size
        self.img_view.cfg = self.cfg
        self.img_view.action_list = self.app_cfg.inference.action_list
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
        ehpi = np.ascontiguousarray(ehpi)
        h, w, ch = ehpi.shape
        qt_img = QImage(ehpi.data, w, h, ch * w, QImage.Format.Format_RGB888)
        # pixel exact (nearest neighbour) scaling to the available space, aspect ratio kept
        pixmap = QPixmap.fromImage(qt_img).scaled(self.img_ehpi.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.FastTransformation)
        self.img_ehpi.setPixmap(pixmap)

    def __clear_ehpi(self):
        self.img_ehpi.clear()
        self.img_ehpi.setText("no selection")

    # Event Handler
    @Slot(int, np.ndarray, list, np.ndarray, int)
    def update_worker_data(self, frame_nr: int, img: np.ndarray, humans: List[Human], object_bbs: np.ndarray, fps: int):
        # frame nr label:
        self.frame_nr_label.setText(f"Frame {frame_nr:05}")
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

        self.fps_label.setText(f"{fps} FPS")
        self.persons_label.setText(f"{len(humans)} person{'s' if len(humans) != 1 else ''}")
        self.worker.frame_consumed()

    @Slot(int)
    def set_selected_human_uid(self, selected_human_uid: int):
        self.selected_human_uid = selected_human_uid
        self.update_selected_human()
        self.img_view.update()

    def __clear_selected_human(self):
        self.__clear_ehpi()
        self.body_orientation_view.clear()
        self.selection_label.setText("no selection")
        self.actions_bar_chart_view.clear_data()

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
        self.actions_bar_chart_view.set_actions(selected_human.action_probabilities, selected_human.actions)

    def __update_metadata(self, selected_human: Human):
        size = get_human_size_from_skeleton_3d(selected_human.skeleton_3d)
        size_text = f"{int(size)} mm" if size > 0 else "- mm"
        self.selection_label.setText(f"selected #{selected_human.uid} · score {int(selected_human.score * 100)}% · "
                                     f"size {size_text}")

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
