import numpy as np
import pyqtgraph.opengl as gl
from qtpy import QtGui

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT
from pedrec.models.data_structures import Color
from pedrec.ui.helper.plot_helper import add_grid, get_joint_positions, get_limb_positions
from pedrec.ui.models.axis_3d import Axis3D


class SkeletonView2p5D(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis: Axis3D = None
        self.setMinimumSize(320, 180)
        self.setMaximumSize(1280, 720)
        self.current_limbs = None
        self.current_joints = None
        self.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=2, elevation=45, azimuth=205)
        add_grid(self)
        self.show()

    def set_axis(self):
        self.axis = Axis3D(self,
                           color_x=Color(255, 0, 0, 90),
                           color_y=Color(0, 0, 255, 90),
                           color_z=Color(0, 255, 0, 90))
        self.axis.setSize(x=3, y=3, z=3)
        self.axis.add_labels()
        self.axis.add_tick_values(x_ticks=[0, 1, 2, 3], y_ticks=[0, 1, 2, 3], z_ticks=[0, 1, 2, 3])
        self.addItem(self.axis)

    def clear(self):
        # the items are reused (setData) instead of being removed and recreated every frame
        if self.current_joints is not None:
            self.current_joints.setVisible(False)
            self.current_limbs.setVisible(False)

    def add_skeleton(self, skeleton: np.ndarray):
        if self.axis is None:
            self.set_axis()
        skeleton = skeleton.copy()
        skeleton[:, :3] /= 1000  # mm -> m
        hip = skeleton[SKELETON_PEDREC_JOINT.hip_center.value]
        skeleton[:, :3] -= hip[:3]
        skeleton[:, [1, 2]] = skeleton[:, [2, 1]]  # z up

        joints, joint_colors = get_joint_positions(skeleton)
        limbs, limb_colors = get_limb_positions(skeleton)
        if self.current_joints is None:
            self.current_joints = gl.GLScatterPlotItem()
            self.addItem(self.current_joints)
            self.current_limbs = gl.GLLinePlotItem(mode="lines", width=1, antialias=False)
            self.addItem(self.current_limbs)
        self.current_joints.setData(pos=joints.reshape(-1, 3), color=joint_colors.reshape(-1, 4))
        self.current_limbs.setData(pos=limbs.reshape(-1, 3), color=limb_colors.reshape(-1, 4))
        self.current_joints.setVisible(len(joints) > 0)
        self.current_limbs.setVisible(len(limbs) > 0)
