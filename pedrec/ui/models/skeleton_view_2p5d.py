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
        # if self.axis is not None:
        #     self.removeItem(self.axis)
        #     self.axis = None
        if self.current_joints is not None:
            self.removeItem(self.current_joints)
            self.current_joints = None
        if self.current_limbs is not None:
            self.removeItem(self.current_limbs)
            self.current_limbs = None

    def add_skeleton(self, skeleton: np.ndarray):
        if self.axis is None:
            self.set_axis()
        # convert from mm to meter for display
        skeleton[:, 0] /= 1000
        skeleton[:, 1] /= 1000
        skeleton[:, 2] /= 1000

        skel = ""
        for joint in SKELETON_PEDREC_JOINT:
            j = skeleton[joint.value].copy() * 10
            skel += f"{j[0]:.2f}/{j[1]:.2f}/{j[2]:.2f}/{1.0}/{1.0},"
        # print(skel)

        hip = skeleton[SKELETON_PEDREC_JOINT.hip_center.value]
        skeleton[:, :3] -= hip[:3]


        # switch x and z
        skeleton[:, [1, 2]] = skeleton[:, [2, 1]]

        joints, joint_colors = get_joint_positions(skeleton)
        limbs, limb_colors = get_limb_positions(skeleton)

        self.current_joints = gl.GLScatterPlotItem(pos=joints, color=joint_colors)
        self.addItem(self.current_joints)

        self.current_limbs = gl.GLLinePlotItem(pos=limbs, color=limb_colors, width=1, antialias=False)
        self.addItem(self.current_limbs)
