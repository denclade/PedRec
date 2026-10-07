import numpy as np
import pyqtgraph.opengl as gl
from qtpy import QtGui

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT
from pedrec.models.data_structures import Color
from pedrec.ui import theme
from pedrec.ui.helper.plot_helper import add_floor, get_joint_positions, get_limb_positions
from pedrec.ui.models.axis_3d import Axis3D


class SkeletonView2p5D(gl.GLViewWidget):
    """3D pose of the selected person (hip centered, in m, z up) above a ground grid."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis: Axis3D = None
        self.setMinimumSize(320, 180)
        self.current_limbs = None
        self.current_joints = None
        self.setBackgroundColor(QtGui.QColor(theme.PANEL))
        self.setCameraPosition(pos=QtGui.QVector3D(0, 0, -0.15), distance=3.4, elevation=14, azimuth=235)
        add_floor(self, z=-1.0)
        self.show()

    def set_axis(self):
        self.axis = Axis3D(self,
                           color_x=Color(255, 90, 90, 200),
                           color_y=Color(90, 140, 255, 200),
                           color_z=Color(110, 220, 120, 200),
                           line_width=0.6)
        self.axis.setSize(x=0.35, y=0.35, z=0.35)
        self.axis.translate(0, 0, -1.0)  # on the floor below the hip
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
            # pyqtgraph blends points / lines additively by default, which makes them invisible on a light background
            self.current_joints = gl.GLScatterPlotItem(size=8, pxMode=True, glOptions="translucent")
            self.addItem(self.current_joints)
            self.current_limbs = gl.GLLinePlotItem(mode="lines", width=3, antialias=True, glOptions="translucent")
            self.addItem(self.current_limbs)
        self.current_joints.setData(pos=joints.reshape(-1, 3), color=joint_colors.reshape(-1, 4))
        self.current_limbs.setData(pos=limbs.reshape(-1, 3), color=limb_colors.reshape(-1, 4))
        self.current_joints.setVisible(len(joints) > 0)
        self.current_limbs.setVisible(len(limbs) > 0)
