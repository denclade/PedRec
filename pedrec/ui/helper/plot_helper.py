from typing import Tuple

import numpy as np
import pyqtgraph.opengl as gl
from qtpy import QtGui

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_LIMB_COLORS, SKELETON_PEDREC, SKELETON_PEDREC_JOINT_COLORS, \
    SKELETON_PEDREC_JOINTS


def add_floor(view: gl.GLViewWidget, z: float = -1.0, size: float = 3.0):
    """Ground grid below the hip centered skeleton (in m)."""
    from pedrec.ui import theme
    floor = gl.GLGridItem(size=QtGui.QVector3D(size, size, 1))
    floor.setSpacing(0.25, 0.25, 1)
    floor.setColor(QtGui.QColor(theme.PANEL_BORDER))
    floor.translate(0, 0, z)
    view.addItem(floor)
    return floor


def get_limb_positions(skeleton: np.array, min_score: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Start / end point pairs of the limbs (GLLinePlotItem mode 'lines') and their colors."""
    limbs = []
    colors = []
    for limb_idx, limb in enumerate(SKELETON_PEDREC):
        if skeleton[limb[0], 3] < min_score or skeleton[limb[1], 3] < min_score:
            continue
        limbs.append(skeleton[limb[0], :3])
        limbs.append(skeleton[limb[1], :3])
        color = SKELETON_PEDREC_LIMB_COLORS[limb_idx].rgba_float_list
        colors.append(color)
        colors.append(color)
    return np.array(limbs, dtype=np.float32), np.array(colors, dtype=np.float32)


def get_joint_positions(skeleton: np.array, min_score: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    joints = []
    colors = []
    for joint_idx, joint in enumerate(SKELETON_PEDREC_JOINTS):
        if skeleton[joint_idx, 3] < min_score:
            continue
        joints.append(skeleton[joint_idx, :3])
        colors.append(SKELETON_PEDREC_JOINT_COLORS[joint_idx].rgba_float_list)
    return np.array(joints), np.array(colors)
