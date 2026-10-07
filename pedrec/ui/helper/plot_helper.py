from typing import Tuple

import numpy as np
import pyqtgraph.opengl as gl
from qtpy import QtGui

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_LIMB_COLORS, SKELETON_PEDREC, SKELETON_PEDREC_JOINT_COLORS, \
    SKELETON_PEDREC_JOINTS


def add_grid(view: gl.GLViewWidget, size: QtGui.QVector3D = QtGui.QVector3D(3, 3, 1)):
    gx = gl.GLGridItem(size=size)
    gx.rotate(90, 0, 1, 0)
    gx.translate(-1.5, 0, 0)
    view.addItem(gx)
    gy = gl.GLGridItem(size=size)
    gy.rotate(90, 1, 0, 0)
    gy.translate(0, -1.5, 0)
    view.addItem(gy)
    gz = gl.GLGridItem(size=size)
    gz.translate(0, 0, -1.5)
    view.addItem(gz)


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
