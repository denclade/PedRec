from typing import List

import numpy as np
import OpenGL.GL as ogl
import pyqtgraph.opengl as gl

from pedrec.models.data_structures import Color
from pedrec.ui.models.gl_arrow_item import GLArrowItem
# from pedrec.ui.models.text_item_3d import TextItem3D


class Axis3D(gl.GLAxisItem):
    """Class defined to extend 'gl.GLAxisItem'."""

    def __init__(self, parent: gl.GLViewWidget,
                 color_x=Color(255, 0, 0, 255),
                 color_y=Color(0, 255, 0, 255),
                 color_z=Color(0, 0, 255, 255),
                 line_width: float = 1
                 ):
        gl.GLAxisItem.__init__(self)
        self.parent_view = parent
        self.line_width = line_width
        self.color_x = color_x
        self.color_y = color_y
        self.color_z = color_z

    def add_labels(self):
        """
        Add x, y and z labels to the axes
        """
        x, y, z = self.size()  # e.g. 1, 2 or 3
        x_label = gl.GLTextItem(pos=(x / 2, -y / 10, -z / 10), text="x", color='white')
        self.parent_view.addItem(x_label)
        # we use z up, thus just switch text to z, data needs to be switched too
        y_label = gl.GLTextItem(pos=(-x / 10, y / 2, -z / 10), text="y", color='white')
        self.parent_view.addItem(y_label)
        z_label = gl.GLTextItem(pos=(-x / 10, -y / 10, z / 2), text="z", color='white')
        self.parent_view.addItem(z_label)

    def add_tick_values(self, x_ticks: List[int], y_ticks: List[int], z_ticks: List[int]):
        """Adds ticks values."""
        x, y, z = self.size()
        x_poss = np.linspace(0, x, len(x_ticks))
        y_poss = np.linspace(0, y, len(y_ticks))
        z_poss = np.linspace(0, z, len(z_ticks))
        # X label
        for i, xt in enumerate(x_ticks):
            label = gl.GLTextItem(pos=(x_poss[i], -y / 20, -z / 20), text=str(xt), color='white')
            self.parent_view.addItem(label)
        # y label
        for i, yt in enumerate(y_ticks):
            label = gl.GLTextItem(pos=(-x / 20, y_poss[i], -z / 20), text=str(yt), color='white')
            self.parent_view.addItem(label)
        # z label
        for i, zt in enumerate(z_ticks):
            label = gl.GLTextItem(pos=(-x / 20, -y / 20, z_poss[i]), text=str(zt), color='white')
            self.parent_view.addItem(label)

    def paint(self):
        self.setupGLState()
        if getattr(self, "antialias", True):
            ogl.glEnable(ogl.GL_LINE_SMOOTH)
            ogl.glHint(ogl.GL_LINE_SMOOTH_HINT, ogl.GL_NICEST)
            ogl.glEnable(ogl.GL_BLEND)
            ogl.glBlendFunc(ogl.GL_SRC_ALPHA, ogl.GL_ONE_MINUS_SRC_ALPHA)
            # ogl.glDisable(ogl.GL_DEPTH_TEST)
        x, y, z = self.size()  # e.g. 1, 2 or 3
        GLArrowItem(0, 0, 0, x, 0, 0, self.line_width, color=self.color_x).paint()
        GLArrowItem(0, 0, 0, 0, y, 0, self.line_width, color=self.color_y).paint()
        GLArrowItem(0, 0, 0, 0, 0, z, self.line_width, color=self.color_z).paint()
