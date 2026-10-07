from typing import List

import numpy as np
import pyqtgraph.opengl as gl

from pedrec.models.data_structures import Color


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the z axis onto ``direction`` (unit vector)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, direction)
    c = float(np.dot(z, direction))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def arrow_mesh(end: np.ndarray, radius: float, cols: int = 16) -> gl.MeshData:
    """Arrow from the origin to ``end``: cylinder shaft + cone head (head radius 2r, head length 4r)."""
    length = float(np.linalg.norm(end))
    head_length = min(4 * radius, 0.5 * length)
    shaft = gl.MeshData.cylinder(rows=1, cols=cols, radius=[radius, radius], length=length - head_length)
    head = gl.MeshData.cylinder(rows=1, cols=cols, radius=[2 * radius, 0.0], length=head_length)
    vertexes = np.vstack([shaft.vertexes(), head.vertexes() + [0, 0, length - head_length]])
    faces = np.vstack([shaft.faces(), head.faces() + len(shaft.vertexes())])
    vertexes = vertexes @ _rotation_from_z(np.asarray(end, dtype=float) / length).T
    return gl.MeshData(vertexes=vertexes, faces=faces)


class Axis3D(gl.GLGraphicsItem.GLGraphicsItem):
    """
    x / y / z arrows as child mesh items. Uses only the shader based pyqtgraph items, so it works with OpenGL core
    profiles (e.g. NVIDIA / Wayland), unlike the former fixed function implementation (glPushMatrix / gluCylinder).
    """

    def __init__(self, parent: gl.GLViewWidget,
                 color_x=Color(255, 0, 0, 255),
                 color_y=Color(0, 255, 0, 255),
                 color_z=Color(0, 0, 255, 255),
                 line_width: float = 1
                 ):
        super().__init__()
        self.parent_view = parent
        self.radius = line_width * 0.01
        self.colors = [color_x.tuplef_rgba, color_y.tuplef_rgba, color_z.tuplef_rgba]
        self.arrows: List[gl.GLMeshItem] = []
        self.__size = [1.0, 1.0, 1.0]
        for color in self.colors:
            gl_options = "opaque" if color[3] >= 1 else "translucent"
            arrow = gl.GLMeshItem(color=color, smooth=False, shader="shaded", glOptions=gl_options)
            arrow.setParentItem(self)
            self.arrows.append(arrow)
        self._update_arrows()

    def setSize(self, x=1.0, y=1.0, z=1.0):
        self.__size = [x, y, z]
        self._update_arrows()

    def size(self):
        return self.__size[:]

    def _update_arrows(self):
        for arrow, end in zip(self.arrows, np.diag(self.__size)):
            arrow.setMeshData(meshdata=arrow_mesh(end, self.radius))
        self.update()

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
        for i, xt in enumerate(x_ticks):
            self.parent_view.addItem(gl.GLTextItem(pos=(x_poss[i], -y / 20, -z / 20), text=str(xt), color='white'))
        for i, yt in enumerate(y_ticks):
            self.parent_view.addItem(gl.GLTextItem(pos=(-x / 20, y_poss[i], -z / 20), text=str(yt), color='white'))
        for i, zt in enumerate(z_ticks):
            self.parent_view.addItem(gl.GLTextItem(pos=(-x / 20, -y / 20, z_poss[i]), text=str(zt), color='white'))
