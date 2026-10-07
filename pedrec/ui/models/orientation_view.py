import math

import pyqtgraph.opengl as gl
from qtpy import QtGui

from pedrec.models.data_structures import Color
from pedrec.ui.helper.plot_helper import add_grid
from pedrec.ui.models.axis_3d import Axis3D
from pedrec.ui.models.pedrec_ui_config import PedRecUIConfig


class OrientationView(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis_body: Axis3D = None
        self.axis_head: Axis3D = None
        self.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=4, elevation=0, azimuth=-90)
        add_grid(self)
        self.show()
        self.cfg: PedRecUIConfig = None

    def get_axis(self):
        axis = Axis3D(self,
                           color_x=Color(255, 0, 0, 255),
                           color_y=Color(0, 0, 255, 255),
                           color_z=Color(0, 255, 0, 255),
                           line_width=4)
        axis.setSize(x=0.25, y=1, z=0.25)

        self.addItem(axis)
        return axis

    def clear(self):
        if self.axis_body is not None:
            self.removeItem(self.axis_body)
            self.axis_body = None
        if self.axis_head is not None:
            self.removeItem(self.axis_head)
            self.axis_head = None

    def set_orientation(self, theta_rad_body: float, phi_rad_body: float, theta_rad_head: float, phi_rad_head: float):
        if self.axis_body is None:
            self.axis_body = self.get_axis()
        if self.axis_head is None:
            self.axis_head = self.get_axis()
            # self.axis_head.translate(0, 0, 1, local=True)
        self.axis_body.resetTransform()
        self.axis_head.resetTransform()
        theta_body = math.degrees(theta_rad_body) - 90  # -90 to have 0 @ horizon
        phi_body = math.degrees(phi_rad_body) - 90  # -90 to have 0 pointing to right

        self.axis_body.rotate(phi_body, 0, 0, 1, local=True)
        self.axis_body.rotate(theta_body, 1, 0, 0, local=True)
        
        theta_head = math.degrees(theta_rad_head) - 90  # -90 to have 0 @ horizon
        phi_head = math.degrees(phi_rad_head) - 90  # -90 to have 0 pointing to right

        self.axis_head.rotate(phi_head, 0, 0, 1, local=True)
        self.axis_head.rotate(theta_head, 1, 0, 0, local=True)

        self.axis_body.translate(0, 0, -0.5, local=True)
        self.axis_head.translate(0, 0, 0.5, local=True)

        # print(f"Head Phi: {phi_head}")
        # print(f"Body Phi: {phi_body}")

        # w.addItem(axis)


