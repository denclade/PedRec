import math
from typing import Optional, Tuple

from qtpy.QtCore import Qt, QPointF, QRectF
from qtpy.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPolygonF
from qtpy.QtWidgets import QWidget

from pedrec.ui import theme
from pedrec.ui.helper.bb_helper import sees_camera


class OrientationView(QWidget):
    """
    Body and head orientation of the selected person as two compasses (top view, phi; up = away from the camera,
    down = towards the camera) with a side view gauge for theta (0 = up, 90 = horizontal, 180 = down).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumSize(260, 160)
        self.orientations: Optional[Tuple[float, float, float, float]] = None
        self.cfg = None

    def clear(self):
        self.orientations = None
        self.update()

    def set_orientation(self, theta_rad_body: float, phi_rad_body: float, theta_rad_head: float, phi_rad_head: float):
        self.orientations = (theta_rad_body, phi_rad_body, theta_rad_head, phi_rad_head)
        self.update()

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(theme.PANEL))
            half = self.width() / 2
            for i, title in enumerate(("Body", "Head")):
                area = QRectF(i * half, 0, half, self.height())
                orientation = None if self.orientations is None else self.orientations[2 * i:2 * i + 2]
                self._draw_dial(painter, area, title, orientation, highlight=i == 1)

    def _draw_dial(self, painter: QPainter, area: QRectF, title: str, orientation, highlight: bool):
        muted, text = QColor(theme.TEXT_MUTED), QColor(theme.TEXT)
        radius = max(10.0, min(area.width() * 0.3, (area.height() - 50) / 2))
        center = QPointF(area.center().x() - radius * 0.25, area.top() + 18 + radius + 8)
        painter.setFont(QFont("Sans Serif", 9, QFont.Weight.DemiBold))
        painter.setPen(muted)
        painter.drawText(QRectF(area.left(), area.top() + 2, area.width(), 16), Qt.AlignmentFlag.AlignCenter, title)

        # compass (top view): phi
        painter.setPen(QPen(QColor(theme.PANEL_BORDER), 1.5))
        painter.setBrush(QBrush(QColor(theme.BACKGROUND)))
        painter.drawEllipse(center, radius, radius)
        painter.setFont(QFont("Sans Serif", 7))
        painter.setPen(muted)
        for label, (dx, dy) in (("away", (0, -1)), ("camera", (0, 1)), ("R", (1, 0)), ("L", (-1, 0))):
            pos = QPointF(center.x() + dx * (radius - 10), center.y() + dy * (radius - 8))
            painter.drawText(QRectF(pos.x() - 22, pos.y() - 7, 44, 14), Qt.AlignmentFlag.AlignCenter, label)

        # side view gauge: theta
        gauge_center = QPointF(center.x() + radius + 14, center.y())
        gauge_radius = radius * 0.55
        painter.setPen(QPen(QColor(theme.PANEL_BORDER), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(QRectF(gauge_center.x() - gauge_radius, gauge_center.y() - gauge_radius,
                               2 * gauge_radius, 2 * gauge_radius), -90 * 16, 180 * 16)

        if orientation is None:
            painter.setPen(muted)
            painter.drawText(QRectF(area.left(), area.bottom() - 18, area.width(), 16), Qt.AlignmentFlag.AlignCenter,
                             "no selection")
            return
        theta, phi = orientation
        color = QColor(theme.ACCENT) if highlight and sees_camera((theta, phi)) else text
        self._needle(painter, center, radius - 4, math.cos(phi), -math.sin(phi), color, 2.5)
        self._needle(painter, gauge_center, gauge_radius, math.sin(theta), -math.cos(theta), color, 1.8)
        painter.setFont(QFont("Sans Serif", 8))
        painter.setPen(text)
        info = f"φ {math.degrees(phi):3.0f}°   θ {math.degrees(theta):3.0f}°"
        if highlight and sees_camera((theta, phi)):
            info += "   sees camera"
        painter.drawText(QRectF(area.left(), area.bottom() - 18, area.width(), 16), Qt.AlignmentFlag.AlignCenter, info)

    @staticmethod
    def _needle(painter: QPainter, center: QPointF, length: float, dx: float, dy: float, color: QColor, width: float):
        tip = QPointF(center.x() + dx * length, center.y() + dy * length)
        painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(center, tip)
        angle = math.atan2(dy, dx)
        head = QPolygonF([tip,
                          QPointF(tip.x() - 8 * math.cos(angle - 0.4), tip.y() - 8 * math.sin(angle - 0.4)),
                          QPointF(tip.x() - 8 * math.cos(angle + 0.4), tip.y() - 8 * math.sin(angle + 0.4))])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(head)
        painter.drawEllipse(center, 3, 3)
