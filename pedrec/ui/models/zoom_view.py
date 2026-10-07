"""
Magnifier for the selected person: the plain camera image of the person (no overlays), enlarged, so one can see with
the own eyes what the person is doing. The magnified region is marked in the video; the view can be popped out into
an own (resizable) window.
"""
import math
from typing import Dict, Optional

import numpy as np
from qtpy.QtCore import Qt, QPointF, QRectF, Signal
from qtpy.QtGui import QPainter, QPen, QColor, QFont, QPainterPath, QPixmap, QBrush
from qtpy.QtWidgets import QWidget, QToolButton, QStyle

from pedrec.ui import theme

SMOOTHING = 0.35  # weight of the new frame in the exponential smoothing of the magnified region (less jitter)
MARGIN = 0.15  # additional context around the person
MAX_ZOOM = 6.0  # small / distant persons are not magnified further (only pixels)


def draw_magnifier_glyph(painter: QPainter, center: QPointF, radius: float, color: QColor, width: float = 2.0):
    """Lens circle with a handle (bottom right)."""
    painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(center, radius, radius)
    offset = radius / math.sqrt(2)
    painter.setPen(QPen(color, width * 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(QPointF(center.x() + offset, center.y() + offset),
                     QPointF(center.x() + offset + radius * 0.9, center.y() + offset + radius * 0.9))


class ZoomView(QWidget):
    pop_out_toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 200)
        self.frame: Optional[QPixmap] = None
        self.region: Optional[QRectF] = None  # magnified region in image pixels
        self.uid: Optional[int] = None
        self.color = QColor(theme.TEXT_MUTED)
        self._smoothed: Dict[int, np.ndarray] = {}
        self.pop_out_button = QToolButton(self)
        self.pop_out_button.setAutoRaise(True)
        self.pop_out_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pop_out_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton))
        self.pop_out_button.setToolTip("Show the magnifier in an own window / dock it again")
        self.pop_out_button.clicked.connect(self.pop_out_toggled.emit)

    # ---------------------------------------------------------------------------------------------------- data
    def clear(self):
        self.frame, self.region, self.uid = None, None, None
        self.update()

    def set_person(self, frame: QPixmap, bb, uid: int):
        """:param bb: center bb (x, y, w, h) of the person in image pixels"""
        lens = self.lens_rect()
        aspect = lens.width() / max(lens.height(), 1.0)
        cx, cy, w, h = (float(v) for v in bb[:4])
        height = max(h, w / aspect) * (1 + 2 * MARGIN)
        height = max(height, lens.height() / MAX_ZOOM)
        target = np.array([cx, cy, height])
        previous = self._smoothed.get(uid) if uid == self.uid else None
        smoothed = target if previous is None else SMOOTHING * target + (1 - SMOOTHING) * previous
        self._smoothed = {uid: smoothed}
        cx, cy, height = smoothed
        width = height * aspect
        # keep the region inside the image where possible (no empty border in the magnifier)
        if width <= frame.width():
            cx = min(max(cx, width / 2), frame.width() - width / 2)
        if height <= frame.height():
            cy = min(max(cy, height / 2), frame.height() - height / 2)
        self.frame, self.uid = frame, uid
        self.region = QRectF(cx - width / 2, cy - height / 2, width, height)
        self.color = theme.track_color(uid)
        self.update()

    @property
    def zoom_factor(self) -> float:
        if self.region is None or self.region.height() <= 0:
            return 1.0
        return self.lens_rect().height() / self.region.height()

    # ---------------------------------------------------------------------------------------------------- drawing
    def lens_rect(self) -> QRectF:
        return QRectF(8, 30, max(self.width() - 16, 1), max(self.height() - 52, 1))

    def resizeEvent(self, event):
        self.pop_out_button.move(self.width() - self.pop_out_button.sizeHint().width() - 4, 2)
        super().resizeEvent(event)

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.fillRect(self.rect(), QColor(theme.PANEL))
            lens = self.lens_rect()
            path = QPainterPath()
            path.addRoundedRect(lens, 10, 10)
            title_glyph = QPointF(16, 15)
            painter.setFont(QFont("Sans Serif", 9, QFont.Weight.DemiBold))
            if self.frame is None or self.region is None:
                painter.fillPath(path, QBrush(QColor(theme.BACKGROUND)))
                draw_magnifier_glyph(painter, title_glyph, 6, QColor(theme.TEXT_MUTED), 1.6)
                painter.setPen(QColor(theme.TEXT_MUTED))
                painter.drawText(QRectF(28, 4, self.width() - 60, 22), Qt.AlignmentFlag.AlignVCenter, "Magnifier")
                draw_magnifier_glyph(painter, QPointF(lens.center().x() - 8, lens.center().y() - 26), 22,
                                     QColor(theme.PANEL_BORDER), 4)
                painter.setPen(QColor(theme.TEXT_MUTED))
                painter.drawText(QRectF(lens.left(), lens.center().y() + 20, lens.width(), 40),
                                 Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                                 "select a person\nto magnify")
                return
            # plain camera image of the person, no overlays
            painter.save()
            painter.setClipPath(path)
            painter.fillRect(lens, QColor(theme.BACKGROUND))
            image_rect = QRectF(self.frame.rect())
            source = self.region.intersected(image_rect)
            if not source.isEmpty():
                scale = lens.width() / self.region.width()
                target = QRectF(lens.left() + (source.left() - self.region.left()) * scale,
                                lens.top() + (source.top() - self.region.top()) * scale,
                                source.width() * scale, source.height() * scale)
                painter.drawPixmap(target, self.frame, source)
            painter.restore()
            # lens ring + handle in the track color
            painter.setPen(QPen(self.color, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            corner = QPointF(lens.right() - 3, lens.bottom() - 3)
            painter.setPen(QPen(self.color, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(corner, QPointF(min(corner.x() + 12, self.width() - 4), min(corner.y() + 14,
                                                                                         self.height() - 4)))
            draw_magnifier_glyph(painter, title_glyph, 6, self.color, 1.6)
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(QRectF(28, 4, self.width() - 60, 22), Qt.AlignmentFlag.AlignVCenter,
                             f"Person #{self.uid}   ×{self.zoom_factor:.1f}")


class ZoomWindow(QWidget):
    """Own window for the popped out magnifier; closing it docks the magnifier again."""
    closed = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)  # the main window decides about quitting
        self.setWindowTitle("PedRec - Magnifier")
        self.resize(420, 640)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
