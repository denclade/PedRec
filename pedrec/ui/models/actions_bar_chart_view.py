from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt

from pedrec.models.constants.action_mappings import ACTION
from pedrec.ui import theme


class ActionsBarChartView(pg.PlotWidget):
    """Horizontal bars with the action probabilities of the selected person; recognized actions are highlighted."""

    def __init__(self, parent=None, **kargs):
        super().__init__(parent, background=theme.PANEL, **kargs)
        self.action_list: Optional[List[ACTION]] = None
        self.bars: Optional[pg.BarGraphItem] = None
        self.threshold_line: Optional[pg.InfiniteLine] = None
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()

    def initialize_actions_chart(self, action_list: List[ACTION], threshold: Optional[float] = None):
        self.action_list = action_list
        count = len(action_list)
        plot = self.getPlotItem()
        plot.invertY(True)  # first action on top
        plot.showGrid(x=True, y=False, alpha=0.12)
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(theme.PANEL_BORDER))
            axis.setTextPen(pg.mkPen(theme.TEXT_MUTED))
        plot.getAxis("left").setTicks([[(i, action.name.lower().replace("_", " "))
                                        for i, action in enumerate(action_list)]])
        plot.getAxis("bottom").setTicks([[(v, f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1.0)]])
        self.bars = pg.BarGraphItem(x0=0, y=np.arange(count), height=0.65, width=np.zeros(count),
                                    brush=theme.INACTIVE, pen=None)
        self.addItem(self.bars)
        if threshold is not None:
            self.threshold_line = pg.InfiniteLine(pos=threshold, angle=90,
                                                  pen=pg.mkPen(theme.ACCENT, width=1, style=Qt.PenStyle.DashLine))
            self.addItem(self.threshold_line)
        self.setXRange(0, 1, padding=0.02)
        self.setYRange(-0.6, count - 0.4, padding=0)

    def set_actions(self, action_probabilities: np.ndarray, active_actions: Optional[List[ACTION]] = None):
        if self.bars is None or action_probabilities is None:
            return
        active = set(active_actions or [])
        brushes = [pg.mkBrush(theme.ACCENT if action in active else theme.INACTIVE) for action in self.action_list]
        self.bars.setOpts(width=np.asarray(action_probabilities, dtype=float), brushes=brushes)

    def clear_data(self):
        if self.bars is not None:
            self.bars.setOpts(width=np.zeros(len(self.action_list)), brush=theme.INACTIVE, brushes=None)
