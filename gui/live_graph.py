"""
gui/live_graph.py  —  compact dark-themed pyqtgraph scrolling plot.

Actual (solid) vs Predicted (dashed) curves on a dark background
matching the new industrial theme.
"""
from __future__ import annotations
from collections import deque
from typing import Deque, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from gui.styles import (
    GRAPH_ACTUAL, GRAPH_PREDICTED,
    BG_CARD, BG_PRIMARY, BORDER_COLOR, TEXT_SECONDARY,
)
from utils.config import APP_CONFIG


class LiveGraph(QWidget):
    """Dark-themed scrolling graph: actual (solid) vs predicted (dashed)."""

    def __init__(
        self,
        title: str,
        unit: str,
        color: str = GRAPH_ACTUAL,
        max_points: int = APP_CONFIG.GRAPH_HISTORY_LEN,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title      = title
        self._unit       = unit
        self._color      = color
        self._max_points = max_points
        self._x:        Deque[float] = deque(maxlen=max_points)
        self._y_actual: Deque[float] = deque(maxlen=max_points)
        self._y_pred:   Deque[float] = deque(maxlen=max_points)
        self._tick: int = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(antialias=True, foreground=TEXT_SECONDARY,
                            background=BG_CARD)

        self._pw = pg.PlotWidget()
        self._pw.setBackground(BG_PRIMARY)
        self._pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pw.setLabel("left", self._unit,
                          color=TEXT_SECONDARY, size="9pt")
        self._pw.showGrid(x=False, y=True, alpha=0.15)
        self._pw.getAxis("left").setPen(pg.mkPen(color=BORDER_COLOR, width=1))
        self._pw.getAxis("bottom").setPen(pg.mkPen(color=BORDER_COLOR, width=1))
        self._pw.getAxis("left").setTextPen(pg.mkPen(color=TEXT_SECONDARY))
        self._pw.getAxis("bottom").setTextPen(pg.mkPen(color=TEXT_SECONDARY))
        self._pw.setMenuEnabled(False)
        self._pw.hideButtons()

        legend = self._pw.addLegend(
            offset=(8, 8),
            labelTextColor=TEXT_SECONDARY,
            colCount=2,
        )
        legend.setParentItem(self._pw.getPlotItem())

        self._curve_actual = self._pw.plot(
            [], [],
            pen=pg.mkPen(color=self._color, width=2),
            name="Actual",
        )
        self._curve_pred = self._pw.plot(
            [], [],
            pen=pg.mkPen(color=GRAPH_PREDICTED, width=1.5,
                         style=Qt.DashLine),
            name="Pred",
        )
        layout.addWidget(self._pw)

    def append(self, actual: float, predicted: float) -> None:
        self._x.append(self._tick)
        self._y_actual.append(actual)
        self._y_pred.append(predicted)
        self._tick += 1
        self._refresh()

    def clear(self) -> None:
        self._x.clear(); self._y_actual.clear(); self._y_pred.clear()
        self._tick = 0
        self._curve_actual.setData([], [])
        self._curve_pred.setData([], [])

    def _refresh(self) -> None:
        x  = np.array(self._x,        dtype=np.float64)
        ya = np.array(self._y_actual,  dtype=np.float64)
        yp = np.array(self._y_pred,    dtype=np.float64)
        self._curve_actual.setData(x, ya)
        self._curve_pred.setData(x, yp)
        if len(x) >= 2:
            self._pw.setXRange(x[0], x[-1], padding=0.02)
