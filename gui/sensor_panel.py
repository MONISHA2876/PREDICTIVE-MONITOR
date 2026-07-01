"""
gui/sensor_panel.py

One complete sensor column: gauge on top, live graph below.
Replaces the old SensorCard widget.
"""
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from gui.gauge_widget import GaugeWidget
from gui.live_graph   import LiveGraph
from gui.styles       import BORDER_COLOR, TEXT_SECONDARY
from model.anomaly_detector import AnomalyLevel


class SensorPanel(QWidget):
    """
    Vertical panel for one sensor channel.
    Top: GaugeWidget (value + dial + LED)
    Bottom: LiveGraph (actual vs predicted scrolling plot)
    """

    def __init__(
        self,
        title: str,
        unit: str,
        color: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._unit  = unit
        self._color = color
        self.setObjectName("SensorPanel")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Gauge
        self._gauge = GaugeWidget(self._title, self._unit, self._color)
        self._gauge.setMinimumHeight(210)
        self._gauge.setMaximumHeight(240)
        layout.addWidget(self._gauge)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER_COLOR}; background: {BORDER_COLOR}; max-height: 1px;")
        layout.addWidget(line)

        # Graph
        self._graph = LiveGraph(self._title, self._unit, self._color)
        self._graph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._graph, stretch=1)

    # ── public API ──────────────────────────────────────────────────
    def update_values(
        self,
        actual: float,
        predicted: float,
        anomaly_pct: float,
        level: AnomalyLevel,
        inference_ms: float,
        model_name: str,
    ) -> None:
        self._gauge.update_values(actual, predicted, anomaly_pct,
                                  level, inference_ms, model_name)
        self._graph.append(actual, predicted)

    def set_idle(self) -> None:
        self._gauge.set_idle()

    def clear_graph(self) -> None:
        self._graph.clear()
