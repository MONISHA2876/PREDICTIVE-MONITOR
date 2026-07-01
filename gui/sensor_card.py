"""
gui/sensor_card.py

Individual sensor display card widget.

Shows:
    - Sensor name + icon
    - Current (actual) reading
    - Predicted next-step value
    - Anomaly score (%)
    - Status badge: Normal / Warning / Critical
    - Model name that produced the prediction
    - Inference time
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.styles import (
    STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL, STATUS_IDLE,
    ACCENT_BLUE, TEXT_SECONDARY,
)
from model.anomaly_detector import AnomalyLevel


class SensorCard(QWidget):
    """
    Compact dashboard card for one sensor channel.

    Args:
        title:     Display name (e.g. "Temperature").
        unit:      Physical unit (e.g. "°C").
        color:     Accent colour used for the value label.
        parent:    Parent widget.
    """

    def __init__(
        self,
        title: str,
        unit: str,
        color: str = ACCENT_BLUE,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._unit = unit
        self._color = color
        self._level = AnomalyLevel.NORMAL
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setObjectName("SensorCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(14, 12, 14, 12)

        # ── Header row ──────────────────────────────────────────────
        header = QHBoxLayout()

        self._lbl_title = QLabel(self._title.upper())
        self._lbl_title.setObjectName("CardTitle")

        self._lbl_status = QLabel("IDLE")
        self._lbl_status.setObjectName("CardStatus")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        self._set_status_style(STATUS_IDLE)

        header.addWidget(self._lbl_title)
        header.addStretch()
        header.addWidget(self._lbl_status)
        root.addLayout(header)

        # ── Current value (large) ────────────────────────────────────
        self._lbl_value = QLabel("—")
        self._lbl_value.setObjectName("CardValue")
        font = self._lbl_value.font()
        font.setPointSize(26)
        font.setBold(True)
        self._lbl_value.setFont(font)
        self._lbl_value.setStyleSheet(f"color: {self._color};")
        root.addWidget(self._lbl_value)

        # ── Grid: predicted / anomaly / inference ────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(2)

        def _caption(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
            return lbl

        grid.addWidget(_caption("Predicted"), 0, 0)
        grid.addWidget(_caption("Anomaly Score"), 0, 1)
        grid.addWidget(_caption("Inference (ms)"), 0, 2)

        self._lbl_predicted = QLabel("—")
        self._lbl_predicted.setObjectName("CardPredicted")

        self._lbl_anomaly = QLabel("— %")
        self._lbl_anomaly.setObjectName("CardAnomalyScore")

        self._lbl_inference = QLabel("—")
        self._lbl_inference.setStyleSheet("font-size: 12px;")

        grid.addWidget(self._lbl_predicted, 1, 0)
        grid.addWidget(self._lbl_anomaly, 1, 1)
        grid.addWidget(self._lbl_inference, 1, 2)
        root.addLayout(grid)

        # ── Model label ──────────────────────────────────────────────
        self._lbl_model = QLabel("Model: —")
        self._lbl_model.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px; font-style: italic;"
        )
        root.addWidget(self._lbl_model)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update_values(
        self,
        actual: float,
        predicted: float,
        anomaly_pct: float,
        level: AnomalyLevel,
        inference_ms: float,
        model_name: str,
    ) -> None:
        """
        Refresh all displayed values.

        Args:
            actual:       Current sensor reading (original units).
            predicted:    Model's next-step prediction.
            anomaly_pct:  Anomaly score as a percentage.
            level:        :class:`AnomalyLevel` classification.
            inference_ms: Time taken for prediction (ms).
            model_name:   Name of the active predictor.
        """
        self._level = level

        self._lbl_value.setText(f"{actual:.2f} {self._unit}")
        self._lbl_predicted.setText(f"{predicted:.2f} {self._unit}")
        self._lbl_anomaly.setText(f"{anomaly_pct:.1f} %")
        self._lbl_inference.setText(f"{inference_ms:.1f}")
        self._lbl_model.setText(f"Model: {model_name}")

        # Status badge
        status_map = {
            AnomalyLevel.NORMAL:   ("NORMAL",   STATUS_NORMAL),
            AnomalyLevel.WARNING:  ("WARNING",  STATUS_WARNING),
            AnomalyLevel.CRITICAL: ("CRITICAL", STATUS_CRITICAL),
        }
        label, colour = status_map[level]
        self._lbl_status.setText(label)
        self._set_status_style(colour)

        # Anomaly score colour
        score_colour = {
            AnomalyLevel.NORMAL:   STATUS_NORMAL,
            AnomalyLevel.WARNING:  STATUS_WARNING,
            AnomalyLevel.CRITICAL: STATUS_CRITICAL,
        }[level]
        self._lbl_anomaly.setStyleSheet(
            f"color: {score_colour}; font-weight: bold; font-size: 13px;"
        )

    def set_idle(self) -> None:
        """Reset card to idle / no-data state."""
        self._lbl_value.setText("—")
        self._lbl_predicted.setText("—")
        self._lbl_anomaly.setText("— %")
        self._lbl_inference.setText("—")
        self._lbl_model.setText("Model: —")
        self._lbl_status.setText("IDLE")
        self._set_status_style(STATUS_IDLE)
        self._lbl_anomaly.setStyleSheet("font-size: 13px;")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_status_style(self, bg_colour: str) -> None:
        self._lbl_status.setStyleSheet(
            f"background-color: {bg_colour}; color: white; "
            "padding: 3px 10px; border-radius: 8px; "
            "font-size: 11px; font-weight: bold;"
        )
