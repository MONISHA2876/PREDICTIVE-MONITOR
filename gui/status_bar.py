"""
gui/status_bar.py  —  compact top bar: title, clock, status badge, model.
"""
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import QDateTime, QTimer, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from gui.styles import (
    STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL, STATUS_IDLE,
    ACCENT_BLUE, TEXT_SECONDARY, BG_HEADER, BORDER_COLOR,
)
from model.anomaly_detector import AnomalyLevel


class TopStatusBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(48)
        self._setup_ui()
        self._start_clock()

    def _setup_ui(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(14)

        title = QLabel("⚡  PREDICTIVE MONITORING SYSTEM")
        title.setObjectName("AppTitle")
        lay.addWidget(title)

        sub = QLabel(" ")
        sub.setObjectName("SubTitle")
        lay.addWidget(sub)

        lay.addStretch()

        self._lbl_tick = QLabel("Tick: 0")
        self._lbl_tick.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:11px;")
        lay.addWidget(self._lbl_tick)

        lay.addSpacing(8)
        self._lbl_model = QLabel("Model: —")
        self._lbl_model.setStyleSheet(f"color:{TEXT_SECONDARY};font-size:11px;")
        lay.addWidget(self._lbl_model)

        lay.addSpacing(8)
        self._lbl_status = QLabel("● IDLE")
        self._set_colour(STATUS_IDLE)
        lay.addWidget(self._lbl_status)

        lay.addSpacing(8)
        self._lbl_clock = QLabel()
        self._lbl_clock.setStyleSheet(
            f"color:white;font-size:11px;font-weight:bold;"
            f"background:{ACCENT_BLUE};padding:3px 10px;border-radius:4px;"
        )
        lay.addWidget(self._lbl_clock)

    def _start_clock(self) -> None:
        def _tick():
            self._lbl_clock.setText(
                QDateTime.currentDateTime().toString("hh:mm:ss")
            )
        t = QTimer(self); t.timeout.connect(_tick); t.start(1000); _tick()

    def set_status(self, level: AnomalyLevel) -> None:
        m = {
            AnomalyLevel.NORMAL:   (STATUS_NORMAL,   "● RUNNING"),
            AnomalyLevel.WARNING:  (STATUS_WARNING,  "⚠ WARNING"),
            AnomalyLevel.CRITICAL: (STATUS_CRITICAL, "🔴 CRITICAL"),
        }
        c, t = m[level]
        self._lbl_status.setText(t)
        self._set_colour(c)

    def set_idle(self) -> None:
        self._lbl_status.setText("● IDLE")
        self._set_colour(STATUS_IDLE)

    def set_model_name(self, name: str) -> None:
        self._lbl_model.setText(f"Model: {name}")

    def set_tick(self, tick: int) -> None:
        self._lbl_tick.setText(f"Tick: {tick}")

    def _set_colour(self, c: str) -> None:
        self._lbl_status.setStyleSheet(
            f"color:white;font-size:11px;font-weight:bold;"
            f"background:{c};padding:3px 10px;border-radius:10px;"
        )
