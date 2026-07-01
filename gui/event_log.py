"""
gui/event_log.py  —  dark-themed compact event log panel.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from gui.styles import (
    STATUS_WARNING, STATUS_CRITICAL,
    BORDER_COLOR, TEXT_SECONDARY, BG_CARD, TEXT_DIM,
)
from model.anomaly_detector import AnomalyLevel


class EventLog(QWidget):
    _WARN_BG  = QColor("#2D2200")
    _CRIT_BG  = QColor("#2D0E0E")
    _INFO_BG  = QColor("#0D1117")
    _WARN_FG  = QColor("#D29922")
    _CRIT_FG  = QColor("#F85149")
    _INFO_FG  = QColor("#8B949E")

    def __init__(self, max_entries: int = 300,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._max = max_entries
        self._count = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        hdr = QHBoxLayout()
        title = QLabel("EVENT LOG")
        title.setObjectName("SectionLabel")

        self._cnt_lbl = QLabel("0")
        self._cnt_lbl.setStyleSheet(
            f"color:{TEXT_SECONDARY};font-size:9px;"
            f"background:{BORDER_COLOR};padding:1px 6px;border-radius:8px;"
        )
        btn_clr = QPushButton("Clear")
        btn_clr.setFixedSize(44, 20)
        btn_clr.setStyleSheet(
            f"background:{BORDER_COLOR};color:{TEXT_SECONDARY};"
            "border-radius:4px;font-size:9px;padding:0;"
        )
        btn_clr.clicked.connect(self.clear)

        hdr.addWidget(title)
        hdr.addWidget(self._cnt_lbl)
        hdr.addStretch()
        hdr.addWidget(btn_clr)
        root.addLayout(hdr)

        self._list = QListWidget()
        self._list.setWordWrap(True)
        self._list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._list)

    def add_anomaly(self, level: AnomalyLevel, score: float,
                    sensor_errors: tuple, model_name: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        t, p, f = sensor_errors
        text = (f"[{ts}] {level.label:8s} {score:5.1f}%  "
                f"T:{t:.1f} P:{p:.1f} F:{f:.1f}  [{model_name[:4]}]")
        bg = self._CRIT_BG if level == AnomalyLevel.CRITICAL else self._WARN_BG
        fg = self._CRIT_FG if level == AnomalyLevel.CRITICAL else self._WARN_FG
        self._add(text, bg, fg)

    def add_info(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._add(f"[{ts}] {msg}", self._INFO_BG, self._INFO_FG)

    def clear(self) -> None:
        self._list.clear(); self._count = 0; self._cnt_lbl.setText("0")

    def _add(self, text: str, bg: QColor, fg: QColor) -> None:
        while self._list.count() >= self._max:
            self._list.takeItem(0)
        item = QListWidgetItem(text)
        item.setBackground(bg); item.setForeground(fg)
        self._list.addItem(item)
        self._list.scrollToBottom()
        self._count += 1
        self._cnt_lbl.setText(str(self._count))
