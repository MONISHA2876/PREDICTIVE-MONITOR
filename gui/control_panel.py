"""
gui/control_panel.py  —  dark sidebar with grouped buttons + model selector.
"""
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QLabel, QProgressBar,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from gui.styles import (
    BORDER_COLOR, TEXT_SECONDARY,
    STATUS_NORMAL, STATUS_CRITICAL,
)


class ControlPanel(QWidget):
    sig_start_sim:    Signal = Signal()
    sig_stop_sim:     Signal = Signal()
    sig_gen_dataset:  Signal = Signal()
    sig_train_model:  Signal = Signal()
    sig_load_model:   Signal = Signal()
    sig_start_pred:   Signal = Signal()
    sig_stop_pred:    Signal = Signal()
    sig_model_changed:Signal = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(185)
        self._setup_ui()
        self._connect()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 14, 12, 14)
        root.setSpacing(6)

        # ── SIMULATION ───────────────────────────────────────────────
        root.addWidget(self._sec("SIMULATION"))
        self.btn_start_sim = self._btn("▶  Start Sim",   "SuccessButton")
        self.btn_stop_sim  = self._btn("■  Stop Sim",    "DangerButton")
        self.btn_stop_sim.setEnabled(False)
        root.addWidget(self.btn_start_sim)
        root.addWidget(self.btn_stop_sim)
        root.addWidget(self._div())

        # ── DATASET ──────────────────────────────────────────────────
        root.addWidget(self._sec("DATASET"))
        self.btn_gen = self._btn("⚙  Generate Data")
        root.addWidget(self.btn_gen)

        self._prog = QProgressBar()
        self._prog.setRange(0, 100)
        self._prog.setFixedHeight(6)
        self._prog.setTextVisible(False)
        self._prog.setVisible(False)
        root.addWidget(self._prog)
        root.addWidget(self._div())

        # ── MODEL ────────────────────────────────────────────────────
        root.addWidget(self._sec("MODEL"))
        self.btn_train      = self._btn("🧠  Train LSTM")
        self.btn_load       = self._btn("📂  Load Model")
        self.btn_start_pred = self._btn("⚡  Start Pred", "SuccessButton")
        self.btn_stop_pred  = self._btn("⛔  Stop Pred",  "DangerButton")
        self.btn_start_pred.setEnabled(False)
        self.btn_stop_pred.setEnabled(False)
        for b in (self.btn_train, self.btn_load,
                  self.btn_start_pred, self.btn_stop_pred):
            root.addWidget(b)
        root.addWidget(self._div())

        # ── ACTIVE MODEL ─────────────────────────────────────────────
        root.addWidget(self._sec("ACTIVE MODEL"))
        self._combo = QComboBox()
        self._combo.addItems(["Moving Average", "LSTM"])
        root.addWidget(self._combo)

        # Status text
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(
            f"font-size:9px;color:{TEXT_SECONDARY};"
        )
        root.addWidget(self._status_lbl)
        root.addStretch()

        # ── THRESHOLDS (read-only info) ──────────────────────────────
        root.addWidget(self._div())
        root.addWidget(self._sec("THRESHOLDS"))
        info = QLabel(
            f"<span style='color:{STATUS_NORMAL}'>●</span> 0–5 %  Normal<br>"
            f"<span style='color:#D29922'>●</span> 5–15 %  Warning<br>"
            f"<span style='color:{STATUS_CRITICAL}'>●</span> &gt;15 %  Critical"
        )
        info.setStyleSheet(f"font-size:9px;color:{TEXT_SECONDARY};line-height:160%;")
        info.setTextFormat(Qt.TextFormat.RichText)  # RichText
        root.addWidget(info)

    def _connect(self) -> None:
        self.btn_start_sim.clicked.connect(self.sig_start_sim)
        self.btn_stop_sim.clicked.connect(self.sig_stop_sim)
        self.btn_gen.clicked.connect(self.sig_gen_dataset)
        self.btn_train.clicked.connect(self.sig_train_model)
        self.btn_load.clicked.connect(self.sig_load_model)
        self.btn_start_pred.clicked.connect(self.sig_start_pred)
        self.btn_stop_pred.clicked.connect(self.sig_stop_pred)
        self._combo.currentTextChanged.connect(self.sig_model_changed)

    # ── Public state helpers ─────────────────────────────────────────
    def set_sim_running(self, v: bool) -> None:
        self.btn_start_sim.setEnabled(not v)
        self.btn_stop_sim.setEnabled(v)

    def set_prediction_running(self, v: bool) -> None:
        self.btn_start_pred.setEnabled(not v)
        self.btn_stop_pred.setEnabled(v)

    def set_prediction_available(self, v: bool) -> None:
        if not self.btn_stop_pred.isEnabled():
            self.btn_start_pred.setEnabled(v)

    def show_progress(self, cur: int, tot: int) -> None:
        self._prog.setVisible(True)
        self._prog.setValue(int(cur / tot * 100) if tot else 0)

    def hide_progress(self) -> None:
        self._prog.setVisible(False); self._prog.setValue(0)

    def set_train_status(self, text: str) -> None:
        self._status_lbl.setText(text)

    @property
    def active_model_name(self) -> str:
        return self._combo.currentText()

    # ── Widget factories ────────────────────────────────────────────
    @staticmethod
    def _btn(label: str, obj: str = "") -> QPushButton:
        b = QPushButton(label)
        b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        b.setFixedHeight(32)
        if obj:
            b.setObjectName(obj)
        return b

    @staticmethod
    def _sec(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionLabel")
        return lbl

    @staticmethod
    def _div() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet(f"background:{BORDER_COLOR};border:none;")
        return f
