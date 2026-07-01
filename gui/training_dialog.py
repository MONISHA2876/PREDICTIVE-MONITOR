"""
gui/training_dialog.py

Modal dialog shown during LSTM training.

Displays:
    - Current epoch / total epochs
    - Training loss
    - Validation loss
    - A progress bar
    - A cancel button (stops the training thread)
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)

from gui.styles import ACCENT_BLUE, TEXT_SECONDARY


class TrainingDialog(QDialog):
    """
    Non-closeable progress dialog for background LSTM training.

    Signals:
        sig_cancel: User requested training cancellation.
    """

    sig_cancel: Signal = Signal()

    def __init__(self, total_epochs: int, parent: Optional[QDialog] = None) -> None:
        super().__init__(parent)
        self._total = total_epochs
        self.setWindowTitle("Training LSTM Model")
        self.setFixedSize(400, 210)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Training LSTM Model…")
        title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {ACCENT_BLUE};")
        layout.addWidget(title)

        self._lbl_epoch = QLabel(f"Epoch 0 / {self._total}")
        layout.addWidget(self._lbl_epoch)

        self._progress = QProgressBar()
        self._progress.setRange(0, self._total)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        metrics_row = QHBoxLayout()
        self._lbl_loss = QLabel("Loss: —")
        self._lbl_val  = QLabel("Val Loss: —")
        for lbl in (self._lbl_loss, self._lbl_val):
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        metrics_row.addWidget(self._lbl_loss)
        metrics_row.addStretch()
        metrics_row.addWidget(self._lbl_val)
        layout.addLayout(metrics_row)

        self._lbl_note = QLabel("Early stopping is active — training may finish sooner.")
        self._lbl_note.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px; font-style: italic;"
        )
        layout.addWidget(self._lbl_note)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedWidth(90)
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update_progress(
        self, epoch: int, total: int, loss: float, val_loss: float
    ) -> None:
        """Called on every epoch completion from the training worker."""
        self._lbl_epoch.setText(f"Epoch {epoch} / {total}")
        self._progress.setValue(epoch)
        self._lbl_loss.setText(f"Loss: {loss:.6f}")
        self._lbl_val.setText(f"Val Loss: {val_loss:.6f}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_cancel(self) -> None:
        self.sig_cancel.emit()
        self.reject()

    def closeEvent(self, event) -> None:
        """Prevent accidental close via window decorations."""
        event.ignore()
