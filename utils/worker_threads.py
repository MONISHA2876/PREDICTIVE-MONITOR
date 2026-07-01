"""
utils/worker_threads.py

PySide6 QThread workers for long-running background operations.

Offloads dataset generation and LSTM training from the GUI thread so
the interface remains responsive during heavy computation.

Each worker emits typed Qt signals for progress, completion, and errors.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from data.generator import DatasetGenerator
from data.preprocessor import DataPreprocessor
from model.lstm_model import LSTMPredictor
from utils.config import APP_CONFIG

log = logging.getLogger("predictive_monitor")


# ──────────────────────────────────────────────────────────────────────
# Dataset generation worker
# ──────────────────────────────────────────────────────────────────────

class DatasetWorker(QThread):
    """
    Generates the sensor dataset in a background thread.

    Signals:
        progress(int, int): (current_sample, total_samples)
        finished(str):      Absolute path of the written CSV.
        error(str):         Error message if generation failed.
    """

    progress: Signal = Signal(int, int)
    finished: Signal = Signal(str)
    error: Signal = Signal(str)

    def __init__(
        self,
        n_samples: int = APP_CONFIG.DATASET_N_SAMPLES,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._n_samples = n_samples

    def run(self) -> None:
        try:
            generator = DatasetGenerator(
                output_dir=str(APP_CONFIG.DATA_DIR),
                n_samples=self._n_samples,
                progress_cb=lambda cur, tot: self.progress.emit(cur, tot),
            )
            path, elapsed = generator.generate_with_timing()
            log.info("Dataset generated in %.1f s → %s", elapsed, path)
            self.finished.emit(str(path))
        except Exception as exc:
            log.exception("Dataset generation failed")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────
# LSTM training worker
# ──────────────────────────────────────────────────────────────────────

class TrainWorker(QThread):
    """
    Runs data preprocessing and LSTM training in a background thread.

    Signals:
        epoch_done(int, int, float, float): (epoch, total, loss, val_loss)
        finished(dict):  Training history dict {loss: [...], val_loss: [...]}.
        error(str):      Error message if training failed.
    """

    epoch_done: Signal = Signal(int, int, float, float)
    finished: Signal = Signal(dict)
    error: Signal = Signal(str)

    def __init__(
        self,
        lstm: LSTMPredictor,
        csv_path: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._lstm = lstm
        self._csv_path = csv_path

    def run(self) -> None:
        try:
            preprocessor = DataPreprocessor(
                sequence_length=APP_CONFIG.SEQUENCE_LENGTH,
                val_ratio=APP_CONFIG.VAL_RATIO,
                test_ratio=APP_CONFIG.TEST_RATIO,
                scaler_path=str(APP_CONFIG.SCALER_PKL),
            )

            log.info("Preprocessing dataset …")
            X_train, y_train, X_val, y_val, X_test, y_test = (
                preprocessor.fit_transform(self._csv_path)
            )
            log.info(
                "Shapes  train=%s  val=%s  test=%s",
                X_train.shape, X_val.shape, X_test.shape,
            )

            self._lstm.build_model()

            history = self._lstm.train(
                X_train, y_train, X_val, y_val,
                preprocessor=preprocessor,
                epochs=APP_CONFIG.LSTM_EPOCHS,
                batch_size=APP_CONFIG.LSTM_BATCH_SIZE,
                progress_cb=lambda ep, tot, loss, vloss: self.epoch_done.emit(
                    ep, tot, float(loss), float(vloss)
                ),
            )

            self._lstm.save_model()
            log.info("LSTM model saved.")
            self.finished.emit(history)

        except Exception as exc:
            log.exception("Training failed")
            self.error.emit(str(exc))
