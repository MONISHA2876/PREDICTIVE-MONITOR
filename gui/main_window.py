"""
gui/main_window.py  —  main window matching the hand-drawn sketch layout.

Layout:
    ┌──────────────────── TopStatusBar ───────────────────────────────┐
    │ Sidebar  │         SensorPanel × 3 (gauge + graph)              │
    │  buttons │  Temp       │  Water Flow   │  Pressure              │
    │          │  [gauge]    │  [gauge]      │  [gauge]               │
    │          │  [graph]    │  [graph]      │  [graph]               │
    │          ├─────────────┴───────────────┴────────────────────────┤
    │          │  Event Log (compact, bottom strip)                    │
    └──────────┴────────────────────────────────────────────────────── ┘
"""
from __future__ import annotations
import logging
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget, QMainWindow, QFrame,
)

from gui.control_panel  import ControlPanel
from gui.event_log      import EventLog
from gui.sensor_panel   import SensorPanel
from gui.status_bar     import TopStatusBar
from gui.training_dialog import TrainingDialog
from gui.styles         import (
    MAIN_STYLESHEET, TEMP_COLOR, PRES_COLOR, FLOW_COLOR,
    BG_CARD, BORDER_COLOR,
)
from model.anomaly_detector  import AnomalyDetector, AnomalyLevel
from model.moving_average    import MovingAveragePredictor
from model.lstm_model        import LSTMPredictor
from model.base_predictor    import BasePredictor
from simulator.sensor_manager import SensorManager, SensorSnapshot
from utils.config            import APP_CONFIG
from utils.worker_threads    import DatasetWorker, TrainWorker

log = logging.getLogger("predictive_monitor")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Predictive Monitoring System")
        self.setMinimumSize(1100, 680)
        self.resize(1380, 820)
        self.setStyleSheet(MAIN_STYLESHEET)

        # Core subsystems
        self._sensor_manager = SensorManager()
        self._anomaly_detector = AnomalyDetector(
            warning_threshold=APP_CONFIG.ANOMALY_WARNING_PCT,
            critical_threshold=APP_CONFIG.ANOMALY_CRITICAL_PCT,
        )
        self._ma_predictor   = MovingAveragePredictor(APP_CONFIG.MA_WINDOW)
        self._lstm_predictor = LSTMPredictor(
            sequence_length=APP_CONFIG.SEQUENCE_LENGTH,
            model_dir=str(APP_CONFIG.MODEL_DIR),
        )
        self._active_predictor: BasePredictor = self._ma_predictor
        self._history: deque = deque(maxlen=APP_CONFIG.SEQUENCE_LENGTH + 1)

        self._sim_running  = False
        self._pred_running = False
        self._tick         = 0
        self._dataset_csv  = str(APP_CONFIG.DATASET_CSV)
        self._dataset_worker: Optional[DatasetWorker] = None
        self._train_worker:   Optional[TrainWorker]   = None

        self._build_ui()
        self._connect_signals()

        self._timer = QTimer(self)
        self._timer.setInterval(APP_CONFIG.TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        log.info("MainWindow ready.")

    # ================================================================ #
    #  UI construction                                                  #
    # ================================================================ #

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────
        self._status_bar = TopStatusBar()
        outer.addWidget(self._status_bar)

        # ── Body (sidebar + main content) ────────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # LEFT: sidebar
        self._control_panel = ControlPanel()
        body.addWidget(self._control_panel)

        # Sidebar border
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setStyleSheet(f"background:{BORDER_COLOR};max-width:1px;")
        body.addWidget(vline)

        # RIGHT: sensor panels + event log
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        # Three sensor columns
        sensors_row = QHBoxLayout()
        sensors_row.setContentsMargins(8, 8, 8, 4)
        sensors_row.setSpacing(8)

        self._panel_temp = SensorPanel("Temperature", "°C",    TEMP_COLOR)
        self._panel_flow = SensorPanel("Water Flow",  "L/min", FLOW_COLOR)
        self._panel_pres = SensorPanel("Pressure",    "bar",   PRES_COLOR)

        for panel in (self._panel_temp, self._panel_flow, self._panel_pres):
            panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            sensors_row.addWidget(panel)

        right.addLayout(sensors_row, stretch=1)

        # Horizontal separator
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setStyleSheet(f"background:{BORDER_COLOR};max-height:1px;")
        right.addWidget(hline)

        # Bottom: compact event log strip
        self._event_log = EventLog(max_entries=300)
        self._event_log.setFixedHeight(130)
        self._event_log.setStyleSheet(f"background:{BG_CARD};")
        right.addWidget(self._event_log)

        body.addLayout(right, stretch=1)
        outer.addLayout(body, stretch=1)

    # ================================================================ #
    #  Signal wiring                                                    #
    # ================================================================ #

    def _connect_signals(self) -> None:
        cp = self._control_panel
        cp.sig_start_sim.connect(self._start_simulation)
        cp.sig_stop_sim.connect(self._stop_simulation)
        cp.sig_gen_dataset.connect(self._generate_dataset)
        cp.sig_train_model.connect(self._train_model)
        cp.sig_load_model.connect(self._load_model)
        cp.sig_start_pred.connect(self._start_prediction)
        cp.sig_stop_pred.connect(self._stop_prediction)
        cp.sig_model_changed.connect(self._on_model_changed)

    # ================================================================ #
    #  Real-time loop                                                   #
    # ================================================================ #

    def _on_tick(self) -> None:
        snapshot = self._sensor_manager.read_all()
        self._tick += 1
        self._status_bar.set_tick(self._tick)

        actual = np.array([
            snapshot.temperature.value,
            snapshot.pressure.value,
            snapshot.flow.value,
        ], dtype=np.float32)
        self._history.append(actual)

        if self._pred_running and self._active_predictor.is_ready():
            self._run_prediction(actual)
        else:
            # Just stream raw values; predicted = actual (flat line)
            for panel, val in zip(
                (self._panel_temp, self._panel_flow, self._panel_pres),
                (actual[0], actual[2], actual[1]),
            ):
                panel.update_values(val, val, 0.0, AnomalyLevel.NORMAL, 0.0,
                                    self._active_predictor.name)
            self._status_bar.set_status(AnomalyLevel.NORMAL)

    def _run_prediction(self, actual: np.ndarray) -> None:
        if len(self._history) < APP_CONFIG.SEQUENCE_LENGTH:
            return
        window = np.array(
            list(self._history)[-APP_CONFIG.SEQUENCE_LENGTH:],
            dtype=np.float32,
        )
        try:
            result = self._active_predictor.predict(window, actual)
        except Exception as exc:
            log.warning("Prediction error: %s", exc)
            return

        level = self._anomaly_detector.evaluate(result)
        score = self._anomaly_detector.score(result)

        # Temp=idx0, Pressure=idx1, Flow=idx2
        self._panel_temp.update_values(
            actual[0], result.predicted[0], result.error_pct[0],
            level, result.inference_ms, result.model_name,
        )
        self._panel_flow.update_values(
            actual[2], result.predicted[2], result.error_pct[2],
            level, result.inference_ms, result.model_name,
        )
        self._panel_pres.update_values(
            actual[1], result.predicted[1], result.error_pct[1],
            level, result.inference_ms, result.model_name,
        )
        self._status_bar.set_status(level)

        if level != AnomalyLevel.NORMAL:
            self._event_log.add_anomaly(
                level, score,
                (result.error_pct[0], result.error_pct[1], result.error_pct[2]),
                result.model_name,
            )

    # ================================================================ #
    #  Simulation                                                       #
    # ================================================================ #

    def _start_simulation(self) -> None:
        self._sim_running = True
        self._timer.start()
        self._control_panel.set_sim_running(True)
        self._status_bar.set_status(AnomalyLevel.NORMAL)
        self._event_log.add_info("Simulation started.")

    def _stop_simulation(self) -> None:
        self._sim_running = False
        self._timer.stop()
        self._pred_running = False
        self._control_panel.set_sim_running(False)
        self._control_panel.set_prediction_running(False)
        self._status_bar.set_idle()
        self._event_log.add_info("Simulation stopped.")
        for p in (self._panel_temp, self._panel_flow, self._panel_pres):
            p.set_idle()

    # ================================================================ #
    #  Prediction                                                       #
    # ================================================================ #

    def _start_prediction(self) -> None:
        if not self._active_predictor.is_ready():
            QMessageBox.warning(self, "Model Not Ready",
                "Moving Average is always ready.\n"
                "For LSTM: train or load a model first.")
            return
        self._pred_running = True
        self._control_panel.set_prediction_running(True)
        self._event_log.add_info(
            f"Prediction started — {self._active_predictor.name}")

    def _stop_prediction(self) -> None:
        self._pred_running = False
        self._control_panel.set_prediction_running(False)
        self._event_log.add_info("Prediction stopped.")

    # ================================================================ #
    #  Model switching                                                  #
    # ================================================================ #

    def _on_model_changed(self, name: str) -> None:
        self._active_predictor = (
            self._ma_predictor if name == "Moving Average"
            else self._lstm_predictor
        )
        self._status_bar.set_model_name(name)
        self._event_log.add_info(f"Model → {name}")
        self._control_panel.set_prediction_available(
            self._active_predictor.is_ready())

    # ================================================================ #
    #  Dataset generation                                               #
    # ================================================================ #

    def _generate_dataset(self) -> None:
        if self._dataset_worker and self._dataset_worker.isRunning():
            return
        self._control_panel.btn_gen.setEnabled(False)
        self._control_panel.show_progress(0, 100)
        self._event_log.add_info("Generating dataset…")

        self._dataset_worker = DatasetWorker(APP_CONFIG.DATASET_N_SAMPLES)
        self._dataset_worker.progress.connect(self._control_panel.show_progress)
        self._dataset_worker.finished.connect(self._on_dataset_done)
        self._dataset_worker.error.connect(self._on_worker_error)
        self._dataset_worker.start()

    def _on_dataset_done(self, path: str) -> None:
        self._dataset_csv = path
        self._control_panel.hide_progress()
        self._control_panel.btn_gen.setEnabled(True)
        self._event_log.add_info(f"Dataset ready → {Path(path).name}")

    # ================================================================ #
    #  LSTM training                                                    #
    # ================================================================ #

    def _train_model(self) -> None:
        if not Path(self._dataset_csv).exists():
            QMessageBox.warning(self, "No Dataset",
                "Generate a dataset first (⚙ Generate Data).")
            return
        if self._train_worker and self._train_worker.isRunning():
            return

        dlg = TrainingDialog(APP_CONFIG.LSTM_EPOCHS, parent=self)
        self._train_worker = TrainWorker(self._lstm_predictor, self._dataset_csv)
        self._train_worker.epoch_done.connect(dlg.update_progress)
        self._train_worker.finished.connect(
            lambda h: self._on_train_done(h, dlg))
        self._train_worker.error.connect(
            lambda e: self._on_train_error(e, dlg))
        dlg.sig_cancel.connect(self._train_worker.terminate)
        self._train_worker.start()
        dlg.exec()

    def _on_train_done(self, history: dict, dlg) -> None:
        dlg.accept()
        epochs = len(history["loss"])
        self._control_panel.set_train_status(f"✓ {epochs} epochs done")
        self._control_panel.set_prediction_available(True)
        self._event_log.add_info(f"LSTM training complete ({epochs} epochs).")
        self._control_panel._combo.setCurrentText("LSTM")

    def _on_train_error(self, msg: str, dlg) -> None:
        dlg.reject()
        QMessageBox.critical(self, "Training Failed", msg)
        self._event_log.add_info(f"Training error: {msg}")

    # ================================================================ #
    #  Load model                                                       #
    # ================================================================ #

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load LSTM Model",
            str(APP_CONFIG.MODEL_DIR),
            "Keras Model (*.keras);;All Files (*)",
        )
        if not path:
            return
        try:
            self._lstm_predictor.load_model(path)
            self._control_panel.set_prediction_available(True)
            self._control_panel._combo.setCurrentText("LSTM")
            self._event_log.add_info(f"Model loaded: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))

    # ================================================================ #
    #  Error / close                                                    #
    # ================================================================ #

    def _on_worker_error(self, msg: str) -> None:
        self._control_panel.hide_progress()
        self._control_panel.btn_gen.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        for w in (self._dataset_worker, self._train_worker):
            if w and w.isRunning():
                w.terminate(); w.wait(2000)
        super().closeEvent(event)
