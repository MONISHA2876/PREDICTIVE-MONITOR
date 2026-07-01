"""
model/lstm_model.py

LSTM-based next-step predictor.

Architecture: two stacked LSTM layers with dropout, followed by a Dense output.

Training utilities:
    - EarlyStopping on validation loss.
    - ModelCheckpoint to save best weights.
    - Training history saved as JSON for later plotting.

Inference:
    - Preprocesses raw window through the fitted MinMaxScaler.
    - Returns predictions in original physical units.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base_predictor import BasePredictor, PredictionResult
from data.preprocessor import DataPreprocessor


class LSTMPredictor(BasePredictor):
    """
    LSTM-based time-series predictor.

    Args:
        sequence_length: Length of the input window (must match preprocessor).
        n_features:      Number of sensor channels.
        model_dir:       Directory for saving model weights and history.
        units:           Number of units in each LSTM layer.
        dropout:         Dropout rate applied after each LSTM layer.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 3,
        model_dir: str = "model/saved",
        units: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(name="LSTM", n_features=n_features)
        self._seq_len = sequence_length
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._units = units
        self._dropout = dropout

        self._model = None          # type: ignore[assignment]
        self._preprocessor: Optional[DataPreprocessor] = None
        self._is_ready: bool = False
        self._history: dict = {}

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._is_ready and self._model is not None

    def predict(self, window: np.ndarray, actual: np.ndarray) -> PredictionResult:
        """
        Predict the next sensor values from a raw history window.

        Args:
            window: Raw (unscaled) values, shape ``(sequence_length, n_features)``.
            actual: True next-step values  (raw), shape ``(n_features,)``.

        Returns:
            :class:`PredictionResult` in original physical units.
        """
        if not self.is_ready():
            raise RuntimeError("LSTM model is not loaded. Call load_model() first.")

        t0 = time.perf_counter()

        # Scale the window
        scaled_window = self._preprocessor.transform_single(window)  # (seq, features)
        x = scaled_window[np.newaxis, ...]                            # (1, seq, features)

        scaled_pred = self._model.predict(x, verbose=0)              # (1, features)
        predicted_raw = self._preprocessor.inverse_transform(scaled_pred)[0]

        elapsed_ms = (time.perf_counter() - t0) * 1_000

        return self._make_result(
            np.array(predicted_raw, dtype=np.float32),
            np.array(actual, dtype=np.float32),
            elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        """Construct the Keras model graph."""
        # Deferred import so the app doesn't crash if TensorFlow is absent
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential(
            [
                LSTM(
                    self._units,
                    input_shape=(self._seq_len, self._n_features),
                    return_sequences=True,
                ),
                Dropout(self._dropout),
                LSTM(self._units // 2, return_sequences=False),
                Dropout(self._dropout),
                Dense(self._n_features),
            ],
            name="sensor_lstm",
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self._model = model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        preprocessor: DataPreprocessor,
        epochs: int = 50,
        batch_size: int = 64,
        progress_cb=None,
    ) -> dict:
        """
        Train the LSTM and save best weights.

        Args:
            X_train:     Training sequences, shape ``(N, seq_len, features)``.
            y_train:     Training targets,   shape ``(N, features)``.
            X_val:       Validation sequences.
            y_val:       Validation targets.
            preprocessor: Fitted DataPreprocessor (needed for inference scaling).
            epochs:      Maximum training epochs.
            batch_size:  Mini-batch size.
            progress_cb: Optional ``fn(epoch, total_epochs, loss, val_loss)``.

        Returns:
            Training history dict with ``loss`` and ``val_loss`` lists.
        """
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

        self._preprocessor = preprocessor

        if self._model is None:
            self.build_model()

        checkpoint_path = str(self._model_dir / "best_weights.keras")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=0,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        # Optional progress callback wrapper
        if progress_cb is not None:
            class _ProgressCB(Callback):
                def on_epoch_end(self_, epoch, logs=None):
                    logs = logs or {}
                    progress_cb(
                        epoch + 1,
                        epochs,
                        logs.get("loss", 0.0),
                        logs.get("val_loss", 0.0),
                    )

            callbacks.append(_ProgressCB())

        history = self._model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,
        )

        self._history = {
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }
        self._save_history()
        self._is_ready = True
        return self._history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> Path:
        """Save Keras model weights to disk."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        save_path = Path(path) if path else self._model_dir / "lstm_model.keras"
        self._model.save(save_path)
        return save_path

    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a previously trained model from disk.

        Also loads the fitted scaler so inference can run.

        Args:
            path: Path to the ``.keras`` model file. Defaults to the model dir.
        """
        import tensorflow as tf

        load_path = Path(path) if path else self._model_dir / "lstm_model.keras"
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")

        self._model = tf.keras.models.load_model(str(load_path))

        # Load scaler
        preprocessor = DataPreprocessor(sequence_length=self._seq_len)
        preprocessor.load_scaler()
        self._preprocessor = preprocessor
        self._is_ready = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        hist_path = self._model_dir / "training_history.json"
        with open(hist_path, "w", encoding="utf-8") as fh:
            json.dump(self._history, fh, indent=2)

    @property
    def history(self) -> dict:
        return self._history
