"""
model/moving_average.py

Simple Moving-Average predictor used as a research baseline.

For each feature, the prediction is the unweighted mean of the last
``window_size`` values in the input sequence.

This intentionally naive model lets researchers quantify how much the LSTM
adds over a trivial statistical baseline.
"""

import time
from typing import Optional

import numpy as np

from .base_predictor import BasePredictor, PredictionResult


class MovingAveragePredictor(BasePredictor):
    """
    Moving-Average baseline predictor.

    No training required – predictions are computed analytically.

    Args:
        window_size: Number of trailing ticks to average.
                     If None, the entire input sequence is used.
        n_features:  Number of sensor channels (default 3).
    """

    def __init__(
        self,
        window_size: Optional[int] = 10,
        n_features: int = 3,
    ) -> None:
        super().__init__(name="Moving Average", n_features=n_features)
        self._window_size = window_size

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Moving average is always ready — no training needed."""
        return True

    def predict(self, window: np.ndarray, actual: np.ndarray) -> PredictionResult:
        """
        Predict next values as the mean of the trailing ``window_size`` ticks.

        Args:
            window: Raw sensor values, shape ``(sequence_length, n_features)``.
            actual: True next-step values,  shape ``(n_features,)``.

        Returns:
            :class:`PredictionResult` with MA prediction and errors.
        """
        t0 = time.perf_counter()

        # Slice the trailing portion
        if self._window_size is not None:
            tail = window[-self._window_size :]
        else:
            tail = window

        predicted = np.mean(tail, axis=0).astype(np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1_000

        return self._make_result(predicted, actual.astype(np.float32), elapsed_ms)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> Optional[int]:
        return self._window_size

    @window_size.setter
    def window_size(self, value: Optional[int]) -> None:
        if value is not None and value < 1:
            raise ValueError("window_size must be >= 1 or None")
        self._window_size = value
