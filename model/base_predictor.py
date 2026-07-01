"""
model/base_predictor.py

Abstract predictor interface shared by the Moving-Average baseline and the
LSTM model. Any new model (Transformer, ARIMA, etc.) just needs to implement
this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PredictionResult:
    """
    Container for one prediction step.

    Attributes:
        predicted:      Predicted next-step values (original units), shape ``(n_features,)``.
        actual:         Actual measured values (original units), shape ``(n_features,)``.
        error_abs:      Per-feature absolute prediction error.
        error_pct:      Per-feature percentage error (relative to |actual|).
        inference_ms:   Time taken to produce the prediction in milliseconds.
        model_name:     Human-readable identifier for the model.
    """

    predicted: np.ndarray
    actual: np.ndarray
    error_abs: np.ndarray
    error_pct: np.ndarray
    inference_ms: float
    model_name: str

    @property
    def mean_error_pct(self) -> float:
        """Average percentage error across all features."""
        return float(np.mean(self.error_pct))

    @property
    def max_error_pct(self) -> float:
        """Worst-case percentage error across features."""
        return float(np.max(self.error_pct))


class BasePredictor(ABC):
    """
    Abstract interface for all predictor models.

    Implementations:
        - :class:`MovingAveragePredictor` (statistical baseline)
        - :class:`LSTMPredictor` (deep-learning model)

    All predictors receive a ``(sequence_length, n_features)`` window of
    *raw* (unscaled) sensor values and return a :class:`PredictionResult`.
    Scaling is handled internally per implementation.
    """

    def __init__(self, name: str, n_features: int = 3) -> None:
        self._name = name
        self._n_features = n_features
        self._prediction_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def prediction_count(self) -> int:
        return self._prediction_count

    @abstractmethod
    def predict(self, window: np.ndarray, actual: np.ndarray) -> PredictionResult:
        """
        Predict the next time-step values given a history window.

        Args:
            window: Raw sensor readings, shape ``(sequence_length, n_features)``.
            actual: The true next-step values for error computation, shape ``(n_features,)``.

        Returns:
            :class:`PredictionResult` with predictions, errors, and timing.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the model is ready to make predictions."""

    def _make_result(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        inference_ms: float,
    ) -> PredictionResult:
        """Factory helper shared by subclasses."""
        self._prediction_count += 1
        error_abs = np.abs(predicted - actual)
        # Guard against division by zero
        denominator = np.where(np.abs(actual) > 1e-9, np.abs(actual), 1e-9)
        error_pct = (error_abs / denominator) * 100.0
        return PredictionResult(
            predicted=predicted.copy(),
            actual=actual.copy(),
            error_abs=error_abs,
            error_pct=error_pct,
            inference_ms=inference_ms,
            model_name=self._name,
        )
