# model/__init__.py
"""
Model package.

Contains:
    - BasePredictor: abstract predictor interface.
    - MovingAveragePredictor: simple statistical baseline.
    - LSTMPredictor: deep-learning LSTM-based predictor.
    - AnomalyDetector: threshold-based anomaly scoring on top of any predictor.
"""

from .base_predictor import BasePredictor, PredictionResult
from .moving_average import MovingAveragePredictor
from .lstm_model import LSTMPredictor
from .anomaly_detector import AnomalyDetector, AnomalyLevel

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "MovingAveragePredictor",
    "LSTMPredictor",
    "AnomalyDetector",
    "AnomalyLevel",
]
