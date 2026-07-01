"""
model/anomaly_detector.py

Threshold-based anomaly classifier that wraps any BasePredictor.

Anomaly score is derived from the mean percentage prediction error.
Three severity levels are supported: NORMAL, WARNING, CRITICAL.

The thresholds are fully configurable so that different operating
regimes (e.g. cryogenic vs. room-temperature) can be set at runtime.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import List, Optional

import numpy as np

from .base_predictor import PredictionResult


class AnomalyLevel(Enum):
    """Severity classification for anomaly events."""

    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def css_color(self) -> str:
        _MAP = {
            AnomalyLevel.NORMAL: "#27ae60",
            AnomalyLevel.WARNING: "#f39c12",
            AnomalyLevel.CRITICAL: "#e74c3c",
        }
        return _MAP[self]


@dataclass
class AnomalyEvent:
    """
    Records a single anomaly (WARNING or CRITICAL) occurrence.

    Attributes:
        timestamp:   UTC time of detection.
        level:       Severity level.
        score_pct:   Aggregate anomaly score as a percentage.
        sensor_errors: Per-feature percentage errors (temp, pressure, flow).
        model_name:  Which predictor triggered the event.
    """

    timestamp: datetime
    level: AnomalyLevel
    score_pct: float
    sensor_errors: np.ndarray
    model_name: str
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.label,
            "score_pct": round(self.score_pct, 2),
            "temp_err_pct": round(float(self.sensor_errors[0]), 2),
            "pres_err_pct": round(float(self.sensor_errors[1]), 2),
            "flow_err_pct": round(float(self.sensor_errors[2]), 2),
            "model": self.model_name,
            "message": self.message,
        }


class AnomalyDetector:
    """
    Converts a :class:`PredictionResult` into an :class:`AnomalyLevel` and
    optionally records events in an internal log.

    Args:
        warning_threshold:  % error above which WARNING is raised (default 5 %).
        critical_threshold: % error above which CRITICAL is raised (default 15 %).
        max_log_size:       Maximum number of events kept in memory.
    """

    def __init__(
        self,
        warning_threshold: float = 5.0,
        critical_threshold: float = 15.0,
        max_log_size: int = 500,
    ) -> None:
        self._warn_thr = warning_threshold
        self._crit_thr = critical_threshold
        self._max_log = max_log_size
        self._event_log: List[AnomalyEvent] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def warning_threshold(self) -> float:
        return self._warn_thr

    @warning_threshold.setter
    def warning_threshold(self, value: float) -> None:
        if value >= self._crit_thr:
            raise ValueError("warning_threshold must be < critical_threshold")
        self._warn_thr = value

    @property
    def critical_threshold(self) -> float:
        return self._crit_thr

    @critical_threshold.setter
    def critical_threshold(self, value: float) -> None:
        if value <= self._warn_thr:
            raise ValueError("critical_threshold must be > warning_threshold")
        self._crit_thr = value

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def evaluate(self, result: PredictionResult) -> AnomalyLevel:
        """
        Classify a prediction result and log if anomalous.

        Args:
            result: Output from any :class:`BasePredictor`.

        Returns:
            :class:`AnomalyLevel` for the current tick.
        """
        score = result.mean_error_pct

        if score >= self._crit_thr:
            level = AnomalyLevel.CRITICAL
        elif score >= self._warn_thr:
            level = AnomalyLevel.WARNING
        else:
            level = AnomalyLevel.NORMAL

        if level != AnomalyLevel.NORMAL:
            event = AnomalyEvent(
                timestamp=datetime.now(timezone.utc),
                level=level,
                score_pct=score,
                sensor_errors=result.error_pct,
                model_name=result.model_name,
                message=self._build_message(level, score, result),
            )
            self._log_event(event)

        return level

    def score(self, result: PredictionResult) -> float:
        """Return the aggregate anomaly score (mean % error) for a result."""
        return result.mean_error_pct

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    @property
    def event_log(self) -> List[AnomalyEvent]:
        """Read-only view of the anomaly event log."""
        return list(self._event_log)

    def clear_log(self) -> None:
        """Remove all events from the in-memory log."""
        self._event_log.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_event(self, event: AnomalyEvent) -> None:
        self._event_log.append(event)
        if len(self._event_log) > self._max_log:
            self._event_log.pop(0)

    @staticmethod
    def _build_message(
        level: AnomalyLevel, score: float, result: PredictionResult
    ) -> str:
        feature_names = ["Temp", "Pressure", "Flow"]
        worst_idx = int(np.argmax(result.error_pct))
        worst_name = feature_names[worst_idx]
        worst_err = float(result.error_pct[worst_idx])
        return (
            f"[{level.label}] Score={score:.1f}%  "
            f"Worst: {worst_name} err={worst_err:.1f}%  "
            f"Model: {result.model_name}"
        )
