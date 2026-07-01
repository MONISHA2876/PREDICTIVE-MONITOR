"""
utils/config.py

Central configuration for the predictive monitoring system.

All tuneable parameters live here so that changing a value propagates
throughout the application without hunting across multiple files.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Application-wide constants. Treat as read-only."""

    # ── Paths ─────────────────────────────────────────────────────────
    DATA_DIR: Path = Path("data")
    DATASET_CSV: Path = Path("data/dataset.csv")
    SCALER_PKL: Path = Path("data/scaler.pkl")
    MODEL_DIR: Path = Path("model/saved")
    HISTORY_JSON: Path = Path("model/saved/training_history.json")

    # ── Simulator ─────────────────────────────────────────────────────
    DATASET_N_SAMPLES: int = 5_000
    TICK_INTERVAL_MS: int = 1_000       # GUI refresh rate (ms)

    # ── Preprocessing ─────────────────────────────────────────────────
    SEQUENCE_LENGTH: int = 30
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # ── LSTM hyper-parameters ─────────────────────────────────────────
    LSTM_UNITS: int = 64
    LSTM_DROPOUT: float = 0.2
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 64

    # ── Moving-Average baseline ───────────────────────────────────────
    MA_WINDOW: int = 10

    # ── Anomaly detection thresholds (%) ──────────────────────────────
    ANOMALY_WARNING_PCT: float = 5.0
    ANOMALY_CRITICAL_PCT: float = 15.0

    # ── Graph ─────────────────────────────────────────────────────────
    GRAPH_HISTORY_LEN: int = 120        # number of ticks visible on graph

    # ── Feature column names (order must match DataPreprocessor) ──────
    FEATURE_NAMES: tuple = ("temperature", "pressure", "flow")
    FEATURE_UNITS: tuple = ("°C", "bar", "L/min")


# Singleton instance used across the application
APP_CONFIG = Config()
