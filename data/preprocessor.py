import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """
    Prepares raw sensor CSV data for LSTM / GRU training.

    Pipeline:
        1. Load CSV → DataFrame.
        2. Drop ``timestamp`` and ``anomaly`` columns (not model inputs).
        3. Fit ``MinMaxScaler`` on training split only (avoids leakage).
        4. Build overlapping windows of length ``sequence_length``.
        5. Targets are the values at the next time-step.
        6. Split into train / validation / test.

    Args:
        sequence_length: Number of past ticks fed as context to the RNN.
        val_ratio:       Fraction of non-test data used for validation.
        test_ratio:      Fraction of total data held out for testing.
        scaler_path:     Where to save / load the fitted scaler (pickle).
    """

    FEATURE_COLS = ["temperature", "pressure", "flow"]

    def __init__(
        self,
        sequence_length: int = 30,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        scaler_path: str = "data/scaler.pkl",
    ) -> None:
        self._seq_len = sequence_length
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._scaler_path = Path(scaler_path)
        self._scaler: Optional[MinMaxScaler] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        return self._scaler

    @property
    def n_features(self) -> int:
        return len(self.FEATURE_COLS)

    @property
    def sequence_length(self) -> int:
        return self._seq_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, csv_path: str
    ) -> Tuple[
        np.ndarray, np.ndarray,   # X_train, y_train
        np.ndarray, np.ndarray,   # X_val,   y_val
        np.ndarray, np.ndarray,   # X_test,  y_test
    ]:
        """
        Full preprocessing pipeline.

        Args:
            csv_path: Path to the raw sensor CSV produced by DatasetGenerator.

        Returns:
            Six numpy arrays: X_train, y_train, X_val, y_val, X_test, y_test.
            Each X has shape ``(N, sequence_length, n_features)``.
            Each y has shape ``(N, n_features)``.
        """
        df = self._load(csv_path)
        features = df[self.FEATURE_COLS].values.astype(np.float32)

        # ---------- chronological train/test split (no shuffling) ----------
        n_test = int(len(features) * self._test_ratio)
        train_val_raw = features[: len(features) - n_test]
        test_raw = features[len(features) - n_test :]

        # ---------- fit scaler on training portion only --------------------
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        train_val_scaled = self._scaler.fit_transform(train_val_raw)
        test_scaled = self._scaler.transform(test_raw)
        self._save_scaler()

        # ---------- build sequences ----------------------------------------
        X_tv, y_tv = self._make_sequences(train_val_scaled)
        X_test, y_test = self._make_sequences(test_scaled)

        # ---------- train / val split (no shuffle — time-series) -----------
        n_val = int(len(X_tv) * self._val_ratio)
        X_train, y_train = X_tv[:-n_val], y_tv[:-n_val]
        X_val, y_val = X_tv[-n_val:], y_tv[-n_val:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def transform_single(self, window: np.ndarray) -> np.ndarray:
        """
        Normalise a single ``(sequence_length, n_features)`` window at inference time.

        Args:
            window: Raw sensor values with shape ``(sequence_length, n_features)``.

        Returns:
            Scaled array of the same shape.
        """
        if self._scaler is None:
            self.load_scaler()
        return self._scaler.transform(window).astype(np.float32)

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled values back to original physical units.

        Args:
            scaled: Array of shape ``(N, n_features)`` in [0, 1].

        Returns:
            Array in original units.
        """
        if self._scaler is None:
            self.load_scaler()
        return self._scaler.inverse_transform(scaled)

    def save_scaler(self) -> None:
        """Public alias for persisting the scaler."""
        self._save_scaler()

    def load_scaler(self) -> None:
        """Load a previously fitted scaler from disk."""
        if not self._scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {self._scaler_path}. "
                "Run fit_transform first to generate it."
            )
        with open(self._scaler_path, "rb") as fh:
            self._scaler = pickle.load(fh)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, csv_path: str) -> pd.DataFrame:
        """Load and basic-validate the CSV."""
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        df = df.dropna(subset=self.FEATURE_COLS)
        return df

    def _make_sequences(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sequences.

        Args:
            data: 2-D array ``(timesteps, features)``.

        Returns:
            X of shape ``(N, seq_len, features)``.
            y of shape ``(N, features)``  — next-step target.
        """
        X, y = [], []
        for i in range(len(data) - self._seq_len):
            X.append(data[i : i + self._seq_len])
            y.append(data[i + self._seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _save_scaler(self) -> None:
        self._scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._scaler_path, "wb") as fh:
            pickle.dump(self._scaler, fh)
