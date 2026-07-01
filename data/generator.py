import csv
import os
import time
from pathlib import Path
from typing import Callable, Optional

from simulator.sensor_manager import SensorManager


class DatasetGenerator:
    """
    Runs the sensor simulator and persists readings as a CSV dataset.

    Args:
        output_dir:    Directory where the CSV file will be written.
        filename:      Output filename (default ``dataset.csv``).
        n_samples:     Number of sensor snapshots to generate.
        progress_cb:   Optional callback ``fn(current, total)`` for progress reporting.
    """

    COLUMNS = ["timestamp", "temperature", "pressure", "flow", "anomaly"]

    def __init__(
        self,
        output_dir: str = "data",
        filename: str = "dataset.csv",
        n_samples: int = 5000,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._output_dir / filename
        self._n_samples = n_samples
        self._progress_cb = progress_cb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def filepath(self) -> Path:
        """Absolute path of the generated CSV file."""
        return self._filepath.resolve()

    def generate(self) -> Path:
        """
        Run simulation and write CSV.

        Returns:
            Path to the written CSV file.
        """
        manager = SensorManager()

        with open(self._filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            writer.writeheader()

            for i in range(self._n_samples):
                snapshot = manager.read_all()
                writer.writerow(snapshot.to_dict())

                if self._progress_cb:
                    self._progress_cb(i + 1, self._n_samples)

                # Flush every 500 rows so the file is readable during generation
                if (i + 1) % 500 == 0:
                    fh.flush()

        return self._filepath

    def generate_with_timing(self) -> "tuple[Path, float]":
        """
        Generate dataset and return (filepath, elapsed_seconds).

        Useful for benchmarking / logging.
        """
        start = time.perf_counter()
        path = self.generate()
        elapsed = time.perf_counter() - start
        return path, elapsed
