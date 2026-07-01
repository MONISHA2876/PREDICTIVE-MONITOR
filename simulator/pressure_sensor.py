"""
simulator/pressure_sensor.py

Simulated pressure sensor.

Normal behaviour: stable pressure around 2.5 bar with gradual fluctuations.
Anomaly modes:
    - spike: sudden sharp pressure spike (valve slam, water hammer).
    - drop:  sudden pressure drop (pipe burst / valve failure).
"""

import math
import random
from datetime import datetime, timezone

from .base_sensor import BaseSensor, SensorReading


class PressureSensor(BaseSensor):
    """
    Simulated pressure sensor with spike and drop anomaly injection.

    Args:
        sensor_id:           Unique identifier string.
        baseline:            Normal operating pressure in bar.
        noise_std:           Standard deviation of Gaussian background noise.
        anomaly_probability: Per-tick probability of triggering an anomaly.
        spike_amplitude:     Bar added during a pressure spike event.
        drop_amplitude:      Bar removed during a pressure drop event.
        event_duration:      Ticks an anomaly event lasts.
    """

    def __init__(
        self,
        sensor_id: str = "PRES-01",
        baseline: float = 2.5,
        noise_std: float = 0.05,
        anomaly_probability: float = 0.025,
        spike_amplitude: float = 1.8,
        drop_amplitude: float = 1.5,
        event_duration: int = 10,
    ) -> None:
        super().__init__(sensor_id, "pressure", "bar")
        self._baseline = baseline
        self._noise_std = noise_std
        self._anomaly_prob = anomaly_probability
        self._spike_amplitude = spike_amplitude
        self._drop_amplitude = drop_amplitude
        self._event_duration = event_duration

        # Slow sinusoidal drift
        self._drift_phase: float = random.uniform(0, 2 * math.pi)
        self._drift_amplitude: float = random.uniform(0.05, 0.15)

        # Anomaly state
        self._in_anomaly: bool = False
        self._anomaly_tick: int = 0
        self._anomaly_mode: str = "none"

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def read(self) -> SensorReading:
        """Generate one pressure reading."""
        self._drift_phase += 0.03

        drift = self._drift_amplitude * math.sin(self._drift_phase)
        noise = random.gauss(0, self._noise_std)
        value = self._baseline + drift + noise

        is_anomaly = False

        # Trigger new anomaly
        if not self._in_anomaly and random.random() < self._anomaly_prob:
            self._in_anomaly = True
            self._anomaly_tick = 0
            self._anomaly_mode = random.choice(["spike", "drop"])

        if self._in_anomaly:
            is_anomaly = True
            progress = self._anomaly_tick / self._event_duration

            if self._anomaly_mode == "spike":
                # Sharp triangular spike
                if progress < 0.3:
                    value += self._spike_amplitude * (progress / 0.3)
                else:
                    value += self._spike_amplitude * max(0, 1.0 - (progress - 0.3) / 0.7)

            elif self._anomaly_mode == "drop":
                # Sustained pressure drop
                if progress < 0.2:
                    value -= self._drop_amplitude * (progress / 0.2)
                elif progress < 0.8:
                    value -= self._drop_amplitude
                else:
                    value -= self._drop_amplitude * (1.0 - (progress - 0.8) / 0.2)

            self._anomaly_tick += 1
            if self._anomaly_tick >= self._event_duration:
                self._in_anomaly = False
                self._anomaly_mode = "none"

        # Clamp to physically plausible range
        value = max(0.0, value)

        reading = SensorReading(
            sensor_id=self._sensor_id,
            sensor_type=self._sensor_type,
            value=round(value, 4),
            unit=self._unit,
            timestamp=datetime.now(timezone.utc),
            is_anomaly=is_anomaly,
            metadata={"mode": self._anomaly_mode if is_anomaly else "normal"},
        )
        return self._record(reading)

    def configure(self, **kwargs) -> None:
        super().configure(**kwargs)
        if "baseline" in kwargs:
            self._baseline = float(kwargs["baseline"])
        if "noise_std" in kwargs:
            self._noise_std = float(kwargs["noise_std"])
        if "anomaly_probability" in kwargs:
            self._anomaly_prob = float(kwargs["anomaly_probability"])
