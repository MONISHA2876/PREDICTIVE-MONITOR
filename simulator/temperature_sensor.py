"""
simulator/temperature_sensor.py

Simulated temperature sensor.

Normal behaviour: ~25 °C with smooth Gaussian noise.
Anomaly modes:
    - overheating: value ramps up to a high temperature then decays.
    - rapid_drop : brief sharp dip (cold-water injection scenario).

Design note: replace this class with an OpenModelica or hardware adapter
without touching any other module.
"""

import math
import random
from datetime import datetime, timezone

from .base_sensor import BaseSensor, SensorReading


class TemperatureSensor(BaseSensor):
    """
    Simulated temperature sensor with configurable anomaly injection.

    Args:
        sensor_id:           Unique identifier string.
        baseline:            Normal operating temperature in °C.
        noise_std:           Standard deviation of Gaussian background noise.
        anomaly_probability: Per-tick probability of triggering an anomaly.
        overheat_amplitude:  Extra °C added at the peak of an overheat event.
        overheat_duration:   Ticks the overheat ramp lasts before decay.
    """

    def __init__(
        self,
        sensor_id: str = "TEMP-01",
        baseline: float = 25.0,
        noise_std: float = 0.4,
        anomaly_probability: float = 0.02,
        overheat_amplitude: float = 35.0,
        overheat_duration: int = 15,
    ) -> None:
        super().__init__(sensor_id, "temperature", "°C")
        self._baseline = baseline
        self._noise_std = noise_std
        self._anomaly_prob = anomaly_probability
        self._overheat_amplitude = overheat_amplitude
        self._overheat_duration = overheat_duration

        # Smooth drift state
        self._drift_phase: float = random.uniform(0, 2 * math.pi)
        self._drift_amplitude: float = random.uniform(0.5, 1.5)

        # Anomaly state machine
        self._in_anomaly: bool = False
        self._anomaly_tick: int = 0
        self._anomaly_mode: str = "none"

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def read(self) -> SensorReading:
        """Generate one temperature reading."""
        self._drift_phase += 0.05  # advance slow sinusoidal drift

        # Smooth background variation
        drift = self._drift_amplitude * math.sin(self._drift_phase)
        noise = random.gauss(0, self._noise_std)
        value = self._baseline + drift + noise

        is_anomaly = False

        # Anomaly state machine
        if not self._in_anomaly and random.random() < self._anomaly_prob:
            self._in_anomaly = True
            self._anomaly_tick = 0
            self._anomaly_mode = random.choice(["overheat", "rapid_drop"])

        if self._in_anomaly:
            is_anomaly = True
            progress = self._anomaly_tick / self._overheat_duration

            if self._anomaly_mode == "overheat":
                # Ramp up, hold, decay
                if progress < 0.4:
                    value += self._overheat_amplitude * (progress / 0.4)
                elif progress < 0.7:
                    value += self._overheat_amplitude
                else:
                    value += self._overheat_amplitude * (1.0 - (progress - 0.7) / 0.3)

            elif self._anomaly_mode == "rapid_drop":
                # Short sharp dip
                value -= 10.0 * math.sin(math.pi * progress)

            self._anomaly_tick += 1
            if self._anomaly_tick >= self._overheat_duration:
                self._in_anomaly = False
                self._anomaly_mode = "none"

        reading = SensorReading(
            sensor_id=self._sensor_id,
            sensor_type=self._sensor_type,
            value=round(value, 3),
            unit=self._unit,
            timestamp=datetime.now(timezone.utc),
            is_anomaly=is_anomaly,
            metadata={"mode": self._anomaly_mode if is_anomaly else "normal"},
        )
        return self._record(reading)

    def configure(self, **kwargs) -> None:
        """Runtime configuration (noise, baseline, thresholds)."""
        super().configure(**kwargs)
        if "baseline" in kwargs:
            self._baseline = float(kwargs["baseline"])
        if "noise_std" in kwargs:
            self._noise_std = float(kwargs["noise_std"])
        if "anomaly_probability" in kwargs:
            self._anomaly_prob = float(kwargs["anomaly_probability"])
