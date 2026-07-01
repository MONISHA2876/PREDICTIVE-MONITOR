"""
simulator/flow_sensor.py

Simulated water-flow sensor.

Normal behaviour: stable flow ~10 L/min with small noise.
Anomaly modes:
    - blockage: gradual reduction toward zero (e.g. debris / closed valve).
    - leakage:  sudden increase above nominal (e.g. pipe rupture).
"""

import math
import random
from datetime import datetime, timezone

from .base_sensor import BaseSensor, SensorReading


class FlowSensor(BaseSensor):
    """
    Simulated water-flow sensor with blockage and leakage anomaly injection.

    Args:
        sensor_id:           Unique identifier string.
        baseline:            Normal flow rate in L/min.
        noise_std:           Standard deviation of Gaussian background noise.
        anomaly_probability: Per-tick probability of triggering an anomaly.
        blockage_depth:      Maximum L/min reduction during a blockage.
        leakage_amplitude:   L/min added during a leakage event.
        event_duration:      Ticks an anomaly event lasts.
    """

    def __init__(
        self,
        sensor_id: str = "FLOW-01",
        baseline: float = 10.0,
        noise_std: float = 0.15,
        anomaly_probability: float = 0.02,
        blockage_depth: float = 8.5,
        leakage_amplitude: float = 7.0,
        event_duration: int = 20,
    ) -> None:
        super().__init__(sensor_id, "flow", "L/min")
        self._baseline = baseline
        self._noise_std = noise_std
        self._anomaly_prob = anomaly_probability
        self._blockage_depth = blockage_depth
        self._leakage_amplitude = leakage_amplitude
        self._event_duration = event_duration

        # Slow sinusoidal drift mimicking pump cycles
        self._drift_phase: float = random.uniform(0, 2 * math.pi)
        self._drift_amplitude: float = random.uniform(0.3, 0.8)

        # Anomaly state
        self._in_anomaly: bool = False
        self._anomaly_tick: int = 0
        self._anomaly_mode: str = "none"

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def read(self) -> SensorReading:
        """Generate one flow reading."""
        self._drift_phase += 0.04

        drift = self._drift_amplitude * math.sin(self._drift_phase)
        noise = random.gauss(0, self._noise_std)
        value = self._baseline + drift + noise

        is_anomaly = False

        # Trigger new anomaly
        if not self._in_anomaly and random.random() < self._anomaly_prob:
            self._in_anomaly = True
            self._anomaly_tick = 0
            self._anomaly_mode = random.choice(["blockage", "leakage"])

        if self._in_anomaly:
            is_anomaly = True
            progress = self._anomaly_tick / self._event_duration

            if self._anomaly_mode == "blockage":
                # Gradual reduction then partial recovery
                if progress < 0.5:
                    reduction = self._blockage_depth * (progress / 0.5)
                else:
                    reduction = self._blockage_depth * (1.0 - (progress - 0.5) / 0.5)
                value -= reduction

            elif self._anomaly_mode == "leakage":
                # Sudden spike then slow decay
                if progress < 0.1:
                    value += self._leakage_amplitude * (progress / 0.1)
                else:
                    value += self._leakage_amplitude * math.exp(
                        -5.0 * (progress - 0.1)
                    )

            self._anomaly_tick += 1
            if self._anomaly_tick >= self._event_duration:
                self._in_anomaly = False
                self._anomaly_mode = "none"

        # Flow cannot be negative
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
