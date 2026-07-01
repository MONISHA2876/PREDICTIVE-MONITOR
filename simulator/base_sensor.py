"""
simulator/base_sensor.py

Abstract base class defining the sensor interface.
All sensor implementations (simulated or real) must conform to this contract.
Replacing with OpenModelica / Arduino / ESP32 requires only implementing this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SensorReading:
    """
    Immutable data container for a single sensor measurement.

    Attributes:
        sensor_id:  Unique identifier for the sensor.
        sensor_type: Human-readable type label (e.g. 'temperature').
        value:       The measured / simulated scalar value.
        unit:        Physical unit string (e.g. '°C', 'bar', 'L/min').
        timestamp:   UTC datetime of the reading.
        is_anomaly:  Ground-truth anomaly flag (used during dataset generation).
        metadata:    Optional extra fields (e.g. fault_mode, severity).
    """

    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_anomaly: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a flat dictionary (useful for CSV / logging)."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "is_anomaly": int(self.is_anomaly),
            **self.metadata,
        }


class BaseSensor(ABC):
    """
    Abstract interface for all sensors.

    Concrete implementations:
        - Simulated sensors (this module)
        - OpenModelica co-simulation adapter
        - PLC / SCADA adapter
        - Arduino / ESP32 serial adapter

    Every subclass must implement :meth:`read` and may optionally
    override :meth:`reset` and :meth:`configure`.
    """

    def __init__(self, sensor_id: str, sensor_type: str, unit: str) -> None:
        self._sensor_id = sensor_id
        self._sensor_type = sensor_type
        self._unit = unit
        self._reading_count: int = 0
        self._last_reading: Optional[SensorReading] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    @property
    def sensor_type(self) -> str:
        return self._sensor_type

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def last_reading(self) -> Optional[SensorReading]:
        return self._last_reading

    @property
    def reading_count(self) -> int:
        return self._reading_count

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def read(self) -> SensorReading:
        """
        Obtain one reading from the sensor.

        Returns:
            A :class:`SensorReading` populated with current data.
        """

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state (e.g. after fault injection ends)."""
        self._reading_count = 0
        self._last_reading = None

    def configure(self, **kwargs) -> None:
        """
        Runtime configuration hook (threshold changes, noise level, etc.).
        Subclasses should call ``super().configure(**kwargs)`` first.
        """

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _record(self, reading: SensorReading) -> SensorReading:
        """Bookkeeping called by every concrete :meth:`read` implementation."""
        self._reading_count += 1
        self._last_reading = reading
        return reading
