"""
simulator/sensor_manager.py

Aggregates readings from all sensors into a single timestamped snapshot.
Acts as the single entry-point that the rest of the system uses.

Replacing the simulator later:
    Subclass SensorManager (or inject different BaseSensor implementations)
    without touching the GUI or model layers.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from .base_sensor import BaseSensor, SensorReading
from .temperature_sensor import TemperatureSensor
from .pressure_sensor import PressureSensor
from .flow_sensor import FlowSensor


@dataclass
class SensorSnapshot:
    """
    A synchronised reading from all three sensors at one instant.

    Attributes:
        timestamp:   UTC datetime when the snapshot was taken.
        temperature: Temperature sensor reading.
        pressure:    Pressure sensor reading.
        flow:        Flow sensor reading.
        any_anomaly: True if at least one sensor is in anomaly state.
    """

    timestamp: datetime
    temperature: SensorReading
    pressure: SensorReading
    flow: SensorReading
    any_anomaly: bool

    def to_dict(self) -> dict:
        """Flat dict for CSV export / logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "temperature": self.temperature.value,
            "pressure": self.pressure.value,
            "flow": self.flow.value,
            "anomaly": int(self.any_anomaly),
        }


class SensorManager:
    """
    Manages a collection of sensors and produces synchronised snapshots.

    Usage::

        manager = SensorManager()
        snapshot = manager.read_all()
        print(snapshot.temperature.value)

    Sensors can be replaced via the ``register_sensor`` method, allowing
    hardware or simulation adapters to be swapped without modifying callers.
    """

    def __init__(self) -> None:
        self._sensors: Dict[str, BaseSensor] = {}
        self._snapshot_count: int = 0
        self._last_snapshot: Optional[SensorSnapshot] = None

        # Default simulated sensors
        self._sensors["temperature"] = TemperatureSensor()
        self._sensors["pressure"] = PressureSensor()
        self._sensors["flow"] = FlowSensor()

    # ------------------------------------------------------------------
    # Sensor registration (dependency injection point)
    # ------------------------------------------------------------------

    def register_sensor(self, key: str, sensor: BaseSensor) -> None:
        """
        Register or replace a sensor under the given key.

        Args:
            key:    Logical name (``'temperature'``, ``'pressure'``, ``'flow'``).
            sensor: Any :class:`BaseSensor` implementation.
        """
        self._sensors[key] = sensor

    # ------------------------------------------------------------------
    # Reading interface
    # ------------------------------------------------------------------

    def read_all(self) -> SensorSnapshot:
        """
        Read all registered sensors and return a synchronised snapshot.

        Returns:
            :class:`SensorSnapshot` with readings from every sensor.
        """
        now = datetime.now(timezone.utc)
        temp = self._sensors["temperature"].read()
        pres = self._sensors["pressure"].read()
        flow = self._sensors["flow"].read()

        any_anomaly = temp.is_anomaly or pres.is_anomaly or flow.is_anomaly

        snapshot = SensorSnapshot(
            timestamp=now,
            temperature=temp,
            pressure=pres,
            flow=flow,
            any_anomaly=any_anomaly,
        )

        self._snapshot_count += 1
        self._last_snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_count

    @property
    def last_snapshot(self) -> Optional[SensorSnapshot]:
        return self._last_snapshot

    def configure_sensor(self, key: str, **kwargs) -> None:
        """Pass runtime configuration to a specific sensor."""
        if key in self._sensors:
            self._sensors[key].configure(**kwargs)

    def reset_all(self) -> None:
        """Reset all sensors to their initial state."""
        for sensor in self._sensors.values():
            sensor.reset()
        self._snapshot_count = 0
        self._last_snapshot = None
