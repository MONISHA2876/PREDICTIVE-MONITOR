# simulator/__init__.py
"""
Sensor simulator package.
Contains abstract sensor interface and concrete implementations.
Designed to be replaceable with OpenModelica, PLC, Arduino, or ESP32 outputs.
"""

from .base_sensor import BaseSensor, SensorReading
from .temperature_sensor import TemperatureSensor
from .pressure_sensor import PressureSensor
from .flow_sensor import FlowSensor
from .sensor_manager import SensorManager

__all__ = [
    "BaseSensor",
    "SensorReading",
    "TemperatureSensor",
    "PressureSensor",
    "FlowSensor",
    "SensorManager",
]
