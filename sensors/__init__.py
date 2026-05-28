"""Virtual IMU and wheel-odometry sensors with selectable noise profiles.

The package re-exports the public API from :mod:`sensors.sensors` plus
the noise-related dataclasses / presets from the project-level
:mod:`config` module, so a single ``from sensors import ...`` is enough
for typical callers.
"""

from .sensors import (
    IMU_PRESETS,
    WHEEL_PRESETS,
    ImuConfig,
    ImuSensor,
    NoiseLevel,
    WheelOdomConfig,
    WheelOdometrySensor,
    make_sensors,
)

__all__ = [
    "NoiseLevel",
    "ImuConfig",
    "WheelOdomConfig",
    "IMU_PRESETS",
    "WHEEL_PRESETS",
    "ImuSensor",
    "WheelOdometrySensor",
    "make_sensors",
]
