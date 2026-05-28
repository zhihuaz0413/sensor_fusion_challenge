"""Central configuration for the sensor-fusion testbed.

All tunable parameters that describe the robot, its sensors, and the
simulation harness live here as plain frozen dataclasses (and an enum
for the named sensor noise presets).  Keeping them all in one place
makes it easy to share, version, or serialise a full scenario without
chasing identical-looking dataclasses across three different modules.

Public surface:

- :class:`RobotShape`, :class:`RobotConfig` - chassis geometry and
  drive-train parameters.
- :class:`ImuConfig`, :class:`WheelOdomConfig` - per-sensor noise models.
- :class:`NoiseLevel` plus :data:`IMU_PRESETS` / :data:`WHEEL_PRESETS`
  and the :func:`imu_config_for` / :func:`wheel_config_for` helpers -
  named sensor noise profiles spanning "ideal" to "cheap MEMS".
- :class:`SimulationConfig` - top-level runtime knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

__all__ = [
    # Robot
    "RobotShape",
    "RobotConfig",
    # Sensors
    "ImuConfig",
    "WheelOdomConfig",
    "NoiseLevel",
    "IMU_PRESETS",
    "WHEEL_PRESETS",
    "imu_config_for",
    "wheel_config_for",
    # Simulation
    "SimulationConfig",
]


# ───────────────────────────────────────────────────────── Robot ────


@dataclass(frozen=True)
class RobotShape:
    """Geometry of the wheeled robot, in metres.

    The chassis is drawn as a rectangle of ``body_length`` x ``body_width``
    centred at the robot's reference point.  The two wheels are drawn as
    smaller rectangles offset to either side by ``wheel_offset_y``.
    """

    body_length: float = 0.30
    body_width: float = 0.22
    wheel_length: float = 0.08
    wheel_width: float = 0.025
    wheel_offset_y: float = 0.13


@dataclass(frozen=True)
class RobotConfig:
    """Physical parameters of the wheeled robot.

    `shape` is the visual / collision geometry; `wheel_base_m` and
    `wheel_radius_m` are the drive-train numbers used by anything that
    converts between wheel encoders and planar motion.
    """

    shape: RobotShape = field(default_factory=RobotShape)
    wheel_base_m: float = 0.26
    wheel_radius_m: float = 0.04


# ─────────────────────────────────────────────────────── Sensors ────


@dataclass(frozen=True)
class ImuConfig:
    """IMU noise model.

    Attributes:
        rate_hz: Sensor output frequency.
        accel_noise_std: White acceleration noise std-dev in m/s^2.
        bias_initial_std: Std-dev of the initial acceleration bias in m/s^2.
        bias_random_walk_std: Bias random walk in m/s^2/sqrt(s).
    """

    rate_hz: float = 1000.0
    accel_noise_std: float = 0.03
    bias_initial_std: float = 0.02
    bias_random_walk_std: float = 0.002


@dataclass(frozen=True)
class WheelOdomConfig:
    """Wheel-odometry noise model.

    Attributes:
        rate_hz: Sensor output frequency.
        velocity_noise_std: White velocity noise std-dev in m/s.
        scale_error: Multiplicative scale error (0.01 = +1 % over-reads).
    """

    rate_hz: float = 100.0
    velocity_noise_std: float = 0.03
    scale_error: float = 0.0


class NoiseLevel(str, Enum):
    """Named sensor noise profiles, ordered from clean to drifty."""

    IDEAL = "ideal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


IMU_PRESETS: Mapping[NoiseLevel, ImuConfig] = {
    NoiseLevel.IDEAL: ImuConfig(
        accel_noise_std=0.0,
        bias_initial_std=0.0,
        bias_random_walk_std=0.0,
    ),
    NoiseLevel.LOW: ImuConfig(
        accel_noise_std=0.02,
        bias_initial_std=0.01,
        bias_random_walk_std=0.001,
    ),
    NoiseLevel.MEDIUM: ImuConfig(
        accel_noise_std=0.10,
        bias_initial_std=0.05,
        bias_random_walk_std=0.01,
    ),
    NoiseLevel.HIGH: ImuConfig(
        accel_noise_std=1.00,
        bias_initial_std=0.80,
        bias_random_walk_std=0.20,
    ),
}

WHEEL_PRESETS: Mapping[NoiseLevel, WheelOdomConfig] = {
    NoiseLevel.IDEAL: WheelOdomConfig(
        velocity_noise_std=0.0,
        scale_error=0.0,
    ),
    NoiseLevel.LOW: WheelOdomConfig(
        velocity_noise_std=0.03,
        scale_error=0.005,
    ),
    NoiseLevel.MEDIUM: WheelOdomConfig(
        velocity_noise_std=0.08,
        scale_error=0.01,
    ),
    NoiseLevel.HIGH: WheelOdomConfig(
        velocity_noise_std=0.20,
        scale_error=0.05,
    ),
}


def _coerce_level(level: NoiseLevel | str) -> NoiseLevel:
    return level if isinstance(level, NoiseLevel) else NoiseLevel(level)


def imu_config_for(
    level: NoiseLevel | str, *, rate_hz: float | None = None
) -> ImuConfig:
    """Return the IMU config for a named noise level, optionally
    overriding the sample rate.
    """
    cfg = IMU_PRESETS[_coerce_level(level)]
    return replace(cfg, rate_hz=rate_hz) if rate_hz is not None else cfg


def wheel_config_for(
    level: NoiseLevel | str, *, rate_hz: float | None = None
) -> WheelOdomConfig:
    """Return the wheel-odometry config for a named noise level,
    optionally overriding the sample rate.
    """
    cfg = WHEEL_PRESETS[_coerce_level(level)]
    return replace(cfg, rate_hz=rate_hz) if rate_hz is not None else cfg


# ──────────────────────────────────────────────────── Simulation ────


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level simulation configuration."""

    duration_s: float = 12.0
    seed: int = 7
    imu_rate_hz: float = 1000.0
    wheel_rate_hz: float = 100.0
