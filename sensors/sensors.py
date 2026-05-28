"""Virtual IMU and wheel-odometry sensors with selectable noise profiles.

The dataclasses describing the sensor noise models (`ImuConfig`,
`WheelOdomConfig`), the named noise profiles (`NoiseLevel`,
`IMU_PRESETS`, `WHEEL_PRESETS`) and the helper lookups
(`imu_config_for`, `wheel_config_for`) all live in the project-wide
:mod:`config` module so the sensors stay a thin wrapper around them.

This module adds the actual *behaviour* on top of those configs:

- `ImuSensor` integrates a drifting bias + white noise on top of an
  analytical acceleration.
- `WheelOdometrySensor` applies a multiplicative scale error and white
  noise on top of an analytical velocity.

Both sensors are constructed either with an explicit config or with a
:class:`NoiseLevel` preset; the preset can also be swapped at runtime
via :meth:`set_noise_level`, which re-samples the initial bias.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# `robot.py` and `config.py` live at the repo root.  Add it to sys.path
# so this module loads cleanly no matter the caller's cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (  # noqa: E402
    IMU_PRESETS,
    WHEEL_PRESETS,
    ImuConfig,
    NoiseLevel,
    WheelOdomConfig,
    _coerce_level,
)
from robot import RobotState  # noqa: E402

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


class ImuSensor:
    """2D accelerometer with white noise and a drifting bias.

    The bias is freshly sampled from `bias_initial_std` on construction
    and again whenever the config is replaced (`set_noise_level` /
    `set_config`), so swapping to a new noise level does not carry over
    the previous run's accumulated drift.
    """

    def __init__(self, config: ImuConfig, rng: np.random.Generator):
        self.rng = rng
        self.config = config
        self.bias = self._sample_initial_bias()

    @classmethod
    def from_noise_level(
        cls,
        level: NoiseLevel | str,
        rng: np.random.Generator,
        *,
        rate_hz: float | None = None,
    ) -> "ImuSensor":
        """Construct from a named preset, optionally overriding the rate."""
        config = IMU_PRESETS[_coerce_level(level)]
        if rate_hz is not None:
            config = replace(config, rate_hz=rate_hz)
        return cls(config, rng)

    def set_noise_level(self, level: NoiseLevel | str) -> None:
        """Swap to a different preset and re-sample the initial bias.

        The configured `rate_hz` is preserved so callers can change the
        noise profile mid-run without disturbing the sampling cadence.
        """
        new_config = IMU_PRESETS[_coerce_level(level)]
        self.set_config(replace(new_config, rate_hz=self.config.rate_hz))

    def set_config(self, config: ImuConfig) -> None:
        self.config = config
        self.bias = self._sample_initial_bias()

    def reset_bias(self) -> None:
        """Re-sample the initial bias without touching the noise stds."""
        self.bias = self._sample_initial_bias()

    def _sample_initial_bias(self) -> np.ndarray:
        return self.rng.normal(0.0, self.config.bias_initial_std, size=2)

    def measure(self, state: RobotState, dt: float) -> np.ndarray:
        """Return a noisy acceleration sample for the given ground-truth state."""
        self.bias += self.rng.normal(
            0.0, self.config.bias_random_walk_std * np.sqrt(dt), size=2
        )
        noise = self.rng.normal(0.0, self.config.accel_noise_std, size=2)
        return state.acceleration + self.bias + noise


class WheelOdometrySensor:
    """Planar wheel-odometry velocity sensor with scale and noise errors."""

    def __init__(self, config: WheelOdomConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    @classmethod
    def from_noise_level(
        cls,
        level: NoiseLevel | str,
        rng: np.random.Generator,
        *,
        rate_hz: float | None = None,
    ) -> "WheelOdometrySensor":
        config = WHEEL_PRESETS[_coerce_level(level)]
        if rate_hz is not None:
            config = replace(config, rate_hz=rate_hz)
        return cls(config, rng)

    def set_noise_level(self, level: NoiseLevel | str) -> None:
        new_config = WHEEL_PRESETS[_coerce_level(level)]
        self.set_config(replace(new_config, rate_hz=self.config.rate_hz))

    def set_config(self, config: WheelOdomConfig) -> None:
        self.config = config

    def measure(self, state: RobotState) -> np.ndarray:
        """Return a noisy planar velocity sample."""
        noise = self.rng.normal(0.0, self.config.velocity_noise_std, size=2)
        return (1.0 + self.config.scale_error) * state.velocity + noise


def make_sensors(
    level: NoiseLevel | str,
    rng: np.random.Generator,
    *,
    imu_rate_hz: float | None = None,
    wheel_rate_hz: float | None = None,
) -> tuple[ImuSensor, WheelOdometrySensor]:
    """Build matched IMU + wheel-odometry sensors at the same noise level."""
    return (
        ImuSensor.from_noise_level(level, rng, rate_hz=imu_rate_hz),
        WheelOdometrySensor.from_noise_level(level, rng, rate_hz=wheel_rate_hz),
    )
